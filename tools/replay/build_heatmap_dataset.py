"""Turn the minimap inventory into heatmap targets for multi-class training.

One sample is a minimap crop plus, for every class, a Gaussian heatmap whose
peaks sit on the markers the replay says are drawn. A heatmap is used rather than
a list of positions because a frame holds SEVERAL markers of the same class
(eleven teammates, several visible enemies), and a fixed-length coordinate list
cannot say that.

Channel set, all replay-derived:

  0 own             1 per frame while alive
  1 friendly        1 per teammate, always drawn
  2 enemy_visible   1 per enemy the TEAM has detected
  3 wreck           1 per sunken ship, kept for the rest of the battle
  4..6 zone_A/B/C   map-static capture points, ownership recorded separately

Deliberately NOT a channel: ``enemy_not_visible``. Those positions are enemies the
minimap does NOT draw, so they are the false-positive test for the model instead of
a target it could learn to paint anyway.

Also NOT a channel, and not a negative either: ``enemy_unknown``. Those are vehicles
whose side or visibility the replay never established, so they may be painted
neither as a target nor asserted as an absence -- an absence nobody observed is not
supervision. ``build_minimap_inventory.py`` emits that class rather than writing a
missing visibility packet into ``enemy_not_visible``.

The range rings are not points: a ring is determined by the own ship's position and
the map scale, so the learnable part is WHICH rings the player has enabled. That is
carried as a multi-label target beside the heatmaps.

Usage:
  build_heatmap_dataset.py --inventory replays/parsed/inventory/<s>.inventory.jsonl \
      --dataset replays/parsed/datasets/minimap-<s>.npz \
      --out replays/parsed/datasets/heatmap-<s>.npz \
      --summary replays/parsed/datasets/heatmap-<s>.summary.json \
      [--grid 88] [--sigma 1.5]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CLASSES = ("own", "friendly", "enemy_visible", "wreck", "zone_A", "zone_B", "zone_C",
           "squadron", "aim", "lock")

# Which inventory class feeds which channel. zone_* needs the letter as well.
CLASS_SOURCE = {
    "own": "own",
    "friendly": "friendly",
    "enemy_visible": "enemy_visible",
    "wreck": "wreck",
    "zone_A": "zone",
    "zone_B": "zone",
    "zone_C": "zone",
    "squadron": "squadron",
    "aim": "aim",
    "lock": "lock",
}

NEGATIVE_CLASS = "enemy_not_visible"

# Which availability flag in the inventory summary governs which channel. The flag is
# per SESSION, not per sample: it says whether that session's data made the class
# measurable at all -- a map with no zone calibration cannot place A/B/C, a replay with
# no lock log cannot carry a lock. The four packet-derived channels (own, friendly,
# enemy_visible, wreck) are always measurable, so they are absent here and stay valid.
#
# Why this exists: an unmeasurable class arrives as an all-zero target, and an all-zero
# target is indistinguishable from "confirmed absent" once the loss sees it. Training
# then teaches "this class never exists" when the truth is "this session never said",
# and the two disagree the moment a measurable session joins the corpus. The mask keeps
# them apart; `train_minimap_heatmap.py` multiplies the per-element loss by it.
CLASS_AVAILABILITY = {
    "zone_A": "zone", "zone_B": "zone", "zone_C": "zone",
    "squadron": "squadron", "aim": "aim", "lock": "lock",
}
ALWAYS_MEASURABLE = ("own", "friendly", "enemy_visible", "wreck")


def gaussian(grid, centre, sigma):
    """Add a normalised Gaussian centred at (column, row) onto a grid."""
    rows = np.arange(grid.shape[0], dtype=np.float32)[:, None]
    columns = np.arange(grid.shape[1], dtype=np.float32)[None, :]
    distance = ((rows - centre[1]) ** 2 + (columns - centre[0]) ** 2) / (2.0 * sigma * sigma)
    patch = np.exp(-distance, dtype=np.float32)
    peak = patch.max()
    if peak > 0:
        patch = patch / peak
    np.maximum(grid, patch, out=grid)
    return grid


def load_inventory(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def targets_for(row, classes, grid, sigma, rect):
    """Return (heatmaps, ground truth list, ring vector) for one sample."""
    heatmaps = np.zeros((len(classes), grid, grid), dtype=np.float32)
    truth = []
    for marker in row["markers"]:
        kind = marker["class"]
        channel = None
        if kind == "zone":
            name = "zone_%s" % marker.get("letter")
            channel = classes.index(name) if name in classes else None
        elif kind in classes:
            channel = classes.index(kind)
        if channel is None:
            continue
        column = marker["u"] * (grid - 1)
        row_index = marker["v"] * (grid - 1)
        gaussian(heatmaps[channel], (column, row_index), sigma)
        truth.append((channel, marker["u"], marker["v"]))
    rings = np.zeros(len(RING_LABELS), dtype=np.float32)
    for label in row["markers"]:
        if label["class"] == "range_ring":
            index = RING_LABELS.index(float(label["label_km"]))
            rings[index] = 1.0
    return heatmaps, truth, rings


RING_LABELS = [8.0, 9.9, 12.0, 16.8, 20.1]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True,
                        help="minimap dataset holding the crops and the splits")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--grid", type=int, default=88)
    parser.add_argument("--sigma", type=float, default=1.5)
    args = parser.parse_args(argv)

    data = np.load(args.dataset, allow_pickle=False)
    images = data["images"]
    splits = np.asarray([str(item) for item in data["splits"]])
    rect = [int(v) for v in data["minimap_rect"]]
    rows = load_inventory(args.inventory)
    if len(rows) != len(images):
        raise SystemExit("inventory holds %d samples but the dataset holds %d; they must "
                         "describe the same frames" % (len(rows), len(images)))

    # Read the availability BEFORE the targets are built, because it decides which of
    # them are supervised at all. It comes from the inventory summary, which is written
    # by the tool that knows what the session could measure.
    inventory_summary = Path(str(args.inventory).replace(".jsonl", ".summary.json"))
    availability = {}
    inventory_map = None
    if inventory_summary.exists():
        recorded = json.loads(inventory_summary.read_text(encoding="utf-8"))
        availability = recorded.get("class_availability") or {}
        inventory_map = recorded.get("map")

    # The channel count belongs to the artifact NAME, because a model's output shape
    # is part of the input contract. Check the name against the channels actually
    # written, so a caller running with a stale count fails loudly instead of leaving a
    # file whose contents contradict its name -- the failure mode this project keeps
    # paying for (a 7-channel set silently reused for a 9-channel run).
    classes = CLASSES
    if args.out:
        stem = Path(args.out).name
        marker = "-c"
        if marker in stem:
            declared = stem.rsplit(marker, 1)[1].split(".", 1)[0]
            if declared.isdigit() and int(declared) != len(classes):
                raise SystemExit(
                    "output name %s declares %s channels but this build writes %d (%s); "
                    "the class count in the name and in the data must agree"
                    % (stem, declared, len(classes), ",".join(classes)))
    heatmaps = np.zeros((len(rows), len(classes), args.grid, args.grid), dtype=np.float32)
    rings = np.zeros((len(rows), len(RING_LABELS)), dtype=np.float32)
    max_markers = max(len(row["markers"]) for row in rows)
    truth = np.full((len(rows), max_markers, 3), np.nan, dtype=np.float32)
    counts = np.zeros((len(rows), len(classes)), dtype=np.int16)
    negatives = np.zeros((len(rows), max_markers, 2), np.float32)
    negative_counts = np.zeros(len(rows), dtype=np.int16)

    for index, row in enumerate(rows):
        sample_maps, sample_truth, ring_vector = targets_for(
            row, classes, args.grid, args.sigma, rect)
        heatmaps[index] = sample_maps
        rings[index] = ring_vector
        for position, (channel, u, v) in enumerate(sample_truth):
            truth[index, position] = (channel, u, v)
            counts[index, channel] += 1
        negs = [(m["u"], m["v"]) for m in row["markers"] if m["class"] == NEGATIVE_CLASS]
        for position, (u, v) in enumerate(negs[:max_markers]):
            negatives[index, position] = (u, v)
        negative_counts[index] = min(len(negs), max_markers)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    session_ids = np.asarray([str(row.get("session_id") or "") for row in rows])
    # Per sample AND per class, so the merged corpus keeps the information per session:
    # 0 means "this session never said", and the loss skips those elements entirely.
    valid = np.ones((len(rows), len(classes)), dtype=np.float32)
    for channel, name in enumerate(classes):
        flag = CLASS_AVAILABILITY.get(name)
        if flag is not None and availability.get(flag) is False:
            valid[:, channel] = 0.0
    ring_valid = np.ones(len(rows), dtype=np.float32)
    if availability.get("range_ring") is False:
        ring_valid[:] = 0.0
    np.savez_compressed(args.out, heatmaps=heatmaps, truth=truth, counts=counts,
                        negatives=negatives, negative_counts=negative_counts,
                        rings=rings, valid=valid, ring_valid=ring_valid,
                        splits=splits, classes=np.asarray(classes),
                        session_ids=session_ids,
                        ring_labels=np.asarray(RING_LABELS),
                        grid=np.asarray([args.grid, args.grid]),
                        sigma=np.asarray([args.sigma]))
    print("wrote %s (%.1f MB)  grid %dx%d  sigma %.2f"
          % (args.out, args.out.stat().st_size / 1e6, args.grid, args.grid, args.sigma))
    print("每类贡献的标记总数:")
    for channel, name in enumerate(classes):
        print("   %-14s %6d   出现在 %d 个采样里"
              % (name, int(counts[:, channel].sum()), int((counts[:, channel] > 0).sum())))
    print("负例（确认未画的敌人位置）: %d 个采样里有 %d 个位置"
          % (int((negative_counts > 0).sum()), int(negative_counts.sum())))
    print("射程环标签的开启比例: %s"
          % {str(label): round(float(rings[:, i].mean()), 3)
             for i, label in enumerate(RING_LABELS)})

    availability = availability or {}
    if args.summary:
        summary = {
            "samples": len(rows),
            "classes": list(classes),
            "grid": [args.grid, args.grid],
            "sigma": args.sigma,
            "crop_rect": rect,
            "channel_totals": {name: int(counts[:, i].sum()) for i, name in enumerate(classes)},
            "channel_present_samples": {name: int((counts[:, i] > 0).sum())
                                        for i, name in enumerate(classes)},
            "negative_positions": int(negative_counts.sum()),
            "negative_samples": int((negative_counts > 0).sum()),
            "supervised_channels": {name: bool(valid[0, i]) for i, name in enumerate(classes)}
                                   if len(rows) else {name: False for name in classes},
            "ring_supervised": bool(ring_valid[0]) if len(rows) else False,
            "validity_note": ("a channel the session could not measure is carried as "
                              "valid=0 and is NOT supervised: its all-zero target must "
                              "not be trained as a confirmed absence"),
            "ring_labels": RING_LABELS,
            "ring_enabled_fraction": {str(label): round(float(rings[:, i].mean()), 4)
                                      for i, label in enumerate(RING_LABELS)},
            "split_counts": {name: int((splits == name).sum())
                             for name in ("train", "validation", "test")},
            "note": ("heatmaps are per class; enemy_not_visible is carried as the "
                     "false-positive test rather than a target; range rings are a "
                     "multi-label target because a ring is not a point"),
            "session_ids": sorted({str(row.get("session_id") or "") for row in rows}),
            "map": inventory_map,
            "class_availability": availability,
            "sources": {"inventory": str(args.inventory), "dataset": str(args.dataset)},
        }
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n",
                                encoding="utf-8")
        print("wrote %s" % args.summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
