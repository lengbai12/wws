"""Build a minimap own-ship localisation dataset from one recording.

Each sample is a minimap crop plus the label the replay implies for it: the own
ship's position, projected into the minimap by the calibrated affine. Nothing is
hand-annotated and nothing comes from a detector, so the labels are replay truth.

Occupancy is carried per sample, because the three cases are not interchangeable:

  0 present         the own ship exists and its marker must be on the minimap
  1 absent          the own ship is dead, so no marker is drawn
  2 no_information  the replay carries no position yet for this time

Splits are contiguous video-time blocks, never a random per-sample draw: adjacent
frames of one battle are strongly correlated, so a random split would put nearly
identical crops in both training and test. Blocks keep every test sample away
from every training sample in time. A session-level split is stronger still and
needs a second recorded session; the summary records which one was used.

Usage:
  build_minimap_dataset.py --video recordings/<s>/capture.mkv \
      --hud-calibration replays/parsed/calibration/hud-2560x1600.json \
      --minimap-calibration replays/parsed/calibration/minimap-2560x1600.json \
      --position replays/parsed/position-<session>.jsonl \
      --alignment replays/parsed/alignment/<recording>.json \
      --events replays/parsed/events.jsonl --session <replay file name> \
      --step-s 1.0 --out replays/parsed/datasets/minimap-<session>.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import calibrate_hud
import calibrate_minimap as cm
import hud_calibration as hud
import minimap_calibration as mini

PRESENT, ABSENT, NO_INFORMATION = 0, 1, 2
# 1 world unit = 29.666 m, measured from the map span and the ring scale.
WORLD_UNIT_METRES = 29.666

OCCUPANCY_NAMES = {PRESENT: "present", ABSENT: "absent", NO_INFORMATION: "no_information"}

WORKERS = 6

# The split cycles labels over CONTIGUOUS time blocks instead of cutting the
# battle into thirds. Measured reason: this battle's own position moves
# monotonically across the map, so plain thirds came out position-disjoint
# (train u >= 0.646, validation 0.538-0.650, test 0.502-0.537) and every test
# sample asked the model to extrapolate to a place it had never seen. Blocks keep
# adjacent frames together, and cycling keeps every split spread over the whole
# battle and the whole map.
BLOCK_CYCLE = ("train", "train", "train", "train", "train", "validation", "test")
BLOCK_S = 12.0


def crop_rect(frame, rect):
    x, y, w, h = rect
    return frame[y:y + h, x:x + w].copy()


def sample_plan(positions, rate, offset_s, dead_at, settle_s, step_s, duration_s):
    """Return the video times to sample, with their replay clocks."""
    first_clock = positions[0][0]
    start_video = rate * first_clock + offset_s + settle_s
    end_clock = dead_at if dead_at is not None else positions[-1][0]
    end_video = rate * end_clock + offset_s
    plan = []
    video = start_video
    while video <= end_video + 1e-9:
        clock = (video - offset_s) / rate
        plan.append((round(video, 3), round(clock, 3)))
        video += step_s
    return plan, start_video, end_video


def assign_splits(plan, block_s=BLOCK_S, cycle=BLOCK_CYCLE):
    """Label every sample by cycling over contiguous time blocks.

    Consecutive samples share a block, so no near-duplicate crop is split across
    two sets, and the cycle makes every set cover the whole battle.
    """
    if not plan:
        return []
    origin = plan[0][0]
    splits = []
    for video_time, _clock in plan:
        block = int((video_time - origin) // block_s)
        splits.append(cycle[block % len(cycle)])
    return splits


def read_crops(video, rect, plan, workers=WORKERS):
    """Read every planned crop, in parallel, preserving order."""
    def one(item):
        video_time = item[0]
        rgb, width, height = calibrate_hud.read_frame(video, video_time)
        return crop_rect(mini.frame_array(rgb, width, height), rect)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, plan))


def split_label_ranges(rows, splits):
    """Report the label span of every split, so coverage is auditable.

    Disjoint spans mean the splits test extrapolation, which is not what a
    localisation milestone measures, so the numbers are part of the artifact.
    """
    ranges = {}
    for name in sorted(set(str(item) for item in splits)):
        selected = [row for row, split in zip(rows, splits)
                    if str(split) == name and row["u"] is not None]
        if not selected:
            continue
        us = [row["u"] for row in selected]
        vs = [row["v"] for row in selected]
        ranges[name] = {"count": len(selected),
                        "u": [round(min(us), 4), round(max(us), 4)],
                        "v": [round(min(vs), 4), round(max(vs), 4)]}
    return ranges


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--hud-calibration", type=Path, required=True)
    parser.add_argument("--minimap-calibration", type=Path, required=True)
    parser.add_argument("--position", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--events", type=Path, help="used to find the own death")
    parser.add_argument("--session", help="replay file name, with --events")
    parser.add_argument("--step-s", type=float, default=1.0)
    parser.add_argument("--settle-s", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--block-s", type=float, default=BLOCK_S,
                        help="length of one contiguous split block, in video seconds")
    parser.add_argument("--out", type=Path, required=True, help="dataset .npz")
    parser.add_argument("--labels-out", type=Path, help="one JSONL row per sample")
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)

    calibration = hud.load_calibration(args.hud_calibration)
    rect = hud.element_rect(calibration, "minimap")
    if rect is None:
        raise SystemExit("element 'minimap' is not calibrated")
    resolution = calibration.get("resolution")
    minimap_record = json.loads(args.minimap_calibration.read_text(encoding="utf-8"))
    matrix = np.asarray(minimap_record["world_to_pixel"]["matrix"], dtype=float)
    translation = np.asarray(minimap_record["world_to_pixel"]["translation"], dtype=float)

    positions = cm.read_positions(args.position)
    rate, offset_s, _alignment = cm.read_alignment(args.alignment)
    dead_at = None
    if args.events and args.session:
        dead_at = cm.own_death_clock(args.events, args.session)
    plan, start_video, end_video = sample_plan(positions, rate, offset_s, dead_at,
                                               args.settle_s, args.step_s, None)
    print("sampling %d times from %.1f to %.1f s (step %.2f, own death %s)"
          % (len(plan), start_video, end_video, args.step_s, dead_at))
    if len(plan) < 50:
        raise SystemExit("only %d samples planned; a dataset needs more" % len(plan))

    crops = read_crops(args.video, rect, plan, args.workers)
    images = np.stack(crops).astype(np.uint8)

    labels = np.full((len(plan), 2), np.nan, dtype=np.float32)
    occupancy = np.zeros(len(plan), dtype=np.int8)
    rows = []
    for index, (video_time, clock) in enumerate(plan):
        world = cm.world_at(positions, clock)
        state = PRESENT
        if world is None:
            state = NO_INFORMATION
        elif dead_at is not None and clock > dead_at:
            state = ABSENT
        occupancy[index] = state
        if state == PRESENT:
            pixel = mini.apply_affine(matrix, translation, world[0], world[1])
            labels[index] = ((pixel[0] - rect[0]) / rect[2],
                             (pixel[1] - rect[1]) / rect[3])
        rows.append({
            "sample_id": "minimap:%05d" % index,
            "session_id": Path(args.session).stem if args.session else args.video.stem,
            "video_time_s": video_time,
            "replay_clock_s": clock,
            "occupancy": OCCUPANCY_NAMES[int(state)],
            "world_x": None if world is None else round(world[0], 3),
            "world_z": None if world is None else round(world[1], 3),
            "pixel_x": (None if state != PRESENT
                        else round(float(labels[index][0]) * rect[2] + rect[0], 3)),
            "pixel_y": (None if state != PRESENT
                        else round(float(labels[index][1]) * rect[3] + rect[1], 3)),
            "u": None if state != PRESENT else round(float(labels[index][0]), 6),
            "v": None if state != PRESENT else round(float(labels[index][1]), 6),
            "source_video": str(args.video),
            "resolution": resolution,
            "minimap_rect": rect,
            "evidence": "replay_position_projected_by_calibrated_affine",
        })

    splits = np.asarray(assign_splits(plan, args.block_s))
    clocks = np.asarray([item[1] for item in plan], dtype=np.float32)
    times = np.asarray([item[0] for item in plan], dtype=np.float32)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, images=images, labels=labels, occupancy=occupancy,
                        clocks=clocks, video_times=times, splits=splits,
                        minimap_rect=np.asarray(rect, dtype=np.int32))
    print("wrote %s (%.1f MB)" % (args.out, args.out.stat().st_size / 1e6))

    if args.labels_out:
        args.labels_out.parent.mkdir(parents=True, exist_ok=True)
        with args.labels_out.open("w", encoding="utf-8") as handle:
            for index, row in enumerate(rows):
                row["split"] = str(splits[index])
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        print("wrote %s" % args.labels_out)

    present = occupancy == PRESENT
    summary = {
        "samples": int(len(plan)),
        "present": int(present.sum()),
        "absent": int((occupancy == ABSENT).sum()),
        "no_information": int((occupancy == NO_INFORMATION).sum()),
        "crop_size": [int(images.shape[1]), int(images.shape[2])],
        "minimap_rect": rect,
        "resolution": resolution,
        "video_window_s": [start_video, end_video],
        "step_s": args.step_s,
        "own_death_clock": dead_at,
        "split_rule": "interleaved contiguous blocks of %.1f s, cycle %s"
                      % (args.block_s, "|".join(BLOCK_CYCLE)),
        "split_leak_note": "adjacent samples stay inside one block; a session-level "
                           "split needs a second recorded session",
        "split_label_ranges": split_label_ranges(rows, splits),
        "split_counts": {name: int((splits == name).sum())
                         for name in ("train", "validation", "test")},
        "label_source": "replay position projected by the calibrated affine",
        "label_noise_floor_px": minimap_record["stats"]["residual_px_mean"],
        # The affine scale is PIXELS PER WORLD UNIT, not per metre: the position stream
        # is in map world units. Publishing 1/scale as "metres per pixel" is the same
        # unit mislabel this project already paid for once (4.358 published as metres),
        # so both quantities are named separately and the metre figure is derived from
        # the measured 1 world unit = 29.666 m. Independent check: the TTaro minimap
        # mod's own source converts km to world units as 1000/30, i.e. 30 m per unit,
        # which agrees to 1 %.
        "world_units_per_pixel": 1.0 / minimap_record["stats"]["scale_px_per_m_x"],
        "world_unit_metres": WORLD_UNIT_METRES,
        "metres_per_pixel": WORLD_UNIT_METRES
        / minimap_record["stats"]["scale_px_per_m_x"],
        "sources": {
            "video": str(args.video),
            "position": str(args.position),
            "alignment": str(args.alignment),
            "hud_calibration": str(args.hud_calibration),
            "minimap_calibration": str(args.minimap_calibration),
        },
    }
    print("samples %d  present %d  absent %d  no_information %d"
          % (summary["samples"], summary["present"], summary["absent"],
             summary["no_information"]))
    print("splits %s" % summary["split_counts"])
    print("scale: %.3f world units/px = %.1f m/px (1 world unit = %.3f m)"
          "  label noise floor %.3f px"
          % (summary["world_units_per_pixel"], summary["metres_per_pixel"],
             summary["world_unit_metres"], summary["label_noise_floor_px"]))
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n",
                                encoding="utf-8")
        print("wrote %s" % args.summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
