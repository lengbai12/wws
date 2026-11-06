"""Calibrate the minimap projection from replay truth, and check it.

The own ship's world position comes from the replay. Its marker's pixel position
comes from the recording. Pairing the two over frames spread across a battle
gives the affine transform between the battle map and the minimap, which is what
turns a replay position into a pixel label.

Both halves are measured, so a fit that looks good proves both at once: a wrong
position decode or a marker detector that locks onto the wrong white shape both
show up as residuals that no affine can absorb.

Usage:
  calibrate_minimap.py fit --video recordings/<s>/capture.mkv \
      --hud-calibration replays/parsed/calibration/hud-2560x1600.json \
      --position replays/parsed/position-<session>.jsonl \
      --alignment replays/parsed/alignment/<recording>.json \
      --frames 40 --out replays/parsed/calibration/minimap-2560x1600.json

  calibrate_minimap.py verify --video recordings/<s>/capture.mkv \
      --minimap replays/parsed/calibration/minimap-2560x1600.json --frames 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import calibrate_hud
import hud_calibration as hud
import minimap_calibration as mini

SUPPORTED_MINIMAP_PX = 4.0


def read_positions(path):
    """Return [(clock, x, z)] sorted by clock from a position event stream."""
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("event") == "position":
                rows.append((float(record["clock"]), float(record["x"]), float(record["z"])))
    rows.sort(key=lambda item: item[0])
    return rows


def read_alignment(path):
    """Return (rate, offset_s) for the clock-to-video mapping."""
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    mapping = record.get("replay_to_video")
    if not isinstance(mapping, dict):
        raise SystemExit("%s holds no replay_to_video mapping" % path)
    return float(mapping["rate"]), float(mapping["offset_s"]), record


def world_at(positions, clock):
    """Return (x, z) in force at a clock, or None before the first sample."""
    value = None
    for sample_clock, x, z in positions:
        if sample_clock > clock:
            break
        value = (x, z)
    return value


def span(positions, rate, offset_s, settle_s, dead_at=None):
    """Return the video window that holds usable samples."""
    first = rate * positions[0][0] + offset_s + settle_s
    last_clock = dead_at if dead_at is not None else positions[-1][0]
    last = rate * last_clock + offset_s
    return first, last


def sample_times(first, last, count):
    if count <= 1:
        return [first]
    stride = (last - first) / float(count - 1)
    return [first + index * stride for index in range(count)]


def collect_samples(video, rect, times, positions, rate, offset_s):
    """Pair marker pixels with replay world positions.

    Returns (world, pixels, audit, missing) where ``missing`` lists the times at
    which no marker could be read. Those are reported, never replaced by a guess.
    """
    world = []
    pixels = []
    audit = []
    missing = []
    for video_time in times:
        clock = (video_time - offset_s) / rate
        position = world_at(positions, clock)
        if position is None:
            missing.append({"video_time_s": round(video_time, 3),
                            "reason": "no position sample at or before this clock"})
            continue
        rgb, width, height = calibrate_hud.read_frame(video, video_time)
        found = mini.detect_marker(mini.frame_array(rgb, width, height), rect)
        if found is None:
            missing.append({"video_time_s": round(video_time, 3),
                            "reason": "no white marker component matched"})
            continue
        px, py, area = found
        world.append(position)
        pixels.append((px, py))
        audit.append({"video_time_s": round(video_time, 3), "replay_clock_s": round(clock, 3),
                      "world_x": round(position[0], 3), "world_z": round(position[1], 3),
                      "pixel_x": px, "pixel_y": py, "marker_area": area})
    return world, pixels, audit, missing


def own_death_clock(events, session):
    """Own ship's death clock, or None when it survived.

    The own ship's marker leaves the minimap when it dies, so a fit window that
    runs past the death collects samples with no marker to pair.
    """
    import derive_labels
    _player, vehicle_id, _max_hp = derive_labels.own_vehicle(events, session)
    return derive_labels.death_clock(events, session, vehicle_id)


def fit(args):
    calibration = hud.load_calibration(args.hud_calibration)
    rect = hud.element_rect(calibration, "minimap")
    if rect is None:
        raise SystemExit("element 'minimap' is not calibrated in %s" % args.hud_calibration)
    positions = read_positions(args.position)
    if len(positions) < 10:
        raise SystemExit("%s holds %d position samples; that is too few"
                         % (args.position, len(positions)))
    rate, offset_s, alignment = read_alignment(args.alignment)
    dead_at = args.dead_at
    if dead_at is None and args.events and args.session:
        dead_at = own_death_clock(args.events, args.session)
        if dead_at is not None:
            print("own ship dies at replay clock %.3f s (from %s)"
                  % (dead_at, args.events))
    first, last = span(positions, rate, offset_s, args.settle_s, dead_at)
    if last <= first:
        raise SystemExit("the usable video window is empty: %.3f..%.3f s" % (first, last))
    times = sample_times(first, last, args.frames)
    print("minimap rect %s  window %.3f..%.3f s  %d sample times"
          % (rect, first, last, len(times)))
    world, pixels, audit, missing = collect_samples(args.video, rect, times,
                                                    positions, rate, offset_s)
    if len(world) < 3:
        raise SystemExit("only %d samples carried a marker; an affine fit needs three"
                         % len(world))
    # Fit on the largest mutually consistent subset, not on every sample: the marker
    # detector is a colour rule, and a map with near-white terrain (a snow operation)
    # feeds it as many snow patches as ship positions. A plain least-squares fit then
    # averages two different relations and reports a map-scale residual, which looks
    # like a modelling problem instead of the detection problem it is.
    fit_result = mini.ransac_affine(world, pixels, threshold_px=SUPPORTED_MINIMAP_PX)
    if fit_result is None:
        print("no consistent subset: none of the %d marker readings agrees with an "
              "affine within %.1f px" % (len(world), SUPPORTED_MINIMAP_PX))
        print("VERDICT FAIL (the marker detector is locking onto something that is not "
              "the ship; no transform is written so the next run refits)")
        if args.samples_out:
            args.samples_out.parent.mkdir(parents=True, exist_ok=True)
            with args.samples_out.open("w", encoding="utf-8") as handle:
                for row in audit:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
            print("wrote %s (%d paired samples, for diagnosis)"
                  % (args.samples_out, len(audit)))
        return 1
    matrix, translation, inliers, distances = fit_result
    outliers = np.setdiff1d(np.arange(len(world)), inliers)
    stats = mini.describe(matrix, translation, distances[inliers])
    print("fit: scale_x %.3f px/m  scale_z %.3f px/m  ratio %.4f  rotation %.3f deg"
          % (stats["scale_px_per_m_x"], stats["scale_px_per_m_z"],
             stats["scale_ratio"], stats["rotation_deg"]))
    print("residual px: mean %.3f  p95 %.3f  max %.3f  over %d consistent samples"
          % (stats["residual_px_mean"], stats["residual_px_p95"],
             stats["residual_px_max"], stats["samples"]))
    print("consistent readings %d of %d (%.1f%%); %d rejected as outliers"
          % (len(inliers), len(world), 100.0 * len(inliers) / len(world), len(outliers)))
    if missing:
        print("no marker read at %d of %d times (reported, not guessed)"
              % (len(missing), len(times)))
    ok = stats["residual_px_mean"] <= SUPPORTED_MINIMAP_PX
    if not 0.9 <= (stats["scale_ratio"] or 0) <= 1.1:
        print("WARNING the two axis scales disagree by more than 10 percent: %s"
              % stats["scale_ratio"])
    print("VERDICT %s (mean residual limit %.1f px)"
          % ("pass" if ok else "FAIL", SUPPORTED_MINIMAP_PX))
    if not ok:
        # A failed fit must leave NO artifact behind: the orchestrator decides whether
        # to run a stage by asking whether its output exists, so a written failure
        # would be consumed as a valid transform by the next pass. That happened —
        # a session was built against a 114 px affine and only the dataset's absurd
        # "label noise floor" betrayed it.
        print("not writing %s: only a PASSING fit may become an input" % args.out)
        if args.samples_out:
            args.samples_out.parent.mkdir(parents=True, exist_ok=True)
            with args.samples_out.open("w", encoding="utf-8") as handle:
                for row in audit:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
        return 1

    record = {
        "schema_version": 1,
        "resolution": calibration.get("resolution"),
        "minimap_rect": rect,
        "world_to_pixel": {
            "matrix": [[float(v) for v in row] for row in matrix],
            "translation": [float(v) for v in translation],
            "model": "pixel = matrix @ (world_x, world_z) + translation",
        },
        "stats": stats,
        # How many marker readings agreed with the written transform, and how many did
        # not: on a map with near-white terrain some readings are terrain, so the
        # rejection rate is part of what this artifact means.
        "marker_consistency": {
            "consistent": int(len(inliers)),
            "total": int(len(world)),
            "rejected": int(len(outliers)),
            "threshold_px": SUPPORTED_MINIMAP_PX,
            "rejected_times_s": [round(float(audit[index]["replay_clock_s"]), 3)
                                 for index in outliers
                                 if index < len(audit)
                                 and audit[index].get("replay_clock_s") is not None],
        },
        "video_window_s": [round(first, 3), round(last, 3)],
        "source": {
            "video": str(args.video),
            "position": str(args.position),
            "alignment": str(args.alignment),
            "hud_calibration": str(args.hud_calibration),
        },
        "missing": missing,
    }
    inverse, inverse_translation = mini.inverse_affine(matrix, translation)
    record["pixel_to_world"] = {
        "matrix": [[float(v) for v in row] for row in inverse],
        "translation": [float(v) for v in inverse_translation],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote %s" % out)
    if args.samples_out:
        args.samples_out.parent.mkdir(parents=True, exist_ok=True)
        with args.samples_out.open("w", encoding="utf-8") as handle:
            for row in audit:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        print("wrote %s (%d paired samples)" % (args.samples_out, len(audit)))
    return 0 if ok else 1


def load_minimap(path):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    matrix = np.asarray(record["world_to_pixel"]["matrix"], dtype=float)
    translation = np.asarray(record["world_to_pixel"]["translation"], dtype=float)
    return record, matrix, translation


def verify(args):
    record, matrix, translation = load_minimap(args.minimap)
    rect = record["minimap_rect"]
    positions = read_positions(args.position)
    rate, offset_s, _alignment = read_alignment(args.alignment)
    first, last = span(positions, rate, offset_s, args.settle_s)
    times = sample_times(first, last, args.frames)
    world, pixels, audit, missing = collect_samples(args.video, rect, times,
                                                    positions, rate, offset_s)
    if not world:
        print("no marker read: nothing to check")
        return 1
    predicted = [mini.apply_affine(matrix, translation, x, z) for x, z in world]
    residuals = np.linalg.norm(np.asarray(predicted) - np.asarray(pixels), axis=1)
    print("verify over %d samples: mean %.3f px  p95 %.3f px  max %.3f px  unread %d"
          % (len(world), residuals.mean(), np.percentile(residuals, 95),
             residuals.max(), len(missing)))
    for row, residual in sorted(zip(audit, residuals.tolist()),
                                key=lambda item: -item[1])[:5]:
        print("  worst %.3fs clock %.1f world(%.1f, %.1f) pixel(%d, %d) residual %.3f px"
              % (row["video_time_s"], row["replay_clock_s"], row["world_x"], row["world_z"],
                 row["pixel_x"], row["pixel_y"], residual))
    ok = float(residuals.mean()) <= SUPPORTED_MINIMAP_PX
    print("VERDICT %s (mean residual limit %.1f px)"
          % ("pass" if ok else "FAIL", SUPPORTED_MINIMAP_PX))
    return 0 if ok else 1


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    for name, function in (("fit", fit), ("verify", verify)):
        node = sub.add_parser(name)
        node.add_argument("--video", type=Path, required=True)
        node.add_argument("--position", type=Path, required=True)
        node.add_argument("--alignment", type=Path, required=True)
        node.add_argument("--frames", type=int, default=40)
        node.add_argument("--settle-s", type=float, default=20.0,
                          help="skip this many seconds of video before the first sample")
        node.add_argument("--dead-at", type=float,
                          help="own ship death clock; the fit window stops there")
        node.add_argument("--events", type=Path,
                          help="event stream used to find the own death automatically")
        node.add_argument("--session", help="replay file name, with --events")
        node.set_defaults(func=function, settle_s=20.0)
        if name == "fit":
            node.add_argument("--hud-calibration", type=Path, required=True)
            node.add_argument("--out", type=Path, required=True)
            node.add_argument("--samples-out", type=Path)
        else:
            node.add_argument("--minimap", type=Path, required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
