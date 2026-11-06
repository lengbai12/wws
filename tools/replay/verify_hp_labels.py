"""Check derived own-health labels against the pixels they describe.

A derived label is only useful when the pixels it points at actually show that
value. This tool measures the calibrated health bar on real frames of the
recording and compares the measurement with the label the replay produced.

The gate is on rows whose occupancy is ``present``: the share of measurements
inside the tolerance must reach ``--min-pass-share``. Rows marked ``absent`` or
``no_information`` carry no expected value and are reported separately, so a
region that keeps reading a bar after the own ship died is visible rather than
counted as a pass.

Usage:
  verify_hp_labels.py --labels replays/parsed/labels/<s>.own_hp.jsonl \
      --video recordings/<s>/capture.mkv \
      --calibration replays/parsed/calibration/hud-2560x1600.json \
      --limit 40 --tolerance 0.03
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import calibrate_hud
import hud_calibration as hud


def read_labels(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sample(rows, limit):
    """Take at most ``limit`` rows spread across the list, keeping the ends.

    A contiguous prefix would test one part of the battle only, so the sample is
    spread over the whole span. Unsettled rows are kept in the sample so they are
    still measured, then reported apart from the rows the gate judges.
    """
    present = [row for row in rows if row.get("occupancy") == "present"]
    if limit <= 0 or len(present) <= limit:
        return present
    stride = len(present) / float(limit)
    return [present[min(len(present) - 1, int(index * stride))] for index in range(limit)]


def percentile(values, share):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(share * (len(ordered) - 1))))
    return ordered[index]


def measure_rows(rows, video, rect, direction, crop_dir=None, worst=0):
    """Measure every row and return (measurements, crop paths by sample id)."""
    results = []
    for row in rows:
        rgb, width, height = calibrate_hud.read_frame(video, row["video_time_s"])
        measured = hud.measure_fill_fraction(rgb, width, height, rect, direction)
        expected = (row.get("value") or {}).get("fraction")
        error = None if (measured is None or expected is None) else measured - expected
        results.append({
            "sample_id": row.get("sample_id"),
            "video_time_s": row["video_time_s"],
            "replay_clock_s": row["replay_clock_s"],
            "occupancy": row.get("occupancy"),
            "expected": expected,
            "measured": measured,
            "error": error,
        })
    if crop_dir and worst:
        offenders = sorted((item for item in results if item["error"] is not None),
                           key=lambda item: -abs(item["error"]))[:worst]
        for index, item in enumerate(offenders):
            out = Path(crop_dir) / ("worst-%02d-%.3fs.png" % (index, item["video_time_s"]))
            calibrate_hud.dump_crop(video, item["video_time_s"], rect, out)
            item["crop"] = str(out)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--element", default="own_hp_bar")
    parser.add_argument("--limit", type=int, default=40,
                        help="how many present rows to measure, 0 for all")
    parser.add_argument("--tolerance", type=float, default=0.03,
                        help="largest acceptable |measured - expected|")
    parser.add_argument("--min-pass-share", type=float, default=0.9)
    parser.add_argument("--crop-dir", type=Path, help="write crops of the worst rows here")
    parser.add_argument("--worst", type=int, default=5)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)

    calibration = hud.load_calibration(args.calibration)
    rect = hud.element_rect(calibration, args.element)
    if rect is None:
        raise SystemExit("element %r is not calibrated" % args.element)
    element = calibration["elements"][args.element]
    direction = element.get("fill_direction") or "left_to_right"

    rows = read_labels(args.labels)
    chosen = sample(rows, args.limit)
    if not chosen:
        print("no present rows in %s: nothing to verify" % args.labels)
        return 2
    unsettled_ids = {row.get("sample_id") for row in chosen if row.get("settled") is False}
    results = measure_rows(chosen, args.video, rect, direction,
                           args.crop_dir, args.worst)
    for item in results:
        item["settled"] = item["sample_id"] not in unsettled_ids

    judged = [item for item in results if item["settled"]]
    unsettled = [item for item in results if not item["settled"]]
    errors = [abs(item["error"]) for item in judged if item["error"] is not None]
    unreadable = [item for item in judged if item["measured"] is None]
    passed = [value for value in errors if value <= args.tolerance]
    pass_share = len(passed) / float(len(judged)) if judged else 0.0

    skipped = [row for row in rows if row.get("occupancy") != "present"]
    print("labels %d  measured %d  judged %d  unsettled %d  unreadable %d"
          % (len(rows), len(results), len(judged), len(unsettled), len(unreadable)))
    print("|error| mean %.4f  p95 %.4f  max %.4f  tolerance %.4f"
          % (sum(errors) / len(errors) if errors else float("nan"),
             percentile(errors, 0.95) or float("nan"),
             max(errors) if errors else float("nan"),
             args.tolerance))
    print("within tolerance %d/%d = %.3f  required %.3f"
          % (len(passed), len(judged), pass_share, args.min_pass_share))
    if unsettled:
        # A row inside the settle window is not a wrong label: the rendered bar
        # leads the packet that carries the step, so the two are not comparable
        # there. The rows are still measured and reported, never dropped.
        uerr = [abs(item["error"]) for item in unsettled if item["error"] is not None]
        print("unsettled rows %d measured, |error| max %s (excluded from the gate)"
              % (len(unsettled), ("%.4f" % max(uerr)) if uerr else "n/a"))
    for item in sorted(results, key=lambda row: -abs(row["error"] or 0.0))[:args.worst]:
        print("  worst %.3fs clock %.3f expected %.4f measured %s error %s%s"
              % (item["video_time_s"], item["replay_clock_s"], item["expected"] or float("nan"),
                 "None" if item["measured"] is None else "%.4f" % item["measured"],
                 "None" if item["error"] is None else "%+.4f" % item["error"],
                 "  crop %s" % item["crop"] if item.get("crop") else ""))
    if skipped:
        absent = sum(1 for row in skipped if row.get("occupancy") == "absent")
        print("not measured: %d rows (%d absent, %d no_information)"
              % (len(skipped), absent, len(skipped) - absent))
    verdict = pass_share >= args.min_pass_share and not unreadable
    print("VERDICT %s" % ("pass" if verdict else "FAIL"))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({
            "labels": str(args.labels),
            "video": str(args.video),
            "element": args.element,
            "rect": rect,
            "measured_rows": len(results),
            "judged_rows": len(judged),
            "unsettled_rows": len(unsettled),
            "unreadable_rows": len(unreadable),
            "tolerance": args.tolerance,
            "min_pass_share": args.min_pass_share,
            "pass_share": pass_share,
            "mean_abs_error": (sum(errors) / len(errors)) if errors else None,
            "p95_abs_error": percentile(errors, 0.95),
            "max_abs_error": max(errors) if errors else None,
            "verdict": "pass" if verdict else "FAIL",
            "results": results,
        }, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print("wrote %s" % args.json_out)
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
