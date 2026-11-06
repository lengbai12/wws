"""Calibrate HUD element rectangles on a captured WoWS frame, and verify them.

Two modes, both against one recorded frame:

  detect   find saturated horizontal bars in a search region, print the
           candidates, and write the chosen one into a calibration record.
  verify   measure a calibrated element on a frame whose expected value is known
           from replay truth, and fail when the measurement disagrees.

Usage:
  calibrate_hud.py detect recordings/<s>/capture.mkv --at 112 \
      --search 0,1000,700,250 --element own_hp_bar --pick 0 \
      --out replays/parsed/calibration/hud-2560x1600.json

  calibrate_hud.py verify recordings/<s>/capture.mkv --at 112 \
      --calibration replays/parsed/calibration/hud-2560x1600.json \
      --element own_hp_bar --expected-fraction 1.0 --tolerance 0.05

A calibration record is written only after detection, and a detected rectangle
is reported as UNVERIFIED until a verify run with a known value passes.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import hud_calibration as hud


def run(argv):
    """Run a command and return its stdout as text."""
    completed = subprocess.run(argv, capture_output=True, check=True)
    return completed.stdout.decode("utf-8", "replace")


def read_frame(video, at_s, rect=None, scale=None):
    """Return (rgb_bytes, width, height) for one frame or crop of a video."""
    argv = hud.cropping_argv(video, at_s, rect, scale)
    raw = subprocess.run(argv, capture_output=True, check=True).stdout
    if rect:
        width, height = rect[2], rect[3]
    else:
        width, height = hud.probe_size(video, run)
    if scale:
        width, height = scale
    expected = width * height * 3
    if len(raw) != expected:
        raise SystemExit("frame is %d bytes, expected %d for %dx%d RGB24"
                         % (len(raw), expected, width, height))
    return raw, width, height


def parse_rect(text):
    parts = [int(p) for p in text.replace(" ", "").split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("expected x,y,w,h")
    return parts


def parse_size(text):
    parts = [int(p) for p in text.lower().replace(" ", "").split("x")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected WxH")
    return parts


def dump_crop(video, at_s, rect, out_path):
    """Write a PNG crop so a human can look at exactly the calibrated pixels."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    argv = ["ffmpeg", "-v", "error", "-ss", "%.6f" % at_s, "-i", str(video),
            "-frames:v", "1", "-vf", "crop=%d:%d:%d:%d" % (rect[2], rect[3], rect[0], rect[1]),
            "-y", str(out_path)]
    subprocess.run(argv, check=True, capture_output=True)
    return out_path


def detect(args):
    rgb, width, height = read_frame(args.video, args.at)
    content = hud.letterbox_rect(rgb, width, height)
    print("frame %dx%d  content[x,y,w,h]=%s" % (width, height, content))
    region = args.search or content
    candidates = hud.candidate_bars(rgb, width, height, region,
                                    min_length=args.min_length)
    if not candidates:
        print("no saturated bar found in region %s" % (region,))
        print("widen --search or lower --min-length; nothing was written")
        return 1
    print("%d candidate bar(s) in region %s, largest first:" % (len(candidates), region))
    for index, candidate in enumerate(candidates):
        print("  [%d] rect=%s colour=%s rows=%d"
              % (index, candidate["rect"], candidate["colour"], candidate["rows"]))
    if args.out is None:
        print("no --out given: detection only, nothing written")
        return 0
    if args.pick >= len(candidates):
        print("--pick %d is out of range" % args.pick)
        return 2
    chosen = candidates[args.pick]["rect"]
    # A detector groups one shade, and a HUD bar is often drawn with a bevel, so
    # the rectangle is grown vertically to hold the whole bar. Horizontal growth
    # is off by default: widening past the bar end would make a full bar measure
    # below 1.0 and hide the error.
    grown = [chosen[0] - args.pad_x, chosen[1] - args.pad_y,
             chosen[2] + 2 * args.pad_x, chosen[3] + 2 * args.pad_y]
    if args.calibration and Path(args.calibration).exists():
        cal = hud.load_calibration(args.calibration)
    else:
        cal = hud.blank_calibration(args.resolution or (width, height), content)
    if args.resolution:
        cal["resolution"] = list(args.resolution)
    if cal.get("content_rect") is None:
        cal["content_rect"] = content
    previous = cal["elements"].get(args.element)
    cal["elements"][args.element] = {
        "rect": grown,
        "fill_direction": args.fill_direction,
        "colour": candidates[args.pick]["colour"],
        "detected_at_s": args.at,
        "verified": False,
        "note": "rect detected on one frame; run the verify mode with a known value",
    }
    path = hud.save_calibration(args.out, cal)
    print("wrote %s: %s rect=%s (detected %s, verified=false)"
          % (path, args.element, grown, chosen))
    if previous:
        print("replaced previous rect %s" % (previous.get("rect"),))
    return 0


def verify(args):
    cal = hud.load_calibration(args.calibration)
    rect = hud.element_rect(cal, args.element)
    if rect is None:
        print("element %r is not calibrated in %s" % (args.element, args.calibration))
        return 2
    element = cal["elements"][args.element]
    direction = element.get("fill_direction") or args.fill_direction
    rgb, width, height = read_frame(args.video, args.at)
    measured = hud.measure_fill_fraction(rgb, width, height, rect, direction)
    if measured is None:
        print("FAIL %s: the bar could not be read at t=%.3f" % (args.element, args.at))
        return 1
    error = abs(measured - args.expected_fraction)
    verdict = "ok" if error <= args.tolerance else "FAIL"
    print("%s %s at t=%.3f: measured=%.4f expected=%.4f error=%.4f tolerance=%.4f"
          % (verdict, args.element, args.at, measured, args.expected_fraction,
             error, args.tolerance))
    if args.crop_out:
        print("crop written to %s" % dump_crop(args.video, args.at, rect, args.crop_out))
    return 0 if verdict == "ok" else 1


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    detect_parser = sub.add_parser("detect", help="find and store a HUD element rectangle")
    detect_parser.add_argument("video", type=Path)
    detect_parser.add_argument("--at", type=float, required=True,
                               help="video time in seconds of the calibration frame")
    detect_parser.add_argument("--search", type=parse_rect,
                               help="x,y,w,h region to search, default the whole content box")
    detect_parser.add_argument("--min-length", type=int, default=40,
                               help="shortest saturated run that counts as a bar")
    detect_parser.add_argument("--element", default="own_hp_bar", choices=hud.ELEMENTS)
    detect_parser.add_argument("--pick", type=int, default=0,
                               help="which candidate to store, largest is 0")
    detect_parser.add_argument("--fill-direction", default="left_to_right",
                               choices=hud.FILL_DIRECTIONS)
    detect_parser.add_argument("--resolution", type=parse_size,
                               help="override the recorded frame size, WxH")
    detect_parser.add_argument("--pad-x", type=int, default=0,
                               help="grow the chosen rectangle left and right by this many px")
    detect_parser.add_argument("--pad-y", type=int, default=0,
                               help="grow the chosen rectangle up and down by this many px")
    detect_parser.add_argument("--calibration", type=Path,
                               help="existing record to update instead of starting empty")
    detect_parser.add_argument("--out", type=Path,
                               help="where to write the calibration record")
    detect_parser.set_defaults(func=detect)

    verify_parser = sub.add_parser("verify", help="measure an element and compare")
    verify_parser.add_argument("video", type=Path)
    verify_parser.add_argument("--at", type=float, required=True)
    verify_parser.add_argument("--calibration", type=Path, required=True)
    verify_parser.add_argument("--element", default="own_hp_bar", choices=hud.ELEMENTS)
    verify_parser.add_argument("--fill-direction", default=None, choices=hud.FILL_DIRECTIONS)
    verify_parser.add_argument("--expected-fraction", type=float, required=True)
    verify_parser.add_argument("--tolerance", type=float, default=0.05)
    verify_parser.add_argument("--crop-out", type=Path,
                               help="also write the calibrated crop as a PNG")
    verify_parser.set_defaults(func=verify)
    return parser


def main(argv=None):
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
