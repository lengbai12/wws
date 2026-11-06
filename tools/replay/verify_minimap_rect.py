#!/usr/bin/env python3
"""Verify that a recording's minimap panel is the size the calibration assumes.

The pipeline asserts the panel rectangle: ``panel_rect.txt`` is written from the
calibration, never measured from the video. So a minimap whose size changed -- it
is an in-game setting, and the ``+`` key resizes it -- would produce labels cropped
from the wrong rectangle, and nothing would report it. This tool measures instead
of assuming.

WHAT IS MEASURABLE, and why only this (all measured on real recordings):

* The panel's TOP border is the edge that encodes the panel's size, and it is
  readable: the minimap's map meets the world behind it with a gradient 16 to 31
  times the local background. A smaller minimap moves this border down, so it is
  read at the calibration's row, within a few pixels.
* The panel's BOTTOM border is strong in every frame (20 to 38 times), but it
  coincides with the bottom of the 2560x1440 content area, so it checks the frame
  layout rather than the minimap's size. Kept as a secondary check for that.
* The panel's LEFT border is NOT readable: the sea behind it can be the same dark
  teal as the minimap's own water. It is reported as unmeasured rather than
  guessed at, and the top border carries the verdict instead.

A frame whose background is too bright to read the border is reported unreadable
and left out of the count, never counted as a failure.

Usage:
  /usr/bin/python3 tools/replay/verify_minimap_rect.py --video recordings/<s>/capture.mkv \
      --calibration replays/parsed/calibration/hud-2560x1600-minimapmax.json
  /usr/bin/python3 tools/replay/verify_minimap_rect.py --video ... --rect 1836,796,720,723
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

# How far from the expected border a read may land and still agree. The panel is
# hundreds of pixels tall, so a few pixels is tolerance enough for a frame
# arriving at a slightly different moment.
TOLERANCE_PX = 4
# The band searched around each expected border, also the base of the strength
# ratio: the border gradient must beat the band's median by this factor.
PAD_PX = 48
MIN_STRENGTH_RATIO = 3.0
# The verdict counts EVERY sampled frame, not only the readable ones. Measured on
# real recordings: a panel of the calibrated size scores 7 to 15 at the expected
# row, while a smaller panel scores under 2 and can only reach 3 or more by
# coincidence in a frame or two. Counting every frame keeps those coincidences
# from carrying a verdict.
AGREE_SHARE = 2 / 3
DEFAULT_FRAMES = 7
# Frames are sampled inside this window, away from the start-up and the harbour.
WINDOW = (0.05, 0.9)


def panel_rect(calibration: Path | None, explicit: str | None) -> list[int]:
    """The panel rectangle, from the calibration the pipeline itself reads.

    ``auto_learn`` takes it from ``elements.minimap.rect``, so this does too:
    a second interpretation of the same file would let the two disagree.
    """
    if explicit:
        values = [int(v) for v in explicit.split(",")]
        if len(values) != 4:
            raise SystemExit("--rect wants x,y,w,h")
        return values
    if calibration is None:
        raise SystemExit("give --rect or --calibration")
    payload = json.loads(Path(calibration).read_text(encoding="utf-8"))
    rect = (payload.get("elements") or {}).get("minimap", {}).get("rect")
    if rect is None:
        raise SystemExit(
            f"{calibration} has no elements.minimap.rect")
    return [int(v) for v in rect]


def video_size(video: Path) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=True).stdout.strip()
    width, height = out.split(",")[:2]
    return int(width), int(height)


def duration(video: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=True).stdout.strip()
    return float(out)


def gray_crop(video: Path, at_s: float, box: list[int]) -> list[list[float]]:
    """The crop at ``box`` as one luminance sample per pixel, or None."""
    x, y, width, height = box
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{at_s:.3f}", "-i", str(video),
         "-frames:v", "1", "-vf", f"crop={width}:{height}:{x}:{y},format=gray",
         "-f", "rawvideo", "-"],
        capture_output=True).stdout
    if len(raw) != width * height:
        return None
    return [list(raw[row * width:(row + 1) * width]) for row in range(height)]


def border_profiles(rows: list[list[float]], vertical: bool,
                    span: tuple[int, int]) -> list[float]:
    """Mean absolute gradient along one axis, over the other axis' span."""
    low, high = span
    if vertical:
        height = len(rows)
        low, high = max(0, low), min(height, high)
        return [sum(abs(rows[row][col + 1] - rows[row][col])
                    for row in range(low, high)) / max(1, high - low)
                for col in range(len(rows[0]) - 1)]
    width = len(rows[0])
    low, high = max(0, low), min(width, high)
    return [sum(abs(rows[row + 1][col] - rows[row][col])
                for col in range(low, high)) / max(1, high - low)
            for row in range(len(rows) - 1)]


def strongest(profile: list[float], expected: int, pad: int) -> dict:
    """The strongest gradient AT the expected border, and its band ratio.

    Restricted to a few pixels of the expected position: a border somewhere else
    is a different panel size, which must be reported rather than adopted. The
    ratio against the band's median says whether a border is there at all, so a
    bright background is reported as an unreadable frame instead of a failure.
    """
    low = max(0, expected - TOLERANCE_PX)
    high = min(len(profile), expected + TOLERANCE_PX + 1)
    band = profile[max(0, expected - pad):min(len(profile), expected + pad + 1)]
    if high - low < 2 or not band:
        return {"position": None, "strength": 0.0, "ratio": 0.0}
    best = max(range(low, high), key=lambda index: profile[index])
    typical = statistics.median(band) or 1e-9
    return {"position": best, "strength": round(profile[best], 3),
            "ratio": round(profile[best] / typical, 2)}


def measure(video: Path, rect: list[int], frames: int) -> dict:
    width, height = video_size(video)
    total = duration(video)
    x0, y0, w, h = rect
    # One crop holds both borders, so each sampled frame costs one decode. It
    # reaches further up than the search band so that a border which moved by a
    # panel resize is still inside the crop, where it can be named in the report.
    left = max(0, x0 - PAD_PX)
    top = max(0, y0 - 3 * PAD_PX)
    right = min(width, x0 + w + PAD_PX)
    bottom = min(height, y0 + h + PAD_PX)
    box = [left, top, right - left, bottom - top]
    inside_columns = (x0 + 10 - left, x0 + w - 10 - left)
    expectations = {"top": y0 - top, "bottom": y0 + h - top}
    samples = []
    for index in range(frames):
        fraction = WINDOW[0] + (WINDOW[1] - WINDOW[0]) * (
            index / max(1, frames - 1))
        at = total * fraction
        rows = gray_crop(video, at, box)
        if rows is None:
            samples.append({"at_s": round(at, 3), "readable": False,
                            "reason": "the frame could not be decoded"})
            continue
        profile = border_profiles(rows, False, inside_columns)
        reads = {name: strongest(profile, value, PAD_PX)
                 for name, value in expectations.items()}
        readable = reads["top"]["ratio"] >= MIN_STRENGTH_RATIO
        agrees = {name: bool(read["position"] is not None
                             and abs(read["position"] - expectations[name])
                             <= TOLERANCE_PX)
                  for name, read in reads.items()}
        # When the expected row holds no border, list the strongest edges in the
        # crop instead: a moved border is a different panel size, and seeing where
        # the edges actually are turns a refusal into a diagnosis. A plain "the
        # strongest one" is not enough, because the bottom border of the content
        # area usually wins that search.
        ranked = sorted(range(len(profile)), key=lambda index: -profile[index])
        peaks = []
        for index in ranked:
            if all(abs(index - taken) > PAD_PX for taken in peaks):
                peaks.append(index)
            if len(peaks) == 3:
                break
        median = statistics.median(profile) or 1e-9
        sample = {
            "at_s": round(at, 3), "readable": readable,
            "strongest_edges_in_frame": [
                {"row": index + top, "ratio": round(profile[index] / median, 2)}
                for index in peaks],
            "expected": expectations, "measured": reads, "agrees": agrees,
            "reason": None if readable else (
                "no border at the expected row; the strongest edges in the crop sit "
                "at frame rows "
                + ", ".join(f"{index + top} (x{profile[index] / median:.1f})"
                            for index in peaks)
                + f", and the panel top is expected at row {expectations['top'] + top}"),
        }
        samples.append(sample)
    readable = [s for s in samples if s.get("readable")]
    needed = max(1, int(AGREE_SHARE * len(samples) + 0.999))
    top_agreeing = [s for s in samples if s.get("readable") and s["agrees"]["top"]]
    bottom_agreeing = [s for s in samples if s.get("readable") and s["agrees"]["bottom"]]
    verdict = ("PASS" if len(top_agreeing) >= needed
               and len(bottom_agreeing) >= needed else "FAIL")
    return {
        "video": str(video),
        "resolution": [width, height],
        "rect": rect,
        "frames_sampled": len(samples),
        "frames_readable": len(readable),
        "top_agreeing": len(top_agreeing),
        "bottom_agreeing": len(bottom_agreeing),
        "frames_required": needed,
        "tolerance_px": TOLERANCE_PX,
        "min_strength_ratio": MIN_STRENGTH_RATIO,
        "left_border": "not measured: the sea behind it can match the minimap's "
                       "own water colour, so it carries no readable gradient",
        "samples": samples,
        "verdict": verdict,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--rect", help="x,y,w,h; overrides the calibration")
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    rect = panel_rect(args.calibration, args.rect)
    result = measure(args.video, rect, args.frames)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"{args.video}")
        print(f"  expected panel {rect} on "
              f"{result['resolution'][0]}x{result['resolution'][1]}")
        for sample in result["samples"]:
            if not sample.get("readable"):
                print(f"  t={sample['at_s']:8.2f}  unreadable ({sample['reason']})")
                continue
            parts = []
            for name in ("top", "bottom"):
                read = sample["measured"][name]
                mark = "ok" if sample["agrees"][name] else "MOVED"
                parts.append(f"{name}@{read['position']}"
                             f"(want {sample['expected'][name]}, x{read['ratio']} {mark})")
            print(f"  t={sample['at_s']:8.2f}  " + "  ".join(parts))
        print(f"  readable {result['frames_readable']}/{result['frames_sampled']}; "
              f"top agrees {result['top_agreeing']}, bottom agrees "
              f"{result['bottom_agreeing']} (need {result['frames_required']})")
        print("VERDICT " + result["verdict"])
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
