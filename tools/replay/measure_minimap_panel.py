#!/usr/bin/env python3
"""Measure the minimap panel's rectangle from a frame, independently of the gate.

WHY THIS EXISTS: ``verify_minimap_rect.py`` reads the panel's size from the
gradient at the calibrated TOP row. That works because the panel's top border is
where the world meets the map, and it is the edge that encodes the size. But the
border is only readable when the world behind it differs in brightness from the
map, so on a bright map the gate reports most frames *unreadable* and cannot
reach its 2/3 quorum -- a legitimate refusal, but one that leaves the question
"is the panel the right size?" unanswered.

This tool answers that question a second way, and it does NOT depend on the
world/map contrast at the top row alone:

* The panel is anchored to the BOTTOM-RIGHT of the content area, so a smaller
  panel moves its TOP and LEFT edges. Measuring either edge locates the panel.
* Each edge is found as the strongest full-width (or full-height) step of the
  mean luminance profile, searched in a window that contains both the calibrated
  position and the default-size position.

It is a DIAGNOSTIC, not a gate: it always reports where it thinks the edges are
and how strong they were, so a human or a later rule can judge. It says nothing
about whether the panel's *content* is valid.

Usage:
  ./.venv/bin/python tools/replay/measure_minimap_panel.py --image state/x.png
  ./.venv/bin/python tools/replay/measure_minimap_panel.py --video recordings/<s>/capture.mkv --at 394
  ... --rect 1836,796,720,723        # compare against an expected rectangle
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Where the panel's edges can be. The calibrated maximum puts them at 796 and
# 1836; the client's default size puts the top near 888 and the left near 1966.
TOP_SEARCH = (600, 1150)
LEFT_SEARCH = (1600, 2200)
# The profile is read from inside the panel's footprint so that the edge being
# searched for is the only strong step in the band.
INSIDE_COLUMNS = (1846, 2546)
INSIDE_ROWS = (1150, 1500)


def gray(video: Path | None, image: Path | None, at_s: float) -> tuple[list[list[int]], int, int]:
    """One frame as 8-bit luminance rows, plus its size."""
    if image is not None:
        cmd = ["ffmpeg", "-v", "error", "-i", str(image)]
    else:
        cmd = ["ffmpeg", "-v", "error", "-ss", f"{at_s:.3f}", "-i", str(video)]
    cmd += ["-frames:v", "1", "-vf", "format=gray", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0", str(image or video)],
        capture_output=True, text=True, check=True).stdout.strip()
    width, height = (int(v) for v in probe.split(",")[:2])
    if len(raw) != width * height:
        raise SystemExit(f"could not decode one {width}x{height} frame")
    return [list(raw[row * width:(row + 1) * width]) for row in range(height)], width, height


def row_profile(rows: list[list[int]], span: tuple[int, int]) -> list[float]:
    """Mean luminance of each row, over the columns ``span``."""
    low, high = span
    return [sum(row[low:high]) / max(1, high - low) for row in rows]


def column_profile(rows: list[list[int]], span: tuple[int, int]) -> list[float]:
    """Mean luminance of each column, over the rows ``span``."""
    low, high = span
    width = len(rows[0])
    return [sum(rows[row][col] for row in range(low, high)) / max(1, high - low)
            for col in range(width)]


def strongest_step(profile: list[float], window: tuple[int, int], minimum_gap: int = 4) -> list[tuple[int, float]]:
    """The strongest |step| positions inside ``window``, most extreme first.

    Reported as (index, signed delta). Several are returned because a moved edge
    and a coincidental edge inside the panel are otherwise indistinguishable from
    one number alone.
    """
    low, high = window
    deltas = [(index, profile[index + 1] - profile[index])
              for index in range(max(0, low), min(len(profile) - 1, high))]
    deltas.sort(key=lambda item: -abs(item[1]))
    peaks: list[tuple[int, float]] = []
    for index, delta in deltas:
        if all(abs(index - taken) > minimum_gap for taken, _ in peaks):
            peaks.append((index, delta))
        if len(peaks) == 3:
            break
    return peaks


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--video", type=Path)
    parser.add_argument("--at", type=float, default=0.0, help="seconds into --video")
    parser.add_argument("--rect", help="x,y,w,h to compare against")
    args = parser.parse_args()

    rows, width, height = gray(args.video, args.image, args.at)
    row_means = row_profile(rows, INSIDE_COLUMNS)
    col_means = column_profile(rows, INSIDE_ROWS)

    top = strongest_step(row_means, TOP_SEARCH)
    left = strongest_step(col_means, LEFT_SEARCH)

    label = str(args.image or f"{args.video}@{args.at:.1f}s")
    print(f"{label}  ({width}x{height})")
    print("  top edge candidates  (frame row, delta): "
          + ", ".join(f"{index} ({delta:+.1f})" for index, delta in top))
    print("  left edge candidates (frame col, delta): "
          + ", ".join(f"{index} ({delta:+.1f})" for index, delta in left))
    if args.rect:
        x0, y0, w, h = (int(v) for v in args.rect.split(","))
        top_row, top_delta = top[0]
        left_col, left_delta = left[0]
        print(f"  expected panel x0={x0} y0={y0} w={w} h={h}; "
              f"measured x0={left_col} ({left_col - x0:+d} px), y0={top_row} ({top_row - y0:+d} px)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
