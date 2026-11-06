#!/usr/bin/env python3
"""Verify image-only main-view perception on real RGB video frames.

Example:
  python verify_main_view.py --video recordings/236387/capture.mkv \
      --times 394,398 --out state/main-view-development

The verifier creates JSON observations and PPM overlays.  It does not use replay
truth or send game input.  `--times` are container seconds.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import math
from PIL import Image
from pathlib import Path
import numpy as np

from main_view_perception import perceive_frame, load_native_templates, FRAME_W, FRAME_H


def read_frame(video: str, seconds: float) -> np.ndarray:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(seconds),
           "-i", video, "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.check_output(cmd)
    expected = FRAME_W * FRAME_H * 3
    if len(raw) != expected:
        raise RuntimeError(f"decoded frame at {seconds}s has {len(raw)} bytes, expected {expected}")
    return np.frombuffer(raw, dtype=np.uint8).reshape(FRAME_H, FRAME_W, 3).copy()


def _line(im: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    # Integer Bresenham, with bounds checks for a dependency-free evidence overlay.
    dx, sx = abs(x1-x0), 1 if x0 < x1 else -1; dy, sy = -abs(y1-y0), 1 if y0 < y1 else -1; err = dx + dy
    while True:
        if 0 <= x0 < im.shape[1] and 0 <= y0 < im.shape[0]: im[y0, x0] = color
        if x0 == x1 and y0 == y1: break
        e2 = 2 * err
        if e2 >= dy: err += dy; x0 += sx
        if e2 <= dx: err += dx; y0 += sy


def _box(im: np.ndarray, box: list[int], color: tuple[int, int, int]) -> None:
    x, y, w, h = box
    _line(im, x, y, x+w, y, color); _line(im, x, y, x, y+h, color)
    _line(im, x+w, y, x+w, y+h, color); _line(im, x, y+h, x+w, y+h, color)


def write_ppm(path: Path, frame: np.ndarray, observation: dict) -> None:
    out = frame.copy()
    for d in observation.get("detections", []):
        _box(out, d["nameplate_bbox"], (255, 255, 0))
        _box(out, d["body_roi"], (0, 255, 0) if d.get("body_visible") else (255, 80, 80))
        if d.get("body_bbox"): _box(out, d["body_bbox"], (100, 255, 100))
    for d in observation.get("markers", []):
        _box(out, d["bbox"], (255, 128, 0) if d["class"] == "torpedo" else (0, 255, 255))
    lock = observation.get("lock_confirmation", {})
    if lock.get("bbox"): _box(out, lock["bbox"], (255, 200, 0))
    path.write_bytes((f"P6\n{FRAME_W} {FRAME_H}\n255\n").encode() + out.tobytes())


def parse_times(values: list[str]) -> list[float]:
    out = []
    for value in values:
        for token in value.split(","):
            if not token.strip(): continue
            t = float(token)
            if not math.isfinite(t) or t < 0: raise ValueError("--times must contain finite nonnegative seconds")
            out.append(t)
    if not out: raise ValueError("--times requires at least one timestamp")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="RGB-compatible video path")
    source.add_argument("--image", help="single actual RGB screenshot path")
    ap.add_argument("--times", nargs="+", help="container seconds, comma-separated or repeated; required for video")
    ap.add_argument("--out", required=True, help="new output directory")
    ap.add_argument("--assets", default="state/toalba/gamedata/data/gui/battle_hud/markers", help="extracted native marker asset root")
    args = ap.parse_args(argv)
    if args.video and not args.times: ap.error("--video requires --times")
    if args.image and args.times: ap.error("--image does not accept --times")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing overwrite of non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    times = parse_times(args.times) if args.video else [0.0]
    if any(b <= a for a, b in zip(times, times[1:])): ap.error("--times must increase strictly")
    templates = load_native_templates(args.assets)
    observations = []; previous = None; previous_time = None
    for i, t in enumerate(times):
        frame = read_frame(args.video, t) if args.video else np.asarray(Image.open(args.image).convert("RGB"))
        dt = None if previous_time is None else t - previous_time
        obs = perceive_frame(frame, previous, templates, dt_s=dt)
        obs["video_time_s"] = t; obs["frame_index_request"] = i
        observations.append(obs); write_ppm(out / f"frame-{i:03d}-{t:.3f}s.ppm", frame, obs)
        previous = frame; previous_time = t
    report = {"source": str(Path(args.video or args.image)), "times_s": times,
              "frame_shape": [FRAME_H, FRAME_W, 3], "coordinate_system": "native_frame_xy_pixels",
              "observations": observations,
              "evidence": {"overlay_format": "PPM", "replay_truth_used": False,
                           "lock_absence": "unknown_not_success",
                           "aircraft": "unverified unless native-color sprite observed"},
              "limitations": ["screen evidence only", "nameplate does not prove visible hull",
                              "screen motion is not world velocity", "no OCR or calibrated 3D"]}
    (out / "observations.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"out": str(out), "frames": len(observations),
                      "detections": sum(len(x["detections"]) for x in observations),
                      "markers": sum(len(x["markers"]) for x in observations)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
