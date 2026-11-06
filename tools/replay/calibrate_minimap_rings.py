"""Measure the minimap's distance scale from the range rings it draws.

The minimap draws the ship's weapon and detection ranges as concentric circles
centred on the own ship, each labelled with its range in kilometres. Those labels
are the only absolute distances the picture carries, so they are what fixes the
scale of everything else: without them a pixel count cannot be converted to
metres, and the replay's own coordinates have no stated unit.

The measurement is a radial scan from the own ship. Ring radii appear as peaks in
"how much of this circle is ring-coloured", and the candidate labels are fitted by
matching RATIOS, so a wrong label set cannot pass: three rings matching to a
fraction of a percent is not something a wrong assignment does.

Usage:
  calibrate_minimap_rings.py --video recordings/<s>/capture.mkv \
      --hud-calibration replays/parsed/calibration/hud-2560x1600.json \
      --minimap-calibration replays/parsed/calibration/minimap-2560x1600.json \
      --position replays/parsed/position-<session>.jsonl \
      --alignment replays/parsed/alignment/<recording>.json \
      --labels-km 8.0,9.9,12.0,16.8,20.1 [--frames 40]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import calibrate_hud
import calibrate_minimap as cm
import hud_calibration as hud
import minimap_calibration as mini

RING_MIN_SHARE = 0.22
RATIO_TOLERANCE = 0.02

# A fit is only accepted when several distinct rings support it AND the implied
# scales agree with each other. A loose window accepts coincidences: with eight
# observed radii and five labels, a 3 % window is satisfied by chance often
# enough to produce a nonsense scale (one frame fitted 25 px/km).
MIN_SUPPORT = 3
MAX_SPREAD = 0.01

# A radius must appear in at least this share of frames to count as a map ring.
FRAME_SUPPORT_SHARE = 0.25


def ring_mask(frame):
    """Pixels that look like a range ring: light, unsaturated, and BRIGHTER than
    the frame's own background.

    The brightness contrast is what makes this usable. Without it the calm sea and
    the map's land both pass a "light and unsaturated" test, half the sampled
    pixels count as ring-coloured, and no peak stands out (measured baseline
    0.494 with 80 frames). The rings are the light line work drawn on top.
    """
    work = frame.astype(np.int32)
    luminance = work.mean(axis=2)
    saturation = work.max(axis=2) - work.min(axis=2)
    background = float(np.median(luminance))
    return ((luminance > background + 12) & (luminance > 80) & (luminance < 235)
            & (saturation < 60))


def radial_profile(frame, centre, max_radius=345):
    """Return the share of ring-coloured pixels at each radius from a centre."""
    mask = ring_mask(frame)
    height, width = mask.shape[:2]
    angles = np.linspace(0.0, 2.0 * np.pi, 1440, endpoint=False)
    cosines, sines = np.cos(angles), np.sin(angles)
    profile = []
    for radius in range(15, min(max_radius, min(height, width) // 2 + 120)):
        columns = np.clip((centre[0] + radius * cosines).round().astype(int), 0, width - 1)
        rows = np.clip((centre[1] + radius * sines).round().astype(int), 0, height - 1)
        profile.append(float(mask[rows, columns].mean()))
    return np.asarray(profile), 15


def peak_radii(profile, origin, minimum, min_radius=40):
    """Return (radius, share) peaks in a radial profile, strongest kept per gap.

    Radii below ``min_radius`` are ignored: close to the centre the own-ship
    marker and the UI clutter make any circle look ring-coloured, so a small
    radius scores high for reasons that have nothing to do with a range ring.
    """
    baseline = float(np.median(profile))
    smoothed = np.convolve(profile, np.ones(3) / 3.0, mode="same")
    peaks = []
    for index in range(2, len(smoothed) - 2):
        radius = origin + index
        if radius < min_radius:
            continue
        window = smoothed[index - 2:index + 3]
        if smoothed[index] < window.max() or smoothed[index] <= baseline + minimum:
            continue
        if peaks and radius - peaks[-1][0] <= 5:
            if smoothed[index] > peaks[-1][1]:
                peaks[-1] = (radius, smoothed[index])
            continue
        peaks.append((radius, smoothed[index]))
    return peaks, baseline


def fit_pixels_per_km(radii, labels_km, tolerance=RATIO_TOLERANCE):
    """Return (pixels_per_km, matches, spread) for the best label assignment.

    Every pair of observed radii is compared with every pair of labels from the
    full set, and the assignment that explains the most consistent ratios wins.
    """
    best = None
    for first in range(len(radii)):
        for second in range(first + 1, len(radii)):
            ratio = radii[second] / radii[first]
            for a in range(len(labels_km)):
                for b in range(a + 1, len(labels_km)):
                    wanted = labels_km[b] / labels_km[a]
                    if abs(ratio - wanted) > tolerance * wanted:
                        continue
                    scales = [radii[second] / labels_km[b], radii[first] / labels_km[a]]
                    for other_radius, other_label in zip(radii, labels_km):
                        if other_label in (labels_km[a], labels_km[b]):
                            continue
                        predicted = other_label * (sum(scales) / len(scales))
                        if abs(other_radius - predicted) <= max(2.0, 0.03 * predicted):
                            scales.append(other_radius / other_label)
                    spread = (max(scales) - min(scales)) / (sum(scales) / len(scales))
                    candidate = {
                        "pixels_per_km": sum(scales) / len(scales),
                        "support": len(scales),
                        "spread": spread,
                        "matches": [(radii[first], labels_km[a]), (radii[second], labels_km[b])],
                    }
                    if best is None or (candidate["support"], -candidate["spread"]) > (
                            best["support"], -best["spread"]):
                        best = candidate
    return best


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--hud-calibration", type=Path, required=True)
    parser.add_argument("--minimap-calibration", type=Path, required=True)
    parser.add_argument("--position", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--labels-km", default="8.0,9.9,12.0,16.8,20.1",
                        help="range values printed on the minimap rings")
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--out", type=Path, help="update this calibration record")
    args = parser.parse_args(argv)

    labels = [float(v) for v in args.labels_km.split(",") if v.strip()]
    calibration = hud.load_calibration(args.hud_calibration)
    rect = hud.element_rect(calibration, "minimap")
    if rect is None:
        raise SystemExit("element 'minimap' is not calibrated")
    positions = cm.read_positions(args.position)
    rate, offset_s, _alignment = cm.read_alignment(args.alignment)
    record = json.loads(args.minimap_calibration.read_text(encoding="utf-8"))
    matrix = np.asarray(record["world_to_pixel"]["matrix"], dtype=float)
    translation = np.asarray(record["world_to_pixel"]["translation"], dtype=float)

    first = rate * positions[0][0] + offset_s + 30.0
    last = rate * positions[-1][0] + offset_s
    times = [first + (last - first) * index / (args.frames - 1)
             for index in range(args.frames)]
    profiles = []
    for video_time in times:
        clock = (video_time - offset_s) / rate
        position = cm.world_at(positions, clock)
        if position is None:
            continue
        rgb, width, height = calibrate_hud.read_frame(args.video, video_time)
        frame = mini.frame_array(rgb, width, height)
        pixel = mini.apply_affine(matrix, translation, position[0], position[1])
        centre = (pixel[0] - rect[0], pixel[1] - rect[1])
        if not (0 <= centre[0] < rect[2] and 0 <= centre[1] < rect[3]):
            continue
        profile, origin = radial_profile(frame, centre)
        profiles.append(profile)
    if len(profiles) < 5:
        raise SystemExit("only %d usable frames; the own ship must be on the minimap"
                         % len(profiles))
    # A range ring sits at the SAME radius in every frame; terrain sits at a radius
    # that moves with the ship, and land is only in some frames. So the radii are
    # counted ACROSS frames and only the ones seen in a good share of frames are
    # kept. Fitting a single frame instead invites a coincidence: its peak list is
    # full of terrain edges (measured: per-frame fits ranged 4.9 to 25 px/km).
    votes = {}
    for profile in profiles:
        peaks, _baseline = peak_radii(profile, 15, 0.06)
        for radius, _share in peaks:
            votes[radius] = votes.get(radius, 0) + 1
    required = max(2, int(round(FRAME_SUPPORT_SHARE * len(profiles))))
    kept = sorted(radius for radius, count in votes.items() if count >= required)
    merged = []
    for radius in kept:
        if merged and radius - merged[-1][-1] <= 3:
            merged[-1].append(radius)
        else:
            merged.append([radius])
    radii = [int(round(sum(group) / len(group))) for group in merged]
    print("frames %d  投票阈值 %d 帧  跨帧稳定半径: %s"
          % (len(profiles), required,
             ", ".join("r=%d(%d帧)" % (radius, votes.get(radius, 0) +
                                       sum(votes.get(n, 0) for n in range(radius-3, radius+4)))
                       for radius in radii)))
    if len(radii) < 2:
        raise SystemExit("fewer than two ring radii are stable across frames: %s" % radii)
    fit = fit_pixels_per_km(radii, labels)
    if fit is None or fit["support"] < MIN_SUPPORT or fit["spread"] > MAX_SPREAD:
        raise SystemExit("no consistent label assignment for radii %s" % radii)
    frame_fits = [(fit, radii, 0.0)]
    spread = fit["spread"]
    print("px/km %.3f  支持度 %d  相对跨度 %.5f"
          % (fit["pixels_per_km"], fit["support"], fit["spread"]))
    metres_per_pixel = 1000.0 / per_km
    unit_m = metres_per_pixel / (1.0 / record["stats"]["scale_px_per_m_x"])
    print("pixels_per_km %.3f  =>  1 px = %.1f m  (1 world unit = %.2f m)"
          % (per_km, metres_per_pixel, unit_m))
    print("matches: %s  support %d  spread %.4f"
          % (best["matches"], best["support"], best["spread"]))
    print("每标签的自检：")
    for label in labels:
        expected = label * per_km
        nearest = min(radii, key=lambda radius: abs(radius - expected))
        print("   %5.1f km -> 预期半径 %6.1f px  最近实测 %s px  (%s)"
              % (label, expected, nearest,
                 "命中" if abs(nearest - expected) <= max(2.0, 0.03 * expected)
                 else "未检出"))

    if args.out:
        record["rings"] = {
            "pixels_per_km": round(per_km, 4),
            "metres_per_pixel": round(metres_per_pixel, 3),
            "world_unit_m": round(unit_m, 3),
            "labels_km": labels,
            "method": ("radial scan from the own ship, fitted per frame and reduced "
                       "by the median over %d fitted frames; the scale is fixed by "
                       "matching observed radius RATIOS to the printed range labels"
                       % len(frame_fits)),
            "frames_averaged": len(profiles),
            "radius_candidates": radii,
            "relative_spread": round(spread, 5),
            "evidence": {"matched_pairs": [[r, l] for r, l in best["matches"]],
                         "support": best["support"],
                         "relative_spread": round(best["spread"], 5),
                         "profile_baseline": round(baseline, 4)},
        }
        Path(args.out).write_text(json.dumps(record, indent=1, sort_keys=True) + "\n",
                                  encoding="utf-8")
        print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
