#!/usr/bin/env python3
"""Write the golden fixture the Rust port of decode() is checked against.

decode() is the reference implementation: a peak is a 3x3 local maximum at or above
the threshold, capped at MAX_PEAKS_PER_CLASS per channel. The Rust runtime agent needs
the same function, and the only honest way to claim parity is to decode the same bytes
with both and compare. This dumps the inputs and the reference's own output.

Two inputs, because they stress different things:

  targets  the corpus' painted heatmaps (sparse: mostly zero with small blobs)
  soft     a deterministic synthetic field with gaussian blobs and noise, which is
           what a model's sigmoid output looks like: dense values, many of them close
           to the threshold

Peak order is canonicalised (score desc, then row, then column) after the per-channel
cap, so the comparison is about which peaks were found and their exact scores, not
about the order two different sort implementations happened to produce.

    ./.venv/bin/python tools/replay/dump_decode_fixture.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_minimap_heatmap as T

HEATMAPS = "replays/parsed/datasets/heatmap-merged-14s-g176.npz"
IMAGES = "replays/parsed/datasets/minimap-merged-14s-g176.npz"
GRID = 176


def canonical(peaks):
    """The same peaks, in an order both implementations can agree on."""
    return sorted(peaks, key=lambda p: (-p[2], p[1], p[0]))


def synthetic(samples, channels, seed):
    """Blobs plus noise: dense, soft values in [0, 1] like a sigmoid output."""
    rng = np.random.default_rng(seed)
    field = rng.random((samples, channels, GRID, GRID), dtype=np.float32) * 0.35
    rows = np.arange(GRID, dtype=np.float32)[:, None]
    columns = np.arange(GRID, dtype=np.float32)[None, :]
    for s in range(samples):
        for c in range(channels):
            for _ in range(6):
                cy, cx = rng.integers(8, GRID - 8, size=2)
                sigma = float(rng.uniform(2.0, 6.0))
                height = float(rng.uniform(0.5, 0.9))
                blob = height * np.exp(-((rows - cy) ** 2 + (columns - cx) ** 2)
                                       / (2.0 * sigma ** 2))
                field[s, c] = np.maximum(field[s, c], blob.astype(np.float32))
    return np.clip(field, 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="rust/wws-perception/tests/fixtures/decode")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--channels", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261012)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = T.load_all(HEATMAPS, IMAGES, 2)
    targets = np.ascontiguousarray(
        data["heatmaps"][np.arange(args.samples)][:, :args.channels]).astype(np.float32)
    soft = np.ascontiguousarray(synthetic(args.samples, args.channels, args.seed))

    fixture = {"grid": GRID, "threshold": T.PEAK_THRESHOLD,
               "max_peaks_per_class": T.MAX_PEAKS_PER_CLASS, "cases": {}}
    for name, field in (("targets", targets), ("soft", soft)):
        (out / ("%s.f32" % name)).write_bytes(field.tobytes(order="C"))
        import torch
        decoded = T.decode(torch.from_numpy(field), T.PEAK_THRESHOLD)
        fixture["cases"][name] = {
            "shape": list(field.shape),
            "file": "%s.f32" % name,
            "peaks": [[[list(p) for p in canonical(channel)] for channel in sample]
                      for sample in decoded],
        }
        counts = sum(len(c) for s in decoded for c in s)
        print("  %-8s %s  peaks=%d  min=%.4f max=%.4f"
              % (name, tuple(field.shape), counts, float(field.min()), float(field.max())))

    (out / "peaks.json").write_text(json.dumps(fixture, indent=1) + "\n", encoding="utf-8")
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
