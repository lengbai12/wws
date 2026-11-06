#!/usr/bin/env python3
"""Check the half-resolution member against the exact 2x2 mean.

The merge stores a 2x2-downsampled copy of the image member so an epoch reads four
times less data, and the trainer prefers it whenever the caller asks for a factor of 2.
That is only safe if the stored copy really is that downsample, so this compares it with
the exact float mean - on a synthetic array always, and on the real corpus when it has
the member.

Both checks go through the merge's own code (``write_half_twin``) instead of a second
implementation, because a checker that reimplements the rule only proves the checker.

    ./.venv/bin/python tools/replay/check_half_member.py [--corpus <images npz>]
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import merge_heatmap_datasets as merge
import train_minimap_heatmap as T

# Round-half-up can be off by exactly half a step, so the bound is inclusive: the first
# run of this check printed FAIL at max|delta| == 0.5/255 because the comparison was
# strict and the float arithmetic landed a hair above it. Compare with a relative slack
# instead of tightening the rule.
BOUND = 0.5 / 255.0
# The comparison itself is float32: the exact mean is rounded to float32 and the stored
# value is divided by 255, which together add about 1e-7 of noise on top of the half
# step. A relative 1e-6 was too small for that (the bound is ~2e-3, so 1e-6 relative is
# 2e-9); 1e-4 relative is 0.01 % of a step, far below any real deviation.
SLACK = 1e-4


def compare(reference_uint8, stored_uint8):
    """Max |difference| between the stored half and the exact float mean, in 0..1."""
    block = np.asarray(reference_uint8)
    rows, columns = block.shape[1] // 2, block.shape[2] // 2
    trimmed = block[:, :rows * 2, :columns * 2, :].astype(np.float32)
    exact = trimmed.reshape(block.shape[0], rows, 2, columns, 2,
                            block.shape[3]).mean(axis=(2, 4)) / 255.0
    got = np.asarray(stored_uint8).astype(np.float32) / 255.0
    if got.shape != exact.shape:
        raise SystemExit("shape differs: stored %s vs exact %s" % (got.shape, exact.shape))
    difference = np.abs(got - exact)
    exact_share = float((difference == 0).mean())
    print("       exact cells %.1f%%  max %.3e" % (100.0 * exact_share, difference.max()))
    return float(difference.max())


def synthetic():
    rng = np.random.default_rng(7)
    crops = rng.integers(0, 256, size=(3, 40, 24, 3), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "source.npy"
        np.save(source, crops)
        target = merge.write_half_twin(source, Path(tmp) / "half.npy")
        stored = np.load(target, mmap_mode="r")
        delta = compare(crops, stored)
        print("  synthetic  %s -> %s   max|delta| = %.3e  (bound %.3e)  %s"
              % (crops.shape, stored.shape, delta, BOUND,
                 "OK" if delta <= BOUND * (1 + SLACK) else "FAIL"))
        return delta <= BOUND * (1 + SLACK)


def corpus(path):
    """Compare a couple of samples of the real member, reading through the twins."""
    images = Path(path)
    half_twin = T.twin_path(images, "images_half")
    full_twin = T.twin_path(images, "images")
    if not half_twin.exists():
        print("  corpus     %s has no images_half member; skipped" % images.name)
        return True
    half = np.load(half_twin, mmap_mode="r")
    full = np.load(full_twin, mmap_mode="r")
    expected = (full.shape[0], full.shape[1] // 2, full.shape[2] // 2, full.shape[3])
    print("  corpus     full %s  half %s  expected %s  %s"
          % (full.shape, half.shape, expected,
             "OK" if half.shape == expected else "FAIL"))
    if half.shape != expected:
        return False
    delta = compare(full[:2], half[:2])
    print("  corpus     first 2 samples  max|delta| = %.3e  (bound %.3e)  %s"
          % (delta, BOUND, "OK" if delta <= BOUND * (1 + SLACK) else "FAIL"))
    return delta <= BOUND * (1 + SLACK)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", action="append", default=[],
                    help="an images npz that may carry the member")
    args = ap.parse_args()
    ok = synthetic()
    for path in args.corpus:
        ok &= corpus(path)
    print("VERDICT %s" % ("pass" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
