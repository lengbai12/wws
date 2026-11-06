#!/usr/bin/env python3
"""Prove that preparing the images on the GPU equals preparing them on the CPU.

The loader hands batches over as raw uint8 crops and prepare_images() does the float
conversion and the downsample on the training device. That is what took the per-batch
CPU cost from 153 ms to 2.6 ms (the GPU step is 32 ms, so the conversion on the CPU was
pinning the card at 21 %). The float values must not change: the model must see exactly
what it saw before.

Heavy: it reads the corpus and uses the GPU, so do not run it beside a training.

    ./.venv/bin/python tools/replay/check_gpu_input_prep.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_minimap_heatmap as T

HEATMAPS = "replays/parsed/datasets/heatmap-merged-14s-g176.npz"
IMAGES = "replays/parsed/datasets/minimap-merged-14s-g176.npz"
BATCH = 16


def main():
    data = T.load_all(HEATMAPS, IMAGES, 2)
    lazy = data["images"]
    raw = lazy[np.arange(BATCH)]
    old = F.avg_pool2d(torch.from_numpy(lazy.array[np.arange(BATCH)]).float()
                       .div_(255.0).permute(0, 3, 1, 2), 2)
    device = T.pick_device("auto")
    new = T.prepare_images(raw, device, 2).cpu()
    print("raw %s %s -> prepared %s on %s"
          % (raw.dtype, tuple(raw.shape), tuple(new.shape), device))
    difference = (old - new).abs().max().item()
    print("max|Δ| against the CPU float path = %.3e" % difference)

    started = time.time()
    for k in range(40):
        lazy[np.arange(k, k + BATCH)]
    gather = (time.time() - started) / 40
    print("CPU gather only              %6.1f ms/batch" % (gather * 1000))
    print("  (the old path, which also converted and pooled here, measured 153.2 ms)")
    return 1 if difference > 1e-5 else 0


if __name__ == "__main__":
    sys.exit(main())
