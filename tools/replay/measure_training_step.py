#!/usr/bin/env python3
"""Measure where a training step spends its time: gathering crops, or the GPU work.

This is the measurement that decided how the GPU would be fed. On this corpus:

    CPU crop + downsample (main thread)   153.2 ms   <- the old loader
    copy to the GPU                         3.6 ms
    forward + backward + optimiser         31.7 ms

so the loader was five times the GPU work and pinned the card near 20 % utilisation.
After moving the conversion and the downsample onto the GPU the gather is about 3 ms,
and what remains (roughly 40 ms per batch unaccounted for) is page-cache work: the
corpus is 37 GB against ~15 GB of usable RAM, so it is re-read and decompressed every
epoch whatever the loader does.

Heavy: it reads the corpus and uses the GPU, so do not run it beside a training.

    ./.venv/bin/python tools/replay/measure_training_step.py [BATCH]
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


def main():
    batch = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    data = T.load_all(HEATMAPS, IMAGES, 2)
    train_index = T.select(data, "train")
    order = np.random.permutation(len(train_index))
    source = T.BatchSource(data["images"], data["heatmaps"], data["rings"],
                           train_index, order, batch)

    started = time.time()
    for k in range(1, 41):
        source[k]
    gather = (time.time() - started) / 40

    device = T.pick_device("auto")
    model = T.MinimapHeatmap(len(data["classes"]), len(data["ring_labels"]),
                             data["grid"]).to(device)
    optimiser = torch.optim.Adam(model.parameters())
    weight = torch.ones(1, len(data["classes"]), 1, 1, device=device)
    held = [source[k] for k in range(1, 9)]
    images = [T.prepare_images(b[0], device, 2) for b in held]
    targets = [b[1].to(device) for b in held]
    rings = [b[2].to(device) for b in held]

    def step(k):
        optimiser.zero_grad()
        heat, ring = model(images[k])
        loss = F.binary_cross_entropy_with_logits(heat, targets[k], pos_weight=weight) \
            + 0.2 * F.binary_cross_entropy_with_logits(ring, rings[k])
        loss.backward()
        optimiser.step()

    for k in range(3):
        step(k)
    torch.cuda.synchronize() if device.type == "cuda" else None
    started = time.time()
    for k in range(60):
        step(k % 8)
    if device.type == "cuda":
        torch.cuda.synchronize()
    gpu = (time.time() - started) / 60

    started = time.time()
    for k in range(60):
        T.prepare_images(held[k % 8][0], device, 2)
    if device.type == "cuda":
        torch.cuda.synchronize()
    prep = (time.time() - started) / 60

    print("a batch of %d samples on %s:" % (batch, device))
    print("  CPU gather                     %6.1f ms" % (gather * 1000))
    print("  prepare on the device          %6.1f ms" % (prep * 1000))
    print("  forward + backward + step      %6.1f ms" % (gpu * 1000))
    print("  => if they serialise, utilisation tops out near %.0f%%"
          % (100 * gpu / max(gpu, gather + prep)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
