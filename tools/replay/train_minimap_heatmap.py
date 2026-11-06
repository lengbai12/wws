"""Train a multi-class minimap heatmap model from replay-derived targets.

One forward pass predicts, for every class, a heatmap whose peaks are the markers
the replay says are drawn, plus which range rings the player has enabled.

The classes are not interchangeable, and the training and the metrics treat them
that way:

  own, friendly, enemy_visible, wreck, zone_A/B/C   heatmap targets
  enemy_not_visible                                 NOT a target: those are the
                                                    places the map does NOT paint,
                                                    so they measure false positives
  enemy_unknown                                     neither: the side or the
                                                    visibility was never established,
                                                    so it is not supervision at all
  range rings                                       multi-label, because a ring is
                                                    a circle and not a point

Usage:
  train_minimap_heatmap.py --heatmaps replays/parsed/datasets/heatmap-<s>.npz \
      --images replays/parsed/datasets/minimap-<s>.npz \
      --model-out replays/parsed/artifacts/minimap-heatmap-<s>.pt \
      --manifest-out replays/parsed/artifacts/minimap-heatmap-<s>.json \
      --epochs 60 --downsample 2

  train_minimap_heatmap.py --validate --model ... --manifest ...
  train_minimap_heatmap.py --preview --model ... --manifest ... --preview-out ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from pathlib import Path
from queue import Queue
from fsutil import twin_path, sha256


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PRESENT = 0
ARCHITECTURE = "conv4-decoder3-rings"
# Fixed on the VALIDATION split, never on test: at 0.2 the phantom rate was 0.181
# and at 0.6 it was 0.104 while recall moved only 0.860 -> 0.854, so 0.6 buys most of
# the precision for almost none of the recall. The full curve is in the manifest.
PEAK_THRESHOLD = 0.60

# The matching radius is stated in CROP pixels, not grid cells, so that a metric
# stays comparable when the grid changes. 24 crop px = 3.1 km at this map's scale.
MATCH_RADIUS_CROP_PX = 24.0
MAX_PEAKS_PER_CLASS = 24
# Measuring a split runs the model on its whole index. At 723x720 crops that is 14 GB
# of float32 input for the corpus' two-session test split, which does not fit beside
# the logits it produces, so the forward pass is batched even when it is only measured.
FORWARD_CHUNK = 32
# Positive pixels are about 0.4 % of a heatmap, so the positive term is scaled by
# the observed negative-to-positive ratio PER CHANNEL. A single weight cannot serve
# channels whose densities differ by an order of magnitude (one own marker against
# eleven teammates).
POSITIVE_WEIGHT_FALLBACK = 200.0
RING_NAMES = ("8.0", "9.9", "12.0", "16.8", "20.1")


class MinimapHeatmap(nn.Module):
    """U-Net with skip connections, a heatmap head and a range-ring head.

    Feature resolution and skips both matter for the dense classes. The earlier
    version encoded four times (176 -> 11) and decoded with no skips, so two
    teammates a few cells apart produced ONE peak: one-to-one recall for teammates
    was capped near 0.44 by that merge and measured 0.540. Skips hand the fine
    encoder levels to the corresponding decoder levels, and the final interpolate
    lands exactly on the target grid so the grid can be raised to the input size.
    """

    def __init__(self, classes: int, rings: int, grid: int, widths=(16, 32, 64, 96)):
        super().__init__()
        self.grid = grid
        self.stem = self._block(3, widths[0], stride=2)          # 176 -> 88
        self.down1 = self._block(widths[0], widths[1], stride=2)  # 88 -> 44
        self.down2 = self._block(widths[1], widths[2], stride=2)  # 44 -> 22
        self.down3 = self._block(widths[2], widths[3], stride=2)  # 22 -> 11
        self.up2 = self._block(widths[3] + widths[2], widths[2])
        self.up1 = self._block(widths[2] + widths[1], widths[1])
        self.up0 = self._block(widths[1] + widths[0], widths[0])
        self.head = nn.Conv2d(widths[0], classes, 1)
        self.ring_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1),
                                       nn.Linear(widths[3], rings))

    @staticmethod
    def _block(incoming, outgoing, stride=1):
        return nn.Sequential(
            nn.Conv2d(incoming, outgoing, 3, stride=stride, padding=1),
            nn.BatchNorm2d(outgoing), nn.ReLU(inplace=True))

    def forward(self, images: torch.Tensor):
        """Return (heatmap logits at grid x grid, ring logits).

        The heatmaps are LOGITS. Sparse targets cannot be learned with a sigmoid and
        a weighted MSE: positives are about half a percent of a heatmap, so that loss
        is minimised by painting nothing, which is exactly what an earlier 60-epoch
        run did (every class except the densest recalled nothing). Binary cross
        entropy on the logits with a per-channel positive weight is what works.
        """
        stem = self.stem(images)
        down1 = self.down1(stem)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        up2 = self.up2(torch.cat([F.interpolate(down3, size=down2.shape[-2:],
                                                mode="bilinear", align_corners=False),
                                  down2], dim=1))
        up1 = self.up1(torch.cat([F.interpolate(up2, size=down1.shape[-2:],
                                                mode="bilinear", align_corners=False),
                                  down1], dim=1))
        up0 = self.up0(torch.cat([F.interpolate(up1, size=stem.shape[-2:],
                                                mode="bilinear", align_corners=False),
                                  stem], dim=1))
        features = F.interpolate(up0, size=(self.grid, self.grid), mode="bilinear",
                                 align_corners=False)
        return self.head(features), self.ring_head(down3)


def open_member(path, member, archive):
    """Return one dataset member, memory-mapped when that is possible.

    ``np.load(..., mmap_mode="r")`` does NOT memory-map a zipped member on the
    installed numpy (2.5.3): the mmap branch is only reached for a plain .npy file,
    and an npz member is read into RAM whether it is stored or deflated -- a 960 MB
    STORED member costs 960 MB of RSS under ``mmap_mode="r"``, with no warning. The
    merged corpus therefore carries each large array twice: inside the npz, for the
    artifact contract, and as a .npy twin, which is the copy mapped here. A dataset
    without a twin (every single-session file) reads from the archive as before.
    """
    twin = twin_path(path, member)
    if twin.exists():
        return np.load(twin, allow_pickle=False, mmap_mode="r")
    return archive[member]


class LazyPictures:
    """Model input built from a stored crop stack one batch at a time.

    The corpus frames are 8.2 GB as uint8 and four times that once they are float32,
    so neither the conversion nor the copy can be done for the whole dataset up front.
    ``shape`` is the shape AFTER the channel move and the downsample, because that is
    what the manifest records and what the model is handed.
    """

    def __init__(self, array, downsample):
        self.array = array
        self.downsample = downsample
        height, width = array.shape[1], array.shape[2]
        self.shape = (array.shape[0], 3, height // downsample, width // downsample)

    def __getitem__(self, index):
        # A batch of ONE must stay a batch. The training loop slices the shuffled
        # index, so its last batch holds len(train) % batch samples, and when that
        # remainder is one, numpy can hand this index over in a shape that reads as a
        # scalar: the crop then came back as (3, H, W) and the model's first
        # BatchNorm rejected it with "expected 4D input (got 3D input)". Only a real
        # scalar index means "one sample"; anything array-like means "a batch".
        as_array = None if isinstance(index, (int, np.integer)) else np.asarray(index)
        single = as_array is None or as_array.ndim == 0
        if single:
            crops = np.ascontiguousarray(self.array[int(index)][None])
        else:
            crops = np.ascontiguousarray(self.array[as_array.reshape(-1)])
        # The crops leave here as uint8, and the float conversion and the downsample
        # happen on the GPU (see prepare_images). Doing them here was 153 ms per batch
        # against 32 ms of GPU work, which pinned the GPU at 21 %; on the GPU the same
        # two steps cost about a millisecond, and the batch crossing the bus is 25 MB
        # instead of 100 MB.
        return torch.from_numpy(crops)


def prepare_images(batch, device, downsample):
    """Turn a raw uint8 (N, H, W, 3) crop batch into the model's float input.

    Runs on the training device so the CPU only ever gathers bytes out of the
    memory-mapped corpus. Also accepts an already-prepared float batch, which is what
    the preview passes in.
    """
    if batch.ndim == 3:
        batch = batch.unsqueeze(0)
    if batch.dtype != torch.uint8:
        return batch.to(device, non_blocking=True)
    pictures = batch.to(device, non_blocking=True)
    pictures = pictures.permute(0, 3, 1, 2).float().div_(255.0)
    if downsample > 1:
        pictures = F.avg_pool2d(pictures, downsample)
    return pictures


class LazyTargets:
    """A float target stack copied out of storage one batch at a time."""

    def __init__(self, array):
        self.array = array
        self.shape = tuple(array.shape)

    def __getitem__(self, index):
        # Same rule as LazyPictures: an array-like index is a batch even when it
        # holds a single row, so the last batch of an epoch keeps its leading axis.
        as_array = None if isinstance(index, (int, np.integer)) else np.asarray(index)
        if as_array is None or as_array.ndim == 0:
            return torch.from_numpy(np.ascontiguousarray(self.array[int(index)])).float()
        return torch.from_numpy(
            np.ascontiguousarray(self.array[as_array.reshape(-1)])).float()


def load_all(heatmap_path, image_path, downsample):
    targets = np.load(heatmap_path, allow_pickle=False)
    images = np.load(image_path, allow_pickle=False)
    # A corpus may carry its 2x2-downsampled crops as a member of their own: reading
    # that is four times less data per epoch, which is exactly what the GPU waits on
    # (76.5 ms of gathering against 42.5 ms of compute, because one crop is 1.5 MB and
    # the corpus does not fit in the page cache). It is only used when the caller asked
    # for that very factor, so the model's input - and therefore the manifest - is
    # unchanged; the member is a storage choice, not a change of task.
    source, effective = open_member(image_path, "images", images), downsample
    if downsample == 2:
        half = twin_path(image_path, "images_half")
        if half.exists():
            source, effective = np.load(half, allow_pickle=False, mmap_mode="r"), 1
    heatmaps = LazyTargets(open_member(heatmap_path, "heatmaps", targets))
    rings = LazyTargets(open_member(heatmap_path, "rings", targets))
    # The validity mask says, per sample and per class, whether that class was
    # MEASURABLE in that session (1) or the session had nothing to say about it (0).
    # A corpus built before the contract carries none, and all-ones is the honest
    # default for it: nothing recorded that anything was unmeasurable.
    samples, channels = heatmaps.shape[0], heatmaps.shape[1]
    validity = (np.asarray(targets["valid"], dtype=np.float32) if "valid" in targets
                else np.ones((samples, channels), dtype=np.float32))
    ring_validity = (np.asarray(targets["ring_valid"], dtype=np.float32)
                     if "ring_valid" in targets
                     else np.ones((samples,), dtype=np.float32))
    return {
        "images": LazyPictures(source, effective),
        "heatmaps": heatmaps,
        "rings": rings,
        "valid": validity,
        "ring_valid": ring_validity,
        # The EFFECTIVE factor: with the half-resolution member in play this is 1, and
        # every prepare_images() call must agree with it or the batch is downsampled
        # twice. eval and the training loop both read it from here.
        "downsample": effective,
        "truth": targets["truth"],
        "counts": targets["counts"],
        "negatives": targets["negatives"],
        "negative_counts": targets["negative_counts"],
        "splits": np.asarray([str(item) for item in targets["splits"]]),
        "classes": [str(item) for item in targets["classes"]],
        "ring_labels": [float(v) for v in targets["ring_labels"]],
        "crop_rect": [int(v) for v in images["minimap_rect"]],
        "grid": int(targets["grid"][0]),
    }


def select(data, name):
    index = np.nonzero(data["splits"] == name)[0]
    return index


def decode(heatmaps, threshold=PEAK_THRESHOLD, chunk=64):
    """Return per-sample, per-channel peak positions in grid coordinates.

    Peaks are local maxima of a 3x3 window above a threshold, capped per channel.
    Decoding is part of the metric, so it is stated here once and used by every
    measurement below.

    The pooled window and its mask are the size of the split being measured (2.3 GB
    for the corpus' two-session test split), and nothing in them crosses a sample
    boundary, so they are built a chunk of samples at a time. The peaks this returns
    are the same either way: pooling pads each sample's own edges, not a neighbour.
    """
    out = []
    for start in range(0, heatmaps.shape[0], chunk):
        block = heatmaps[start:start + chunk]
        pooled = F.max_pool2d(block, 3, stride=1, padding=1)
        keep = (block >= pooled - 1e-6) & (block >= threshold)
        del pooled
        for sample in range(block.shape[0]):
            per_channel = []
            for channel in range(block.shape[1]):
                scores = block[sample, channel]
                mask = keep[sample, channel]
                rows, columns = torch.nonzero(mask, as_tuple=True)
                values = scores[rows, columns]
                order = torch.argsort(values, descending=True)[:MAX_PEAKS_PER_CLASS]
                per_channel.append([(float(columns[i]), float(rows[i]), float(values[i]))
                                    for i in order])
            out.append(per_channel)
        del block, keep
    return out


def match_radius(data):
    """Matching radius in grid cells for this dataset's grid."""
    return MATCH_RADIUS_CROP_PX * data["grid"] / float(data["crop_rect"][2])


def pick_device(name):
    """Resolve the requested device, refusing a GPU request this build cannot serve.

    The venv's torch was a CPU-only wheel for most of this project, so "cuda is
    unavailable" was an install fact, not a hardware fact. Say which it is instead of
    silently training on the CPU.
    """
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "--device %s was asked for, but this torch cannot see a GPU (torch %s, "
            "cuda build %s); install a CUDA wheel or use --device cpu"
            % (name, torch.__version__, torch.version.cuda))
    return torch.device(name)


def to_device(tensor, device):
    """Move a batch to the training device, or leave it alone when there is none."""
    return tensor if device is None else tensor.to(device, non_blocking=True)


def evaluate(model, index, data, threshold=PEAK_THRESHOLD, stream=False, device=None):
    """Precision, recall and position error per class, plus the negative test.

    ``stream`` decodes one forward chunk at a time and never materialises the split
    as a whole. The training split's logits are 12.4 GB as a single tensor, and
    building them after the last epoch is what OOM-killed a 40-epoch run (18 GB of
    anonymous memory, killed at 16:27:20 with the model not yet written). ``decode``
    pads each sample's own edges and never crosses a sample boundary, so decoding
    chunk by chunk returns exactly the same peaks.
    """
    model.eval()
    if device is None:
        # Callers that do not name a device (the sweep) still have to feed the model
        # where the model lives, or the first forward fails on a device mismatch.
        device = next(model.parameters()).device
    if stream:
        peaks = []
        ring_chunks = []
        with torch.no_grad():
            for start in range(0, len(index), FORWARD_CHUNK):
                batch_heat, batch_ring = model(prepare_images(
                    data["images"][index[start:start + FORWARD_CHUNK]], device,
                    data["downsample"]))
                # Decoding happens on the CPU: it walks peaks element by element, which
                # would sync the GPU thousands of times per chunk. Only the forward is
                # worth the device, and one chunk at a time keeps VRAM flat.
                peaks.extend(decode(torch.sigmoid(batch_heat).cpu(), threshold))
                ring_chunks.append(batch_ring.cpu())
                del batch_heat
        ring_logits = torch.cat(ring_chunks)
        del ring_chunks
        heatmaps = None
    else:
        heat_logits, ring_logits = [], []
        with torch.no_grad():
            for start in range(0, len(index), FORWARD_CHUNK):
                batch_heat, batch_ring = model(prepare_images(
                    data["images"][index[start:start + FORWARD_CHUNK]], device,
                    data["downsample"]))
                heat_logits.append(batch_heat.cpu())
                ring_logits.append(batch_ring.cpu())
        heatmap_logits = torch.cat(heat_logits)
        del heat_logits
        ring_logits = torch.cat(ring_logits)
        heatmaps = torch.sigmoid(heatmap_logits)
        # The chunks, their concatenation and the sigmoid are each the size of the split
        # (2.3 GB for the corpus' two-session test split). Releasing each one as soon as it
        # is consumed is what keeps measuring a split to one copy instead of three.
        del heatmap_logits
        peaks = decode(heatmaps, threshold)
    classes = data["classes"]
    radius = match_radius(data)
    validity = data.get("valid")
    metrics = {"match_radius_grid_px": round(radius, 3),
               "match_radius_crop_px": MATCH_RADIUS_CROP_PX}
    for channel, name in enumerate(classes):
        # A class the session could not measure is NOT supervised: its target is all
        # zeros, so a peak on it would be scored as a false positive against a truth
        # nobody stated. Those samples are skipped and `supervised` says so in the
        # manifest, instead of leaving a 0/0 to be misread as a broken model.
        allowed = [bool(validity[sample][channel]) if validity is not None else True
                   for sample in index]
        supervised = any(allowed)
        matched = 0
        predicted = 0
        truth_total = 0
        errors = []
        for position, sample in enumerate(index):
            if not allowed[position]:
                continue
            predicted_peaks = peaks[position][channel]
            predicted += len(predicted_peaks)
            truth = [(row[1], row[2]) for row in data["truth"][sample]
                     if not np.isnan(row[0]) and int(row[0]) == channel]
            truth_total += len(truth)
            used = set()
            for column, row_index, _score in predicted_peaks:
                best = None
                for number, (u, v) in enumerate(truth):
                    if number in used:
                        continue
                    distance = float(np.hypot(column - u * (data["grid"] - 1),
                                              row_index - v * (data["grid"] - 1)))
                    if best is None or distance < best[0]:
                        best = (distance, number)
                if best is not None and best[0] <= radius:
                    used.add(best[1])
                    matched += 1
                    errors.append(best[0])
        # One-to-one matching, and a pair of set metrics beside it.
        #
        # Markers of one class are often closer together than the matching radius:
        # 56 % of teammates have another teammate within it (median gap 5 grid px).
        # A single peak then legitimately stands for a whole cluster, and one-to-one
        # matching counts the rest as misses, which caps recall near 0.44 for
        # teammates no matter how good the model is. The set metrics answer the
        # question that actually matters: did a peak land near each marker, and did
        # every peak land near some marker.
        covered = 0
        peaks_near_truth = 0
        for position, sample in enumerate(index):
            if not allowed[position]:
                continue
            predicted_peaks = peaks[position][channel]
            truth = [(row[1], row[2]) for row in data["truth"][sample]
                     if not np.isnan(row[0]) and int(row[0]) == channel]
            for u, v in truth:
                if any(float(np.hypot(column - u * (data["grid"] - 1),
                                      row_index - v * (data["grid"] - 1))) <= radius
                       for column, row_index, _score in predicted_peaks):
                    covered += 1
            for column, row_index, _score in predicted_peaks:
                if any(float(np.hypot(column - u * (data["grid"] - 1),
                                      row_index - v * (data["grid"] - 1))) <= radius
                       for u, v in truth):
                    peaks_near_truth += 1
        metrics[name] = {
            "supervised": supervised,
            "predicted": predicted,
            "truth": truth_total,
            "matched": matched,
            "precision": (matched / predicted) if predicted else None,
            "recall": (matched / truth_total) if truth_total else None,
            "covered_truth": covered,
            "covered_recall": (covered / truth_total) if truth_total else None,
            "peaks_near_a_marker": peaks_near_truth,
            "peak_precision": (peaks_near_truth / predicted) if predicted else None,
            "error_grid_px_mean": float(np.mean(errors)) if errors else None,
            "error_grid_px_p95": float(np.percentile(errors, 95)) if errors else None,
        }

    # The false-positive test: a peak where the map paints NOTHING.
    #
    # An unpainted enemy is not automatically an empty place. Enemy positions sit
    # next to capture points, wrecks and teammates, so a peak there is often a
    # correct marker for something else. A position counts only when no true marker
    # of ANY class is within the matching radius, which is what isolates "the model
    # invented an enemy that is not on the map".
    phantom = 0
    negative_total = 0
    explained = 0
    for position, sample in enumerate(index):
        targets = data["negatives"][sample][:int(data["negative_counts"][sample])]
        negative_total += len(targets)
        true_points = [(row[1], row[2]) for row in data["truth"][sample]
                       if not np.isnan(row[0])]
        # Only peaks from SUPERVISED channels count. A channel the session could not
        # measure is not being trained, so what it emits is neither right nor wrong, and
        # counting it charges the model for a claim nobody made. Measured the moment the
        # mask landed (2026-10-14): the phantom rate went 0.318 -> 0.698 because the four
        # unconstrained channels (zone_A/B/C, lock) spray peaks that land on unpainted
        # enemies. This is the same rule the per-class metrics already follow.
        if validity is None:
            allowed_channels = range(len(classes))
        else:
            allowed_channels = [channel for channel in range(len(classes))
                                if validity[sample][channel]]
        all_peaks = [(column, row_index) for channel in allowed_channels
                     for column, row_index, _score in peaks[position][channel]]
        for u, v in targets:
            column, row_index = u * (data["grid"] - 1), v * (data["grid"] - 1)
            nearby_true = any(np.hypot(column - tu * (data["grid"] - 1),
                                       row_index - tv * (data["grid"] - 1)) <= radius
                              for tu, tv in true_points)
            nearby_peak = any(np.hypot(column - pc, row_index - pr) <= radius
                              for pc, pr in all_peaks)
            if nearby_true:
                explained += 1
                continue
            if nearby_peak:
                phantom += 1
    metrics["phantom_peaks_on_unpainted_enemies"] = {
        "positions": negative_total,
        "explained_by_a_true_marker": explained,
        "judged": negative_total - explained,
        "phantom_hits": phantom,
        "share": (phantom / (negative_total - explained)) if negative_total > explained else None,
    }

    with torch.no_grad():
        probabilities = torch.sigmoid(ring_logits)
        predicted_rings = (probabilities >= 0.5).float()
    # The rings are masked per SAMPLE, and an unsupervised sample carries an all-zero
    # ring target: "this session never measured the rings", not "no ring was enabled".
    # Scoring those as negatives is the same mistake the heatmap mask exists to remove.
    ring_validity = data.get("ring_valid")
    keep = [position for position, sample in enumerate(index)
            if ring_validity is None or ring_validity[sample]]
    ring_metrics = {}
    if keep:
        ring_truth = data["rings"][np.asarray([index[position] for position in keep])]
        guess_all = predicted_rings[keep]
        for column, label in enumerate(data["ring_labels"]):
            truth = ring_truth[:, column]
            guess = guess_all[:, column]
            true_positive = float(((guess == 1) & (truth == 1)).sum())
            false_positive = float(((guess == 1) & (truth == 0)).sum())
            false_negative = float(((guess == 0) & (truth == 1)).sum())
            ring_metrics[str(label)] = {
                "samples": int(len(keep)),
                "enabled_share": float(truth.mean()),
                "precision": true_positive / (true_positive + false_positive)
                             if (true_positive + false_positive) else None,
                "recall": true_positive / (true_positive + false_negative)
                          if (true_positive + false_negative) else None,
                "accuracy": float((guess == truth).float().mean()),
            }
    metrics["range_rings"] = ring_metrics
    return metrics, probabilities, heatmaps


def sweep_thresholds(model, index, data, thresholds):
    """Report precision and recall at several peak thresholds.

    The threshold is an operating point, not a property of the model, so the
    numbers at one value alone cannot say whether the model is good. Recording the
    curve keeps that visible.
    """
    table = {}
    for threshold in thresholds:
        metrics, _probabilities, _heatmaps = evaluate(model, index, data, threshold)
        entry = {name: {"precision": metrics[name]["precision"],
                        "recall": metrics[name]["recall"],
                        "covered_recall": metrics[name]["covered_recall"],
                        "peak_precision": metrics[name]["peak_precision"]}
                 for name in data["classes"]}
        entry["phantom"] = metrics["phantom_peaks_on_unpainted_enemies"]
        table[str(threshold)] = entry
        # Each sweep point is a full split's worth of tensors; holding the previous
        # one while the next is computed costs another copy of the split for nothing.
        del metrics, _probabilities, _heatmaps
    return table


def write_model(model, model_path, classes, ring_labels):
    """Write the scripted model and its state dict; returns the state dict path.

    Called both right after training and again at the end: the epochs are the
    expensive part, so they are on disk before the measurement tail runs. That tail
    used to be able to kill the process (it did), and it took 40 epochs with it.
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.script(model).save(str(model_path))
    state_path = model_path.with_suffix(".state_dict.pt")
    torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "architecture": ARCHITECTURE, "classes": list(classes),
                "ring_labels": list(ring_labels)}, str(state_path))
    return state_path


def as_float_batch(array):
    """A small per-sample array (a mask) as a torch float tensor."""
    return torch.from_numpy(np.ascontiguousarray(array)).float()


def masked_bce(logits, targets, mask, pos_weight=None):
    """Binary cross entropy over the elements the mask keeps, normalised by their count.

    ``reduction="mean"`` cannot express "part of this target is not supervision", and an
    unmeasurable class arrives as an all-zero target -- indistinguishable from a
    confirmed absence. That is how a session with no zone calibration or no lock log
    taught the model "these classes never exist".

    With nothing masked this calls the canonical ``reduction="mean"`` path, so a corpus
    that masks nothing trains EXACTLY as before -- and "exactly" has to include the
    gradients. The masked expression agrees with the mean in value but not bit for bit
    in the backward pass, because the reduction and the sum are different operators with
    different accumulation orders; that was measured, not assumed.
    """
    if mask is None:
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight,
                                                 reduction="mean")
    weights = mask.expand_as(targets)
    kept = weights.sum()
    if float(kept) == float(weights.numel()):
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight,
                                                 reduction="mean")
    if float(kept) == 0.0:
        # Nothing in this batch is supervised. Keep the graph connected so backward()
        # still works, and contribute exactly zero gradient.
        return logits.sum() * 0.0
    raw = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight,
                                            reduction="none")
    return (raw * weights).sum() / kept


class BatchSource:
    """One item per training batch: the crops, the targets, the rings and the masks.

    The crops leave this object as uint8 and are prepared on the GPU (see
    prepare_images), so the only per-batch cost left on the CPU is gathering bytes out
    of the memory-mapped corpus. A background thread does that gathering while the GPU
    works; the float conversion and the downsample that used to happen here cost 153 ms
    against 32 ms of GPU work and pinned the GPU at 21 %.

    Multi-process loading is deliberately not used: forked workers inherited CUDA
    state, spawned ones re-imported torch per worker, and both took a run to about
    20 GB of anonymous memory and an OOM kill within half a minute.
    """

    def __init__(self, images, heatmaps, rings, valid, ring_valid, train_index, order,
                 batch):
        self.images = images
        self.heatmaps = heatmaps
        self.rings = rings
        self.valid = valid
        self.ring_valid = ring_valid
        self.train_index = train_index
        self.order = order
        self.batch = batch

    def __len__(self):
        return int(np.ceil(len(self.order) / self.batch))

    def __getitem__(self, number):
        start = number * self.batch
        # atleast_1d: a batch of ONE must stay array-like. Indexing an ndarray with a
        # one-element index can come back as a scalar, which the lazy loader then reads
        # as a single sample (3D crops) and the model's BatchNorm rejects.
        index = np.atleast_1d(
            self.train_index[self.order[start:start + self.batch]])
        return (self.images[index], self.heatmaps[index], self.rings[index],
                as_float_batch(self.valid[index]), as_float_batch(self.ring_valid[index]))


def prefetch(source, depth):
    """Yield batches from one background thread, so crop work overlaps the GPU step.

    Threads, NOT DataLoader workers. Forking a process that has already initialised
    CUDA gives every child its own piece of driver state plus a copy-on-write view of
    the dataset, and eight workers took the run to 20 GB of anonymous memory and an
    OOM kill in 30 seconds. The loading itself is numpy, which releases the GIL, so a
    single thread does overlap the GPU work, and a bounded queue caps the memory at
    ``depth`` batches. Yields the same tuples as the source.
    """
    queue = Queue(maxsize=depth)
    finished = object()

    def produce():
        for number in range(len(source)):
            queue.put(source[number])
        queue.put(finished)

    threading.Thread(target=produce, daemon=True, name="prefetch").start()
    while True:
        item = queue.get()
        if item is finished:
            return
        yield item


def positive_weights(data, index, batch, cap):
    """Count negatives/positives within each channel's supervised training pixels.

    Keep counters integral and stream targets: a shared pixel total multiplies a
    channel's negative count by the number of other supervised channels.
    """
    channels = len(data["classes"])
    positives = torch.zeros(channels, dtype=torch.int64)
    pixels = torch.zeros_like(positives)
    for start in range(0, len(index), batch):
        rows = np.atleast_1d(index[start:start + batch])
        chunk = data["heatmaps"][rows]
        valid = torch.from_numpy(np.ascontiguousarray(data["valid"][rows])).bool()
        positives += ((chunk > 0.2) & valid[:, :, None, None]).sum(dim=(0, 2, 3))
        pixels += valid.sum(dim=0) * chunk.shape[2] * chunk.shape[3]
    ratios = (pixels - positives).double() / positives.clamp(min=1)
    return ratios.clamp(max=cap).float().view(1, -1, 1, 1)


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data = load_all(args.heatmaps, args.images, args.downsample)
    train_index = select(data, "train")
    validation_index = select(data, "validation")
    test_index = select(data, "test")
    classes = data["classes"]
    print("samples train %d validation %d test %d  grid %d  classes %s"
          % (len(train_index), len(validation_index), len(test_index), data["grid"],
             classes))

    device = pick_device(args.device)
    model = MinimapHeatmap(len(classes), len(data["ring_labels"]), data["grid"]).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    print("device %s  (%s)" % (device, torch.cuda.get_device_name(device)
                               if device.type == "cuda" else "no GPU in this build"))
    positive_weight = positive_weights(data, train_index, args.batch,
                                       args.max_pos_weight).to(device)
    print("每通道 pos_weight: %s"
          % {name: round(float(positive_weight[0, i, 0, 0]), 1)
             for i, name in enumerate(classes)})
    started = time.time()
    best_state = None
    best_score = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        # numpy, not torch.randperm: the order indexes a numpy index array, and the
        # seed is already set for numpy above.
        order = np.random.permutation(len(train_index))
        total = 0.0
        # One in-process thread prefetches the gather; the float conversion and the
        # downsample happen on the GPU instead of here. Multi-process loading was tried
        # twice (fork, then spawn) and OOM-killed this machine both times at ~20 GB of
        # anonymous memory, so nothing in this path starts a second process.
        source = BatchSource(data["images"], data["heatmaps"], data["rings"],
                             data["valid"], data["ring_valid"], train_index, order,
                             args.batch)
        for images, targets, rings, valid_batch, ring_valid_batch in prefetch(
                source, args.prefetch):
            optimiser.zero_grad()
            heatmaps, ring_logits = model(
                prepare_images(images, device, data["downsample"]))  # logits
            rows = valid_batch.shape[0]
            mask = valid_batch.to(device).view(rows, -1, 1, 1)
            heat_loss = masked_bce(heatmaps, to_device(targets, device), mask,
                                   pos_weight=positive_weight)
            ring_loss = masked_bce(ring_logits, to_device(rings, device),
                                   ring_valid_batch.to(device).view(rows, -1))
            loss = heat_loss + args.ring_weight * ring_loss
            loss.backward()
            optimiser.step()
            total += float(loss) * len(images)
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics, _probabilities, _heatmaps = evaluate(
                model, validation_index, data, stream=True, device=device)
            own = metrics["own"]
            friendly = metrics["friendly"]
            score = (own["error_grid_px_mean"] or 99.0)
            marker = ""
            if score < best_score:
                best_score = score
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                # Write it the moment it improves, not only after the loop. A run that
                # died at epoch 31 (the machine rebooted at 18:00:31) lost 30 epochs
                # because the only save was past the loop; now any death costs at most
                # one validation interval.
                write_model(model, args.model_out, classes, data["ring_labels"])
                marker = " *"
            print("epoch %3d  loss %.6f  validation own err %s grid px  own P/R %.3f/%.3f  "
                  "friendly P/R %.3f/%.3f  %.0fs%s"
                  % (epoch, total / len(order),
                     "%.2f" % own["error_grid_px_mean"] if own["error_grid_px_mean"] else "n/a",
                     own["precision"] or 0.0, own["recall"] or 0.0,
                     friendly["precision"] or 0.0, friendly["recall"] or 0.0,
                     time.time() - started, marker))
            del metrics, _probabilities, _heatmaps
    elapsed = time.time() - started
    if best_state is None:
        raise SystemExit("training produced no checkpoint")
    model.load_state_dict(best_state)
    write_model(model, args.model_out, classes, data["ring_labels"])
    print("wrote %s before the measurement tail" % args.model_out)

    # Measuring three splits in a row keeps each one's tensors alive until the next
    # call RETURNS, so the previous split is dropped before the next one is measured.
    # All three stream: the training split alone is 12.4 GB as one tensor.
    train_metrics, _probabilities, _heatmaps = evaluate(
        model, train_index, data, stream=True, device=device)
    _probabilities = _heatmaps = None
    validation_metrics, _probabilities, _heatmaps = evaluate(
        model, validation_index, data, stream=True, device=device)
    _probabilities = _heatmaps = None
    test_metrics, _probabilities, _heatmaps = evaluate(
        model, test_index, data, stream=True, device=device)
    _probabilities = _heatmaps = None

    # Convert grid error to crop pixels and metres using the measured map scale.
    scale = json.loads(Path(args.minimap_calibration).read_text(encoding="utf-8"))
    metres_per_pixel = float(scale["rings"]["metres_per_pixel"])
    grid_to_crop = data["crop_rect"][2] / float(data["grid"])
    def in_metres(metrics):
        for name, entry in metrics.items():
            if not isinstance(entry, dict) or "error_grid_px_mean" not in entry:
                continue
            entry["grid_to_crop_px"] = round(grid_to_crop, 3)
            for key in ("error_grid_px_mean", "error_grid_px_p95"):
                if entry.get(key) is not None:
                    entry[key.replace("grid_px", "crop_px")] = round(entry[key] * grid_to_crop, 3)
                    entry[key.replace("grid_px", "m")] = round(
                        entry[key] * grid_to_crop * metres_per_pixel, 1)
        return metrics
    in_metres(train_metrics), in_metres(validation_metrics), in_metres(test_metrics)

    print("test:")
    for name in data["classes"]:
        entry = test_metrics[name]
        print("   %-14s 覆盖召回 %.3f  峰精度 %.3f  一对一 P/R %.3f/%.3f  误差 %s 格 = %s m"
              "  (真值 %d, 预测 %d)"
              % (name, entry["covered_recall"] or 0.0, entry["peak_precision"] or 0.0,
                 entry["precision"] or 0.0, entry["recall"] or 0.0,
                 "%.2f" % entry["error_grid_px_mean"] if entry["error_grid_px_mean"] else "n/a",
                 "%.0f" % entry["error_m_mean"] if entry.get("error_m_mean") else "n/a",
                 entry["truth"], entry["predicted"]))
    negative = test_metrics["phantom_peaks_on_unpainted_enemies"]
    print("   phantom peaks on unpainted enemies: %d of %d judged positions (%.3f); "
          "%d positions were explained by another true marker and are excluded"
          % (negative["phantom_hits"], negative["judged"], negative["share"] or 0.0,
             negative["explained_by_a_true_marker"]))
    sweep = sweep_thresholds(model, test_index, data, (0.2, 0.3, 0.4, 0.5))
    print("   阈值扫描（test, precision/recall）:")
    for threshold, entry in sweep.items():
        print("      t=%s  %s" % (threshold, "  ".join(
            "%s %.2f/%.2f" % (name, item["precision"] or 0.0, item["recall"] or 0.0)
            for name, item in entry.items() if name in ("own", "friendly", "enemy_visible"))))
    for label, entry in test_metrics["range_rings"].items():
        print("   ring %-5s accuracy %.3f precision %s recall %s (enabled %.2f)"
              % (label, entry["accuracy"],
                 "%.3f" % entry["precision"] if entry["precision"] is not None else "n/a",
                 "%.3f" % entry["recall"] if entry["recall"] is not None else "n/a",
                 entry["enabled_share"]))

    state_path = write_model(model, args.model_out, classes, data["ring_labels"])
    model_path = Path(args.model_out)

    manifest = {
        "schema_version": 1,
        "task": "minimap_multi_class_heatmap",
        "architecture": ARCHITECTURE,
        "input": {
            "shape": [1, 3, data["images"].shape[2], data["images"].shape[3]],
            "dtype": "float32", "range": [0.0, 1.0],
            "note": "minimap crop divided by 255, averaged %dx%d from the calibrated "
                    "rectangle" % (args.downsample, args.downsample),
            "source_rect_xywh": data["crop_rect"],
        },
        "output": {
            "heatmaps": {"channels": list(classes), "grid": [data["grid"], data["grid"]],
                         "range": [0.0, 1.0],
                         "note": "one heatmap per class; the model outputs LOGITS, so "
                                 "apply sigmoid before decoding peaks"},
            "range_rings": {"labels_km": list(data["ring_labels"]),
                            "note": "logits, apply sigmoid; 1 means the player has that "
                                    "range circle enabled"},
        },
        "decoding": {"peak_threshold": PEAK_THRESHOLD,
                     "match_radius_grid_px": round(match_radius(data), 3),
                     "max_peaks_per_class": MAX_PEAKS_PER_CLASS,
                     "grid_to_crop_px": round(grid_to_crop, 3),
                     "crop_px_to_metres": metres_per_pixel},
        "training": {"epochs": args.epochs, "batch": args.batch, "learning_rate": args.lr,
                     "device": str(device),
                     "positive_weight_rule": "per_channel_supervised_pixels_v1",
                     "positive_weight_per_channel": {
                         name: round(float(positive_weight[0, i, 0, 0]), 2)
                         for i, name in enumerate(classes)},
                     "positive_weight_cap": args.max_pos_weight,
                     "ring_weight": args.ring_weight,
                     "seed": args.seed, "downsample": args.downsample,
                     "seconds": round(elapsed, 1)},
        "dataset": {"heatmaps": str(args.heatmaps), "heatmaps_sha256": sha256(args.heatmaps),
                    "images": str(args.images),
                    "samples": {"train": int(len(train_index)),
                                "validation": int(len(validation_index)),
                                "test": int(len(test_index))},
                    "split_rule": "contiguous interleaved blocks from the dataset"},
        "metrics": {"train": train_metrics, "validation": validation_metrics,
                    "test": test_metrics,
                    "threshold_sweep_test": sweep},
        "artifact": {"path": str(model_path), "sha256": sha256(model_path),
                     "format": "torch.jit.script", "state_dict": str(state_path),
                     "state_dict_sha256": sha256(state_path)},
    }
    manifest_path = Path(args.manifest_out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n",
                             encoding="utf-8")
    print("wrote %s and %s" % (model_path, manifest_path))
    return 0


def validate(args):
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    heatmap_path = Path(manifest["dataset"]["heatmaps"])
    if not heatmap_path.exists():
        raise SystemExit("heatmap dataset %s is missing" % heatmap_path)
    if sha256(heatmap_path) != manifest["dataset"]["heatmaps_sha256"]:
        raise SystemExit("heatmap dataset sha256 changed; the claim cannot be re-checked")
    data = load_all(heatmap_path, manifest["dataset"]["images"],
                    manifest["training"]["downsample"])
    # The manifest names the device the metrics were recorded on (an older manifest
    # without the key was trained on the CPU, which is what "cpu" says). Recompute on
    # the same one: the comparison tolerance is 0.02 grid px, but there is no reason
    # to spend it on a device mismatch.
    device = pick_device(manifest.get("training", {}).get("device") or "cpu")
    model = torch.jit.load(args.model).to(device)
    index = select(data, "test")
    metrics, _probabilities, _heatmaps = evaluate(model, index, data, stream=True, device=device)
    expected = manifest["metrics"]["test"]
    print("recorded vs recomputed (test):")
    failed = []
    for name in data["classes"]:
        recorded = expected[name]["error_grid_px_mean"]
        actual = metrics[name]["error_grid_px_mean"]
        line = "   %-14s %s -> %s" % (name,
                                      "n/a" if recorded is None else "%.4f" % recorded,
                                      "n/a" if actual is None else "%.4f" % actual)
        if recorded is not None and actual is not None and abs(recorded - actual) > 0.02:
            line += "  FAIL"
            failed.append(name)
        print(line)
    if failed:
        print("VERDICT FAIL: %s drifted" % ", ".join(failed))
        return 1
    position = data["crop_rect"]
    print("grid %.0fx%.0f maps to a %dx%d crop; 1 grid px = %.1f m"
          % (data["grid"], data["grid"], position[2], position[3],
             manifest["decoding"]["grid_to_crop_px"] * manifest["decoding"]["crop_px_to_metres"]))
    print("VERDICT pass")
    return 0


def preview(args):
    import subprocess
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    data = load_all(manifest["dataset"]["heatmaps"], manifest["dataset"]["images"],
                    manifest["training"]["downsample"])
    device = pick_device(manifest.get("training", {}).get("device") or "cpu")
    model = torch.jit.load(args.model).to(device)
    index = select(data, "test")
    with torch.no_grad():
        heatmap_logits, _logits = model(prepare_images(
            data["images"][index], device, data["downsample"]))
    peaks = decode(torch.sigmoid(heatmap_logits).cpu())
    count = min(args.preview_count, len(index))
    stride = max(1, len(index) // count)
    picks = list(range(0, len(index), stride))[:count]
    colours = [(255, 255, 255), (0, 255, 255), (255, 40, 40), (170, 120, 255),
               (255, 235, 40), (120, 220, 120), (255, 150, 30)]
    grid = data["grid"]
    side = data["crop_rect"][2]
    tiles = []
    for position, sample_in_split in zip(picks, [index[i] for i in picks]):
        tile = torch.nn.functional.interpolate(
            data["images"][sample_in_split].unsqueeze(0), size=(side, side),
            mode="bilinear", align_corners=False)[0]
        tile = np.ascontiguousarray((tile.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        for channel, name in enumerate(data["classes"]):
            for row in data["truth"][sample_in_split]:
                if np.isnan(row[0]) or int(row[0]) != channel:
                    continue
                draw_cross(tile, row[1], row[2], colours[channel % len(colours)], arm=5)
        for channel, entries in enumerate(peaks[position]):
            for column, row_index, _score in entries:
                draw_cross(tile, column / (grid - 1), row_index / (grid - 1),
                           colours[channel % len(colours)], arm=2)
        tiles.append(tile)
    columns = 2
    rows = (len(tiles) + columns - 1) // columns
    height, width = tiles[0].shape[0], tiles[0].shape[1]
    canvas = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for position, tile in enumerate(tiles):
        row, column = divmod(position, columns)
        canvas[row * height:(row + 1) * height,
               column * width:(column + 1) * width] = tile
    out = Path(args.preview_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", "%dx%d" % (canvas.shape[1], canvas.shape[0]), "-i", "-",
                    "-frames:v", "1", "-y", str(out)],
                   input=np.ascontiguousarray(canvas).tobytes(), check=True)
    print("large cross = replay truth, small cross = model peak; wrote %s" % out)
    return 0


def draw_cross(array, u, v, colour, arm=4):
    height, width = array.shape[0], array.shape[1]
    column = int(round(u * (width - 1)))
    row_index = int(round(v * (height - 1)))
    for offset in range(-arm, arm + 1):
        if 0 <= row_index + offset < height:
            array[row_index + offset, column] = colour
        if 0 <= column + offset < width:
            array[row_index, (column + offset) % width] = colour
    return array


def sweep_only(args):
    """Report the operating-point trade-off of a saved model without retraining."""
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    data = load_all(manifest["dataset"]["heatmaps"], manifest["dataset"]["images"],
                    manifest["training"]["downsample"])
    model = torch.jit.load(args.model)
    index = select(data, args.split)
    print("split=%s  阈值  类别            一对一 P/R     覆盖召回/峰精度" % args.split)
    for threshold in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        metrics, _probabilities, _heatmaps = evaluate(model, index, data, threshold)
        phantom = metrics["phantom_peaks_on_unpainted_enemies"]
        for name in ("own", "friendly", "enemy_visible"):
            entry = metrics[name]
            print("  %.1f  %-14s %.3f/%.3f     %.3f/%.3f"
                  % (threshold, name, entry["precision"] or 0.0, entry["recall"] or 0.0,
                     entry["covered_recall"] or 0.0, entry["peak_precision"] or 0.0))
        print("       phantom %d/%d = %.3f"
              % (phantom["phantom_hits"], phantom["judged"], phantom["share"] or 0.0))
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--sweep", action="store_true",
                        help="report the threshold/phantom trade-off of a saved model")
    parser.add_argument("--split", default="validation",
                        help="which split to measure with, default validation: an "
                             "operating point must not be chosen on the test set")
    parser.add_argument("--heatmaps", type=Path)
    parser.add_argument("--images", type=Path)
    parser.add_argument("--model-out", type=Path)
    parser.add_argument("--manifest-out", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--preview-out", type=Path)
    parser.add_argument("--preview-count", type=int, default=4)
    parser.add_argument("--minimap-calibration", type=Path, default=Path(
        "replays/parsed/calibration/minimap-2560x1600.json"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--device", default="auto",
                        help="auto (GPU when this torch build can see one), cpu, cuda, "
                             "or cuda:N")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--prefetch", type=int, default=6,
                        help="batches a background thread gathers ahead; 0 gathers on "
                             "the main thread. Loading is never spread over worker "
                             "processes: forked or spawned, that cost this machine "
                             "20 GB and an OOM kill")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--ring-weight", type=float, default=0.2)
    parser.add_argument("--max-pos-weight", type=float, default=2000.0)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.sweep:
        if not args.model or not args.manifest:
            raise SystemExit("--sweep needs --model and --manifest")
        return sweep_only(args)
    if args.validate or args.preview:
        if not args.model or not args.manifest:
            raise SystemExit("--validate/--preview need --model and --manifest")
        return validate(args) if args.validate else preview(args)
    if not args.heatmaps or not args.images or not args.model_out or not args.manifest_out:
        raise SystemExit("training needs --heatmaps, --images, --model-out and --manifest-out")
    return train(args)


if __name__ == "__main__":
    sys.exit(main())
