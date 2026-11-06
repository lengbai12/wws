#!/usr/bin/env python3
"""Train BowNet on real, reviewed 256px RGB bow annotations with named provenance.

Three disjoint session splits are deterministic; decoded-pixel and source-image
leakage is rejected before training. Keypoints use normalized spatial targets
and log-softmax cross entropy, not sparse sigmoid MSE. Validation selects the
lowest mean bow/stern argmax pixel error (mask IoU breaks ties). Camera-relative
orientation is a separately reviewed annotation: never derive it from 2D keypoints.

A training run is NOT a fire-control acceptance test. Known orientation bins
describe coverage only; an unknown-only dataset cannot validate orientation.
Confidence and cross-session accuracy require separate operational evaluation.
Every exported package remains unaccepted and unauthorized for fire control.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

# Must precede CUDA initialization for deterministic CUDA matrix operations.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import bow_dataset
from bow_dataset import (SIZE, ORIENTATION_BINS, ORIENTATION_CONVENTION,
                         PIXEL_HASH_CONVENTION, CROP_CONVENTION,
                         load_annotations, pixel_sha256)
from fsutil import sha256

SCHEMA_VERSION = 1
MODEL_TYPE = "bow-net-v3"
BASE = 16
SPLITS = ("train", "validation", "test")


class BowNet(nn.Module):
    def __init__(self, base=BASE):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, base, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(base, base, 3, padding=1), nn.ReLU())
        self.down = nn.Sequential(nn.MaxPool2d(2),
                                  nn.Conv2d(base, base * 2, 3, padding=1), nn.ReLU())
        self.context = nn.Sequential(nn.MaxPool2d(2),
            nn.Conv2d(base * 2, base * 4, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base * 4, base * 4, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(base * 4, base * 4, 3, padding=4, dilation=4), nn.ReLU(),
            nn.Conv2d(base * 4, base * 4, 3, padding=8, dilation=8), nn.ReLU())
        self.up_half = nn.Sequential(nn.Conv2d(base * 6, base * 2, 3, padding=1), nn.ReLU())
        self.up_full = nn.Sequential(nn.Conv2d(base * 3, base, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(base, base, 3, padding=1), nn.ReLU())
        self.mask = nn.Conv2d(base, 1, 1)
        self.bow = nn.Conv2d(base, 1, 1)
        self.stern = nn.Conv2d(base, 1, 1)
        self.orientation = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                         nn.Linear(base * 4, ORIENTATION_BINS))

    def forward(self, x):
        full = self.stem(x)
        half = self.down(full)
        features = self.context(half)
        decoded = self.up_half(torch.cat((F.interpolate(features, size=half.shape[-2:],
            mode="bilinear", align_corners=False), half), dim=1))
        dense = self.up_full(torch.cat((F.interpolate(decoded, size=full.shape[-2:],
            mode="bilinear", align_corners=False), full), dim=1))
        return self.mask(dense), self.bow(dense), self.stern(dense), self.orientation(features)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cv2.setNumThreads(1)


def split_sessions(rows, seed):
    """Split complete sessions; input ordering cannot affect membership."""
    sessions = sorted({row["session"] for row in rows})
    if len(sessions) < 3:
        raise ValueError("at least three sessions are required for train/validation/test")
    random.Random(seed).shuffle(sessions)
    held_out = max(1, len(sessions) // 5)
    assignments = {session: "test" for session in sessions[:held_out]}
    assignments.update({session: "validation" for session in sessions[held_out:2 * held_out]})
    assignments.update({session: "train" for session in sessions[2 * held_out:]})
    return {name: [row for row in rows if assignments[row["session"]] == name] for name in SPLITS}


def prepare_inventory(splits, root):
    """Hash actual files and decoded pixels; reject leakage, not just path reuse.

    Row-local runtime digests are also checked whenever a crop/mask is loaded,
    so files changed during training cannot silently invalidate the inventory.
    """
    files, sample_records, seen_pixels, seen_sources = {}, [], {}, {}
    for split, rows in splits.items():
        for row in rows:
            image = cv2.imread(str(root / row["image"]), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(root / row["mask"]), cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape != (SIZE, SIZE, 3) or mask is None or mask.shape != (SIZE, SIZE):
                raise ValueError(f"invalid crop or mask: {row['image']}")
            image_digest, mask_digest = pixel_sha256(image), pixel_sha256(mask)
            previous = seen_pixels.setdefault(image_digest, split)
            if previous != split:
                raise ValueError(f"decoded pixel leakage between {previous} and {split}: {row['image']}")
            source_digest = row["source_sha256"]
            previous = seen_sources.setdefault(source_digest, split)
            if previous != split:
                raise ValueError(f"source image leakage between {previous} and {split}: {row['source_image']}")
            if image_digest != row["image_sha256"]:
                raise ValueError(f"crop changed since annotation validation: {row['image']}")
            row["_image_pixels_sha256"] = image_digest
            row["_mask_pixels_sha256"] = mask_digest
            for field in ("image", "mask", "source_image"):
                relative = row[field]
                if relative not in files:
                    path = root / relative
                    files[relative] = {"sha256": sha256(path), "bytes": path.stat().st_size}
            sample_records.append({"split": split, "session": row["session"],
                                   "annotation_source": row["annotation_source"], "reviewer": row["reviewer"],
                                   "image": row["image"], "mask": row["mask"],
                                   "image_pixels_sha256": image_digest,
                                   "mask_pixels_sha256": mask_digest,
                                   "source_image": row["source_image"],
                                   "source_pixels_sha256": source_digest,
                                   "labels": {key: row[key] for key in ("label_status", "bow_point", "stern_point", "orientation_bin")}})
    return {"files": files, "samples": sample_records,
            "pixel_hash_convention": PIXEL_HASH_CONVENTION,
            "file_hash_convention": "sha256 exact file bytes"}


def heatmap(point, size=SIZE, sigma=5.0):
    """A probability distribution, normalized even for boundary keypoints."""
    y, x = np.ogrid[:size, :size]
    weights = np.exp(-((x - point[0]) ** 2 + (y - point[1]) ** 2) / (2 * sigma ** 2))
    return (weights / weights.sum()).astype(np.float32)


def augment_sample(image, mask, points, generator):
    """Translate/scale existing labels without inventing a camera orientation."""
    for _ in range(16):
        scale = generator.uniform(.65, 1.25)
        shift = generator.uniform(-45., 45., 2)
        matrix = np.array([[scale, 0., (1-scale)*127.5+shift[0]],
                           [0., scale, (1-scale)*127.5+shift[1]]], np.float32)
        transformed = np.asarray(points, np.float32).reshape(-1, 2) * scale + matrix[:, 2]
        if len(transformed) and not ((transformed >= 6).all() and (transformed < SIZE-6).all()):
            continue
        return (cv2.warpAffine(image, matrix, (SIZE, SIZE), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT_101),
                cv2.warpAffine(mask, matrix, (SIZE, SIZE), flags=cv2.INTER_NEAREST),
                transformed.tolist())
    return image, mask, points


def load_batch(rows, root, generator=None):
    images, masks, bows, sterns, orientations, positives, points = [], [], [], [], [], [], []
    for row in rows:
        image = cv2.imread(str(root / row["image"]), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(root / row["mask"]), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (SIZE, SIZE, 3) or mask is None or mask.shape != (SIZE, SIZE):
            raise ValueError(f"invalid crop or mask: {row['image']}")
        for field, array in (("_image_pixels_sha256", image), ("_mask_pixels_sha256", mask)):
            if pixel_sha256(array) != row[field]:
                raise ValueError(f"dataset pixels changed during training: {row['image']}")
        positive = row["label_status"] == "positive"
        keypoints = [row["bow_point"], row["stern_point"]] if positive else []
        if generator is not None:
            image, mask, keypoints = augment_sample(image, mask, keypoints, generator)
        images.append(image[:, :, ::-1].astype(np.float32) / 255.0)
        masks.append((mask > 0).astype(np.float32))
        bows.append(heatmap(keypoints[0]) if positive else np.zeros((SIZE, SIZE), np.float32))
        sterns.append(heatmap(keypoints[1]) if positive else np.zeros((SIZE, SIZE), np.float32))
        orientations.append(row["orientation_bin"] if positive and isinstance(row["orientation_bin"], int) else -1)
        positives.append(positive)
        points.append(keypoints if positive else [[0., 0.], [0., 0.]])
    return (torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2),
            torch.from_numpy(np.stack(masks))[:, None],
            torch.from_numpy(np.stack(bows))[:, None],
            torch.from_numpy(np.stack(sterns))[:, None],
            torch.tensor(orientations, dtype=torch.long), torch.tensor(positives, dtype=torch.bool),
            torch.tensor(points, dtype=torch.float32))


def spatial_cross_entropy(logits, target):
    return -(target.flatten(1) * F.log_softmax(logits.flatten(1), dim=1)).sum(1)


def balanced_mask_loss(logits, target):
    """Give hull and sea equal weight; an empty hull still supervises sea."""
    raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    foreground = target > .5
    counts = foreground.flatten(1).sum(1)
    background_counts = (~foreground).flatten(1).sum(1)
    hull = (raw * foreground).flatten(1).sum(1) / counts.clamp_min(1)
    sea = (raw * ~foreground).flatten(1).sum(1) / background_counts.clamp_min(1)
    regions = (counts > 0).float() + (background_counts > 0).float()
    return ((hull + sea) / regions).mean()


def loss_terms(outputs, batch):
    _, mask, bow, stern, orient, positive, _ = batch
    pm, pb, ps, po = outputs
    # All terms are per-sample sums divided by batch size. Negative samples
    # supervise only the mask; unknown headings never act as invented labels.
    count = mask.shape[0]
    zero = pb.sum() * 0.0
    bow_loss = spatial_cross_entropy(pb[positive], bow[positive]).sum() / count if positive.any() else zero
    stern_loss = spatial_cross_entropy(ps[positive], stern[positive]).sum() / count if positive.any() else ps.sum() * 0.0
    known = orient >= 0
    orient_loss = F.cross_entropy(po[known], orient[known], reduction="sum") / count if known.any() else po.sum() * 0.0
    return {"mask": balanced_mask_loss(pm, mask), "bow": bow_loss,
            "stern": stern_loss, "orientation": orient_loss}


def decode_keypoints(logits):
    flat = logits.flatten(1).argmax(1)
    return torch.stack((flat % logits.shape[-1], flat // logits.shape[-1]), dim=1).float()


class Metrics:
    def __init__(self):
        self.samples = self.positives = self.negative_samples = self.known = self.correct = 0
        self.mask_iou = self.positive_mask_iou = self.bow_error = self.stern_error = self.negative_mask_fraction = 0.0
        self.losses = {name: 0.0 for name in ("mask", "bow", "stern", "orientation")}
        self.orientation_counts = [0] * ORIENTATION_BINS
        self.orientation_correct = [0] * ORIENTATION_BINS

    def update(self, outputs, batch, losses):
        _, mask, _, _, orient, positive, points = batch
        pm, pb, ps, po = outputs
        count = mask.shape[0]
        self.samples += count
        for name, value in losses.items():
            self.losses[name] += float(value.detach()) * count
        predicted, target = pm[:, 0] >= 0, mask[:, 0] > 0
        intersection = (predicted & target).flatten(1).sum(1)
        union = (predicted | target).flatten(1).sum(1)
        iou = torch.where(union > 0, intersection.float() / union.clamp_min(1), 1.0)
        self.mask_iou += float(iou.sum())
        n_positive = int(positive.sum())
        self.positives += n_positive
        self.positive_mask_iou += float(iou[positive].sum())
        if n_positive:
            self.bow_error += float(torch.linalg.vector_norm(decode_keypoints(pb)[positive] - points[positive, 0], dim=1).sum())
            self.stern_error += float(torch.linalg.vector_norm(decode_keypoints(ps)[positive] - points[positive, 1], dim=1).sum())
        self.negative_samples += count - n_positive
        self.negative_mask_fraction += float(predicted[~positive].float().flatten(1).mean(1).sum())
        known = orient >= 0
        labels = orient[known]
        correct = po.argmax(1)[known] == labels
        self.known += int(known.sum())
        self.correct += int(correct.sum())
        counts = torch.bincount(labels, minlength=ORIENTATION_BINS).tolist()
        successes = torch.bincount(labels[correct], minlength=ORIENTATION_BINS).tolist()
        self.orientation_counts = [a + b for a, b in zip(self.orientation_counts, counts)]
        self.orientation_correct = [a + b for a, b in zip(self.orientation_correct, successes)]

    def result(self):
        losses = {name: value / self.samples for name, value in self.losses.items()}
        bow = self.bow_error / self.positives if self.positives else None
        stern = self.stern_error / self.positives if self.positives else None
        return {"samples": self.samples, "positive_samples": self.positives,
                "negative_samples": self.negative_samples, "losses": losses, "loss": sum(losses.values()),
                "mask_iou": self.mask_iou / self.samples,
                "positive_mask_iou": self.positive_mask_iou / self.positives if self.positives else None,
                "negative_mask_positive_fraction": self.negative_mask_fraction / self.negative_samples if self.negative_samples else None,
                "bow_error_px": bow, "stern_error_px": stern,
                "mean_keypoint_error_px": (bow + stern) / 2 if bow is not None else None,
                "orientation_known_samples": self.known,
                "orientation_accuracy": self.correct / self.known if self.known else None,
                "orientation_counts": self.orientation_counts, "orientation_correct": self.orientation_correct}


def evaluate(model, rows, root, batch_size, device):
    model.eval()
    metrics = Metrics()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = tuple(t.to(device) for t in load_batch(rows[start:start + batch_size], root))
            outputs = model(batch[0])
            metrics.update(outputs, batch, loss_terms(outputs, batch))
    return metrics.result()


def split_summary(rows):
    counts = [0] * ORIENTATION_BINS
    positive = unknown = 0
    for row in rows:
        if row["label_status"] == "positive":
            positive += 1
            if isinstance(row["orientation_bin"], int):
                counts[row["orientation_bin"]] += 1
            else:
                unknown += 1
    return {"samples": len(rows), "sessions": sorted({r["session"] for r in rows}),
            "annotation_source_counts": {source: sum(r["annotation_source"] == source for r in rows)
                                         for source in sorted({r["annotation_source"] for r in rows})},
            "positive_samples": positive, "negative_samples": len(rows) - positive,
            "unknown_orientation_positives": unknown, "orientation_counts": counts,
            "missing_orientation_bins": [i for i, n in enumerate(counts) if not n]}


def atomic_save(path, value, *, checkpoint=False):
    """An interrupted write cannot replace the previously complete best file."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            if checkpoint:
                torch.save(value, handle)
            else:
                handle.write((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def metadata(*, orientation_supervised=True):
    return {"orientation_supervised": orientation_supervised,
            "schema_version": SCHEMA_VERSION, "model_type": MODEL_TYPE, "base": BASE,
            "input_size": SIZE, "input_channels": "RGB", "input_scale": "uint8 / 255.0",
            "crop_transform": CROP_CONVENTION,
            "mask_loss": "per-sample foreground/background balanced BCE; empty mask supervises background",
            "orientation_bins": ORIENTATION_BINS, "orientation_convention": ORIENTATION_CONVENTION,
            "keypoint_output": "spatial logits; argmax flat index -> [index % 256, index // 256]",
            "keypoint_loss": "spatial log-softmax cross entropy; normalized sigma=5px Gaussian target",
            "mask_output": "binary logits; probability sigmoid; metric threshold 0.5",
            "accepted": False, "fire_control_authorized": False,
            "acceptance_reason": "training only; no independent operational/confidence acceptance performed"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch < 1 or not 0 <= args.seed < 2**32:
        parser.error("epochs/batch must be positive and seed must be in 0..2**32-1")
    try:
        rows = load_annotations(args.annotations)
        root = args.annotations.parent
        splits = split_sessions(rows, args.seed)
        summaries = {name: split_summary(part) for name, part in splits.items()}
        orientation_supervised = any(summaries["train"]["orientation_counts"])
        if any(not summary["positive_samples"] for summary in summaries.values()):
            raise ValueError("every split needs reviewed positive keypoints; add independent positive sessions")
        inventory = prepare_inventory(splits, root)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    model = BowNet(base=BASE).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    generator = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=False)
    source_paths = (Path(__file__).resolve(), Path(bow_dataset.__file__).resolve())
    manifest = {**metadata(orientation_supervised=orientation_supervised), "status": "training", "seed": args.seed,
                "deterministic_algorithms": True, "device": str(device),
                "versions": {"torch": str(torch.__version__), "numpy": np.__version__, "opencv": cv2.__version__},
                "epochs_requested": args.epochs, "batch_size": args.batch,
                "optimizer": {"name": "AdamW", "lr": 3e-4, "weight_decay": 0.01},
                "augmentation": {"train_only": True, "scale": [.65, 1.25], "translation_px": [-45, 45],
                                 "orientation_unchanged": True, "creates_independent_sessions": False},
                "annotations": {"path": str(args.annotations.resolve()), "sha256": sha256(args.annotations)},
                "source_sha256": {path.name: sha256(path) for path in source_paths},
                "dataset": inventory, "splits": summaries,
                "all_orientation_bins_in_each_split": all(not s["missing_orientation_bins"] for s in summaries.values()),
                "coverage_is_acceptance": False,
                "checkpoint_selection": "minimum validation mean_keypoint_error_px; tie minimum 1-positive_mask_iou",
                "model": None}
    # Hash the inventory itself so a checkpoint identifies its exact provenance.
    manifest["dataset_sha256"] = hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest()
    atomic_save(args.out / "manifest.json", manifest)
    best_key, best_epoch = None, None
    history = []
    train_rows = splits["train"]
    for epoch in range(1, args.epochs + 1):
        model.train()
        metrics = Metrics()
        order = generator.permutation(len(train_rows))
        for start in range(0, len(order), args.batch):
            selected = [train_rows[int(i)] for i in order[start:start + args.batch]]
            batch = tuple(t.to(device) for t in load_batch(selected, root, generator))
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch[0])
            terms = loss_terms(outputs, batch)
            loss = sum(terms.values())
            if not bool(torch.isfinite(loss)):
                raise ValueError("non-finite loss; no checkpoint acceptance possible")
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                metrics.update(outputs, batch, terms)
        validation = evaluate(model, splits["validation"], root, args.batch, device)
        record = {"epoch": epoch, "train": metrics.result(), "validation": validation}
        history.append(record)
        atomic_save(args.out / "history.json", history)
        print(json.dumps(record, allow_nan=False), flush=True)
        key = (validation["mean_keypoint_error_px"], 1.0 - validation["positive_mask_iou"])
        if best_key is None or key < best_key:
            checkpoint = {**metadata(orientation_supervised=orientation_supervised), "state_dict": model.state_dict(), "epoch": epoch,
                          "seed": args.seed, "validation": validation,
                          "dataset_sha256": manifest["dataset_sha256"],
                          "annotations_sha256": manifest["annotations"]["sha256"],
                          "source_sha256": manifest["source_sha256"]}
            checkpoint_path = args.out / "bow-model.pt"
            atomic_save(checkpoint_path, checkpoint, checkpoint=True)
            best_key, best_epoch = key, epoch
            manifest["model"] = {"path": checkpoint_path.name, "sha256": sha256(checkpoint_path),
                                 "bytes": checkpoint_path.stat().st_size, "epoch": epoch,
                                 "validation": validation}
            atomic_save(args.out / "manifest.json", manifest)
    # Test is evaluated once, on the selected validation checkpoint, not each epoch.
    checkpoint = torch.load(args.out / "bow-model.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    test = evaluate(model, splits["test"], root, args.batch, device)
    manifest.update(status="complete_unaccepted", best_epoch=best_epoch, test=test)
    atomic_save(args.out / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "best_epoch": best_epoch, "test": test,
                      "fire_control_authorized": False}, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
