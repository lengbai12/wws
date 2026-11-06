"""Train a minimap own-ship localiser from replay-derived labels.

Input  : one minimap crop, taken from the recording at the calibrated rectangle.
Output : the own ship's marker position as (u, v), normalised inside that
         rectangle, so the target does not depend on the crop's pixel size.

The labels are replay truth projected by the calibrated affine, so the model is
not imitating a detector. The affine-residual noise floor is carried into the
manifest: a model cannot beat the labels it was given, and reporting an error
below that floor would mean the test set leaked.

Splits come from the dataset, which uses contiguous video-time blocks. A
per-sample random split is refused here as well, because adjacent crops of one
battle are nearly identical.

Usage:
  train_minimap.py --dataset replays/parsed/datasets/minimap-<session>.npz \
      --model-out replays/parsed/artifacts/minimap-localizer-<session>.pt \
      --manifest-out replays/parsed/artifacts/minimap-localizer-<session>.json \
      --epochs 40 --downsample 2

  train_minimap.py --validate \
      --model replays/parsed/artifacts/minimap-localizer-<session>.pt \
      --manifest replays/parsed/artifacts/minimap-localizer-<session>.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from fsutil import sha256


import numpy as np
import torch
import torch.nn as nn

PRESENT = 0
ARCHITECTURE = "conv4-avgpool4-linear"


class MinimapLocalizer(nn.Module):
    """A small convolutional regressor for one marker position."""

    def __init__(self, channels=(16, 32, 64, 64)):
        super().__init__()
        blocks = []
        incoming = 3
        for outgoing in channels:
            blocks += [
                nn.Conv2d(incoming, outgoing, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(outgoing),
                nn.ReLU(inplace=True),
            ]
            incoming = outgoing
        blocks += [nn.AdaptiveAvgPool2d(4), nn.Flatten(start_dim=1)]
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(4 * 4 * channels[-1], 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            nn.Sigmoid(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(images))


def load_dataset(path, downsample):
    data = np.load(path, allow_pickle=False)
    images = torch.from_numpy(data["images"]).permute(0, 3, 1, 2).float() / 255.0
    if downsample > 1:
        images = nn.functional.avg_pool2d(images, downsample)
    labels = torch.from_numpy(data["labels"]).float()
    splits = np.asarray([str(item) for item in data["splits"]])
    occupancy = torch.from_numpy(data["occupancy"]).long()
    return {
        "images": images,
        "labels": labels,
        "splits": splits,
        "occupancy": occupancy,
        "rect": [int(v) for v in data["minimap_rect"]],
    }


def masked_selection(data, name):
    """Return the images and labels of one split that carry a present marker."""
    present = data["occupancy"].numpy() == PRESENT
    index = np.nonzero((data["splits"] == name) & present)[0]
    return data["images"][index], data["labels"][index]


def pixel_errors(predicted, truth, rect):
    """Per-sample Euclidean error in minimap pixels."""
    delta = (predicted - truth) * torch.tensor([rect[2], rect[3]], dtype=torch.float32)
    return torch.linalg.norm(delta, dim=1)


def describe_errors(errors):
    values = errors.detach().cpu().numpy()
    return {
        "samples": int(values.size),
        "mean_px": float(values.mean()),
        "median_px": float(np.median(values)),
        "p95_px": float(np.percentile(values, 95)),
        "max_px": float(values.max()),
    }


def mean_baseline(train_images, train_labels, test_labels, rect):
    """Error of always predicting the training mean position."""
    mean = train_labels.mean(dim=0, keepdim=True).expand_as(test_labels)
    return describe_errors(pixel_errors(mean, test_labels, rect))


def evaluate(model, images, labels, rect):
    model.eval()
    with torch.no_grad():
        predicted = model(images)
    return predicted, describe_errors(pixel_errors(predicted, labels, rect))


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data = load_dataset(args.dataset, args.downsample)
    rect = data["rect"]
    train_images, train_labels = masked_selection(data, "train")
    validation_images, validation_labels = masked_selection(data, "validation")
    test_images, test_labels = masked_selection(data, "test")
    if min(len(train_images), len(validation_images), len(test_images)) < 20:
        raise SystemExit("a split holds fewer than 20 samples: %d/%d/%d"
                         % (len(train_images), len(validation_images), len(test_images)))
    print("samples train %d  validation %d  test %d  crop %s  rect %s"
          % (len(train_images), len(validation_images), len(test_images),
             list(data["images"].shape[2:]), rect))

    model = MinimapLocalizer()
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_function = nn.SmoothL1Loss(beta=0.02)

    best = None
    best_validation = float("inf")
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train_images))
        total = 0.0
        for start in range(0, len(order), args.batch):
            index = order[start:start + args.batch]
            optimiser.zero_grad()
            loss = loss_function(model(train_images[index]), train_labels[index])
            loss.backward()
            optimiser.step()
            total += float(loss) * len(index)
        train_loss = total / len(order)
        _predicted, validation = evaluate(model, validation_images, validation_labels, rect)
        marker = ""
        if validation["mean_px"] < best_validation:
            best_validation = validation["mean_px"]
            best = {name: value.detach().clone() for name, value in model.state_dict().items()}
            marker = " *"
        print("epoch %3d  loss %.6f  validation mean %.3f px  p95 %.3f px%s"
              % (epoch, train_loss, validation["mean_px"], validation["p95_px"], marker))
    elapsed = time.time() - started

    if best is None:
        raise SystemExit("training produced no checkpoint")
    model.load_state_dict(best)
    _train_predicted, train_metrics = evaluate(model, train_images, train_labels, rect)
    _validation_predicted, validation_metrics = evaluate(model, validation_images,
                                                         validation_labels, rect)
    _test_predicted, test_metrics = evaluate(model, test_images, test_labels, rect)
    metrics = {
        "train": train_metrics,
        "validation": validation_metrics,
        "test": test_metrics,
        "mean_baseline_test": mean_baseline(train_images, train_labels, test_labels, rect),
    }
    print("test mean %.3f px  median %.3f px  p95 %.3f px  max %.3f px"
          % (test_metrics["mean_px"], test_metrics["median_px"],
             test_metrics["p95_px"], test_metrics["max_px"]))
    print("mean-predictor baseline on test: %.3f px (a model must beat this)"
          % metrics["mean_baseline_test"]["mean_px"])

    scripted = torch.jit.script(model)
    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(model_path))
    # A state dict is written beside the scripted module because TorchScript
    # loading is deprecated from Python 3.14: the weights stay usable if the
    # serialised program stops loading.
    state_path = model_path.with_suffix(".state_dict.pt")
    torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "architecture": ARCHITECTURE,
                "channels": [16, 32, 64, 64]}, str(state_path))

    crop = [int(data["images"].shape[2]), int(data["images"].shape[3])]
    manifest = {
        "schema_version": 1,
        "task": "minimap_own_ship_localisation",
        "architecture": ARCHITECTURE,
        "input": {
            "shape": [1, 3, crop[0], crop[1]],
            "dtype": "float32",
            "range": [0.0, 1.0],
            "note": "RGB minimap crop, divided by 255, averaged %dx%d from the "
                    "calibrated rectangle" % (args.downsample, args.downsample),
            "source_rect_xywh": rect,
            "source_resolution": None,
        },
        "output": {
            "shape": [1, 2],
            "meaning": ["u", "v"],
            "range": [0.0, 1.0],
            "note": "own-ship marker position inside source_rect_xywh; multiply by the "
                    "rectangle size and add its origin to obtain frame pixels",
        },
        "training": {
            "epochs": args.epochs,
            "batch": args.batch,
            "learning_rate": args.lr,
            "seed": args.seed,
            "downsample": args.downsample,
            "loss": "SmoothL1(beta=0.02) on normalised coordinates",
            "seconds": round(elapsed, 1),
        },
        "dataset": {
            "path": str(args.dataset),
            "sha256": sha256(args.dataset),
            "samples": {"train": int(len(train_images)),
                        "validation": int(len(validation_images)),
                        "test": int(len(test_images))},
            "split_rule": "contiguous video-time blocks from the dataset builder",
        },
        "metrics": metrics,
        "label_noise_floor_px": None,
        "artifact": {"path": str(model_path), "sha256": sha256(model_path),
                     "format": "torch.jit.script",
                     "state_dict": str(state_path), "state_dict_sha256": sha256(state_path)},
    }
    summary_path = Path(args.dataset).with_suffix(".summary.json")
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        manifest["label_noise_floor_px"] = summary.get("label_noise_floor_px")
        manifest["metres_per_pixel"] = summary.get("metres_per_pixel")
        manifest["input"]["source_resolution"] = summary.get("resolution")
    manifest_path = Path(args.manifest_out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n",
                             encoding="utf-8")
    print("wrote %s and %s" % (model_path, manifest_path))
    return 0


def validate(args):
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    expected = manifest["metrics"]["test"]
    path = Path(manifest["dataset"]["path"])
    if not path.exists():
        raise SystemExit("dataset %s is missing, so the claim cannot be re-checked" % path)
    recorded = manifest["dataset"]["sha256"]
    actual = sha256(path)
    if recorded != actual:
        raise SystemExit("dataset sha256 changed: %s != %s" % (recorded, actual))
    model = torch.jit.load(args.model)
    data = load_dataset(path, manifest["training"]["downsample"])
    images, labels = masked_selection(data, "test")
    _predicted, metrics = evaluate(model, images, labels, data["rect"])
    print("recorded test mean %.4f px   recomputed %.4f px"
          % (expected["mean_px"], metrics["mean_px"]))
    floor = manifest.get("label_noise_floor_px")
    if floor is not None and metrics["mean_px"] < floor * 0.5:
        print("REFUSED the measured error is below half the label noise floor (%.3f px); "
              "the check or the split is wrong" % floor)
        return 1
    if abs(metrics["mean_px"] - expected["mean_px"]) > 0.05:
        print("FAIL recomputed error differs from the manifest by more than 0.05 px")
        return 1
    metres = manifest.get("metres_per_pixel")
    if metres:
        print("test mean %.3f px = %.2f m on the battle map (1 px = %.3f m)"
              % (metrics["mean_px"], metrics["mean_px"] * metres, metres))
    print("VERDICT pass")
    return 0


def draw_cross(array, u, v, colour, arm=7):
    """Draw a plus mark at normalised (u, v) inside an image array."""
    height, width = array.shape[0], array.shape[1]
    column = int(round(u * (width - 1)))
    row = int(round(v * (height - 1)))
    for offset in range(-arm, arm + 1):
        if 0 <= row + offset < height:
            array[(row + offset) % height, column] = colour
        if 0 <= column + offset < width:
            array[row, (column + offset) % width] = colour
    return array


def preview(args):
    """Tile test crops with the true and predicted markers drawn on them.

    Numbers alone cannot show whether a model reads the minimap or happens to
    guess well, so the check is written out as an image a person can look at.
    """
    import subprocess
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    model = torch.jit.load(args.model)
    data = load_dataset(manifest["dataset"]["path"], manifest["training"]["downsample"])
    images, labels = masked_selection(data, "test")
    with torch.no_grad():
        predicted = model(images)
    count = min(args.preview_count, len(images))
    stride = max(1, len(images) // count)
    picks = list(range(0, len(images), stride))[:count]
    failures = []
    tiles = []
    for index in picks:
        tile = (images[index].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8).copy()
        tile = np.ascontiguousarray(tile)
        draw_cross(tile, float(labels[index][0]), float(labels[index][1]), (0, 255, 0))
        draw_cross(tile, float(predicted[index][0]), float(predicted[index][1]), (255, 40, 40))
        error = float(pixel_errors(predicted[index:index + 1],
                                   labels[index:index + 1], data["rect"])[0])
        failures.append((error, index))
        tiles.append(tile)
    columns = 4
    rows = (len(tiles) + columns - 1) // columns
    height, width = tiles[0].shape[0], tiles[0].shape[1]
    canvas = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for position, tile in enumerate(tiles):
        row, column = divmod(position, columns)
        canvas[row * height:(row + 1) * height, column * width:(column + 1) * width] = tile
    out = Path(args.preview_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = canvas.tobytes()
    subprocess.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", "%dx%d" % (canvas.shape[1], canvas.shape[0]), "-i", "-",
                    "-frames:v", "1", "-y", str(out)], input=raw, check=True)
    worst = sorted(failures, reverse=True)[:5]
    print("green = replay truth, red = model prediction; wrote %s" % out)
    for error, index in worst:
        print("  worst sample %d  error %.2f px" % (index, error))
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--validate", action="store_true",
                        help="re-check a saved artifact instead of training")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--model-out", type=Path)
    parser.add_argument("--manifest-out", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preview", action="store_true",
                        help="draw the true and predicted markers on test crops")
    parser.add_argument("--preview-out", type=Path)
    parser.add_argument("--preview-count", type=int, default=8)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.validate:
        if not args.model or not args.manifest:
            raise SystemExit("--validate needs --model and --manifest")
        return validate(args)
    if args.preview:
        if not args.model or not args.manifest or not args.preview_out:
            raise SystemExit("--preview needs --model, --manifest and --preview-out")
        return preview(args)
    if not args.dataset or not args.model_out or not args.manifest_out:
        raise SystemExit("training needs --dataset, --model-out and --manifest-out")
    return train(args)


if __name__ == "__main__":
    sys.exit(main())
