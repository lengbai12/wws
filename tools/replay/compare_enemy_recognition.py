#!/usr/bin/env python3
"""Compare enemy peaks and optional red-pixel confirmation on held-out crops.

Choose threshold/confirmation only on validation; measure test once per selected
configuration. Uses the trainer's actual streaming evaluation and matching rules.
No training, large member materialisation, or input injection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import train_minimap_heatmap as T


class RedConfirmed(torch.nn.Module):
    """Keep enemy logits only near red pixels in the actual normalised input.

    Five input pixels at half resolution allow for icon-centre/projection offsets.
    This is visual colour evidence, not a ship identity or permission to fire.
    """
    def __init__(self, model, channel):
        super().__init__()
        self.model = model
        self.channel = channel

    @staticmethod
    def support(images, grid):
        red, green, blue = images.unbind(dim=1)
        mask = ((red >= 100 / 255) & (red - green > 30 / 255)
                & (red - blue > 30 / 255)).float().unsqueeze(1)
        mask = F.max_pool2d(mask, 11, stride=1, padding=5)
        return F.interpolate(mask, size=grid, mode="nearest")[:, 0].bool()

    def forward(self, images):
        heatmaps, rings = self.model(images)
        keep = self.support(images, heatmaps.shape[-2:])
        # Confirm already-existing peaks: masking must not create new maxima at
        # the edge of a red support region.
        probabilities = heatmaps[:, self.channel:self.channel+1].sigmoid()
        pooled = F.max_pool2d(probabilities, 3, stride=1, padding=1)
        keep &= (probabilities >= pooled - 1e-6)[:, 0]
        heatmaps[:, self.channel].masked_fill_(~keep, -100)
        return heatmaps, rings


def enemy_metrics(model, data, index, threshold, device):
    metrics, _, _ = T.evaluate(model, index, data, threshold, stream=True, device=device)
    entry = metrics["enemy_visible"]
    p, r = entry["precision"] or 0.0, entry["recall"] or 0.0
    entry["f1"] = 2 * p * r / (p + r) if p + r else 0.0
    return entry


def colour_audit(data, index, channel):
    full_path = T.twin_path(data["image_path"], "images")
    if not full_path.exists():
        raise SystemExit("full-resolution image twin required for colour audit: %s" % full_path)
    full = np.load(full_path, allow_pickle=False, mmap_mode="r")
    totals = {"truth": 0, "full_red": 0, "input_red": 0}
    # CPU-only, one sample at a time: never gather the crop corpus as a whole.
    for sample in index:
        raw = full[int(sample)]
        half = T.prepare_images(data["images"][int(sample)], torch.device("cpu"),
                                data["downsample"])[0].permute(1, 2, 0).numpy() * 255
        for row in data["truth"][sample]:
            if np.isnan(row[0]) or int(row[0]) != channel:
                continue
            totals["truth"] += 1
            for image, key, radius in ((raw, "full_red", 10), (half, "input_red", 5)):
                x = round(float(row[1]) * (image.shape[1] - 1))
                y = round(float(row[2]) * (image.shape[0] - 1))
                patch = image[max(0, y-radius):y+radius+1,
                              max(0, x-radius):x+radius+1].astype(np.float32)
                if patch.size:
                    r, g, b = patch[..., 0], patch[..., 1], patch[..., 2]
                    totals[key] += int(((r >= 100) & (r-g > 30) & (r-b > 30)).any())
    del full
    return totals


def preview(model, data, index, channel, config, device, path):
    """Six distributed validation crops: cyan truth, yellow confirmed predictions."""
    eligible = [int(sample) for sample in index if any(
        not np.isnan(row[0]) and int(row[0]) == channel for row in data["truth"][sample])]
    if not eligible:
        return None
    samples = np.asarray(eligible)[np.linspace(0, len(eligible)-1, min(6, len(eligible)), dtype=int)]
    raw = data["images"][samples]
    active = RedConfirmed(model, channel) if config["red_confirmed"] else model
    with torch.no_grad():
        heatmaps, _ = active(T.prepare_images(raw, device, data["downsample"]))
        peaks = T.decode(heatmaps.sigmoid().cpu(), config["threshold"])
    pictures = raw.numpy().copy()
    height, width = pictures.shape[1:3]
    def circle(image, u, v, colour, radius):
        x, y = round(u*(width-1)), round(v*(height-1))
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                if abs(dx*dx+dy*dy-radius*radius) <= radius and 0 <= y+dy < height and 0 <= x+dx < width:
                    image[y+dy, x+dx] = colour
    for position, sample in enumerate(samples):
        for row in data["truth"][sample]:
            if not np.isnan(row[0]) and int(row[0]) == channel:
                circle(pictures[position], float(row[1]), float(row[2]), [0, 255, 255], 6)
        for x, y, _ in peaks[position][channel]:
            circle(pictures[position], x/(data["grid"]-1), y/(data["grid"]-1), [255, 255, 0], 3)
    canvas = np.zeros((3*height, 2*width, 3), dtype=np.uint8)
    for position, image in enumerate(pictures):
        canvas[(position//2)*height:(position//2+1)*height,
               (position%2)*width:(position%2+1)*width] = image
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(("P6\n%d %d\n255\n" % (2*width, 3*height)).encode() + canvas.tobytes())
    return {"path": str(path), "sample_indices": samples.tolist(),
            "legend": {"cyan": "enemy truth", "yellow": "predicted enemy"},
            "split": "validation", "threshold": config["threshold"],
            "red_confirmed": config["red_confirmed"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thresholds", default="0.6,0.7,0.8,0.9")
    args = ap.parse_args(argv)
    output = Path(args.out)
    if output.exists():
        raise SystemExit("refusing to overwrite %s" % output)
    manifests = {name: json.loads(Path(path).read_text()) for name, path in
                 (("baseline", args.baseline), ("candidate", args.candidate))}
    baseline, candidate = manifests.values()
    for key in ("heatmaps", "images", "heatmaps_sha256"):
        if baseline["dataset"][key] != candidate["dataset"][key]:
            raise SystemExit("different dataset identity: %s" % key)
    if baseline["training"]["downsample"] != candidate["training"]["downsample"]:
        raise SystemExit("different input preprocessing")
    if T.sha256(candidate["dataset"]["heatmaps"]) != candidate["dataset"]["heatmaps_sha256"]:
        raise SystemExit("heatmap dataset sha256 mismatch")
    torch.set_num_threads(4)
    device = T.pick_device("auto")
    data = T.load_all(candidate["dataset"]["heatmaps"], candidate["dataset"]["images"],
                      candidate["training"]["downsample"])
    data["image_path"] = candidate["dataset"]["images"]
    channel = data["classes"].index("enemy_visible")
    validation, test = T.select(data, "validation"), T.select(data, "test")
    thresholds = [float(value) for value in args.thresholds.split(",")]
    report = {"selection_split": "validation", "selection_rule": "highest_enemy_f1",
              "thresholds": thresholds, "colour_confirmation": {
                  "rgb_min": 100, "red_excess": 30, "input_radius_px": 5},
              "dataset": candidate["dataset"], "models": {},
              "colour_audit_validation": colour_audit(data, validation, channel)}
    print("colour audit", json.dumps(report["colour_audit_validation"]), flush=True)
    for name, manifest in manifests.items():
        path = manifest["artifact"]["path"]
        if T.sha256(path) != manifest["artifact"]["sha256"]:
            raise SystemExit("model sha256 mismatch: %s" % path)
        model = torch.jit.load(path).to(device).eval()
        curves = []
        for confirmed in (False, True):
            active = RedConfirmed(model, channel) if confirmed else model
            for threshold in thresholds:
                entry = {"threshold": threshold, "red_confirmed": confirmed,
                         "metrics": enemy_metrics(active, data, validation, threshold, device)}
                curves.append(entry)
                print(name, "validation", json.dumps(entry), flush=True)
        best = max(curves, key=lambda entry: entry["metrics"]["f1"])
        # Also retain a pure-model choice so colour confirmation has an explicit control.
        pure = max((entry for entry in curves if not entry["red_confirmed"]),
                   key=lambda entry: entry["metrics"]["f1"])
        selected = {}
        for label, entry in (("pure_model", pure), ("best_validation", best)):
            active = RedConfirmed(model, channel) if entry["red_confirmed"] else model
            selected[label] = {"threshold": entry["threshold"],
                               "red_confirmed": entry["red_confirmed"],
                               "validation": entry["metrics"],
                               "test": enemy_metrics(active, data, test, entry["threshold"], device)}
            print(name, label, json.dumps(selected[label]), flush=True)
        # Fixed 0.6 is the existing operating point, not a choice made from test.
        fixed = enemy_metrics(model, data, test, 0.6, device)
        report["models"][name] = {"manifest": args.baseline if name == "baseline" else args.candidate,
                                 "model_sha256": manifest["artifact"]["sha256"],
                                 "validation_curve": curves, "selected": selected,
                                 "test_at_existing_0.6": fixed}
        report["models"][name]["preview"] = preview(
            model, data, validation, channel, best, device,
            output.with_name(output.stem + "-" + name + ".ppm"))
        del active
        del model
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("wrote", output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
