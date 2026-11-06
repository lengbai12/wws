#!/usr/bin/env python3
"""Learn to reject red objective/UI symbols among enemy heatmap candidates.

train: frozen base model, train-only patches, validation-only checkpoint/threshold.
infer: raw RGB minimap crop -> identical integer half input -> enemy detections.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import train_minimap_heatmap as T
from compare_enemy_recognition import RedConfirmed
from ship_icon_templates import ShipIconMatcher
from enemy_glyph_dedup import deduplicate

PATCH = 32


def exact_float32():
    # Batch-size-dependent TF32 convolution rounding moves local-maximum ties.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


class MarkerClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(3, 12, 3, padding=1), nn.ReLU(),
                                    nn.MaxPool2d(2), nn.Conv2d(12, 16, 3, padding=1),
                                    nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(),
                                    nn.Linear(16*8*8, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, images):
        return self.layers(images).flatten()


def half_crop(rgb):
    """Same uint16 sum and half-up rounding as images_half corpus member."""
    height, width = rgb.shape[:2]
    block = rgb[:height//2*2, :width//2*2].astype(np.uint16)
    return ((block[::2, ::2] + block[1::2, ::2] + block[::2, 1::2]
             + block[1::2, 1::2] + 2)//4).astype(np.uint8)


def patches(image, peaks, grid):
    padded = np.pad(image, ((PATCH//2, PATCH//2), (PATCH//2, PATCH//2), (0, 0)))
    result = []
    for x, y, _ in peaks:
        column = round(x/(grid-1)*(image.shape[1]-1))
        row = round(y/(grid-1)*(image.shape[0]-1))
        result.append(padded[row:row+PATCH, column:column+PATCH])
    return np.asarray(result, dtype=np.uint8).reshape(-1, PATCH, PATCH, 3)


def prepared(array, device):
    return torch.from_numpy(np.ascontiguousarray(array)).to(device).permute(0, 3, 1, 2).float()/255


def guard():
    free = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                    if line.startswith('MemAvailable:')))/1048576
    if free < 4:
        raise SystemExit('stopping classifier below 4 GiB available')


def score(records, scores, threshold, radius):
    matched = predicted = truth_count = 0
    errors = []
    for record in records:
        used = set()
        truth = record['truth']
        truth_count += len(truth)
        for i, (x, y, _) in enumerate(record['peaks']):
            if scores[record['offset']+i] < threshold:
                continue
            predicted += 1
            distances = sorted((float(np.hypot(x-tx, y-ty)), k)
                               for k, (tx, ty) in enumerate(truth) if k not in used)
            if distances and distances[0][0] <= radius:
                used.add(distances[0][1]); matched += 1; errors.append(distances[0][0])
    precision = matched/predicted if predicted else 0.
    recall = matched/truth_count if truth_count else 0.
    return dict(predicted=predicted, truth=truth_count, matched=matched,
                precision=precision, recall=recall,
                f1=2*precision*recall/(precision+recall) if precision+recall else 0.,
                error_grid_px_mean=float(np.mean(errors)) if errors else None)


def collect(base, data, index, channel, device):
    blocks, labels, records = [], [], []
    offset = 0
    with torch.no_grad():
        for start in range(0, len(index), 16):
            guard()
            ids = index[start:start+16]
            raw = data['images'][ids]
            heat, _ = base(T.prepare_images(raw, device, data['downsample']))
            peaks = T.decode(heat.sigmoid().cpu(), .9)
            for position, sample in enumerate(ids):
                found = peaks[position][channel]
                truth = [(float(v[1])*(data['grid']-1), float(v[2])*(data['grid']-1))
                         for v in data['truth'][sample] if not np.isnan(v[0]) and int(v[0]) == channel]
                blocks.append(patches(raw[position].numpy(), found, data['grid']))
                labels.extend(float(any(np.hypot(x-tx, y-ty) <= T.match_radius(data)
                                        for tx, ty in truth)) for x, y, _ in found)
                records.append(dict(sample=int(sample), offset=offset, peaks=found, truth=truth))
                offset += len(found)
    return np.concatenate(blocks), np.asarray(labels, dtype=np.float32), records


def predict(model, array, device):
    output = []
    with torch.no_grad():
        for start in range(0, len(array), 256):
            output.extend(model(prepared(array[start:start+256], device)).sigmoid().cpu().tolist())
    return np.asarray(output)


def train(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4); torch.manual_seed(0)
    exact_float32()
    rng = np.random.default_rng(0)
    manifest = json.loads(Path(args.manifest).read_text())
    base_path = manifest['artifact']['path']
    if T.sha256(base_path) != manifest['artifact']['sha256']:
        raise SystemExit('base model identity mismatch')
    if T.sha256(manifest['dataset']['heatmaps']) != manifest['dataset']['heatmaps_sha256']:
        raise SystemExit('dataset identity mismatch')
    device = T.pick_device('auto')
    data = T.load_all(manifest['dataset']['heatmaps'], manifest['dataset']['images'], 2)
    if data['downsample'] != 1:
        raise SystemExit('integer half-resolution member required')
    channel = data['classes'].index('enemy_visible')
    base = RedConfirmed(torch.jit.load(base_path).to(device).eval(), channel)
    datasets = {}
    for split in ('train', 'validation'):
        datasets[split] = collect(base, data, T.select(data, split), channel, device)
        print(split, len(datasets[split][0]), 'positive', int(datasets[split][1].sum()), flush=True)
    x, y, _ = datasets['train']; vx, _, vr = datasets['validation']
    model = MarkerClassifier().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=.001)
    best = -1.; curve = []; thresholds = (.25, .5, .75, .9, .95)
    for epoch in range(1, args.epochs+1):
        guard(); model.train(); order = rng.permutation(len(x)); loss_sum = 0.
        for start in range(0, len(order), 256):
            batch = order[start:start+256]
            opt.zero_grad()
            logits = model(prepared(x[batch], device))
            loss = F.binary_cross_entropy_with_logits(logits, torch.from_numpy(y[batch]).to(device))
            loss.backward(); opt.step(); loss_sum += loss.detach().item()*len(batch)
        model.eval(); probabilities = predict(model, vx, device)
        entries = [dict(epoch=epoch, threshold=t, metrics=score(vr, probabilities, t, T.match_radius(data)))
                   for t in thresholds]
        chosen = max(entries, key=lambda r:r['metrics']['f1']); curve.extend(entries)
        print('epoch', epoch, 'loss', loss_sum/len(x), json.dumps(chosen), flush=True)
        if chosen['metrics']['f1'] > best:
            best = chosen['metrics']['f1']; selected = chosen
            torch.jit.script(model).save(str(out/'classifier.pt'))
    # Test is evaluated only after selecting both checkpoint and threshold on validation.
    model = torch.jit.load(str(out/'classifier.pt')).to(device).eval()
    tx, _, tr = collect(base, data, T.select(data, 'test'), channel, device)
    probabilities = predict(model, tx, device)
    test = score(tr, probabilities, selected['threshold'], T.match_radius(data))
    report = dict(schema_version=1, base_manifest=args.manifest,
                  base_model=base_path, base_sha256=T.sha256(base_path),
                  classifier_path=str(out/'classifier.pt'), classifier_sha256=T.sha256(out/'classifier.pt'),
                  input=dict(crop_rect=data['crop_rect'], colour='RGB', shape=[723,720,3],
                             half_rule='uint16_sum_plus_2_div_4_floor', patch=PATCH),
                  grid=data['grid'], candidate_threshold=.9, channel=channel,
                  classes=data['classes'], selection_split='validation', selection=selected,
                  validation_curve=curve, test=test, dataset=manifest['dataset'],
                  label_limit='replay enemy_visible; unresolved visibility may disagree with pixels')
    (out/'package.json').write_text(json.dumps(report, indent=2)+'\n')
    np.savez_compressed(out/'test-predictions.npz', scores=probabilities)
    (out/'test-records.json').write_text(json.dumps(tr))
    print('test', json.dumps(test), flush=True)
    return 0


class EnemyRecognizer:
    def __init__(self, package, device='cpu'):
        self.package = json.loads(Path(package).read_text())
        exact_float32()
        expected_input = dict(crop_rect=[1836,796,720,723], colour='RGB',
                              shape=[723,720,3], half_rule='uint16_sum_plus_2_div_4_floor', patch=PATCH)
        if self.package['schema_version'] != 1 or self.package['input'] != expected_input:
            raise ValueError('unsupported enemy recognizer input contract')
        if self.package['classes'][self.package['channel']] != 'enemy_visible':
            raise ValueError('enemy channel identity mismatch')
        self.device = torch.device(device)
        for key, digest in (('base_model', 'base_sha256'), ('classifier_path','classifier_sha256')):
            if T.sha256(self.package[key]) != self.package[digest]:
                raise ValueError('model identity mismatch: '+key)
        self.base = RedConfirmed(torch.jit.load(self.package['base_model']).to(self.device).eval(),
                                 self.package['channel'])
        self.classifier = torch.jit.load(self.package['classifier_path']).to(self.device).eval()
        self.ship_icons = ShipIconMatcher()

    def detect(self, rgb):
        if rgb.dtype != np.uint8 or list(rgb.shape) != self.package['input']['shape']:
            raise ValueError('expected uint8 RGB minimap crop of shape '+str(self.package['input']['shape']))
        half = half_crop(rgb)
        with torch.no_grad():
            heat, _ = self.base(prepared(half[None], self.device))
            found = T.decode(heat.sigmoid().cpu(), self.package['candidate_threshold'])[0][self.package['channel']]
            probabilities = predict(self.classifier, patches(half, found, self.package['grid']), self.device)
        detections = []
        for (x, y, confidence), probability in zip(found, probabilities):
            if probability < self.package['selection']['threshold']:
                continue
            u, v = x/(self.package['grid']-1), y/(self.package['grid']-1)
            detections.append(dict(u=u, v=v, score=float(probability),
                                   heatmap_score=float(confidence),
                                   **self.ship_icons.classify(rgb, u, v)))
        return deduplicate(rgb, detections, self.ship_icons)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    fit = sub.add_parser('train'); fit.add_argument('--manifest', required=True)
    fit.add_argument('--out', required=True); fit.add_argument('--epochs', type=int, default=12)
    infer = sub.add_parser('infer'); infer.add_argument('--package', required=True)
    infer.add_argument('--rgb-npy', required=True, help='uint8 HWC minimap crop, RGB')
    args = ap.parse_args()
    if args.command == 'train':
        return train(args)
    recognizer = EnemyRecognizer(args.package)
    print(json.dumps(recognizer.detect(np.load(args.rgb_npy, allow_pickle=False))))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
