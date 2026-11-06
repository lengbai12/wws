#!/usr/bin/env python3
"""Train a small U-Net that segments the enemy hull in the main view.

Input: the dataset built by build_hull_masks.py (256x256 crops under a
nameplate, machine-labelled hull masks). Split is by recording session, so the
validation score says whether the model generalises to a battle it never saw.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyUNet(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout),
                                 nn.ReLU(inplace=True))
        self.e1 = block(3, base)
        self.e2 = block(base, base * 2)
        self.e3 = block(base * 2, base * 4)
        self.b = block(base * 4, base * 6)
        self.d3 = block(base * 6 + base * 4, base * 4)
        self.d2 = block(base * 4 + base * 2, base * 2)
        self.d1 = block(base * 2 + base, base)
        self.out = nn.Conv2d(base, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        b = self.b(self.pool(e3))
        d3 = self.d3(torch.cat([F.interpolate(b, scale_factor=2, mode='nearest'), e3], 1))
        d2 = self.d2(torch.cat([F.interpolate(d3, scale_factor=2, mode='nearest'), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, scale_factor=2, mode='nearest'), e1], 1))
        return self.out(d1)


def dice_loss(logits, target):
    prob = torch.sigmoid(logits)
    num = 2 * (prob * target).sum(dim=(1, 2, 3)) + 1
    den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1
    return 1 - (num / den).mean()


def iou(logits, target, threshold=0.5):
    pred = torch.sigmoid(logits) > threshold
    inter = (pred & (target > .5)).sum(dim=(1, 2, 3)).float()
    union = (pred | (target > .5)).sum(dim=(1, 2, 3)).float()
    return (inter / union.clamp(min=1)).mean().item()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--base', type=int, default=16)
    ap.add_argument('--device', default='auto')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=not args.out.exists())
    images = np.load(args.data / 'images.npy', mmap_mode='r')
    masks = np.load(args.data / 'masks.npy', mmap_mode='r')
    rows = [json.loads(l) for l in (args.data / 'index.jsonl').read_text().splitlines() if l]
    sessions = np.array([r['session'] for r in rows])
    unique = sorted(set(sessions.tolist()))
    if len(unique) < 2:
        print('need at least two sessions'); return 1
    val_sessions = set(unique[-max(1, len(unique) // 4):])
    train_idx = np.where([s not in val_sessions for s in sessions])[0]
    val_idx = np.where([s in val_sessions for s in sessions])[0]
    device = torch.device('cuda' if (args.device == 'auto' and torch.cuda.is_available())
                          else args.device if args.device != 'auto' else 'cpu')
    model = TinyUNet(args.base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.permutation(train_idx)
        losses = []
        for start in range(0, len(order), args.batch):
            idx = order[start:start + args.batch]
            x = torch.from_numpy(np.ascontiguousarray(images[idx])).to(device).float().permute(0, 3, 1, 2) / 255
            y = torch.from_numpy(np.ascontiguousarray(masks[idx])).to(device).float() / 255
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y) + dice_loss(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss))
        model.eval()
        total_iou, total_acc, batches = 0.0, 0.0, 0
        with torch.no_grad():
            for start in range(0, len(val_idx), args.batch):
                idx = val_idx[start:start + args.batch]
                x = torch.from_numpy(np.ascontiguousarray(images[idx])).to(device).float().permute(0, 3, 1, 2) / 255
                y = torch.from_numpy(np.ascontiguousarray(masks[idx])).to(device).float() / 255
                logits = model(x)
                total_iou += iou(logits, y)
                total_acc += float(((torch.sigmoid(logits) > .5) == (y > .5)).float().mean())
                batches += 1
        val_iou = total_iou / max(batches, 1)
        val_acc = total_acc / max(batches, 1)
        history.append(dict(epoch=epoch, train_loss=round(float(np.mean(losses)), 5),
                            val_iou=round(val_iou, 4), val_pixel_acc=round(val_acc, 4)))
        print(json.dumps(history[-1]))
        if val_iou > best:
            best = val_iou
            torch.save({'state_dict': model.state_dict(), 'base': args.base,
                        'val_iou': val_iou, 'epoch': epoch}, args.out / 'hull-seg.pt')
    (args.out / 'history.json').write_text(json.dumps(history, indent=1))
    (args.out / 'manifest.json').write_text(json.dumps(dict(
        data=str(args.data), samples=len(rows), train=len(train_idx), val=len(val_idx),
        val_sessions=sorted(val_sessions), epochs=args.epochs, base=args.base,
        best_val_iou=best, device=str(device)), indent=1))
    print(json.dumps(dict(best_val_iou=best, out=str(args.out))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
