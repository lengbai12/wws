#!/usr/bin/env python3
"""Build main-view hull crops with pseudo masks from saved frames.

The project has no learned hull model: the main-view detectors are the red
nameplate bar and a *dark structure* verifier, and the latter returns null on a
light-grey hull against bright water (measured on a 7 km live plate). This tool
starts the dataset for a learned hull segmenter.

For every frame it finds the nameplate, crops the region the hull must occupy,
and writes a pseudo mask from the one separation measured on live frames: hull
V (157) against water V (189) with a desaturated S band. The mask under-covers
the bright lower hull, so it is a *starting* label, never ground truth.
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import main_view_perception as M

CROP = 256


def pseudo_hull_mask(crop_rgb):
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV).astype(np.int16)
    value, sat = hsv[:, :, 2], hsv[:, :, 1]
    mask = ((value < 178) & (sat > 25) & (sat < 95)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return None
    biggest = 1 + int(np.argmax(stats[1:, 4]))
    if stats[biggest, 4] < 400:
        return None
    kept = (labels == biggest).astype(np.uint8)
    return cv2.dilate(kept, np.ones((3, 3), np.uint8))


def hull_crop(rgb, plate):
    x, y, w, h = plate['nameplate_bbox']
    cx = x + w / 2
    top = y + 0.2 * h
    height = 2.6 * w
    x0 = int(max(0, cx - height / 2))
    y0 = int(max(0, top))
    size = int(min(height, rgb.shape[0] - y0, rgb.shape[1] - x0))
    if size < 96:
        return None, None
    crop = rgb[y0:y0 + size, x0:x0 + size]
    return cv2.resize(crop, (CROP, CROP), interpolation=cv2.INTER_AREA), (x0, y0, size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', nargs='+', required=True, help='frames (png/ppm/jpg)')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max', type=int, default=400)
    parser.add_argument('--min-width', type=int, default=60,
                        help='smallest nameplate width treated as a usable hull anchor')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=not args.out.exists())
    index = []
    for path in args.images:
        if len(index) >= args.max:
            break
        image = cv2.imread(str(path))
        if image is None:
            continue
        rgb = np.ascontiguousarray(image[:, :, ::-1])
        try:
            plates = [p for p in M.detect_nameplates(rgb) if p['nameplate_bbox'][2] >= args.min_width]
        except ValueError:
            continue
        for plate in plates[:2]:
            crop, box = hull_crop(rgb, plate)
            if crop is None:
                continue
            mask = pseudo_hull_mask(crop)
            if mask is None:
                continue
            stem = f'{len(index):05d}'
            cv2.imwrite(str(args.out / f'{stem}.png'), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(args.out / f'{stem}.mask.png'), mask * 255)
            index.append(dict(id=stem, source=str(path), plate=plate['nameplate_bbox'],
                              crop_box=list(box), mask_pixels=int(mask.sum()),
                              mask_fraction=round(float(mask.mean()), 4)))
            if len(index) >= args.max:
                break
    (args.out / 'index.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in index))
    fractions = [row['mask_fraction'] for row in index]
    print(json.dumps(dict(images=len(index), out=str(args.out),
                          mask_fraction_min=min(fractions) if fractions else None,
                          mask_fraction_median=float(np.median(fractions)) if fractions else None,
                          mask_fraction_max=max(fractions) if fractions else None), indent=1))
    return 0 if index else 1


if __name__ == '__main__':
    raise SystemExit(main())
