#!/usr/bin/env python3
"""Harvest main-view hull crops with automatic masks from the battle recordings.

The project has no learned hull model, and the bar-derived aim point proved
unreliable once the hull was not rendered. This builds the dataset for one.

Labels are *machine* labels: under a detected nameplate, the hull is the largest
dark blob (measured 25+ levels below the local sea median, closed along x). Only
crops where that blob is large, wide and ship-shaped are kept, so the label is a
strong hypothesis rather than a guess. Frames where nothing plausible appears are
dropped instead of being labelled as empty.
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

SIZE = 256


def sample_frames(video, stride_s, per_video, start_s=60.0):
    import av
    out = []
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or 30)
        step = max(1, int(round(fps * stride_s)))
        index = 0
        for frame in container.decode(stream):
            if frame.time is not None and frame.time < start_s:
                continue
            if index % step == 0:
                out.append(frame.to_ndarray(format='bgr24'))
                if len(out) >= per_video:
                    break
            index += 1
    return out


def _crop_once(rgb, bar, margin):
    x, y, w, h = bar
    cx, cy = x + w / 2, y + h
    half = margin * w
    x0, y0 = int(max(0, cx - half)), int(max(0, cy - 0.15 * w))
    x1, y1 = int(min(rgb.shape[1], cx + half)), int(min(rgb.shape[0], cy + 1.4 * margin * w))
    return x0, y0, x1, y1


def _label_crop(crop, bar_w):
    gray = crop.astype(np.int16).mean(axis=2)
    sea = float(np.median(gray))
    mask = (gray < sea - 25).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 25), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    centre_x = crop.shape[1] / 2
    best = None
    for bx, by, bw, bh, area in stats[1:]:
        if area < 1200 or bw < 60 or bh < 22:
            continue
        if not 1.1 <= bw / bh <= 12:
            continue
        if abs((bx + bw / 2) - centre_x) > 1.6 * bar_w:
            continue
        if best is None or area > best[0]:
            best = (area, bx, by, bw, bh)
    if best is None:
        return None, None
    area, bx, by, bw, bh = best
    crowded = (bx == 0 or by == 0 or bx + bw >= mask.shape[1] - 2
               or by + bh >= mask.shape[0] - 2)
    keep = np.zeros_like(mask)
    keep[by:by + bh, bx:bx + bw] = mask[by:by + bh, bx:bx + bw]
    return keep, crowded


def hull_crop_and_mask(rgb, bar, margin=1.6):
    """Crop around the plate, growing until the hull fits inside the window.

    A hull at 6.7 km fills the window and touches its edges; a label that is cut
    off teaches the model a false boundary, so the crop grows until the blob is
    fully inside (or the frame limits are reached).
    """
    for attempt in range(4):
        x0, y0, x1, y1 = _crop_once(rgb, bar, margin)
        if x1 - x0 < 96 or y1 - y0 < 96:
            return None, None
        crop = rgb[y0:y1, x0:x1]
        keep, crowded = _label_crop(crop, bar[2])
        if keep is None:
            margin *= 1.6
            continue
        if not crowded:
            break
        margin *= 1.6
    else:
        return None, None
    if keep is None or keep.mean() < 0.01:
        return None, None
    size = max(crop.shape[0], crop.shape[1])
    canvas = np.zeros((size, size, 3), np.uint8)
    canvas[:crop.shape[0], :crop.shape[1]] = crop
    mcanvas = np.zeros((size, size), np.uint8)
    mcanvas[:keep.shape[0], :keep.shape[1]] = keep
    image = cv2.resize(canvas, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    label = cv2.resize(mcanvas, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    if label.mean() < 0.02:
        return None, None
    return image, (label > 0).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--videos', nargs='+', default=[])
    ap.add_argument('--images', nargs='+', default=[],
                    help='PNG frames to label directly (as harvested from a live battle)')
    ap.add_argument('--per-video', type=int, default=40)
    ap.add_argument('--stride-s', type=float, default=15.0)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=not args.out.exists())
    images, masks, meta = [], [], []
    for image_path in args.images:
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        session = Path(image_path).parent.name
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        try:
            plates = M.detect_nameplates(rgb)
        except ValueError:
            continue
        for plate in plates[:2]:
            image, label = hull_crop_and_mask(rgb, plate['nameplate_bbox'])
            if image is None:
                continue
            images.append(image)
            masks.append(label)
            meta.append(dict(session=session, plate=plate['nameplate_bbox'],
                             aspect=round(plate['nameplate_bbox'][2] / max(1, plate['nameplate_bbox'][3]), 2),
                             mask_fraction=round(float(label.mean()), 4)))
            break
    for video in args.videos:
        video = Path(video)
        session = video.parent.name
        kept = 0
        try:
            frames = sample_frames(video, args.stride_s, args.per_video)
        except Exception as error:                     # noqa: BLE001
            print(json.dumps(dict(video=str(video), error=str(error)[:120])))
            continue
        for frame in frames:
            rgb = np.ascontiguousarray(frame[:, :, ::-1])
            try:
                plates = M.detect_nameplates(rgb)
            except ValueError:
                continue
            for plate in plates[:2]:
                image, label = hull_crop_and_mask(rgb, plate['nameplate_bbox'])
                if image is None:
                    continue
                images.append(image)
                masks.append(label)
                meta.append(dict(session=session, plate=plate['nameplate_bbox'],
                                 mask_fraction=round(float(label.mean()), 4)))
                kept += 1
                break
        print(json.dumps(dict(session=session, frames=len(frames), kept=kept)))
    if not images:
        print('no samples'); return 1
    np.save(args.out / 'images.npy', np.stack(images))
    np.save(args.out / 'masks.npy', np.stack(masks)[:, None])
    (args.out / 'index.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in meta))
    fractions = [r['mask_fraction'] for r in meta]
    print(json.dumps(dict(samples=len(meta), sessions=len({r['session'] for r in meta}),
                          mask_fraction_min=min(fractions),
                          mask_fraction_median=float(np.median(fractions)),
                          mask_fraction_max=max(fractions)), indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
