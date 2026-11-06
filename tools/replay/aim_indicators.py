"""Separate the commanded white aim point from actual gun-direction rings.

Coordinates are measured from current RGB pixels. Relative search bounds and
scale bounds are only proposal priors, not returned positions or readiness.
Orange rings never authorize firing; missing or competing geometry is unknown.
"""
import math

import cv2
import numpy as np


def read_aim_indicators(rgb):
    if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or
            rgb.ndim != 3 or rgb.shape[2] != 3):
        raise ValueError('requires uint8 RGB frame')
    height, width = rgb.shape[:2]
    scale = min(width / 2560, height / 1600)
    left, top = int(width * .35), int(height * .35)
    right, bottom = int(width * .65), int(height * .65)
    patch = rgb[top:bottom, left:right]
    low = patch.min(axis=2).astype(np.float32)
    high = patch.max(axis=2).astype(np.float32)
    contrast = low - cv2.GaussianBlur(low, (0, 0), max(.5, 2 * scale))
    white = ((low > 150) & (high-low < 55) & (contrast > 15)).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(white)
    points = []
    for index, (x, y, w, h, area) in enumerate(stats[1:], 1):
        if not (12*scale <= w <= 35*scale and 8*scale <= h <= 26*scale and
                1.05 <= w/h <= 2.2 and area >= 14*scale*scale):
            continue
        ink = (labels[y:y+h, x:x+w] == index).astype(np.uint8)
        expected = np.zeros((h, w), np.uint8)
        thickness = max(1, round(w*.11))
        cv2.polylines(expected, [np.array([[0,h-1],[(w-1)//2,0],[w-1,h-1]])],
                      False, 1, thickness)
        # Both sloping arms and the apex must agree; a white line/digit is not
        # an aim point merely because it occupies the same central band.
        overlap = int((ink & expected).sum())
        precision = overlap / int(ink.sum())
        skeleton = np.zeros_like(expected)
        cv2.polylines(skeleton, [np.array([[0,h-1],[(w-1)//2,0],[w-1,h-1]])], False, 1, 1)
        nearby_ink = cv2.dilate(ink, np.ones((3,3),np.uint8))
        recall = int((nearby_ink & skeleton).sum()) / max(1,int(skeleton.sum()))
        if precision >= .78 and recall >= .65:
            points.append(dict(center=[float(left+x+(w-1)/2), float(top+y+(h-1)/2)],
                               bbox=[int(left+x),int(top+y),int(w),int(h)],
                               precision=precision, recall=recall,
                               source='current_white_chevron_geometry'))
    aim = points[0] if len(points) == 1 else None
    red, green, blue = (patch[...,i].astype(np.int16) for i in range(3))
    masks = dict(green=(green > 120) & (green-red > 30) & (green-blue > 20),
                 orange=(red > 140) & (green > 90) & (red-green > 25) &
                        (green-blue > 25) & (red-blue > 65))
    rings = []
    gap = max(3, round(9*scale))
    if gap % 2 == 0:
        gap += 1
    for color, mask in masks.items():
        grouped = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                                   np.ones((gap,gap), np.uint8))
        _, _, stats, _ = cv2.connectedComponentsWithStats(grouped)
        for x,y,w,h,area in stats[1:]:
            if not (22*scale <= w <= 65*scale and 22*scale <= h <= 65*scale and
                    .8 <= w/h <= 1.25):
                continue
            cx, cy = x+(w-1)/2, y+(h-1)/2
            radius = (w+h)/4 - .5
            yy, xx = np.ogrid[y:y+h, x:x+w]
            inside = (xx-cx)**2+(yy-cy)**2 < (.5*radius)**2
            if mask[y:y+h,x:x+w][inside].mean() > .15:
                continue
            angles = np.arange(128)*2*np.pi/128
            support = np.zeros(128, bool)
            for delta in np.linspace(-max(1,2*scale),max(1,2*scale),5):
                xs = np.rint(cx+(radius+delta)*np.cos(angles)).astype(int)
                ys = np.rint(cy+(radius+delta)*np.sin(angles)).astype(int)
                if xs.min()<0 or ys.min()<0 or xs.max()>=mask.shape[1] or ys.max()>=mask.shape[0]:
                    continue
                support |= mask[ys,xs]
            quadrants = support.reshape(4,32).mean(axis=1)
            if support.mean() >= .5 and quadrants.min() >= .35:
                rings.append(dict(center=[float(left+cx),float(top+cy)],color=color,
                                  radius_px=float(radius),support=float(support.mean()),
                                  quadrants=quadrants.tolist(),
                                  source='current_full_gun_direction_ring'))
    nearby = [] if aim is None else [r for r in rings if math.dist(r['center'],aim['center']) <= 30*scale]
    green = [ring for ring in rings if ring['color'] == 'green'] if aim is not None else []
    ready = bool(green)
    selected = (min(green, key=lambda ring: math.dist(ring['center'],aim['center']))
                if green else nearby[0] if len(nearby)==1 else None)
    return dict(aim_point=aim, aim_candidates=points, gun_rings=rings,
                gun_direction=dict(status='observed' if selected is not None else 'unknown',
                                   ready=ready, ring=selected,
                                   reason=('at_least_one_green_gun_ring' if ready else
                                           'no_green_gun_ring')),
                frame_size=[width,height], semantics='white_commanded_aim;green_actual_gun;orange_unavailable')
