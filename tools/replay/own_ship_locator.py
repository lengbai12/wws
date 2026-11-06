#!/usr/bin/env python3
"""Locate one native white own-ship arrow in an RGB minimap; ambiguity is unknown."""
from io import BytesIO
from pathlib import Path
import hashlib

import cairosvg
import cv2
import numpy as np
from PIL import Image

ICON_PATH = (Path(__file__).resolve().parents[2] /
             'state/toalba/gamedata/data/gui/fla/minimap/ship_icons_self/minimap_self_alive.svg')


class OwnShipLocator:
    def __init__(self, icon_path=ICON_PATH, min_score=.65, min_margin=.08):
        self.min_score = min_score
        self.min_margin = min_margin
        source = Path(icon_path).read_bytes()
        self.icon_sha256 = hashlib.sha256(source).hexdigest()
        rendered = np.asarray(Image.open(BytesIO(cairosvg.svg2png(
            bytestring=source, scale=4))).convert('RGBA'))
        white = ((rendered[..., :3].min(axis=2) >= 200) &
                 (rendered[..., 3] >= 200)).astype(np.float32)
        self.templates = []
        for scale in (.9, 1., 1.1):
            for angle in range(0, 360, 5):
                matrix = cv2.getRotationMatrix2D((32, 40), angle, scale/4)
                matrix[:, 2] += np.array([15-32, 15-40])
                template = (cv2.warpAffine(white, matrix, (31, 31),
                                          flags=cv2.INTER_LINEAR) >= .5).astype(np.float32)
                self.templates.append((angle, scale, template, float(template.sum())))

    def detect(self, rgb):
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError('expected uint8 HWC RGB minimap crop')
        mask = (rgb.min(axis=2) >= 200).astype(np.uint8)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        candidates = []
        for component in range(1, count):
            x, y, width, height, area = stats[component]
            if not (65 <= area <= 210 and 10 <= width <= 28 and 10 <= height <= 28):
                continue
            cx, cy = np.rint(centroids[component]).astype(int)
            # A component's own support excludes nearby text/lines from the fit.
            left, top = cx-18, cy-18
            patch = np.zeros((37, 37), dtype=np.float32)
            x0, y0 = max(0, left), max(0, top)
            x1, y1 = min(mask.shape[1], left+37), min(mask.shape[0], top+37)
            patch[y0-top:y1-top, x0-left:x1-left] = labels[y0:y1, x0:x1] == component
            best = None
            for angle, scale, template, template_area in self.templates:
                overlap = cv2.matchTemplate(patch, template, cv2.TM_CCORR)
                iou = overlap / np.maximum(area+template_area-overlap, 1)
                _, score, _, location = cv2.minMaxLoc(iou)
                if best is None or score > best['score']:
                    px, py = int(left+location[0]+15), int(top+location[1]+15)
                    best = dict(u=px/(rgb.shape[1]-1), v=py/(rgb.shape[0]-1),
                                pixel_x=px, pixel_y=py, score=float(score),
                                angle_deg=angle, scale=scale)
            candidates.append(best)
        candidates.sort(key=lambda row: row['score'], reverse=True)
        if not candidates or candidates[0]['score'] < self.min_score:
            return None
        margin = candidates[0]['score'] - (candidates[1]['score'] if len(candidates) > 1 else 0)
        if margin < self.min_margin:
            return None
        return dict(**candidates[0], margin=margin, source='native_white_arrow')
