#!/usr/bin/env python3
"""Rotation/scale tolerant native minimap ship icons; uncertain evidence is unknown.

Dependencies: numpy, opencv-python-headless, cairosvg, pillow. Templates are rendered
once per matcher from the local gamedata SVGs; no replay information is used at infer.
"""
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
import hashlib

import cairosvg
import cv2
import numpy as np
from PIL import Image

ICON_ROOT = Path(__file__).resolve().parents[2] / 'state/toalba/gamedata/data/gui/fla/minimap/ship_icons'
SPECIES = {'destroyer': 'Destroyer', 'cruiser': 'Cruiser', 'battleship': 'Battleship',
           'aircarrier': 'AirCarrier', 'submarine': 'Submarine'}
PATCH = 48
# OpenCV releases the GIL for matching. Bound concurrent native work across all
# matchers; workers are started lazily and reused, never per frame.
_MATCH_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix='ship-icons')


def _match_templates(search, templates):
    winners = {}
    for species, angle, scale, template in templates:
        result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, location = cv2.minMaxLoc(result)
        if species not in winners or score > winners[species]['score']:
            winners[species] = dict(species=species, score=float(score), angle=angle,
                                    scale=scale, offset=[location[0]-8, location[1]-8])
    return winners


def red_signal(rgb):
    """Colour contrast retains antialiasing and internal gaps, rejects white/green."""
    colour = rgb.astype(np.float32)
    return np.maximum(colour[..., 0] - np.maximum(colour[..., 1], colour[..., 2]), 0) / 255


class ShipIconMatcher:
    def __init__(self, icon_root=ICON_ROOT, min_score=.75, min_margin=.06):
        self.min_score = min_score
        self.min_margin = min_margin
        self.templates = []
        self.hashes = {}
        # Fine angles preserve the one-pixel gaps that distinguish ship classes.
        for icon, species in SPECIES.items():
            path = Path(icon_root) / ('minimap_' + icon + '.svg')
            source = path.read_bytes()
            self.hashes[species] = hashlib.sha256(source).hexdigest()
            rendered = Image.open(BytesIO(cairosvg.svg2png(bytestring=source, scale=4))).convert('RGBA')
            rgba = np.asarray(rendered, dtype=np.float32) / 255
            white = rgba[..., :3].mean(axis=2) * rgba[..., 3]
            # The native dark hull is a background, not part of the coloured glyph.
            white = np.maximum(white - .2 * rgba[..., 3], 0) / .8
            for scale in (.85, 1., 1.15, 1.3):
                for angle in range(0, 360, 5):
                    matrix = cv2.getRotationMatrix2D(((white.shape[1]-1)/2, (white.shape[0]-1)/2), angle, scale/4)
                    matrix[:, 2] += np.array([12-(white.shape[1]-1)/2, 12-(white.shape[0]-1)/2])
                    template = cv2.warpAffine(white, matrix, (25, 25), flags=cv2.INTER_LINEAR)
                    self.templates.append((species, angle, scale, template))
        # Contiguous batches preserve template order, including exact-score ties.
        batch_size = (len(self.templates) + 3) // 4
        self._template_batches = tuple(tuple(self.templates[start:start+batch_size])
                                       for start in range(0, len(self.templates), batch_size))

    def classify(self, rgb, u, v):
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError('expected uint8 HWC RGB crop')
        if not np.isfinite([u, v]).all() or not (0 <= u <= 1 and 0 <= v <= 1):
            raise ValueError('icon location must be finite normalized crop coordinates')
        x = round(u * (rgb.shape[1]-1)); y = round(v * (rgb.shape[0]-1))
        # Clip before padding: do not pad/copy a whole 720x723 frame per detection.
        crop = rgb[max(0, y-24):min(rgb.shape[0], y+24), max(0, x-24):min(rgb.shape[1], x+24)]
        crop = np.pad(crop, ((max(0, 24-y), max(0, y+24-rgb.shape[0])),
                             (max(0, 24-x), max(0, x+24-rgb.shape[1])), (0, 0)))
        return self.classify_patch(crop)

    def classify_patch(self, rgb):
        if rgb.dtype != np.uint8 or rgb.shape != (PATCH, PATCH, 3):
            raise ValueError('expected uint8 RGB 48x48 icon patch')
        signal = red_signal(rgb)
        # Search centres within +/-8 px of the detector's point; distant neighbours
        # must not supply the class of this detection.
        search = signal[4:45, 4:45]
        winners = {}
        futures = [_MATCH_POOL.submit(_match_templates, search, batch)
                   for batch in self._template_batches]
        # Reduce in submission order, not completion order, to retain the first
        # matching template and species when scores are equal.
        for future in futures:
            for species, candidate in future.result().items():
                if species not in winners or candidate['score'] > winners[species]['score']:
                    winners[species] = candidate
        ranked = sorted(winners.values(), key=lambda entry: entry['score'], reverse=True)
        best, second = ranked[:2]
        margin = best['score'] - second['score']
        accepted = best['score'] >= self.min_score and margin >= self.min_margin
        return dict(species=best['species'] if accepted else 'unknown',
                    icon_score=best['score'], icon_margin=margin,
                    icon_candidate=best['species'], icon_angle_deg=best['angle'],
                    icon_scale=best['scale'], icon_offset_px=best['offset'])
