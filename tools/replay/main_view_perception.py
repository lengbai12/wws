#!/usr/bin/env python3
"""Image-only perception for the WoWS full battle view.

The consumer deliberately reports screen evidence, not world state.  Coordinates are
pixels in a 2560x1600 RGB frame (x, y), with the 80px letterbox bands retained.
HUD and minimap pixels are excluded before any candidate is accepted.  No replay
truth, roster, or model output is used here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from aircraft_perception import detect as detect_aircraft
from torpedo_perception import detect as detect_torpedoes

try:
    import hud_projection
except ModuleNotFoundError:  # imported by path, without tools/replay on sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import hud_projection

FRAME_W, FRAME_H = 2560, 1600
MINIMAP_ROI = (1836, 796, 720, 723)  # x, y, width, height, native capture coordinates
DEFAULT_ASSET_ROOT = "state/toalba/gamedata/data/gui/battle_hud/markers"

def load_native_templates(root: str = DEFAULT_ASSET_ROOT) -> dict[str, np.ndarray]:
    """Load extracted RGBA battle-HUD sprites; missing assets remain unknown."""
    from pathlib import Path
    from PIL import Image
    base = Path(root)
    paths = {
        "target_lock_arc": base / "target_lock/icon_target_locked_arc.png",
        "target_lock_serif": base / "target_lock/icon_target_locked_serif.png",
    }
    result = {}
    for key, path in paths.items():
        if path.exists(): result[key] = np.asarray(Image.open(path).convert("RGBA"))
    return result


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    def as_list(self) -> list[int]:
        return [self.x, self.y, self.w, self.h]


def _rect_intersects(a: Rect, b: Rect) -> bool:
    return a.x < b.x2 and b.x < a.x2 and a.y < b.y2 and b.y < a.y2
def _template_match(rgb: np.ndarray, rgba: np.ndarray, threshold: float, search_roi: Optional[Rect] = None,
                    projection=None) -> list[dict[str, Any]]:
    """Match premultiplied native shape including its transparent surroundings."""
    t = np.asarray(rgba)
    if t.ndim != 3 or t.shape[2] not in (3, 4) or not t.size: return []
    alpha = t[..., 3].astype(np.float32) / 255 if t.shape[2] == 4 else np.ones(t.shape[:2], np.float32)
    color = t[..., :3].astype(np.float32)
    colored = float(np.max(np.ptp(color, axis=2) * alpha)) > 35
    x0, y0, x1, y1 = 0, 0, rgb.shape[1], rgb.shape[0]
    if search_roi is not None:
        r = _clip_rect(search_roi, rgb.shape[1], rgb.shape[0]); x0, y0, x1, y1 = r.x, r.y, r.x2, r.y2
    source = rgb[y0:y1, x0:x1]
    if not source.size: return []
    if colored:
        signal = np.maximum(0, source[..., 0].astype(np.float32) - np.maximum(source[..., 1], source[..., 2]))
        shape = np.maximum(0, color[..., 0] - np.maximum(color[..., 1], color[..., 2])) * alpha
    else:
        signal = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY).astype(np.float32)
        shape = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY) * alpha
    pad = 2
    shape = np.pad(shape, pad)
    if float(shape.std()) < 1: return []
    src = signal; th, tw = shape.shape
    if th > src.shape[0] or tw > src.shape[1]: return []
    scores = cv2.matchTemplate(src, shape, cv2.TM_CCOEFF_NORMED)
    excluded = hud_mask(rgb.shape, projection)[y0:y1, x0:x1].astype(np.uint8)
    integral = cv2.integral(excluded)
    sums = integral[th:, tw:] - integral[:-th, tw:] - integral[th:, :-tw] + integral[:-th, :-tw]
    scores[(sums > 0) | ~np.isfinite(scores)] = -1
    peaks = (scores >= threshold) & (scores >= cv2.dilate(scores, np.ones((3, 3), np.uint8)))
    ys, xs = np.where(peaks); order = np.argsort(scores[ys, xs])[::-1][:256]; picks = []
    for k in order:
        xx, yy = int(xs[k]), int(ys[k]); val = float(scores[yy, xx])
        if any(abs(xx-px) < tw//2 and abs(yy-py) < th//2 for px, py, _ in picks): continue
        patch = src[yy:yy+th, xx:xx+tw]; support = shape > shape.max() * .45
        if float(patch[support].mean() - patch[~support].mean()) < (22 if colored else 18): continue
        picks.append((xx, yy, val))
        if len(picks) >= 64: break
    return [{"bbox": [x0+x+pad, y0+y+pad, tw-2*pad, th-2*pad], "score": s} for x, y, s in picks]




REFERENCE_HUD_MASKS = (
    (0, 80, 560, 185),           # task panel
    (560, 80, 1420, 130),        # top-center team/objective status, not scene targets
    (0, 1240, 360, 279),         # own ship/status panel
    # Score, kill feed and the ribbons attached to them live in the top-right
    # corner. The old 690 px height also hid the sea below them: a live plate at
    # 7.0 km landed on (2220, 702) and was masked out, so the detector reported
    # no enemy while one was plainly drawn on screen. Keep the corner, not the
    # whole right column.
    (1980, 80, 580, 250),
    (940, 1320, 1020, 199),      # turret/reload and weapon/consumable bars
    MINIMAP_ROI,
)


def hud_mask(shape: tuple[int, int] | tuple[int, int, int] = (FRAME_H, FRAME_W),
             projection=None) -> np.ndarray:
    """Return True where screen UI is excluded (including minimap and black bars).

    ``REFERENCE_HUD_MASKS`` are reference-space rectangles: they are projected
    onto this frame, so a client that renders a smaller surface masks its own
    minimap and HUD instead of the reference ones. Without a projection the
    identity is used (reference-canvas callers), which is declared, not assumed
    silently, by ``hud_projection.identity``.
    """
    h, w = shape[:2]
    projection = projection or hud_projection.identity(shape)
    mask = np.zeros((h, w), dtype=bool)
    def fill(x: int, y: int, rw: int, rh: int) -> None:
        x0, x1 = max(0, x), min(w, x + rw)
        y0, y1 = max(0, y), min(h, y + rh)
        if x0 < x1 and y0 < y1:
            mask[y0:y1, x0:x1] = True
    # Letterbox and fixed HUD areas.  Central sea remains available.
    fill(0, 0, w, 75); fill(0, h - 90, w, 90)
    for rect in REFERENCE_HUD_MASKS:
        # Masks must not abort a live loop, so they clip instead of raising.
        x, y, rw, rh = projection.clip_rect(rect)
        fill(x, y, rw, rh)
    return mask


def _validate_frame(frame: np.ndarray) -> np.ndarray:
    a = np.asarray(frame)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError("frame must be HxWx3 RGB")
    if a.shape[:2] != (FRAME_H, FRAME_W):
        raise ValueError(f"frame must be {FRAME_W}x{FRAME_H} RGB")
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    return a


def _components(binary: np.ndarray, min_area: int = 1, max_area: int = 100000) -> list[Rect]:
    """Extract the same 8-connected components in row-major discovery order."""
    # SAUF labels retain the original flood fill's first-pixel ordering.
    _, _, stats, _ = cv2.connectedComponentsWithStatsWithAlgorithm(
        binary.astype(np.uint8), 8, cv2.CV_32S, cv2.CCL_WU)
    return [Rect(int(x), int(y), int(w), int(h))
            for x, y, w, h, area in stats[1:] if min_area <= area <= max_area]


def _red_mask(rgb: np.ndarray) -> np.ndarray:
    r, g, b = [rgb[..., i].astype(np.int16) for i in range(3)]
    # Red is closer to the red axis than to yellow: orange detection captions
    # otherwise satisfy the broad brightness/dominance checks and mimic bars.
    return (r >= 135) & (r > g + 45) & (r > b + 30) & (g < 190) & (r + b > 2*g)


def _clip_rect(rect: Rect, w: int = FRAME_W, h: int = FRAME_H) -> Rect:
    x = max(0, min(w, rect.x)); y = max(0, min(h, rect.y))
    x2 = max(x, min(w, rect.x2)); y2 = max(y, min(h, rect.y2))
    return Rect(x, y, x2 - x, y2 - y)


def _healthbar_backing(gray: np.ndarray, bar: Rect, excluded: np.ndarray) -> Optional[Rect]:
    """Require both dark bar edges; extend partial fill only on matching edges."""
    x0, x1 = max(0, bar.x - 6), min(FRAME_W, bar.x2 + 420)
    candidates = []; supported_edges = 0
    for y in (bar.y - 1, bar.y2):
        if y < 4 or y + 4 >= FRAME_H:
            continue
        edge = gray[y, x0:x1].astype(np.int16)
        above = gray[y-4:y-2, x0:x1].mean(axis=0)
        below = gray[y+2:y+4, x0:x1].mean(axis=0)
        support = (np.maximum(above, below) - edge >= 6) & ~excluded[y, x0:x1]
        inside = support[bar.x-x0:bar.x2-x0]
        if not len(inside) or inside.mean() < .85:
            continue
        supported_edges += 1
        left, right = bar.x-x0, bar.x2-x0
        while left > 0 and support[left-1]:
            left -= 1
        while right < len(support) and support[right]:
            right += 1
        width = right-left
        if bar.w <= width <= 420 and right < len(support):
            candidates.append(Rect(x0+left, bar.y, width, bar.h))
    if supported_edges != 2:
        return None  # shell tracers and red text are not health bars
    # Text-filled thick bars have shadow/antialias extensions. Bound endpoint
    # disagreement by the bar height; thin tracers cannot use that allowance
    # to join unrelated upper and lower shadow segments.
    edge_tolerance = max(3, bar.h)
    if len(candidates) == 2 and abs(candidates[0].x-candidates[1].x) <= edge_tolerance and abs(candidates[0].x2-candidates[1].x2) <= edge_tolerance:
        return candidates[0]
    return None  # unmatched edge endpoints do not establish a rectangular bar


def _numeric_healthbar_backing(gray: np.ndarray, fill: Rect, excluded: np.ndarray) -> Optional[Rect]:
    """Recover a compact fill only when two long, aligned backing edges surround it."""
    x0, x1 = max(0, fill.x - 6), min(FRAME_W, fill.x2 + 420)
    edges = []
    for rows in ((fill.y - 2, fill.y - 1, fill.y), (fill.y2 - 1, fill.y2, fill.y2 + 1)):
        candidates = []
        for y in rows:
            if y < 4 or y + 4 >= FRAME_H:
                continue
            edge = gray[y, x0:x1].astype(np.int16)
            support = (np.maximum(gray[y-4:y-2, x0:x1].mean(axis=0),
                                  gray[y+2:y+4, x0:x1].mean(axis=0)) - edge >= 6)
            support &= ~excluded[y, x0:x1]
            if support[fill.x-x0:fill.x2-x0].mean() < .85:
                continue
            left, right = fill.x-x0, fill.x2-x0
            while left > 0 and support[left-1]:
                left -= 1
            while right < len(support) and support[right]:
                right += 1
            if max(48, 2.4 * fill.h) <= right-left <= 420 and left > 0 and right < len(support):
                candidates.append((x0+left, x0+right, y))
        edges.append(candidates)
    # The live client draws the plate's top edge longer than the bar, so the
    # two endpoints rarely agree to 3 px. What identifies the bar is the
    # interval both edges support: the pairing accepts endpoints that agree
    # (with the old slack) or runs that overlap by at least one bar length,
    # and the rectangle returned is that overlap.
    need = max(48, 2.4 * fill.h)
    pairs = []
    for top in edges[0]:
        for bottom in edges[1]:
            if not 8 <= bottom[2]-top[2] <= 26:
                continue
            agree = abs(top[0]-bottom[0]) <= 3 and abs(top[1]-bottom[1]) <= 3
            overlap = min(top[1], bottom[1]) - max(top[0], bottom[0])
            if agree or overlap >= need - 3:
                pairs.append((top, bottom))
    if not pairs:
        return None
    top, bottom = max(pairs, key=lambda pair: min(pair[0][1]-pair[0][0], pair[1][1]-pair[1][0]))
    x, right = max(top[0], bottom[0]), min(top[1], bottom[1])
    return Rect(x, top[2]+1, right-x, bottom[2]-top[2]-1)


def _lone_dark_top_edge(gray, bar, drop=25, max_top=115, relative=8, scope_like=False):
    """A bar whose single dark edge is still proof of a drawn health bar.

    Two criteria, because the scoped view is bright: an absolute dark top edge,
    or a nearby edge that is measurably darker than the fill itself (measured
    119 outside against 133 inside on a live scoped plate).
    """
    if bar.h < 8:
        return False
    inside = gray[bar.y + max(1, bar.h // 3):bar.y2 - 1, bar.x:bar.x2].mean()
    if bar.y - 1 >= 1:
        above = gray[bar.y - 1, bar.x:bar.x2].mean()
        if above <= max_top and inside - above >= drop:
            return True
        if scope_like and inside - above >= relative:
            return True
    if scope_like and bar.y2 + 1 < gray.shape[0]:
        below = gray[bar.y2 + 1, bar.x:bar.x2].mean()
        if inside - below >= relative:
            return True
    return False


def detect_nameplates(frame: np.ndarray, *, scope_observed=False, projection=None) -> list[dict[str, Any]]:
    """Find red enemy health/nameplate evidence, never infer ship body from it alone."""
    a = _validate_frame(frame)
    projection = projection or hud_projection.resolve(a)
    excluded = hud_mask(a.shape, projection)
    red = _red_mask(a) & ~excluded
    gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    # Preserve the broad-bar path; compact numeric fills must independently
    # prove a complete backing and must not merge glyph fragments into bars.
    components = _components(red, 10, 10000)
    comps = [c for c in components
             if 24 <= c.w <= 420 and 2 <= c.h <= 24 and c.w / c.h >= 2.4]
    compact = [c for c in components if 12 <= c.w <= 420 and 8 <= c.h <= 24
               and .8 <= c.w / c.h < 2.4 and red[c.y:c.y2, c.x:c.x2].mean() >= .55]
    comps.sort(key=lambda c: (c.y, c.x))
    merged: list[Rect] = []
    for c in comps:
        if merged and abs(c.y - merged[-1].y) <= 12 and c.x <= merged[-1].x2 + 36:
            p = merged.pop(); merged.append(Rect(min(p.x, c.x), min(p.y, c.y),
                max(p.x2, c.x2) - min(p.x, c.x), max(p.y2, c.y2) - min(p.y, c.y)))
        else: merged.append(c)
    candidates = []
    # The binocular view darkens the corners far below the frame centre; only
    # there is a single soft edge accepted as plate evidence.
    corner = float(gray[100:180, 60:260].mean())
    centre = float(gray[700:900, 1150:1410].mean())
    scope_like = scope_observed is True or corner < centre - 20
    for bar in merged:
        backing = _healthbar_backing(gray, bar, excluded)
        if backing is None and _lone_dark_top_edge(gray, bar, scope_like=scope_like):
            # The scope darkens the plate, so the paired-edge backing fails while
            # a real bar still shows its own dark top edge (measured on a live
            # scoped frame: row above the 121x17 fill = 93.9, fill rows 124).
            candidates.append((bar, bar))
            continue
        if backing is None:
            # A wide numeric bar can fail the plain top/bottom pairing while its
            # backing is still complete: measured on a live 3.2 km plate whose
            # 120x17 fill was rejected by the broad path and by the compact
            # width rule, so both gates missed a bar the numeric rule proves.
            backing = _numeric_healthbar_backing(gray, bar, excluded)
        candidates.append((bar, backing))
    for fill in compact:
        backing = _numeric_healthbar_backing(gray, fill, excluded)
        if backing is not None and not any(bar is not None and _rect_intersects(backing, bar)
                                           for _, bar in candidates):
            candidates.append((fill, backing))
    # The binocular view draws the plate as a filled red value box with white
    # digits, which the thin-bar shape rules reject. Measured live at 10.3 km in
    # scope: 113x21 px with the digits punching holes in the fill, so the mask is
    # closed across them before the box is measured.
    import cv2 as _cv2
    closed = _cv2.morphologyEx(red.astype(np.uint8), _cv2.MORPH_CLOSE,
                                np.ones((9, 25), np.uint8))
    count2, _, stats2, _ = _cv2.connectedComponentsWithStats(closed)
    for bx, by, bw, bh, area in stats2[1:]:
        # Tight: a scope value box is 20..30 px tall with white digits inside.
        # A looser box swallowed ordinary red ribbons and made a bar without its
        # edges score as a plate, which the edge-erasure test caught.
        if not (90 <= bw <= 400 and 20 <= bh <= 30 and area >= 800):
            continue
        if not 2.5 <= bw / bh <= 8:
            continue
        white = (gray[by:by + bh, bx:bx + bw] > 200).mean()
        if white < .05:
            continue
        box = Rect(int(bx), int(by), int(bw), int(bh))
        if any(bar is not None and _rect_intersects(box, bar) for _, bar in candidates):
            continue
        candidates.append((box, box))
    result = []
    for fill_bar, bar in candidates:
        if bar is None:
            continue
        # Body is only a verification window; its evidence is evaluated separately.
        body = _clip_rect(Rect(bar.x - max(20, bar.w // 3), bar.y + bar.h + 3,
                               bar.w + max(40, bar.w // 2), max(45, min(220, bar.w))))
        red_pixels = int(red[bar.y:bar.y2, bar.x:bar.x2].sum())
        result.append({"class": "enemy_ship", "evidence": "nameplate_healthbar",
                       "nameplate_bbox": bar.as_list(), "red_fill_bbox": fill_bar.as_list(), "body_roi": body.as_list(),
                       "red_pixels": red_pixels, "confidence": min(0.99, .55 + .4 * min(1, bar.w / 120)),
                       "body_visible": False, "body_status": "unverified"})
    return result


def verify_ship_bodies(frame: np.ndarray, nameplates: Iterable[dict[str, Any]], templates=None,
                       projection=None) -> list[dict[str, Any]]:
    """Localize enclosed dark candidates; image geometry cannot confirm a hull."""
    a = _validate_frame(frame)
    projection = projection or hud_projection.resolve(a)
    excluded = hud_mask(a.shape, projection); out = []
    gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    for item in nameplates:
        r = _clip_rect(Rect(*map(int, item["body_roi"])))
        row = dict(item, body_visible=False, body_status="occluded_or_unresolved", body_confidence=None,
                   body_bbox=None, body_association="unknown", body_roi=r.as_list())
        if r.w >= 32 and r.h >= 24:
            crop = gray[r.y:r.y2, r.x:r.x2]
            # Local contrast proposes structure, including interior island shadows.
            background = cv2.GaussianBlur(crop, (0, 0), 9)
            dark = ((background.astype(np.int16)-crop.astype(np.int16) >= 12) & (crop < 115)).astype(np.uint8)
            dark[excluded[r.y:r.y2, r.x:r.x2]] = 0
            dark[:min(32, r.h)] = 0  # distance labels immediately under healthbar
            joined = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 11), np.uint8))
            contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            candidates = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w < 40 or h < 7 or w / h < 2.5 or w / h > 18: continue
                if x == 0 or x+w >= r.w or y+h >= r.h: continue
                area = cv2.contourArea(contour)
                if area / (w*h) < .2: continue
                seed = dark[y:y+h, x:x+w].astype(bool)
                if not seed.any(): continue
                # Recover absolute-intensity support in surrounding pixels. A
                # high-pass island interior must not become a detached object
                # merely because the verification window cropped its terrain.
                threshold = min(115, float(np.median(crop[y:y+h, x:x+w][seed])) + 12)
                margin = max(32, h)
                context = _clip_rect(Rect(r.x+x-margin, r.y+y-margin,
                                          w+2*margin, h+2*margin))
                support = (gray[context.y:context.y2, context.x:context.x2] <= threshold).astype(np.uint8)
                support[excluded[context.y:context.y2, context.x:context.x2]] = 0
                _, labels, stats, _ = cv2.connectedComponentsWithStats(support, 8)
                sx, sy = r.x+x-context.x, r.y+y-context.y
                seed_labels = labels[sy:sy+h, sx:sx+w][seed]
                counts = np.bincount(seed_labels)
                counts[0] = 0
                label = int(counts.argmax())
                if counts[label] < .75 * len(seed_labels): continue
                bx, by, bw, bh, support_area = map(int, stats[label])
                if bx == 0 or by == 0 or bx+bw >= context.w or by+bh >= context.h: continue
                bbox = Rect(context.x+bx, context.y+by, bw, bh)
                if bbox.x <= r.x or bbox.x2 >= r.x2 or bbox.y < r.y+32 or bbox.y2 >= r.y2: continue
                if bw < 40 or bh < 7 or not 2.5 <= bw / bh <= 18: continue
                if support_area / (bw*bh) < .2: continue
                if bbox.as_list() not in candidates:
                    candidates.append(bbox.as_list())
            if len(candidates) == 1:
                row.update(body_bbox=candidates[0], body_status="visible_silhouette_candidate",
                           body_evidence="enclosed_dark_component_below_nameplate",
                           body_association="unknown", body_visible=False)
            elif len(candidates) > 1:
                row["body_status"] = "ambiguous_silhouettes"
        out.append(row)
    return out


REFERENCE_CAMERA_EXCLUSIONS = (
    (560, 80, 1420, 155),        # top status strip
    (430, 740, 1710, 95),        # reticle/ruler band
    (760, 445, 110, 105),        # left corner warning label
    (1570, 445, 110, 105),       # right corner warning label
)


def _camera_exclusion(shape, projection=None):
    """Fixed UI may be detected elsewhere but cannot support scene motion.

    The reticle/ruler, central warning labels and top status strip contain
    dense stationary corners even when the scenery moves. Keep this separate
    from detection masks: ship and hazard glyphs can occupy these regions.
    """
    projection = projection or hud_projection.identity(shape)
    excluded = hud_mask(shape, projection)
    for rect in REFERENCE_CAMERA_EXCLUSIONS:
        x, y, w, h = projection.clip_rect(rect)
        excluded[y:y + h, x:x + w] = True
    # Exclude corner/flow support crossing a UI boundary, not only its center.
    return cv2.dilate(excluded.astype(np.uint8), np.ones((35, 35), np.uint8)) > 0


def estimate_camera_shift(previous: np.ndarray, current: np.ndarray, exclude: Optional[np.ndarray] = None) -> tuple[float, float, float]:
    """Estimate global observed screen translation with Lucas-Kanade flow.

    This is camera/screen motion in pixels, never world displacement. HUD and
    minimap features are excluded; confidence is robust inlier coverage.
    """
    p = _validate_frame(previous); c = _validate_frame(current)
    pg = cv2.cvtColor(p, cv2.COLOR_RGB2GRAY); cg = cv2.cvtColor(c, cv2.COLOR_RGB2GRAY)
    excluded = _camera_exclusion(pg.shape)
    if exclude is not None:
        supplied = np.asarray(exclude, dtype=bool)
        if supplied.shape != pg.shape:
            raise ValueError('camera exclusion must match native frame shape')
        excluded |= supplied
    mask = (~excluded).astype(np.uint8) * 255
    points = cv2.goodFeaturesToTrack(pg, maxCorners=500, qualityLevel=.01,
                                     minDistance=12, blockSize=7, mask=mask)
    if points is None or len(points) < 4: return 0.0, 0.0, 0.0
    nxt, status, err = cv2.calcOpticalFlowPyrLK(pg, cg, points, None,
        winSize=(31, 31), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
    if nxt is None or status is None: return 0.0, 0.0, 0.0
    good = status.reshape(-1).astype(bool)
    flow = (nxt.reshape(-1, 2) - points.reshape(-1, 2))[good]
    errors = err.reshape(-1)[good] if err is not None else np.zeros(len(flow))
    flow = flow[np.isfinite(flow).all(axis=1) & (errors < 45)]
    if len(flow) < 4: return 0.0, 0.0, float(len(flow) / max(len(points), 1))
    med = np.median(flow, axis=0)
    spread = np.linalg.norm(flow - med, axis=1)
    inliers = spread <= max(2.5, float(np.percentile(spread, 70)))
    if inliers.sum() < 3: return float(med[0]), float(med[1]), 0.0
    shift = np.median(flow[inliers], axis=0)
    return float(shift[0]), float(shift[1]), float(inliers.mean() * len(flow) / max(len(points), 1))


def body_motion(previous: np.ndarray, current: np.ndarray, body_roi: Iterable[int], camera_shift: Optional[tuple[float, float]] = None) -> dict[str, Any]:
    """Track body ROI with optical flow after removing camera translation."""
    p = _validate_frame(previous); c = _validate_frame(current)
    x, y, w, h = map(int, body_roi); r = _clip_rect(Rect(x, y, w, h))
    if x < 0 or y < 0 or w < 12 or h < 12 or x+w > FRAME_W or y+h > FRAME_H:
        return {"state": "unknown", "reason": "out_of_bounds"}
    dx, dy = camera_shift[:2] if camera_shift is not None else estimate_camera_shift(p, c)[:2]
    pg = cv2.cvtColor(p, cv2.COLOR_RGB2GRAY); cg = cv2.cvtColor(c, cv2.COLOR_RGB2GRAY)
    roi_mask = np.zeros(pg.shape, dtype=np.uint8); roi_mask[r.y:r.y2, r.x:r.x2] = 255
    points = cv2.goodFeaturesToTrack(pg, maxCorners=80, qualityLevel=.01,
                                     minDistance=5, blockSize=5, mask=roi_mask)
    semantics = "observed_screen_motion_after_camera_shift; not_world_velocity"
    if points is None or len(points) < 3:
        return {"state": "lost", "reason": "insufficient_body_features",
                "camera_shift_removed": [float(dx), float(dy)], "velocity_semantics": semantics}
    nxt, status, err = cv2.calcOpticalFlowPyrLK(pg, cg, points, None, winSize=(21, 21), maxLevel=2)
    if nxt is None or status is None:
        return {"state": "lost", "reason": "optical_flow_failed", "camera_shift_removed": [float(dx), float(dy)],
                "velocity_semantics": semantics}
    back, back_status, _ = cv2.calcOpticalFlowPyrLK(cg, pg, nxt, None, winSize=(21, 21), maxLevel=2)
    if back is None or back_status is None:
        return {"state": "lost", "reason": "backward_flow_failed", "velocity_semantics": semantics}
    good = status.reshape(-1).astype(bool) & back_status.reshape(-1).astype(bool)
    good &= np.linalg.norm(back.reshape(-1, 2) - points.reshape(-1, 2), axis=1) < 1.5
    if err is not None: good &= np.isfinite(err.reshape(-1)) & (err.reshape(-1) < 25)
    flow = (nxt.reshape(-1, 2) - points.reshape(-1, 2))[good]
    flow = flow[np.isfinite(flow).all(axis=1)]
    if len(flow) < 3:
        state, residual, confidence = "lost", [0.0, 0.0], 0.0
    else:
        residual_flow = np.median(flow, axis=0) - np.asarray([dx, dy], dtype=np.float32)
        residual = [float(residual_flow[0]), float(residual_flow[1])]
        spread = float(np.median(np.linalg.norm(flow - np.median(flow, axis=0), axis=1)))
        confidence = float(min(1.0, len(flow) / max(len(points), 1)))
        state = "tracked" if confidence >= .35 and spread < 8 else ("ambiguous" if confidence >= .2 else "lost")
    return {"state": state, "screen_displacement": residual,
            "camera_shift_removed": [float(dx), float(dy)], "tracked_features": int(len(flow)),
            "confidence": round(confidence, 4), "velocity_semantics": semantics}


def detect_markers(frame: np.ndarray) -> list[dict[str, Any]]:
    """Return native hazard glyph evidence; absence remains unknown."""
    a = _validate_frame(frame)
    out = detect_torpedoes(a) + detect_aircraft(a)
    for hit in out:
        hit["status"] = "observed"
    return sorted(out, key=lambda hit: (hit["bbox"][1], hit["bbox"][0], hit["class"]))


def detect_target_lock(frame: np.ndarray, templates: Optional[dict[str, np.ndarray]] = None) -> dict[str, Any]:
    """Confirm lock only when native arc and serif sprites compose together."""
    templates = templates or {}; arc = templates.get("target_lock_arc"); serif = templates.get("target_lock_serif")
    if arc is None or serif is None:
        return {"state": "unknown", "confirmed": False, "evidence": "native_lock_sprite_pair_unavailable",
                "reason": "absence_is_not_success"}
    a = _validate_frame(frame); roi = Rect(int(FRAME_W*.25), int(FRAME_H*.25), int(FRAME_W*.5), int(FRAME_H*.5))
    arcs = _template_match(a, arc, .80, roi)
    # A serif alone can never establish the required sprite pair.
    serifs = _template_match(a, serif, .80, roi) if arcs else []
    for ar in arcs:
        ax, ay, aw, ah = ar["bbox"]
        for se in serifs:
            sx, sy, sw, sh = se["bbox"]
            if abs(sx - (ax + aw - sw)) <= 8 and abs((sy + sh) - (ay + ah)) <= 12:
                return {"state": "confirmed", "confirmed": True, "evidence": "paired_native_lock_arc_and_serif",
                        "bbox": [min(ax, sx), min(ay, sy), max(ax+aw, sx+sw)-min(ax, sx), max(ay+ah, sy+sh)-min(ay, sy)],
                        "confidence": min(ar["score"], se["score"])}
    return {"state": "unknown", "confirmed": False, "evidence": "lock_sprite_pair_not_verified",
            "reason": "absence_is_not_success"}


def perceive_frame(frame: np.ndarray, previous: Optional[np.ndarray] = None, templates: Optional[dict[str, np.ndarray]] = None,
                   dt_s: Optional[float] = None, *, previous_view: Optional[dict[str, Any]] = None,
                   aircraft_job=None, torpedo_job=None) -> dict[str, Any]:
    a = _validate_frame(frame)
    templates = load_native_templates() if templates is None else templates
    plates = verify_ship_bodies(a, detect_nameplates(a))
    for plate in plates:
        plate.update({"class": "ship", "team": "enemy", "identity": None})
    if aircraft_job is None:
        markers = detect_markers(a)
    else:
        # Finish independent lock/camera work before waiting for either hazard.
        markers = detect_torpedoes(a) if torpedo_job is None else []
    lock = detect_target_lock(a, templates)
    lock.update(target_association="unknown", target_track_id=None)
    shift = None; motions = []; previous_plates = []; previous_markers = []
    valid_dt = dt_s is not None and math.isfinite(dt_s) and 0 < dt_s <= 1
    if previous is not None and valid_dt:
        excluded = hud_mask(a.shape)
        if previous_view is None:
            previous_plates = verify_ship_bodies(previous, detect_nameplates(previous))
            for plate in previous_plates:
                plate.update({"class": "ship", "team": "enemy", "identity": None})
        else:
            # The continuous consumer owns this exact previous-image result.
            previous_plates = previous_view["detections"]
        for d in plates + previous_plates:
            for key in ("nameplate_bbox", "body_roi"):
                x, y, w, h = d[key]; excluded[y:y+h, x:x+w] = True
        shift = estimate_camera_shift(previous, a, excluded)
        previous_markers = detect_markers(previous) if previous_view is None else previous_view["markers"]
    if aircraft_job is not None and torpedo_job is not None:
        markers = torpedo_job.result()
    if aircraft_job is not None:
        markers += aircraft_job.result()
        for hit in markers:
            hit["status"] = "observed"
        markers.sort(key=lambda hit: (hit["bbox"][1], hit["bbox"][0], hit["class"]))
    for i, d in enumerate(plates + markers):
        box = d.get("nameplate_bbox", d.get("bbox")); x, y, w, h = box
        d.update(position_px=[x+w/2, y+h/2], bbox=box, track_state="observed", screen_velocity_px_s=None,
                 velocity_semantics="observed_screen_pixels_per_second; not_world_velocity", track_id=None)
        if d.get("marker_family") == "central_dangers":
            # Warning position describes the HUD, not an associated plane entity.
            d["velocity_semantics"] = "screen_hud_danger_indicator_only; entity_motion_unknown"
            continue
        prior = previous_plates if "nameplate_bbox" in d else previous_markers
        matches = []
        if shift is not None and shift[2] >= .35:
            for j, old in enumerate(prior):
                if old["class"] != d["class"] or old.get("team") != d.get("team"): continue
                ox, oy, ow, oh = old.get("nameplate_bbox", old.get("bbox"))
                delta = np.array([x+w/2-(ox+ow/2)-shift[0], y+h/2-(oy+oh/2)-shift[1]])
                if float(np.linalg.norm(delta)) < 60: matches.append((old, delta))
        if len(matches) == 1:
            ox, oy, ow, oh = matches[0][0].get("nameplate_bbox", matches[0][0].get("bbox"))
            contenders = 0
            for other in plates + markers:
                if other["class"] != d["class"] or other.get("team") != d.get("team"): continue
                tx, ty, tw, th = other.get("nameplate_bbox", other.get("bbox"))
                if math.hypot(tx+tw/2-(ox+ow/2)-shift[0], ty+th/2-(oy+oh/2)-shift[1]) < 60:
                    contenders += 1
            if contenders != 1:
                d["track_state"] = "ambiguous"
                continue
            old, delta = matches[0]; d["track_state"] = "tracked"
            d["screen_velocity_px_s"] = (delta / dt_s).tolist()
            if d.get("body_bbox") is not None and old.get("body_bbox") is not None:
                motion = body_motion(previous, a, old["body_bbox"], shift[:2])
                motion.update(detection_index=i, evidence="silhouette_candidate_flow", ship_association="unverified")
                if motion.get("state") == "tracked":
                    motion["screen_velocity_px_s"] = (np.asarray(motion["screen_displacement"]) / dt_s).tolist()
                motions.append(motion)
        elif len(matches) > 1: d["track_state"] = "ambiguous"
        elif previous is not None: d["track_state"] = "unknown" if shift is None or shift[2] < .35 else "new"
    return {"frame_shape": [FRAME_H, FRAME_W, 3], "coordinate_system": "native_frame_xy_pixels",
            "detections": plates, "markers": markers, "observations": plates + markers, "lock_confirmation": lock,
            "dt_s": dt_s if dt_s is None or math.isfinite(dt_s) else None,
            "camera_shift": None if shift is None else {"dx": shift[0], "dy": shift[1], "score": shift[2]},
            "body_motion": motions, "class_status": {
                kind: "observed" if any(x["class"] == kind for x in markers) else "unknown"
                for kind in ("torpedo", "aircraft")},
            "limitations": ["screen evidence only", "no OCR identity", "silhouette association unverified",
                            "translation camera compensation only; no calibrated 3D/world velocity", "lock target identity unknown"]}


__all__ = ["FRAME_W", "FRAME_H", "MINIMAP_ROI", "Rect", "hud_mask", "detect_nameplates",
           "verify_ship_bodies", "estimate_camera_shift", "body_motion", "detect_markers",
           "detect_target_lock", "perceive_frame"]
