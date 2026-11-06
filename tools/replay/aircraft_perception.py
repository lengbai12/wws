"""Conservative image-only recognition of native battle-view aircraft markers.

The public API takes a native 1600x2560 RGB uint8 capture. Coordinates retain the
letterbox bands. A result proves a HUD glyph, not a visible aircraft body, plane
count, identity, range, or world velocity. Absence is unknown. Enemy and allied
marker colors are measured from matched pixels. Central danger colors are a
warning palette, so their team and aircraft subtype remain unknown.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

FRAME_SHAPE = (1600, 2560, 3)
ASSET_ROOT = Path(__file__).resolve().parents[2] / "state/toalba/gamedata/data/gui/battle_hud/markers"
# Native 21px glyphs plus the enlarged capture renderings. Equal rounded sizes
# are deliberately omitted so a glyph cannot be duplicated by equivalent scales.
SCALES = (1.0, 1.25, 1.5, 1.6, 2.0)
# Exact native stamps retain the original strict NCC gate. Rendered scout and
# rotated warning families use separate full-shape coverage gates below.
MATCH_THRESHOLD = .95
NEGATIVE_MARGIN = .035
_NEIGHBOR_KERNEL = np.ones((3, 3), np.uint8)
# Preserve the original float64 palette boundaries, including values such as
# 90 * .7 just below 63. LUTs avoid full-frame float64 products per palette.
_ENEMY_GREEN_LIMIT = np.floor(np.arange(256) * .46).astype(np.uint8)
_DANGER_GREEN_LIMIT = np.floor(np.arange(256) * .7).astype(np.uint8)


@dataclass(frozen=True)
class _Template:
    asset: str
    family: str
    symbol: str
    team: str
    scale: float
    shape: np.ndarray
    core: np.ndarray
    width: int
    height: int
    rendering: str = "native"

    @cached_property
    def color_core(self):
        return self.core.astype(np.float32)

    @cached_property
    def color_support(self):
        x, y, w, h = cv2.boundingRect(self.core.view(np.uint8))
        return x, y, self.color_core[y:y+h, x:x+w]

    @cached_property
    def required_color(self):
        return .8 * np.count_nonzero(self.core)

    @cached_property
    def core_key(self):
        return self.core.shape, self.core.tobytes()

    @cached_property
    def support_key(self):
        # The padded score domain matters as well as the unpadded color core.
        return "supported_crop", self.core_key, self.shape.shape

    @cached_property
    def expected(self):
        return self.shape[2:-2, 2:-2] >= self.shape.max()*.35

    @cached_property
    def expected_near(self):
        return cv2.dilate(self.expected.view(np.uint8), _NEIGHBOR_KERNEL).astype(bool)


@lru_cache(maxsize=1)
def _rendered_templates() -> tuple[_Template, ...]:
    """Full native scout assets and freely rotated central plane warnings."""
    loaded = []
    paths = [(ASSET_ROOT / f"plane/consumables/scout_{team}.png", team, "scout")
             for team in ("ally", "enemy")]
    danger_root = ASSET_ROOT.parent / "central_dangers/icons"
    paths += [(danger_root / f"icon_danger_plane_{range_}.png", "enemy", "danger")
              for range_ in ("near", "far")]
    for path, team, rendering in paths:
        if not path.is_file():
            continue
        rgba = np.asarray(Image.open(path).convert("RGBA"))
        shape = (_signal(rgba[..., :3], team) * rgba[..., 3] / 255).astype(np.float32)
        scales = (1.25, 4/3, 1.5) if rendering == "scout" else (4/3, 1.5, 1.6, 1.7)
        for scale in scales:
            h, w = shape.shape
            resized = cv2.resize(shape, (round(w*scale), round(h*scale)))
            angles = (0,) if rendering == "scout" else range(0, 360, 5)
            for angle in angles:
                glyph = resized
                if rendering == "danger":
                    h, w = resized.shape
                    extent = int(np.ceil(np.hypot(h, w))) + 4
                    canvas = np.zeros((extent, extent), np.float32)
                    oy, ox = (extent-h)//2, (extent-w)//2
                    canvas[oy:oy+h, ox:ox+w] = resized
                    matrix = cv2.getRotationMatrix2D(((extent-1)/2, (extent-1)/2), angle, 1)
                    glyph = cv2.warpAffine(canvas, matrix, (extent, extent))
                    ys, xs = np.where(glyph >= glyph.max()*.08)
                    glyph = glyph[ys.min():ys.max()+1, xs.min():xs.max()+1]
                h, w = glyph.shape
                asset = (path.relative_to(ASSET_ROOT).as_posix() if rendering == "scout"
                         else "central_dangers/icons/" + path.name)
                loaded.append(_Template(asset, "consumables" if rendering == "scout" else "central_dangers",
                                        "scout" if rendering == "scout" else "unknown", team, scale,
                                        np.pad(glyph, 2), glyph >= glyph.max()*.7, w, h, rendering))
    return tuple(loaded)


def _shape_support(patch: np.ndarray, template: _Template) -> tuple[float, float, np.ndarray]:
    """Bidirectional whole-glyph coverage, allowing one compression pixel."""
    if template.rendering == "danger":
        observed = _danger_pixels(patch)
    elif template.team == "ally":
        r, g, b = (patch[..., i].astype(np.int16) for i in range(3))
        observed = (g >= 110) & (g-r >= 55) & (g-b >= -8)
    else:
        observed = _team_pixels(patch, "enemy")
    expected = template.expected
    expected_near = template.expected_near
    observed_near = cv2.dilate(observed.view(np.uint8), _NEIGHBOR_KERNEL).astype(bool)
    precision = float((observed & expected_near).sum() / max(1, observed.sum()))
    recall = float(observed_near[expected].mean())
    return precision, recall, observed


def _signal(rgb: np.ndarray, team: str) -> np.ndarray:
    a = rgb.astype(np.float32)
    if team == "enemy":
        return np.maximum(0, a[..., 0] - np.maximum(a[..., 1], a[..., 2]))
    return np.maximum(0, np.minimum(a[..., 1], a[..., 2]) - a[..., 0])


def _signal_channels(r: np.ndarray, g: np.ndarray, b: np.ndarray, team: str) -> np.ndarray:
    """Compute the same integer-valued signal without recasting an RGB crop."""
    if team == "enemy":
        return np.maximum(0, r - np.maximum(g, b)).astype(np.float32)
    return np.maximum(0, np.minimum(g, b) - r).astype(np.float32)


def _rendered_signal_channels(r: np.ndarray, g: np.ndarray, b: np.ndarray, team: str) -> np.ndarray:
    if team == "enemy":
        return _signal_channels(r, g, b, team)
    # Native green scout arcs are compressed over blue sky. Subtract the sky's
    # green/red baseline rather than treating its blue channel as glyph ink.
    return np.maximum(0, g - r - 40).astype(np.float32)


@lru_cache(maxsize=1)
def _templates() -> tuple[tuple[_Template, ...], tuple[_Template, ...]]:
    """Load every native plane family; no synthetic substitute for missing assets."""
    planes = sorted((ASSET_ROOT / "plane").rglob("*.png"))
    negatives = sorted((ASSET_ROOT / "ship").rglob("*enemy*.png"))
    negatives += sorted((ASSET_ROOT / "ship").rglob("*ally*.png"))
    negatives += sorted((ASSET_ROOT / "building_icons").rglob("*.png"))
    for directory in ("mines", "torpedoes", "missile", "state_markers", "depth_charges"):
        negatives += sorted((ASSET_ROOT / directory).glob("*.png"))
    groups = []
    for paths, is_plane in ((planes, True), (negatives, False)):
        loaded = []
        for path in paths:
            if is_plane and not path.stem.endswith(("_enemy", "_ally")):
                continue
            rgba = np.asarray(Image.open(path).convert("RGBA"))
            alpha = rgba[..., 3].astype(np.float32) / 255
            teams = (path.stem.rsplit("_", 1)[1],) if is_plane else ("enemy", "ally")
            for team in teams:
                shape = _signal(rgba[..., :3], team) * alpha
                if float(shape.max()) < 35:
                    continue
                for scale in SCALES:
                    h, w = shape.shape
                    width, height = round(w * scale), round(h * scale)
                    if max(width, height) > 100:
                        continue
                    resized = cv2.resize(shape, (width, height), interpolation=cv2.INTER_LINEAR)
                    core = resized >= resized.max() * .7
                    if int(core.sum()) < 4:
                        continue
                    padded = np.pad(resized, 2).astype(np.float32)
                    loaded.append(_Template(
                        path.relative_to(ASSET_ROOT).as_posix(), path.parent.name,
                        path.stem.rsplit("_", 1)[0], team, scale, padded,
                        core, width, height))
        groups.append(tuple(loaded))
    return groups[0], groups[1]


def _hud_mask() -> np.ndarray:
    mask = np.zeros(FRAME_SHAPE[:2], dtype=bool)
    for x, y, w, h in ((0, 0, 2560, 75), (0, 1510, 2560, 90),
                       (0, 75, 560, 190), (0, 1240, 360, 360),
                       (1980, 65, 580, 690), (940, 1320, 1020, 280),
                       (1836, 796, 720, 723)):
        mask[y:y+h, x:x+w] = True
    return mask


def _team_pixels(rgb: np.ndarray, team: str) -> np.ndarray:
    r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
    return _team_pixel_channels(r, g, b, team)


def _team_pixel_channels(r, g, b, team):
    if team == "enemy":
        # Extracted enemy plane cores are about RGB(255,64,0). A high green
        # fraction (ordinary orange fire/projectiles) cannot establish enemy.
        return ((r >= 145) & (r-g >= 70) &
                (g <= cv2.LUT(r.astype(np.uint8), _ENEMY_GREEN_LIMIT)) & (b <= g+25))
    return (g >= 130) & (b >= 90) & (g-r >= 65) & (b-r >= 35)


def _danger_pixels(rgb: np.ndarray) -> np.ndarray:
    # Native near/far warnings use red (255,54,0) and amber (255,138,0).
    # This palette only proposes complete danger shapes, never enemy planes.
    r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
    return _danger_pixel_channels(r, g, b)


def _danger_pixel_channels(r, g, b):
    return ((r >= 145) & (r-g >= 60) &
            (g <= cv2.LUT(r.astype(np.uint8), _DANGER_GREEN_LIMIT)) & (b <= g+25))


def _negative_score(signal: np.ndarray, box: list[int], team: str,
                    negatives: tuple[_Template, ...]) -> float:
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    best = -1.0
    for template in negatives:
        if template.team != team:
            continue
        th, tw = template.shape.shape
        # Competition must describe this glyph, rather than an unrelated ship
        # elsewhere in the proposal window. Allow two pixels of registration.
        x0, y0 = int(round(cx-tw/2))-2, int(round(cy-th/2))-2
        x1, y1 = x0+tw+4, y0+th+4
        if x0 < 0 or y0 < 0 or x1 > signal.shape[1] or y1 > signal.shape[0]:
            continue
        scores = cv2.matchTemplate(signal[y0:y1, x0:x1], template.shape, cv2.TM_CCOEFF_NORMED)
        finite = scores[np.isfinite(scores)]
        if finite.size:
            best = max(best, float(finite.max()))
    return best


def prepare() -> None:
    """Load static recognition evidence before capture-age accounting starts."""
    planes, _ = _templates()
    for template in planes + _rendered_templates():
        template.color_core
        template.color_support
        template.required_color
        template.core_key
        template.support_key
    for template in _rendered_templates():
        template.expected
        template.expected_near


def _supported_crop(crop: np.ndarray, colored: np.ndarray,
                    template: _Template, cache: dict) -> tuple[np.ndarray, int, int] | None:
    """Prune translations that cannot reach the unchanged .8 core-color gate.

    Count colored core pixels before expensive normalized shape matching. Keep
    one score pixel around their feasible envelope, preserving the original
    local-maximum neighbors; the final shape/color gates remain unchanged.
    """
    key = template.support_key
    if key not in cache:
        cache[key] = _supported_bounds(crop, colored, template, cache)
    bounds = cache[key]
    if bounds is None:
        return None
    x0, y0, x1, y1 = bounds
    return crop[y0:y1, x0:x1], x0, y0


def _supported_bounds(crop: np.ndarray, colored: np.ndarray,
                      template: _Template, cache: dict) -> tuple[int, int, int, int] | None:
    # Equivalent cores share feasibility and its envelope within this proposal.
    # Cache only four coordinates (or rejection), never another image allocation.
    th, tw = template.shape.shape
    if crop.shape[0] < th or crop.shape[1] < tw:
        return None
    # Every core is a subset of this proposal's colored pixels. Reject a
    # count shortage before allocating integral/rectangle sums per core size.
    if "color_total" not in cache:
        cache["color_total"] = cv2.countNonZero(colored)
    if cache["color_total"] < template.required_color - .1:
        return None
    # Shared groups compute rectangle bounds and exact core feasibility below.
    # Only independent windows need their own rectangle scan.
    shared = cache.get("shared")
    if shared is None:
        bound_key = ("rectangle_max", template.core.shape)
        if bound_key not in cache:
            h, w = template.core.shape
            if "color_integral" not in cache:
                cache["color_integral"] = cv2.integral(colored, sdepth=cv2.CV_32F)
            integral = cache["color_integral"]
            counts = integral[h:, w:] - integral[:-h, w:]
            counts -= integral[h:, :-w]
            counts += integral[:-h, :-w]
            cache[bound_key] = float(counts.max())
        if cache[bound_key] < template.required_color - .1:
            return None
    key = ("core_translations", template.core_key, template.shape.shape)
    if key not in cache:
        # Overlapping proposal windows share this pixel-local correlation.
        # Restrict the shared feasible translations back to this exact window;
        # shape matching still sees the original local-maximum neighbours.
        if "shared" in cache:
            shared_color, shared_cache, dx, dy = cache["shared"]
            shared_key = template.core_key
            if shared_key not in shared_cache:
                # The core is contained in its rectangle. Exact binary
                # rectangle counts bound every possible core correlation;
                # correlate only the enclosing feasible rectangle, once for
                # this group. Coordinates remain in the shared source domain.
                if "integral" not in shared_cache:
                    # Binary sums stay exact: a native frame has fewer than
                    # 2**24 pixels, the float32 consecutive-integer limit.
                    shared_cache["integral"] = cv2.integral(shared_color, sdepth=cv2.CV_32F)
                h, w = template.core.shape
                rectangle_key = ("rectangle_counts", h, w)
                if rectangle_key not in shared_cache:
                    integral = shared_cache["integral"]
                    counts = integral[h:, w:] - integral[:-h, w:]
                    counts -= integral[h:, :-w]
                    counts += integral[:-h, :-w]
                    shared_cache[rectangle_key] = counts
                # Equal rectangle sizes and color thresholds have the same
                # feasible envelope even when their core shapes differ. Scan
                # the binary mask once, without allocating every coordinate.
                envelope_key = ("rectangle_envelope", h, w, template.required_color)
                if envelope_key not in shared_cache:
                    feasible = shared_cache[rectangle_key] >= template.required_color-.1
                    shared_cache[envelope_key] = cv2.boundingRect(feasible.view(np.uint8))
                x0, y0, width, height = shared_cache[envelope_key]
                if width:
                    x1, y1 = x0+width, y0+height
                    # Zero border pixels contribute nothing. Shift the source
                    # by the same offset so all original translations survive.
                    cx, cy, core = template.color_support
                    ch, cw = core.shape
                    counts = cv2.matchTemplate(
                        shared_color[y0+cy:y1+cy+ch-1, x0+cx:x1+cx+cw-1],
                        core, cv2.TM_CCORR)
                    mask = (counts >= template.required_color-.1).view(np.uint8)
                    bounds = cv2.boundingRect(mask)
                    shared_cache[shared_key] = (mask, x0, y0, bounds) if bounds[2] else None
                else:
                    shared_cache[shared_key] = None
            feasible = shared_cache[shared_key]
            if feasible is None:
                return None
            # Reject disjoint proposals or reuse a fully contained envelope
            # without rescanning the shared mask. Partial intersections still
            # scan the exact original score domain and preserve its halo.
            mask, mx, my, (bx, by, bw, bh) = feasible
            left, top = max(0, dx+2-mx), max(0, dy+2-my)
            right = min(mask.shape[1], dx+crop.shape[1]-tw+3-mx)
            bottom = min(mask.shape[0], dy+crop.shape[0]-th+3-my)
            if right <= left or bottom <= top:
                return None
            if right <= bx or bottom <= by or left >= bx+bw or top >= by+bh:
                return None
            if left <= bx and top <= by and right >= bx+bw and bottom >= by+bh:
                x, y, width, height = bx-left, by-top, bw, bh
            else:
                x, y, width, height = cv2.boundingRect(mask[top:bottom, left:right])
                if not width:
                    return None
            x += left+mx-dx-2
            y += top+my-dy-2
            return (max(0, x-1), max(0, y-1),
                    min(crop.shape[1], x+width+tw),
                    min(crop.shape[0], y+height+th))
        else:
            cx, cy, core = template.color_support
            ch, cw = core.shape
            h, w = template.core.shape
            counts = cv2.matchTemplate(
                colored[cy:colored.shape[0]-h+cy+ch, cx:colored.shape[1]-w+cx+cw],
                core, cv2.TM_CCORR)
            cache[key] = np.where(counts[2:crop.shape[0]-th+3, 2:crop.shape[1]-tw+3]
                                  >= template.required_color-.1)
    ys, xs = cache[key]
    if not xs.size:
        return None
    x0, y0 = max(0, int(xs.min())-1), max(0, int(ys.min())-1)
    x1 = min(crop.shape[1], int(xs.max())+tw+1)
    y1 = min(crop.shape[0], int(ys.max())+th+1)
    return x0, y0, x1, y1


def _window_groups(windows):
    """Keep the original sorted-window and first-seen group order."""
    groups = {}
    for window in sorted(windows):
        groups.setdefault((window[0] // 128, window[1] // 128), []).append(window)
    return groups.values()


def _shared_windows(group, color_mask, scout_color, danger_color, team):
    """Own correlation caches for exactly one independent nearby group."""
    gx, gy = min(w[0] for w in group), min(w[1] for w in group)
    right, bottom = max(w[2] for w in group), max(w[3] for w in group)
    colored = color_mask[gy:bottom, gx:right].astype(np.float32)
    rendered = colored if team == "enemy" else scout_color[gy:bottom, gx:right].astype(np.float32)
    core_cache = {}
    rendered_cache = core_cache if team == "enemy" else {}
    danger = danger_color[gy:bottom, gx:right].astype(np.float32) if team == "enemy" else None
    danger_cache = {}
    for window in group:
        x, y, _, _ = window
        yield (window, (colored, core_cache, x-gx, y-gy),
               (rendered, rendered_cache, x-gx, y-gy),
               (danger, danger_cache, x-gx, y-gy))


def detect(rgb: np.ndarray, excluded_mask: np.ndarray | None = None) -> list[dict]:
    """Return measured aircraft HUD markers, with conservative unknown subtypes.

    Each marker contains class='aircraft', bbox=[x,y,w,h], score in [0,1],
    evidence='native_aircraft_sprite_shape_and_measured_color', team,
    aircraft_type (or 'unknown'), marker_family (or 'unknown'), matched_assets,
    measured_color_rgb, color_support_fraction and negative_score. Central
    danger glyphs retain class='aircraft' for existing consumers but have
    marker_family='central_dangers', team/type='unknown', and semantics
    'screen_hud_danger_indicator_only'. No body, count, range or world state is
    inferred. Malformed frames/masks raise ValueError. The built-in HUD/minimap/
    letterbox exclusion is always unioned with excluded_mask; a glyph whose
    full bounding box touches excluded pixels is rejected.
    """
    frame = np.asarray(rgb)
    if frame.shape != FRAME_SHAPE or frame.dtype != np.uint8:
        raise ValueError("rgb must be a 1600x2560x3 RGB uint8 frame")
    excluded = _hud_mask()
    if excluded_mask is not None:
        extra = np.asarray(excluded_mask)
        if extra.shape != FRAME_SHAPE[:2] or extra.dtype != np.bool_:
            raise ValueError("excluded_mask must be a 1600x2560 bool array")
        excluded |= extra
    planes, negatives = _templates()
    if not planes:
        return []
    rendered = _rendered_templates()
    r, g, b = (frame[..., i].astype(np.int16) for i in range(3))
    scout_color = (g >= 110) & (g-r >= 55) & (g-b >= -8)
    danger_color = _danger_pixel_channels(r, g, b)
    def prepare_team(team):
        team_planes = tuple(t for t in planes if t.team == team)
        team_rendered = tuple(t for t in rendered if t.team == team)
        scout_rendered = tuple(t for t in team_rendered if t.rendering != "danger")
        # Signals are pixel-local: evaluate only proposal crops, rather than
        # repeatedly allocating float RGB copies of the entire native frame.
        color_mask = _team_pixel_channels(r, g, b, team)
        # Components only propose search windows; they never establish class.
        proposal_color = color_mask | danger_color if team == "enemy" else color_mask
        n, _, stats, _ = cv2.connectedComponentsWithStats((proposal_color & ~excluded).astype(np.uint8))
        windows = {}
        for x, y, w, h, area in stats[1:n]:
            if area < 6 or not (3 <= w <= 75 and 3 <= h <= 75):
                continue
            window = (max(0, int(x)-45), max(0, int(y)-45),
                      min(2560, int(x+w)+45), min(1600, int(y+h)+45))
            windows[window] = windows.get(window, False) or (area >= 100 and w >= 15 and h >= 15)
        return [(team, group, windows, color_mask, team_planes, team_rendered, scout_rendered)
                for group in _window_groups(windows)]

    def detect_group(job):
        team, group, windows, color_mask, team_planes, team_rendered, scout_rendered = job
        hits = []
        for (x0, y0, x1, y1), shared_core, shared_rendered, shared_danger in _shared_windows(
                group, color_mask, scout_color, danger_color, team):
            r_crop = r[y0:y1, x0:x1]
            g_crop = g[y0:y1, x0:x1]
            b_crop = b[y0:y1, x0:x1]
            crop = _signal_channels(r_crop, g_crop, b_crop, team)
            rendered_crop = (crop if team == "enemy"
                             else _rendered_signal_channels(r_crop, g_crop, b_crop, team))
            # Shared groups already hold float32 masks for correlation. Slice views
            # for this window instead of re-casting the same boolean masks.
            core_group, _, core_dx, core_dy = shared_core
            colored = core_group[core_dy:core_dy + y1 - y0,
                                 core_dx:core_dx + x1 - x0]
            core_cache = {"shared": shared_core}
            if team == "enemy":
                rendered_color = colored
            else:
                rendered_group, _, rendered_dx, rendered_dy = shared_rendered
                rendered_color = rendered_group[rendered_dy:rendered_dy + y1 - y0,
                                                rendered_dx:rendered_dx + x1 - x0]
            rendered_cache = core_cache if team == "enemy" else {"shared": shared_rendered}
            if team == "enemy":
                danger_group, _, danger_dx, danger_dy = shared_danger
                danger_colored = danger_group[danger_dy:danger_dy + y1 - y0,
                                              danger_dx:danger_dx + x1 - x0]
            else:
                danger_colored = None
            danger_cache = {"shared": shared_danger}
            for template in team_planes:
                supported = _supported_crop(crop, colored, template, core_cache)
                if supported is None:
                    continue
                matching_crop, sx, sy = supported
                scores = cv2.matchTemplate(matching_crop, template.shape, cv2.TM_CCOEFF_NORMED)
                if not np.any(scores >= MATCH_THRESHOLD):
                    continue
                peaks = ((scores >= MATCH_THRESHOLD) & np.isfinite(scores) &
                         (scores >= cv2.dilate(scores, _NEIGHBOR_KERNEL)))
                ys, xs = np.where(peaks)
                for yy, xx in zip(ys, xs):
                    x, y = x0+sx+int(xx)+2, y0+sy+int(yy)+2
                    w, h = template.width, template.height
                    if excluded[y:y+h, x:x+w].any():
                        continue
                    patch = frame[y:y+h, x:x+w]
                    fraction = float(color_mask[y:y+h, x:x+w][template.core].mean())
                    if fraction < .8:
                        continue
                    score = float(scores[yy, xx])
                    hits.append((score, [x, y, w, h], template, fraction,
                                 np.median(patch[template.core], axis=0).tolist()))
            candidates = team_rendered if windows[(x0, y0, x1, y1)] else scout_rendered
            for template in candidates:
                support_color = danger_colored if template.rendering == "danger" else rendered_color
                support_cache = danger_cache if template.rendering == "danger" else rendered_cache
                supported = _supported_crop(rendered_crop, support_color, template, support_cache)
                if supported is None:
                    continue
                matching_crop, sx, sy = supported
                scores = cv2.matchTemplate(matching_crop, template.shape, cv2.TM_CCOEFF_NORMED)
                # These are proposal gates only; full foreground and background
                # geometry must agree in both directions to accept a rendering.
                threshold = .76 if template.rendering == "scout" else .86
                if not np.any(scores >= threshold):
                    continue
                peaks = ((scores >= threshold) & np.isfinite(scores) &
                         (scores >= cv2.dilate(scores, _NEIGHBOR_KERNEL) - .015))
                ys, xs = np.where(peaks)
                for yy, xx in zip(ys, xs):
                    x, y = x0+sx+int(xx)+2, y0+sy+int(yy)+2
                    w, h = template.width, template.height
                    if excluded[y:y+h, x:x+w].any():
                        continue
                    patch = frame[y:y+h, x:x+w]
                    precision, recall, observed = _shape_support(patch, template)
                    if precision < .94 or recall < .94:
                        continue
                    # Every quarter of the asset must be present; a partial arc
                    # or ship label cannot win by matching only the bright core.
                    expected = template.expected
                    near = cv2.dilate(observed.view(np.uint8), _NEIGHBOR_KERNEL).astype(bool)
                    quadrants = [(slice(0, h//2), slice(0, w//2)),
                                 (slice(0, h//2), slice(w//2, w)),
                                 (slice(h//2, h), slice(0, w//2)),
                                 (slice(h//2, h), slice(w//2, w))]
                    if any(expected[q].any() and near[q][expected[q]].mean() < .8 for q in quadrants):
                        continue
                    fraction = float(observed[template.core].mean())
                    if fraction < .8:
                        continue
                    hits.append((float(scores[yy, xx]), [x, y, w, h], template, fraction,
                                 np.median(patch[observed & expected], axis=0).tolist()))
        return hits

    # Each active job owns one group's caches and original ROI sizes. Queue
    # only small job descriptors; two workers keep independent groups moving
    # without batch barriers while HUD/torpedo work runs alongside this detector.
    jobs = prepare_team("enemy") + prepare_team("ally")
    hits = []
    with ThreadPoolExecutor(max_workers=2) as workers:
        pending = [workers.submit(detect_group, job) for job in jobs]
        failure = None
        for future in pending:
            try:
                hits.extend(future.result())
            except BaseException as error:
                if failure is None:
                    failure = error
        if failure is not None:
            raise failure
    hits.sort(key=lambda hit: (-hit[0], hit[1][1], hit[1][0], hit[2].asset))
    groups: list[list[tuple]] = []
    for hit in hits:
        x, y, w, h = hit[1]
        for group in groups:
            gx, gy, gw, gh = group[0][1]
            if abs(x+w/2-gx-gw/2) < min(w, gw)*.4 and abs(y+h/2-gy-gh/2) < min(h, gh)*.4:
                group.append(hit)
                break
        else:
            groups.append([hit])
    result = []
    negative_extents = {}
    if groups:
        for team in ("enemy", "ally"):
            shapes = [t.shape.shape for t in negatives if t.team == team]
            negative_extents[team] = (max((h for h, _ in shapes), default=0),
                                      max((w for _, w in shapes), default=0))
    for group in groups:
        score, box, template, fraction, measured = group[0]
        # Negative competition reads only each template's original centered
        # registration window. Bound their union before constructing a signal.
        # Even origins preserve round-to-even registration at half pixels.
        x, y, w, h = box
        nh, nw = negative_extents[template.team]
        cx, cy = x+w/2, y+h/2
        nx = max(0, int(cx-nw/2)-4) & ~1
        ny = max(0, int(cy-nh/2)-4) & ~1
        right = min(frame.shape[1], int(cx+nw/2)+5)
        bottom = min(frame.shape[0], int(cy+nh/2)+5)
        channels = r[ny:bottom, nx:right], g[ny:bottom, nx:right], b[ny:bottom, nx:right]
        signal = (_signal_channels(*channels, template.team) if template.rendering == "native"
                  else _rendered_signal_channels(*channels, template.team))
        negative = _negative_score(signal, [x-nx, y-ny, w, h], template.team, negatives)
        if negative >= score-NEGATIVE_MARGIN:
            continue
        alternatives = [hit[2] for hit in group if hit[0] >= score-.04]
        symbols = {t.symbol for t in alternatives}
        families = {t.family for t in alternatives}
        result.append({"class": "aircraft", "bbox": box, "score": min(1.0, score),
                       "evidence": "native_aircraft_sprite_shape_and_measured_color",
                       "team": "unknown" if template.rendering == "danger" else template.team,
                       "aircraft_type": next(iter(symbols)) if len(symbols) == 1 else "unknown",
                       "marker_family": next(iter(families)) if len(families) == 1 else "unknown",
                       "matched_assets": sorted({t.asset for t in alternatives}),
                       "measured_color_rgb": measured, "color_support_fraction": fraction,
                       "negative_score": negative, "world_velocity": None,
                       "observation_semantics": ("screen_hud_danger_indicator_only" if template.rendering == "danger"
                                                 else "screen_hud_marker_only")})
    return sorted(result, key=lambda marker: (marker["bbox"][1], marker["bbox"][0]))
