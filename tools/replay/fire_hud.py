#!/usr/bin/env python3
"""Conservative image-only fire HUD readouts for any captured resolution.

Every HUD rectangle in this module is written in *reference* space: the
2560x1440 game surface inside the native 2560x1600 capture (80 px bars). The
readers hand those rectangles to ``hud_projection`` and touch the pixels of the
frame they were given, so a client that renders 1924x1058 or 1920x1080 is read
with its own boxes instead of the reference numbers taken as raw coordinates.
On a reference-size frame the projection is the identity, and the numbers and
evidence are exactly what the pre-projection reader produced. The switch is
explicit: pass ``FireHudReader(projection=...)`` to force one projection for a
burst of frames, or set ``WWS_HUD_PROJECTION=reference`` to force the identity
everywhere (``hud_projection.projection_disabled``), which restores the
pre-projection path for a live run without editing code.

Bboxes are [x, y, width, height]. Values are None when unresolved. Confidence
is a shape agreement score, not a calibrated probability. No replay fields are
accepted. The bundled selector/caret references are masked screen pixels from
state/tracking-mainview-236374.png; other anchors and fonts are native assets.
Battle evidence needs two ammo sprites in one supported selector layout and an
independent mount layout or reticle, or a selected native gun sprite with both
a full colored mount layout and a native reticle. A gun sprite plus the torpedo
sprite remains usable while a cooldown overlay obscures the other gun sprite.

White HUD text drawn on a grey sea (storm, night) defeats the absolute
brightness rules: the sea is ink, and every digit reads "malformed_stroke".
``battle_time_left``, ``nameplate_range_caption``/``nameplate_range_km``/
``target_range_km`` and ``impact_readout`` therefore keep their existing path
first and retry the same pixels through a local-background front end
(``low_contrast_residual`` and ``canonical_ink``) only when that fails. The
front end supplies contrast and segmentation; which glyph a stroke is stays the
reader's own template, hole-topology and margin decision, and a read is accepted
only when several ink thresholds agree on it. ``impact_readout`` also reports
the drawn "- - km" placeholder as ``'beyond'`` instead of estimating a range.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import re
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from main_view_perception import MINIMAP_ROI

try:
    import hud_projection
except ModuleNotFoundError:  # imported by path, without tools/replay on sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import hud_projection

ROOT = Path(__file__).resolve().parents[2]
GUI = ROOT / "state/toalba/gamedata/data/gui"
REFERENCES = Path(__file__).with_name("fire_hud_assets")
# The reference capture the constants below were measured on. It is a shape,
# not a requirement: readers accept any HxWx3 frame and project the rectangles.
FRAME_SHAPE = (1600, 2560, 3)
# Reference space only: a 2560x1440 game surface inside a 2560x1600 canvas.
# Every reader call projects these through hud_projection before touching
# pixels, so a client that renders a different surface still gets its own HUD
# rectangles instead of these numbers read as raw coordinates.
RELOAD_ROIS = [(1118 + 96*i, 1338, 42, 25) for i in range(4)]
# The frame-space layout those rectangles belong to: four mounts, 96 px stride,
# 60 px bar bodies, bar row at y=1365 on the reference surface.
REFERENCE_RELOAD_LAYOUT = (4, 1108, 96, 60, 1365)
SELECTOR_ROIS = [(940, 1395, 80, 115), (1022, 1395, 90, 115), (1170, 1395, 90, 115)]
# Reference ammo-bar row and the sprite offsets the live clients shift it by.
REFERENCE_SELECTOR_ROW = (1395, 1510)
# Measured offsets of the ammo-panel pair for each supported client layout.
# -85 comes from X Linhai (live frame 2026-10-20): its HE/AP sprites sit at
# x≈852 and x≈950, while every existing offset sampled between them.
# Measured offsets of the ammo-panel pair for each supported client layout.
# +25: X Vermont in the 2026-10-20 12:40 training battle -- the highlighted
#      first slot sits at x 965..1049 while the reference row starts at 940.
REFERENCE_SELECTOR_OFFSETS = (0, -44, 71, -138, -28, 28, -85, 25)
REFERENCE_SELECTOR_BBOX = (896, 1395, 364, 115)
REFERENCE_DEPTH_CHARGE_ROI = (1259, 1395, 90, 115)
REFERENCE_BATTLE_ROIS = {
    "caret": (900, 550, 700, 550),
    "caret_wide": (120, 200, 2320, 1300),
    "ring": (1140, 660, 280, 280),
    "anchored_number": (700, 550, 1000, 550),
}
REFERENCE_SCOPE_RULER_ROWS = (790, 830)
REFERENCE_RETICLE_CENTRE = (1280, 800)
# Slot-selection search band: the bottom ammo bar and the number-plate row that
# hangs under it, nothing of the scope or the minimap. Measured on the
# 2560x1600 capture state/s_g.png, where the highlighted first slot occupies
# x 965..1049, y 1384..1468 and its number plate 995..1022, y 1466..1496.
REFERENCE_SLOT_BAND = (820, 1290, 1080, 290)
# The band height those size limits are quoted against; every limit below is
# rescaled by measured_band_height/290 so a scaled client still reads.
SLOT_BAND_REF_HEIGHT = 290.0
# A slot's number plate is a square of this side, in reference pixels.
SLOT_PLATE_SIDE = (15.0, 38.0)
# Consumable columns follow the shell-type columns, so only the first two
# left-to-right slot columns hold main-battery ammunition. That is a property
# of the HUD, not of any hull: the ordinals come from this frame's own plates.
MAIN_BATTERY_COLUMNS = 2


def evidence(status="unknown", bbox=None, confidence=0.0, **values):
    return {"status": status, "bbox": bbox, "confidence": float(confidence),
            "raw": {}, **values}


def _trim(mask):
    yy, xx = np.where(mask > 0)
    if not len(xx):
        return None
    return mask[yy.min():yy.max()+1, xx.min():xx.max()+1]


def _normalized(mask):
    return cv2.resize(mask.astype(np.uint8), (20, 32), interpolation=cv2.INTER_NEAREST) > 0


def _glyph_holes(mask):
    """Significant enclosed background centroids in normalized glyph coordinates.

    Ignore one-pixel raster pinholes; an open stroke has no enclosed background.
    Padding makes the exterior explicit even when foreground touches every edge.
    """
    height, width = mask.shape
    _, _, stats, centres = cv2.connectedComponentsWithStats(
        np.pad((~mask).astype(np.uint8), 1, constant_values=1))
    return sorted([
        ((centres[k][0] - .5) / width, (centres[k][1] - .5) / height)
        for k, (x, y, w, h, area) in enumerate(stats[1:], 1)
        if x > 0 and y > 0 and x+w < width+2 and y+h < height+2
        and area >= max(2, .015 * width * height)
    ], key=lambda point: point[1])


class NativeNumericReader:
    """Connected stroke OCR using extracted WarHelios numeric font shapes.

    Reject the entire field when any substantial stroke is unrecognized. Decimal
    punctuation is accepted only on the baseline between two recognized digits.
    """
    def __init__(self, font_root=GUI / "fonts", small_fonts=False, distance_fonts=False):
        self.templates = []
        font_root = Path(font_root)
        self.unit_templates = []
        self.small_fonts = small_fonts
        self.distance_fonts = distance_fonts
        # 8 px is measured on the live nameplate range caption ("7.7 公里"),
        # which is one size below the previous floor and therefore unreadable.
        sizes = (18, 20, 22, 24) + ((8, 9, 10, 11, 12, 13, 14, 15, 16)
                                  if small_fonts or distance_fonts else ())
        for name in ("Warhelios_Regular.ttf", "Warhelios_Bold.ttf"):
            path = font_root / name
            if not path.exists():
                continue
            for size in sizes:
                font = ImageFont.truetype(str(path), size)
                for char in "0123456789skm":
                    if size < 18 and not small_fonts and char not in ('1', '4', '5'):
                        continue
                    canvas = Image.new("L", (45, 45))
                    ImageDraw.Draw(canvas).text((3, -2), char, font=font, fill=255)
                    # Thin ones, open-top fours and compact fives need native
                    # small rasterizations; retain other digit competitors.
                    thresholds = (110, 160, 200) if (small_fonts or distance_fonts) and char == '4' else (110,)
                    for threshold in thresholds:
                        mask = _trim(np.asarray(canvas) > threshold)
                        if mask is not None:
                            target = self.templates if char.isdigit() else self.unit_templates
                            target.append((char, _normalized(mask), mask.shape[1]/mask.shape[0]))
        for size in ("medium", "big"):
            meta = font_root / size / "WarHeliosCond_Textile.json"
            png = meta.with_suffix(".png")
            if not meta.exists() or not png.exists():
                continue
            frames = json.loads(meta.read_text(encoding="utf-8-sig"))["frames"]
            atlas = np.asarray(Image.open(png).convert("RGBA"))
            for char in "0123456789":
                r = frames[char]["frame"]; x, y, w, h = (r[k] for k in ("x", "y", "w", "h"))
                mask = _trim(atlas[y:y+h, x:x+w, 3] > 110)
                if mask is not None:
                    self.templates.append((char, _normalized(mask), mask.shape[1]/mask.shape[0]))
        self.template_pixels = {id(template): np.count_nonzero(template)
                                for _, template, _ in self.templates + self.unit_templates}
        self.template_holes = {id(template): _glyph_holes(template)
                               for _, template, _ in self.templates + self.unit_templates}

    def read(self, rgb, bbox, decimal=False, maximum=999, unit=None, min_height=8,
             match_holes=False, any_polarity=False):
        x, y, w, h = bbox
        patch = rgb[y:y+h, x:x+w]
        result = evidence(bbox=list(bbox), value=None, text=None)
        if not self.templates:
            result["raw"] = {"reason": "native_fonts_missing"}; return result
        if patch.ndim != 3 or patch.shape[0] < 3 or patch.shape[1] < 3:
            # A caller can hand in a box that falls outside the frame (a plate
            # near the edge, or an upscaled-band crop). OpenCV raises on an
            # empty source in GaussianBlur, which used to kill the aiming run.
            result["raw"] = {"reason": "bbox_outside_frame",
                             "shape": list(getattr(patch, 'shape', ()))}
            return result
        p = patch.astype(np.int16)
        # HUD white digits can be dimmed during reload (actual capture peaks at 166).
        gray = np.minimum(np.minimum(p[..., 0], p[..., 1]), p[..., 2])
        spread = np.maximum(np.maximum(p[..., 0], p[..., 1]), p[..., 2]) - gray
        contrast = gray - cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 3)
        if any_polarity:
            # The reticle readout is drawn as grey digits over bright water,
            # where local contrast is negative; the bright-on-dark rule finds
            # nothing there (measured: "6.4 s / 12.1 km" visible, reader said
            # unknown). Accept either sign, and only for this readout.
            mask = ((gray > 60) & (spread < 60)
                    & (np.abs(contrast) > 12)).astype(np.uint8)
        else:
            mask = ((gray > 95) & (spread < 42) & (contrast > 12)).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if decimal and self.distance_fonts:
            # Antialiasing can diagonally join a baseline decimal to a digit.
            # Split a complete digit plus a baseline dot on either side only
            # when another full-height digit brackets the dot. Keep all ink.
            extra = []
            for label in range(1, n):
                xx, yy, ww, hh, area = map(int, stats[label])
                if hh < min_height:
                    continue
                region = labels[yy:yy+hh, xx:xx+ww]
                count4, parts, boxes, _ = cv2.connectedComponentsWithStats(
                    (region == label).astype(np.uint8), connectivity=4)
                if count4 != 3:
                    continue
                ordered = sorted(range(1, count4), key=lambda k: boxes[k][4], reverse=True)
                body, dot = (list(map(int, boxes[k])) for k in ordered)
                bx, by, bw, bh, ba = body
                dx, dy, dw, dh, da = dot
                baseline = yy+by+bh
                right_dot = dx >= bx+bw and any(
                    sx >= xx+dx+dw and sh >= min_height and abs(sy+sh-baseline) <= 3
                    for sx, sy, sw, sh, sa in stats[1:])
                left_dot = dx+dw <= bx and any(
                    sx+sw <= xx+dx and sh >= min_height and abs(sy+sh-baseline) <= 3
                    for sx, sy, sw, sh, sa in stats[1:])
                if not (bh >= min_height and dw <= 4 and dh <= 4 and da >= 3 and
                        by+bh-4 <= dy+dh <= by+bh+2 and (right_dot or left_dot)):
                    continue
                region[parts == ordered[1]] = n + len(extra)
                stats[label] = [xx+bx, yy+by, bw, bh, ba]
                extra.append([xx+dx, yy+dy, dw, dh, da])
            if extra:
                stats = np.concatenate((stats, np.asarray(extra, dtype=stats.dtype)))
        components = sorted([(int(xx), int(yy), int(ww), int(hh), int(area), k)
                             for k, (xx, yy, ww, hh, area) in enumerate(stats[1:], 1)
                             if area >= 3], key=lambda r: r[0])
        raw = {"foreground_pixels": int(mask.sum()), "components": [], "method": "native_font_stroke_ocr"}
        result["raw"] = raw
        if not components:
            raw["reason"] = "no_digits"; return result
        digit_heights = [r[3] for r in components if r[3] >= min_height]
        if not digit_heights:
            raw["reason"] = "no_digit_height"; return result
        base = max(yy+hh for _, yy, _, hh, _, _ in components if hh >= min_height)
        if any(abs(yy+hh-base) > 3 for _, yy, _, hh, _, _ in components if hh >= min_height):
            raw["reason"] = "inconsistent_digit_baseline"; return result
        chars, scores = [], []
        for i, (xx, yy, ww, hh, area, label) in enumerate(components):
            cb = [x+xx, y+yy, ww, hh]
            if hh <= 4 and ww <= 4 and decimal and 0 < i < len(components)-1 and base-4 <= yy+hh <= base+2:
                chars.append("."); scores.append(.95)
                raw["components"].append({"bbox": cb, "char": ".", "score": .95}); continue
            if hh < min_height or ww < 2 or hh > 27:
                raw["reason"] = "malformed_stroke"; return result
            glyph = labels[yy:yy+hh, xx:xx+ww] == label
            normalized = _normalized(glyph)
            foreground_pixels = np.count_nonzero(normalized)
            holes = _glyph_holes(normalized) if match_holes else None
            candidates = {}
            for char, template, aspect in self.templates + (self.unit_templates if unit else []):
                if holes is not None:
                    expected = self.template_holes[id(template)]
                    if (len(holes) != len(expected) or
                            any(max(abs(a-b) for a, b in zip(p, q)) > .10
                                for p, q in zip(holes, expected))):
                        continue
                dice = 2*np.count_nonzero(normalized & template)/(foreground_pixels+self.template_pixels[id(template)])
                score = float(dice - .18*abs(np.log((ww/hh)/aspect)))
                candidates[char] = max(candidates.get(char, 0), score)
            if not candidates:
                raw['reason'] = 'glyph_topology_unmatched'; return result
            ranking = sorted(candidates.items(), key=lambda z: z[1], reverse=True)
            char, score = ranking[0]
            margin = score - (ranking[1][1] if len(ranking) > 1 else 0)
            raw["components"].append({"bbox": cb, "char": char, "score": score, "margin": margin})
            if holes is not None:
                raw['components'][-1]['holes'] = [list(point) for point in holes]
            if score < (.72 if char in "skm" else .74) or margin < .035:
                raw["reason"] = "ambiguous_glyph"; return result
            chars.append(char); scores.append(score)
        text = "".join(chars); raw["candidate_text"] = text
        pattern = r"\d{1,3}(?:\.\d{1,2})?" if decimal else r"\d{1,3}"
        if unit and text.endswith(unit):
            text = text[:-len(unit)]
            raw["unit"] = unit
        if re.fullmatch(pattern, text) is None:
            raw["reason"] = "malformed_number"; return result
        value = float(text)
        if not 0 <= value <= maximum:
            raw["reason"] = "number_out_of_range"; return result
        result.update(status="observed", value=value, text=text, confidence=min(scores))
        return result


def _load(path):
    return np.asarray(Image.open(path).convert("RGBA")) if path.exists() else None


def _validate_frame(rgb):
    """Any HxWx3 uint8 capture; every HUD coordinate comes from the projection.

    The reference-capture size is no longer a requirement: a client that
    renders 1924x1058 is read with its own boxes, taken from
    ``hud_projection``. What remains required is the pixel format, because the
    readers index the array directly, and a plausible minimum size.
    """
    if (not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3
            or rgb.dtype != np.uint8):
        raise ValueError("FireHudReader requires an HxWx3 uint8 RGB frame")
    if min(rgb.shape[0], rgb.shape[1]) < hud_projection.PANEL_MIN_SIDE:
        raise ValueError(f"frame {rgb.shape[:2]} is too small to hold a HUD")
    return rgb


# A measured projection on a reference-size client lands within a per-cent or
# two of the reference scale (the 2560x1600 capture measures 0.992..1.001). That
# is rasterisation and letterbox measurement slack, not a smaller HUD: sampling
# sprites or glyph templates at 0.999 of their native size would blur them for
# no gain. Anything beyond this band is a genuinely different HUD size.
REFERENCE_SCALE_TOLERANCE = 0.015


def _effective(scale):
    """HUD scale to sample at: 1.0 for anything near the reference size."""
    return 1.0 if abs(scale - 1.0) <= REFERENCE_SCALE_TOLERANCE else scale


def _scale_px(value, scale):
    """Reference pixels -> frame pixels, exactly unchanged at the reference size."""
    scale = _effective(scale)
    return value if scale == 1.0 else max(1, int(round(value * scale)))


def _scale_area(value, x_scale, y_scale):
    """Reference pixels^2 -> frame pixels^2, exactly unchanged at the reference size."""
    x_scale, y_scale = _effective(x_scale), _effective(y_scale)
    if x_scale == 1.0 and y_scale == 1.0:
        return value
    return max(1, int(round(value * x_scale * y_scale)))


def _green_selection(rgb, min_area=500, max_area=60000, scale_x=1.0, scale_y=1.0):
    """Bounding box of the green frame the client draws round the chosen slot.

    ``min_area``/``max_area`` and the plate-sized extent are reference pixels;
    they are rescaled for a client that draws the HUD smaller.
    """
    import numpy as _np
    a = _np.asarray(rgb).astype(_np.int16)
    if a.ndim != 3:
        return None
    min_area = _scale_area(min_area, scale_x, scale_y)
    max_area = _scale_area(max_area, scale_x, scale_y)
    green = ((a[..., 1] > a[..., 0] + 25) & (a[..., 1] > a[..., 2] + 25)
             & (a[..., 1] > 90)).astype(_np.uint8)
    # A thin frame: close along each axis separately so the border survives
    # without merging it into the neighbouring icons.
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE,
                             _np.ones((_scale_px(3, scale_y), _scale_px(9, scale_x)),
                                      _np.uint8))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE,
                             _np.ones((_scale_px(9, scale_y), _scale_px(3, scale_x)),
                                      _np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(green)
    low = _scale_px(40, min(scale_x, scale_y))
    high = _scale_px(200, max(scale_x, scale_y))
    best = None
    for bx, by, bw, bh, area in stats[1:]:
        if not (min_area <= area <= max_area):
            continue
        if not (low <= bw <= high and low <= bh <= high):
            continue
        if not .5 <= bw / bh <= 2.2:
            continue
        if best is None or area > best[0]:
            best = (area, int(bx), int(by), int(bw), int(bh))
    if best is None:
        return None
    return best[1:]


def _hollow_rect(component, min_side):
    """The rectangle a hollow selection outline draws, read from its borders.

    A component's bounding box is not the frame: the client also fills the
    selected slot's number plate with the same green, and that plate hangs
    below the bottom border (measured on state/s_g.png: component bbox is
    84x108 while the frame is 84x84 and the plate sits underneath). The four
    borders are the only rows and columns with near-total coverage, so they
    are read as coverage peaks instead of an extent.
    """
    rows = component.sum(axis=1)
    cols = component.sum(axis=0)
    width = component.shape[1]
    solid_rows = [y for y in range(component.shape[0]) if rows[y] >= .55 * width]
    if len(solid_rows) < 2:
        return None
    top, bottom = solid_rows[0], solid_rows[-1]
    height = bottom - top
    if height < min_side:
        return None
    solid_cols = [x for x in range(width) if cols[x] >= .55 * height]
    if len(solid_cols) < 2:
        return None
    left, right = solid_cols[0], solid_cols[-1]
    inside = component[top + 4:bottom - 4, left + 4:right - 4]
    return {"rect": (int(left), int(top), int(right - left + 1), int(height + 1)),
            "interior_fill": float(inside.mean()) if inside.size else 1.0,
            "border_rows": [int(top), int(bottom)],
            "border_columns": [int(left), int(right)]}


def _selection_frame(rgb, band, scale):
    """The selected slot's hollow rectangle inside the bottom HUD band.

    The client marks the chosen slot with a thin green outline (a yellow one is
    accepted too). Searched structurally -- hue, hollow interior, near-square
    aspect, plate-sized extent -- so no slot coordinate is assumed. Returns the
    evidence marks or None.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, sat, val = (hsv[..., i].astype(int) for i in range(3))
    hues = (("green", (hue >= 55) & (hue <= 105) & (sat >= 120) & (val >= 80)),
            ("yellow", (hue >= 15) & (hue <= 40) & (sat >= 140) & (val >= 150)))
    x0, y0, w, h = band
    best = None
    for name, mask in hues:
        local = np.zeros(mask.shape, np.uint8)
        local[y0:y0 + h, x0:x0 + w] = mask[y0:y0 + h, x0:x0 + w]
        count, labels, stats, _ = cv2.connectedComponentsWithStats(local, 8)
        for index in range(1, count):
            bx, by, bw, bh, area = (int(v) for v in stats[index])
            if not (45 * scale <= bw <= 190 * scale and 55 * scale <= bh <= 200 * scale):
                continue
            if not .55 <= bw / float(bh) <= 1.35:
                continue
            if area / float(bw * bh) > .45:
                continue
            rect = _hollow_rect(labels[by:by + bh, bx:bx + bw] == index, 20 * scale)
            if rect is None:
                continue
            fw, fh = rect["rect"][2], rect["rect"][3]
            if not .6 <= fw / float(fh) <= 1.35 or rect["interior_fill"] > .3:
                continue
            marks = {"hue": name, "bbox": [bx + rect["rect"][0], by + rect["rect"][1], fw, fh],
                     "interior_fill": round(rect["interior_fill"], 3),
                     "border_rows": [by + v for v in rect["border_rows"]],
                     "border_columns": [bx + v for v in rect["border_columns"]],
                     "outline_pixels": int(area)}
            if best is None or fw * fh > best["bbox"][2] * best["bbox"][3]:
                best = marks
    return best


def _slot_plate_columns(rgb, band, row_band, scale):
    """Slot number plates on the row under the selected frame, as x centres.

    Plates are uniform light-grey squares about 25 px on a 290 px band; the
    selected slot's plate is instead filled with the highlight colour, and it
    is read too so that its column is numbered even when the selection frame
    is elsewhere. Their own coordinates differ per hull, so the row is
    anchored on the frame the client drew rather than on a fixed y: each plate
    is a square blob inside ``row_band`` whose extent and fill agree with a
    plate rather than with hull plating or water.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, sat, val = (hsv[..., i].astype(int) for i in range(3))
    x0, y0, w, h = band
    py0, py1 = row_band
    low, high = (v * scale for v in SLOT_PLATE_SIDE)
    columns = []
    plate = (sat <= 115) & (val >= 95) & (val <= 170)
    mask = np.zeros(sat.shape, np.uint8)
    mask[py0:py1, x0:x0 + w] = plate[py0:py1, x0:x0 + w]
    count, labels, stats, centres = cv2.connectedComponentsWithStats(mask, 8)
    for index in range(1, count):
        bx, by, bw, bh, area = (int(v) for v in stats[index])
        if not (low <= bw <= high and low <= bh <= high * 1.1):
            continue
        if area / float(bw * bh) < .45:
            continue
        if not py0 <= centres[index][1] <= py1:
            continue
        columns.append({"center_x": float(centres[index][0]),
                        "bbox": [bx, by, bw, bh], "source": "number_plate"})
    # The highlighted slot's plate is filled with the selection colour and its
    # top edge touches the frame's bottom border, so a component would be the
    # whole 84 px frame width. A column profile separates them: only the plate
    # is tall, while the border contributes the width of the frame at 3 rows.
    highlight = ((hue >= 55) & (hue <= 105) & (sat >= 140) & (val >= 120))
    roi = highlight[py0:py1, x0:x0 + w]
    coverage = roi.sum(axis=0).astype(np.int32)
    tall = (coverage >= max(6, 12 * scale)).astype(np.uint8)
    # The plate's own glyph interrupts its fill (measured: the "1" splits the
    # run into 995..1005 and 1008..1018), so close a glyph-wide gap before the
    # run is measured. Neighbouring slots are a pitch apart and stay separate.
    gap = max(3, int(round(7 * scale)))
    tall = cv2.morphologyEx(tall[None, :], cv2.MORPH_CLOSE,
                            np.ones((1, gap), np.uint8))[0]
    run = None
    for x in range(len(tall) + 1):
        inside = x < len(tall) and tall[x]
        if inside and run is None:
            run = x
        elif not inside and run is not None:
            width = x - run
            if low <= width <= high:
                weights = coverage[run:x].astype(np.float64)
                centre = (x0 + run + float((weights * np.arange(width)).sum()
                                           / max(1.0, weights.sum())))
                columns.append({"center_x": centre,
                                "bbox": [x0 + run, py0, width, py1 - py0],
                                "source": "highlighted_plate"})
            run = None
    return columns


def slot_selection(rgb, projection=None):
    """Which slot the client's selection frame is on, and whether it is a gun.

    The bottom HUD numbers its slots; the client highlights the chosen one with
    a hollow rectangle. Slot positions differ per hull and per client, so both
    the frame and the slot columns are found in the frame itself: the frame as
    a hollow green/yellow rectangle inside the ammo bar band, the columns as
    the number plates on the row beneath it. The frame centre is then ranked
    against those columns left to right; the first two ordinals hold the
    shell types, later ones hold torpedoes, depth charges and consumables.

    Returns ``selected_slot_index`` (1-based ordinal, or None),
    ``main_battery`` (True/False/None) plus ``slot_columns``, ``frame_marks``
    and the reason the answer is what it is.
    """
    result = {"status": "no_selection", "selected_slot_index": None, "main_battery": None,
              "confidence": 0.0, "slot_columns": [], "frame_marks": None,
              "reason": "no_selection_frame"}
    if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] < 3:
        return result
    try:
        projection = projection if projection is not None else hud_projection.resolve(rgb)
        band = projection.from_ref(REFERENCE_SLOT_BAND)
    except (hud_projection.MappingError, ValueError):
        result["reason"] = "slot_band_unavailable"      # an unmeasurable HUD
        return result
    x0, y0 = max(0, band[0]), max(0, band[1])
    w = min(band[2], rgb.shape[1] - x0)
    h = min(band[3], rgb.shape[0] - y0)
    if w <= 0 or h <= 0:
        result["reason"] = "slot_band_unavailable"
        return result
    band = (x0, y0, w, h)
    scale = h / SLOT_BAND_REF_HEIGHT
    frame = _selection_frame(rgb, band, scale)
    if frame is None:
        return result
    result["frame_marks"] = frame
    fx, fy, fw, fh = frame["bbox"]
    # The number-plate row belongs to this frame: plates begin at its bottom
    # border and sit below it (measured: frame bottom 1467, plates 1467..1492).
    # The band stops just under the plates so a consumable icon that reaches
    # down to the row (the T slot's green cross ends at 1463) is not a plate.
    row_band = (int(max(y0, fy + fh - max(3, .04 * fh))), int(min(y0 + h, fy + fh + .4 * fh)))
    columns = _slot_plate_columns(rgb, band, row_band, scale)
    # A client may leave the selected plate grey while drawing the frame over
    # it; that yields two detections of one slot. Slot columns are a pitch
    # apart, so anything closer than a plate is the same column.
    columns.sort(key=lambda c: c["center_x"])
    merged = []
    for column in columns:
        if merged and abs(column["center_x"] - merged[-1]["center_x"]) < 12 * scale:
            keep = (column if merged[-1]["source"] != "highlighted_plate"
                    else merged[-1])
            merged[-1] = keep
            continue
        merged.append(column)
    columns = merged
    centre = fx + fw / 2.0
    # The selected slot's own plate carries the highlight fill, not the plate
    # grey, so when it is missing from the plate row its column is the frame
    # centre itself. When the client leaves that plate grey, the frame sits on
    # the plate and the two must not be counted as separate slots.
    gaps = np.diff([c["center_x"] for c in columns])
    pitch = float(np.median(gaps)) if len(gaps) else 0.0
    nearest = min(columns, key=lambda c: abs(c["center_x"] - centre)) if columns else None
    if nearest is not None and abs(nearest["center_x"] - centre) <= max(8 * scale, .35 * pitch):
        selected = nearest
        selected["selection_frame_bbox"] = [int(fx), int(fy), int(fw), int(fh)]
    else:
        selected = {"center_x": float(centre),
                    "bbox": [int(fx), int(fy), int(fw), int(fh)],
                    "source": "selection_frame"}
        columns.append(selected)
        columns.sort(key=lambda c: c["center_x"])
        gaps = np.diff([c["center_x"] for c in columns])
        pitch = float(np.median(gaps)) if len(gaps) else 0.0
    # A plate can be lost to a hull behind it (measured: the R plate merges
    # into the ship superstructure). One missing column shows up as a gap of a
    # clean multiple of the row's own pitch; restore it so ordinals stay true.
    restored = []
    for left, right, gap in zip(columns, columns[1:], gaps):
        if pitch > 0 and 1.5 <= gap / pitch <= 2.5:
            restored.append({"center_x": (left["center_x"] + right["center_x"]) / 2.0,
                             "bbox": None, "source": "gap_inferred"})
    columns.extend(restored)
    columns.sort(key=lambda c: c["center_x"])
    for ordinal, column in enumerate(columns, 1):
        column["index"] = ordinal
    result["slot_columns"] = columns
    selected = [c for c in columns
                if c["source"] == "selection_frame"
                or c.get("selection_frame_bbox") is not None][0]
    result["selected_slot_index"] = selected["index"]
    result["confidence"] = round(.6 + .3 * (1.0 - frame["interior_fill"]), 3)
    if len(columns) < 2:
        # A single column cannot be ranked; say so instead of guessing a gun.
        result.update(status="unresolved", selected_slot_index=None,
                      reason="slot_columns_unresolved")
        return result
    result["status"] = "observed"
    result["main_battery"] = selected["index"] <= MAIN_BATTERY_COLUMNS
    result["reason"] = ("shell_slot_ordinal" if result["main_battery"]
                        else "ordinal_beyond_shell_slots")
    return result


def edge_rows(rgb, y, x0, x1, height):
    """Count rows in one vertical band that carry the native selection outline.

    The band may extend outside the sprite window, so coordinates are clamped to
    the frame. A row counts once no matter how many outline pixels it holds.
    """
    x0, x1 = max(0, int(x0)), min(rgb.shape[1], int(x1))
    band = rgb[y:y+height, x0:x1].astype(np.int16)
    if band.size == 0:
        return 0
    green = ((band[..., 1] > 125) & (band[..., 1]-band[..., 0] > 40) & (band[..., 2] > 60))
    return int(np.count_nonzero(green.any(axis=1)))


def scope_ruler_center(rgb, rows=None, min_ticks=20, min_span=800, centre_tol=6,
                       projection=None):
    """Centre of the binocular ruler, which is drawn symmetrically on screen.

    Measured on a live scoped frame: 80 bright stroke columns spanning
    705..1853 with midpoint 1279 against a frame centre of 1280. The
    third-person shell-fall ruler is asymmetric (midpoints 1342 and 1530 on two
    live frames), so symmetry is the discriminator, not brightness. The zoom
    level is never inferred from this. ``rows`` is a reference-space band and
    is projected before use unless the caller supplies frame rows directly.
    """
    a = np.asarray(rgb)
    if rows is None:
        projection = projection or hud_projection.resolve(a)
        y0, y1 = projection.project_span_y(*REFERENCE_SCOPE_RULER_ROWS)
        rows = (y0, max(y1, y0 + 8))
    if projection is not None:
        # The ruler is a reference-size element: its tick span and the
        # symmetry tolerance are reference pixels too.
        x_scale = _effective(projection.x_scale)
        min_span = min_span if x_scale == 1.0 else min_span * x_scale
        centre_tol = (centre_tol if x_scale == 1.0
                      else max(2, centre_tol * x_scale))
    if a.ndim != 3 or a.shape[0] < rows[1]:
        return None
    gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
    band = gray[rows[0]:rows[1], :]
    if band.size == 0:
        return None
    threshold = float(np.percentile(band, 99.5))
    if threshold < 80:
        return None              # a blank or near-black band draws no ticks
    strong = band >= threshold
    columns = np.where(strong.sum(axis=0) >= 3)[0]
    if len(columns) < min_ticks or len(columns) > .6 * band.shape[1]:
        return None              # every column "bright" is a flat surface
    span = float(columns[-1] - columns[0])
    if span < min_span:
        return None
    centre_x = float(columns[0] + columns[-1]) / 2.0
    width = gray.shape[1]
    if abs(centre_x - width / 2.0) > centre_tol:
        return None
    rows_hit = np.bincount(np.where(strong.sum(axis=1) >= max(3, len(columns) // 4))[0],
                           minlength=band.shape[0])
    centre_y = rows[0] + float(rows_hit.argmax())
    return dict(status='observed', center=[centre_x, centre_y], ticks=int(len(columns)),
                span_px=span, centre_offset_px=round(centre_x - width / 2.0, 2))


def green_reticle(frame, centre=None, window=140, min_area=18, max_area=400,
                  projection=None):
    """Find the aim reticle: a small green ring near the screen centre.

    The native caret template misses on some live frames, so this measures the
    green circle the game draws at the middle of the screen instead of assuming
    a fixed pixel. The default centre is the *reference* screen centre and is
    projected onto this frame. Returns None when nothing matches the colour and
    geometry.
    """
    if centre is None:
        projection = projection or hud_projection.resolve(frame)
        if (_effective(projection.x_scale) == 1.0
                and _effective(projection.y_scale) == 1.0):
            # At reference size the frame's own coordinates are reference
            # coordinates, and the ring is drawn at the screen centre. Using a
            # measured anchor that is 1-2 px off would move the search window
            # for no gain, and can flip which of two symmetric rings is nearest.
            centre = REFERENCE_RETICLE_CENTRE
        else:
            centre = projection.project_point(REFERENCE_RETICLE_CENTRE)
    if projection is not None:
        # Window, ring area and ring extent are reference pixels: the ring the
        # client draws at another resolution is the same element, smaller.
        scale_x, scale_y = _effective(projection.x_scale), _effective(projection.y_scale)
        if (scale_x, scale_y) != (1.0, 1.0):
            window = _scale_px(window, max(scale_x, scale_y))
            min_area = _scale_area(min_area, scale_x, scale_y)
            max_area = max(min_area, _scale_area(max_area, scale_x, scale_y))
    x0, y0 = max(0, centre[0]-window), max(0, centre[1]-window)
    x1, y1 = min(frame.shape[1], centre[0]+window), min(frame.shape[0], centre[1]+window)
    if x1 - x0 < 40 or y1 - y0 < 40:
        return None
    ring_side = (8, 40) if projection is None else (
        _scale_px(8, min(projection.x_scale, projection.y_scale)),
        _scale_px(40, max(projection.x_scale, projection.y_scale)))
    patch = frame[y0:y1, x0:x1]
    r, g, b = (patch[..., i].astype(np.int16) for i in range(3))
    mask = ((g > 120) & (g - r > 30) & (g - b > 20)).astype(np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    best = None
    for index in range(1, count):
        area = int(stats[index][4])
        width, height = int(stats[index][2]), int(stats[index][3])
        if (not min_area <= area <= max_area or not ring_side[0] <= width <= ring_side[1]
                or not ring_side[0] <= height <= ring_side[1]):
            continue
        if not 0.6 <= width / height <= 1.7:
            continue
        cx, cy = float(centroids[index][0]) + x0, float(centroids[index][1]) + y0
        distance = math.hypot(cx - centre[0], cy - centre[1])
        if best is None or distance < best[0]:
            best = (distance, cx, cy, width, height, area)
    if best is None:
        return None
    _, cx, cy, width, height, area = best
    return dict(centre=[cx, cy], bbox=[int(cx-width/2), int(cy-height/2), width, height],
                area=area, source='green_ring')


def _gray_signal(rgb, roi):
    x, y, w, h = roi
    signal = cv2.cvtColor(rgb[y:y+h, x:x+w].astype(np.float32), cv2.COLOR_RGB2GRAY)
    signal -= cv2.GaussianBlur(signal, (0, 0), 3)
    return signal


def _template_variants(rgba, scales):
    """Build the immutable pieces of a template match once per reader."""
    if rgba is None:
        return ()
    color = rgba[..., :3].astype(np.float32)
    alpha = rgba[..., 3].astype(np.float32)/255
    spread = np.ptp(color, axis=2)
    colored = float((spread*alpha).max()) > 40
    if colored:
        support = alpha > .3
        means = color[support].mean(axis=0) if support.any() else np.zeros(3)
        channel = int(np.argmax(means))
        others = [k for k in range(3) if k != channel]
        shape = np.maximum(0, color[..., channel]-color[..., others].mean(axis=2))*alpha
    else:
        shape = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)*alpha
    variants = []
    for scale in scales:
        shape_scaled = cv2.resize(shape, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        support_scaled = cv2.resize((alpha > .3).astype(np.uint8),
                                    (shape_scaled.shape[1], shape_scaled.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
        variants.append((scale, np.pad(shape_scaled, 2), np.pad(support_scaled, 2), colored))
    return tuple(variants)


def _match(rgb, rgba, roi, threshold=.80, scales=(1.0,), gray_signal=None, template_variants=None):
    if rgba is None:
        return []
    x, y, w, h = roi
    if template_variants is None:
        template_variants = _template_variants(rgba, scales)
    colored = template_variants[0][3] if template_variants else False
    if colored:
        color = rgba[..., :3].astype(np.float32)
        alpha = rgba[..., 3].astype(np.float32)/255
        spread = np.ptp(color, axis=2)
        support = alpha > .3
        means = color[support].mean(axis=0) if support.any() else np.zeros(3)
        channel = int(np.argmax(means))
        others = [k for k in range(3) if k != channel]
        src = rgb[y:y+h, x:x+w].astype(np.float32)
        # Preserve the sprite's dominant color, avoiding similarly bright labels.
        signal = np.maximum(0, src[..., channel] - src[..., others].mean(axis=2))
    else:
        signal = _gray_signal(rgb, roi) if gray_signal is None else gray_signal
    hits = []
    for scale, shape_scaled, support_scaled, variant_colored in template_variants:
        th, tw = shape_scaled.shape
        if th > h or tw > w or shape_scaled.std() < 1:
            continue
        # Selection outlines can occupy transparent sprite pixels. They are not
        # part of the icon and must not change its shape agreement.
        scores = cv2.matchTemplate(signal, shape_scaled, cv2.TM_CCOEFF_NORMED,
                                   mask=support_scaled if variant_colored else None)
        scores[~np.isfinite(scores)] = -1
        peaks = (scores >= threshold) & (scores == cv2.dilate(scores, np.ones((7, 7), np.uint8)))
        ys, xs = np.where(peaks)
        for yy, xx in zip(ys, xs):
            patch = signal[yy:yy+th, xx:xx+tw]; on = shape_scaled > shape_scaled.max()*.4
            if float(patch[on].mean() - patch[~on].mean()) < 16:
                continue
            box = [x+int(xx)+2, y+int(yy)+2, tw-4, th-4]
            hits.append(evidence("observed", box, scores[yy, xx], raw={"method": "shape_template", "scale": scale}))
    picks = []
    for hit in sorted(hits, key=lambda z: z["confidence"], reverse=True):
        xx, yy, ww, hh = hit["bbox"]
        if not any(abs(xx-p["bbox"][0]) < ww/2 and abs(yy-p["bbox"][1]) < hh/2 for p in picks):
            picks.append(hit)
    return picks


class FireHudReader:
    def __init__(self, gui_root=GUI, reference_root=REFERENCES, projection=None):
        gui_root, reference_root = Path(gui_root), Path(reference_root)
        self.projection = projection
        self._resolved = None
        self.numeric = NativeNumericReader(gui_root / "fonts")
        # The reticle readout ("3.4 s / 7.7 km") is set in the small distance
        # font with a decimal point; the default reader only carries 18-24 px
        # glyphs and returns unknown on it (measured on a live pre-fire frame
        # whose text is plainly "7.7 km"). Keep the default reader untouched so
        # ammo/mount reads are unchanged and give the anchored numbers their own.
        self.numeric_distance = NativeNumericReader(gui_root / "fonts", distance_fonts=True)
        self.aim_units = {unit: _load(reference_root/f'aim-unit-{unit}-20261019.png')
                          for unit in ('s','km')}
        # Resize alpha before thresholding: native HUD scaling changes the
        # rasterized strokes, which normalization of a binary mask cannot undo.
        self._unit_templates = {}
        for unit, reference in self.aim_units.items():
            variants = []
            if reference is not None:
                for scale in (1.0, 1.25, 4/3, 1.4, 1.5):
                    alpha = cv2.resize(reference[..., 3], None, fx=scale, fy=scale,
                                       interpolation=cv2.INTER_LINEAR)
                    ink = _trim(alpha > 110)
                    if ink is not None:
                        variants.append((_normalized(ink), ink.shape[1]/ink.shape[0]))
            self._unit_templates[unit] = variants
        self.icons = {k: _load(reference_root / f"{k}.png") for k in ("he", "ap", "torpedo", "aim_caret")}
        self.icons['green_aim_ring'] = _load(reference_root / 'green-aim-ring-20261018.png')
        self.icons['amber_aim_ring'] = _load(reference_root / 'amber-aim-ring-20261018.png')
        indicators = gui_root / "battle_hud/crosshair/indicators"
        self.anchors = {k: _load(indicators / f"crosshair_indicator_{v}.png")
                        for k, v in (("time", "bullettime"), ("distance", "distance"))}
        self._match_templates = {
            **{k: _template_variants(v, (1.0,)) for k, v in self.icons.items()},
            "time": _template_variants(self.anchors["time"], (1.0, 1.25, 4/3, 1.4, 1.5)),
            "distance": _template_variants(self.anchors["distance"], (1.0, 1.25, 4/3, 1.4, 1.5)),
        }
        # Native sprites are reference-size. A client that renders a smaller
        # surface draws them smaller, so a frame whose projection is not the
        # reference needs the same shape sampled at that scale; the reference
        # set above is used unchanged at scale 1.
        self._scaled_templates = {}
        self.depth_charge = _load(reference_root / "depth-charge-selected.png")
        self._depth_charge_templates = _template_variants(self.depth_charge, (1.0,))
        # Cooldown text/fill hides the center and bottom, but not both upper
        # outer wings. Keep their native spacing; neither wing alone is proof.
        self._depth_charge_wings = ()
        if self.depth_charge is not None:
            gray = cv2.cvtColor(self.depth_charge[..., :3].astype(np.float32), cv2.COLOR_RGB2GRAY)
            gray -= cv2.GaussianBlur(gray, (0, 0), 3)
            self._depth_charge_wings = tuple(
                (x, gray[:24, x:x+10].copy()) for x in (0, gray.shape[1]-10))

    def resolve(self, rgb):
        """Projection for this frame: supplied, measured, or declared.

        A supplied projection wins (a caller may have measured once for a
        burst of frames). Otherwise the anchors are measured in this frame; a
        frame that draws no HUD at all falls back to the declared content
        projection, which is recorded in the evidence, never silent. One frame
        is read by several HUD helpers, so the result is memoised against the
        frame's object id, its shape and a cheap fingerprint of the array. The
        shape is part of the key because Python reuses object ids: a caller
        that reads frames of different sizes would otherwise be handed the
        previous frame's projection.
        """
        if self.projection is not None:
            return self.projection
        fingerprint = (rgb[64, 64].tobytes() + rgb[-64, -64].tobytes()
                       + rgb[rgb.shape[0] // 2, rgb.shape[1] // 2].tobytes())
        key = (id(rgb), rgb.shape, fingerprint)
        cached = self._resolved
        if cached is not None and cached[0] == key:
            return cached[1]
        projection = hud_projection.resolve(rgb)
        self._resolved = (key, projection)
        return projection

    def reload_regions(self, rgb):
        """Reference turret-number rectangles projected onto this frame."""
        return self.resolve(rgb).project_rects(RELOAD_ROIS)

    def scaled_match_templates(self, key, projection):
        """Template variants for this frame's HUD scale.

        At the reference projection the prebuilt native set is returned
        unchanged, so a reference frame matches exactly as before. Off the
        reference the same shapes are resampled at ``sqrt(sx*sy)`` times their
        native sizes, because the client draws the sprite smaller rather than
        the reader reading it at the wrong size.
        """
        cached = self._match_templates.get(key)
        if projection is None:
            return cached
        factor = math.sqrt(_effective(projection.x_scale) * _effective(projection.y_scale))
        if abs(factor - 1.0) < 1e-6:
            return cached
        token = (key, round(factor, 4))
        scaled = self._scaled_templates.get(token)
        if scaled is None:
            base = (1.0,) if key in self.icons else (1.0, 1.25, 4/3, 1.4, 1.5)
            template = self.icons.get(key)
            if template is None:
                template = self.anchors.get(key)
            if template is None and key == "depth_charge":
                template = self.depth_charge
                base = (1.0,)
            scales = tuple(sorted({round(factor * s, 4) for s in base}))
            scaled = _template_variants(template, scales)
            self._scaled_templates[token] = scaled
        return scaled

    def _weapon(self, rgb):
        projection = self.resolve(rgb)
        result = evidence(bbox=projection.project_rect(REFERENCE_SELECTOR_BBOX),
                          kind=None, ammo=None, selected=False)
        slots = []
        # 236387 has one extra consumable and shifts these three sprites left by 44 px.
        # The live client centres the bar on this hull's slot set, which moves the
        # same sprites right by 71 px (measured: HE/AP match across 57..83, and this
        # hull's bar carries no torpedo sprite at all).
        # AP-only San Martin shares the same AP/torpedo spacing at -138 px.
        # The current San Martin single-ammo frame uses a measured -28px slot.
        # Current live HUD: native HE/AP at x=979/1068;
        # the selected HE frame has both vertical edges in the +28 slot.
        # The offsets are reference-space shifts, so they are applied to the
        # reference rect and the result is projected once.
        for offset in REFERENCE_SELECTOR_OFFSETS:
            for key, base_roi in zip(("he", "ap", "torpedo"), SELECTOR_ROIS):
                roi = projection.project_rect(
                    (base_roi[0] + offset, base_roi[1], base_roi[2], base_roi[3]))
                hits = _match(rgb, self.icons[key], roi, threshold=.76,
                              template_variants=self.scaled_match_templates(key, projection))
                x, y, w, h = roi
                # The selected slot's native outline can sit a couple of pixels
                # outside this sprite window (measured on the live client: the
                # outline occupies 1009..1091 while the offset-71 slot is
                # 1011..1091). Inspect a small margin so both edges are seen.
                margin = _scale_px(4, projection.x_scale)
                edge = _scale_px(12, projection.x_scale)
                left = edge_rows(rgb, y, x-margin, x+edge, h)
                right = edge_rows(rgb, y, x+w-edge, x+w+margin, h)
                min_rows = 55 if _effective(projection.y_scale) == 1.0 else max(
                    10, int(round(55 * projection.y_scale)))
                selected = bool(left >= min_rows and right >= min_rows)
                slots.append({"ammo": key.upper(), "bbox": list(roi), "icon": hits[0] if hits else None,
                              "selected_outline": selected, "outline_edge_rows": [int(left), int(right)],
                              "layout_offset_px": offset})
        result["raw"] = {"slots": slots, "reference_source": "masked_actual_screen"}
        # The live client marks the chosen slot with a GREEN frame around the
        # sprite (reported by the operator and measured here). That frame is a
        # far stronger cue than the pale edge rows, which several layouts share.
        # Search only supported ammo-slot regions. Green sea highlights and
        # consumable icons elsewhere are not selection frames.
        left = min(s['bbox'][0] for s in slots) - _scale_px(4, projection.x_scale)
        right = max(s['bbox'][0]+s['bbox'][2] for s in slots) + _scale_px(4, projection.x_scale)
        top, bottom = projection.project_span_y(*REFERENCE_SELECTOR_ROW)
        local_green = _green_selection(rgb[top:bottom, left:right],
                                       scale_x=projection.x_scale,
                                       scale_y=projection.y_scale)
        green = ((local_green[0]+left, local_green[1]+top, *local_green[2:])
                 if local_green is not None else None)
        if green is not None:
            # A green frame is only a selection cue. Name the weapon from a
            # native icon match inside it; slot geometry alone is insufficient
            # and would mislabel the selected depth-charge panel as a torpedo.
            centre = (green[0] + green[2] / 2, green[1] + green[3] / 2)
            hits = [s for s in slots
                    if s["bbox"][0] <= centre[0] <= s["bbox"][0] + s["bbox"][2]
                    and s["bbox"][1] <= centre[1] <= s["bbox"][1] + s["bbox"][3]]
            with_icon = [s for s in hits if s["icon"]]
            if with_icon:
                strongest = max(with_icon, key=lambda s: float(s["icon"]["confidence"]))
                score = float(strongest["icon"]["confidence"])
                ammo = strongest["ammo"]
                result.update(status="observed", bbox=list(green), selected=True,
                              kind="torpedo" if ammo == "TORPEDO" else "gun",
                              ammo=ammo, confidence=min(.95, .6 + .4 * score))
                result["raw"]["reason"] = "green_selection_native_icon"
                result["raw"]["green_box"] = list(green)
                return result
            result["raw"]["reason"] = "green_selection_native_icon_unresolved"
            result["raw"]["green_box"] = list(green)
        outlined = [s for s in slots if s["selected_outline"]]
        selected = [s for s in outlined if s["icon"]]
        if len(selected) == 1:
            s = selected[0]; result.update(status="observed", bbox=s["bbox"], selected=True,
                kind="torpedo" if s["ammo"] == "TORPEDO" else "gun", ammo=s["ammo"], confidence=s["icon"]["confidence"])
        else:
            # The scope darkens the bottom HUD, so the native sprite can fail to
            # match while the selection outline is plainly drawn (measured on a
            # live scoped frame: HE slot outlined, edge rows 59/73, icon null).
            # A unique outline is independent selection evidence. Without a
            # native icon it must not be converted into an ammo or weapon kind.
            icons = [s for s in outlined if s["icon"]]
            if len(icons) == 1:
                # Several offsets can appear outlined in the live panel while
                # only one of them also carries the native ammo sprite; that
                # sprite plus its outline is the selection (measured after a
                # reload cycle: HE icon observed at the -44 layout).
                s = icons[0]
                result.update(status="observed", bbox=s["bbox"], selected=True,
                              kind="torpedo" if s["ammo"] == "TORPEDO" else "gun",
                              ammo=s["ammo"], confidence=s["icon"]["confidence"])
                result["raw"]["reason"] = "outline_with_sprite"
            elif len(outlined) == 1:
                s = outlined[0]
                result.update(status="observed", bbox=s["bbox"], selected=True,
                              confidence=0.6)
                result["raw"]["reason"] = "selected_outline_without_native_icon"
            else:
                if not self._selection_frame_weapon(rgb, projection, result):
                    result["raw"]["reason"] = "selection_unresolved"
        return result

    def _selection_frame_weapon(self, rgb, projection, result):
        """Name the weapon from the slot selection frame when no sprite matched.

        The frame is independent evidence of *which* slot is live: it is drawn
        by the client, not read off a sprite, so it survives a hull whose ammo
        sprites this reader has never matched. It says nothing about the shell
        type, so ammo stays unset and only the gun/torpedo split is claimed.
        """
        selection = slot_selection(rgb, projection)
        if selection["main_battery"] is not True:
            result["raw"]["slot_selection"] = {
                "status": selection["status"], "reason": selection["reason"],
                "selected_slot_index": selection["selected_slot_index"],
                "main_battery": selection["main_battery"]}
            return False
        result.update(status="observed", bbox=selection["frame_marks"]["bbox"],
                      selected=True, kind="gun", ammo=None,
                      confidence=selection["confidence"])
        result["raw"].update(selected_slot_index=selection["selected_slot_index"],
                             main_battery_evidence="selection_frame",
                             slot_columns=selection["slot_columns"],
                             frame_marks=selection["frame_marks"],
                             reason="selection_frame_main_battery")
        return True

    def _classify_green(self, rgb, box, threshold=.55):
        """Which native ammo sprite sits inside the green selection frame."""
        x, y, w, h = [int(v) for v in box]
        pad = 6
        x0, y0 = max(0, x - pad), max(0, y - pad)
        roi = rgb[y0:y + h + pad, x0:x + w + pad]
        if roi.size == 0:
            return None, 0.0
        signal = _gray_signal(roi, (0, 0, roi.shape[1], roi.shape[0]))
        best, best_score = None, 0.0
        for key in ("he", "ap", "torpedo"):
            template = self.icons.get(key)
            if template is None:
                continue
            kernel = _gray_signal(np.asarray(template)[..., :3], (0, 0, template.shape[1], template.shape[0])) \
                if np.asarray(template).ndim == 3 else None
            if kernel is None or kernel.shape[0] >= signal.shape[0] or kernel.shape[1] >= signal.shape[1]:
                continue
            best_key = -1.0
            for scale in (1.0, 1.2, 1.4, 1.6, 1.8, 2.0):
                scaled = cv2.resize(kernel, None, fx=scale, fy=scale,
                                    interpolation=cv2.INTER_LINEAR)
                if scaled.shape[0] >= signal.shape[0] or scaled.shape[1] >= signal.shape[1]:
                    continue
                scores = cv2.matchTemplate(signal, scaled, cv2.TM_CCOEFF_NORMED)
                scores = np.where(np.isfinite(scores), scores, -1.0)
                best_key = max(best_key, float(scores.max()))
            score = best_key
            if score > best_score:
                best, best_score = key, score
        if best is None or best_score < threshold:
            return None, best_score
        return best.upper(), best_score

    def _reload(self, rgb, weapon):
        # Fixed native layouts only: the fourth-slot view raises the three
        # mount bars by 16 px. All mounts must agree on one shared baseline.
        # The live client centres three 60 px bars at x=1252/1348/1444 on
        # y=1368 (measured on a live battle frame), which is not the 15.8
        # geometry; both are kept so either client can establish the layout.
        # Each tuple is a candidate layout in *reference* space, measured on
        # some client at reference size; the projection moves it onto this
        # frame, so a client that draws the row smaller reads its own bars.
        projection = self.resolve(rgb)
        sx, sy = _effective(projection.x_scale), _effective(projection.y_scale)
        layouts = [REFERENCE_RELOAD_LAYOUT,
                   # X Linhai: five 60 px bars at x=1061/1157/1253/1349/1445 on
                   # y=1349 with dim "1".."5" labels just above (live frame).
                   # Kept after the reference row: when no candidate is
                   # supported the reader falls back to layouts[0].
                   (5, 1061, 96, 60, 1352),
                   # Fourth-slot view on the same client: the four bars sit at
                   # y=1352 (measured on a live Vermont frame: green 1108..1263,
                   # amber 1309..1358, 2-4 px tall), 13 px above the plain row.
                   (4, 1108, 96, 60, 1351),
                   (3, 1084, 144, 108, 1365),
                   (3, 1084, 144, 108, 1349), (3, 1252, 96, 60, 1368),
                   # Native 2560x1440 window content, padded by 80px.
                   # San Martin: four complete 56px bars on a 90px stride.
                   (4, 1119, 90, 56, 1375),
                   (3, 1096, 135, 101, 1375)]

        # The candidate layouts are reference-pixel measurements of the row,
        # so they only need the projection when the HUD is drawn at a different
        # size. At the reference size the reader samples exactly the reference
        # coordinates, which is also what keeps a reference frame's readout
        # identical to the pre-projection reader.
        at_reference_size = sx == 1.0 and sy == 1.0

        def frame_rect(rect):
            return list(rect) if at_reference_size else projection.from_ref(rect)

        def number_roi(start, stride, bar_y, i):
            """Reference turret-number plate -> this frame."""
            return frame_rect((start + i*stride - 24, bar_y - 5, 20, 22))

        def bar_patch(start, stride, width, bar_y, i):
            """One bar body, sampled at reference geometry.

            The layout masks below are written in reference pixels, so a frame
            that draws the bar smaller is sampled at the reference patch size
            (or resampled back to it) instead of re-tuning every row and column
            index.
            """
            x, y, w, h = frame_rect((start + i*stride - 2, bar_y - 4,
                                     width + 4, 11))
            patch = rgb[y:y+h, x:x+w]
            if patch.shape[1] != width + 4 or patch.shape[0] != 11:
                patch = cv2.resize(patch.astype(np.float32), (width + 4, 11),
                                   interpolation=cv2.INTER_AREA)
            return patch.astype(np.int16)

        supported = []
        amber_layouts = []
        colored_layouts = []
        for count, start, stride, width, bar_y in layouts:
            labels = [self.numeric.read(rgb, number_roi(start, stride, bar_y, i), maximum=4)
                      for i in range(count)]
            if sum(n["status"] == "observed" and n["value"] == i+1 for i, n in enumerate(labels)) >= 2:
                supported.append((count, start, stride, width, bar_y))
            # Full bars need their raised right endcap. An amber progress
            # bar can instead establish layout through a raised left endcap
            # and a continuous filled prefix. Neither proves readiness.
            bars = []
            amber_bars = []
            for i in range(count):
                patch = bar_patch(start, stride, width, bar_y, i)
                amber = ((patch[..., 0] > 110) &
                         (patch[..., 0]-patch[..., 1] > 25) &
                         (patch[..., 1]-patch[..., 2] > 25))
                green = ((patch[..., 1] > 135) &
                         (patch[..., 1]-patch[..., 0] > 45) &
                         (patch[..., 2] > 60))
                def complete(mask):
                    body = mask[4:10, 2:width+2]
                    endcap = mask[4:6, width-2:width+4]
                    return bool(body.any(axis=0).mean() >= .95 and
                                body.sum() >= 2*width and endcap.any())
                def thin_native_layout(mask):
                    # This native four-mount skin has one continuous body row;
                    # both raised ends must remain. Layout only, never ready.
                    if (count, start, stride, width, bar_y) != (4, 1119, 90, 56, 1375):
                        return False
                    return bool(mask[7, 2:width+2].mean() >= .95 and
                                mask[1:8, 1:4].sum() >= 2 and
                                mask[1:8, width:width+3].sum() >= 2)
                amber_complete = complete(amber)
                body = amber[6:10, 2:width+2]
                columns = body.any(axis=0)
                gaps = np.flatnonzero(~columns)
                filled = int(gaps[0]) if len(gaps) else width
                amber_progress = bool(
                    filled >= max(12, width//4) and
                    body[:, :filled].sum() >= 2*filled and
                    amber[4:6, 1:4].any())
                amber_bars.append(amber_complete or amber_progress)
                # A traversing mount can contain both existing colors within
                # one complete bar. This establishes layout, never readiness.
                bars.append(amber_progress or complete(amber | green) or
                            thin_native_layout(amber | green))
            if all(amber_bars):
                amber_layouts.append((count, start, stride, width, bar_y))
            if all(bars):
                colored_layouts.append((count, start, stride, width, bar_y))
        supported = list(dict.fromkeys(supported + colored_layouts))
        count, start, stride, width, bar_y = supported[0] if len(supported) == 1 else layouts[0]
        ambiguous = len(supported) > 1
        turrets = []
        # A two-row filled prefix is the readiness floor in reference pixels;
        # a scaled frame has proportionally fewer rows to give.
        fill_rows = 2 if sy == 1.0 else max(1, int(round(2 * sy)))
        for i in range(count):
            roi = frame_rect((start+i*stride+width//2-20, bar_y-27, 42, 25))
            number = self.numeric.read(rgb, roi, maximum=180)
            bar_bbox = frame_rect((start+i*stride, bar_y, width, 6))
            bx, by, bw, bh = bar_bbox; p = rgb[by:by+bh, bx:bx+bw].astype(np.int16)
            green = (p[..., 1] > 135) & (p[..., 1]-p[..., 0] > 45) & (p[..., 2] > 60)
            coverage = float(np.count_nonzero(green.any(axis=0))/bw)
            full_green = bool(coverage >= .95 and green.sum() >= fill_rows*bw and not ambiguous)
            # Amber fill means "this mount is reloading". Measured 2026-10-23: a
            # 20-frame sample was amber on every frame, with no green and no
            # countdown, and the reader reported unknown -- so the fire gate could
            # not tell "wait a second" from "cannot see". Classifying the amber
            # bar makes the state observable and lets the caller wait instead of
            # failing.
            amber = ((p[..., 0] > 140) & (p[..., 1] > 70) & (p[..., 2] < 90)
                     & (p[..., 0] - p[..., 1] > 30))
            amber_cov = float(np.count_nonzero(amber.any(axis=0))/bw)
            ready = None; status = "unknown"; amber_only = False
            if number["status"] == "observed" and number["value"] > 0 and not ambiguous:
                ready, status = False, "reloading"
            elif full_green and weapon["status"] == "observed" and weapon["kind"] == "gun":
                ready, status = True, "ready"
            elif amber_cov >= .60 and not ambiguous:
                # Operator's rule (2026-10-23): an amber mount is *traversing*,
                # a countdown means reloading, and any single green mount is
                # enough to fire. Amber therefore answers "not yet", and saying
                # so is strictly better than "unknown" -- but only when the
                # amber spans the whole bar, because a third of a bar is
                # background, not a state.
                ready, status, amber_only = False, "traversing", True
            reason = ("ambiguous_mount_layout" if ambiguous else
                      "amber_bar_traversing" if amber_only else
                      "no_countdown_or_full_green_bar" if status == "unknown" and not full_green else
                      "selected_gun_unresolved" if status == "unknown" else None)
            turrets.append(evidence(status, list(roi), number["confidence"] if status == "reloading" else (coverage if ready else 0),
                index=i, ready=ready, remaining_s=number["value"] if number["status"] == "observed" and not ambiguous else None,
                raw={"number": number, "bar_bbox": bar_bbox, "green_pixels": int(green.sum()),
                          "amber_column_coverage": round(amber_cov, 3),
                     "green_column_coverage": coverage, "full_green_bar": full_green, "reason": reason}))
        any_ready = any(t["ready"] is True for t in turrets)
        known = all(t["ready"] is not None for t in turrets)
        # Firing needs one green mount. If no mount is green the answer is a
        # definite "not now" even when another mount is unreadable -- measured
        # 2026-10-23 a live frame gave [False, None, None, False] and reporting
        # unknown there made the gate unable to tell "wait" from "cannot see".
        # Only claim a definite "not ready" when the main battery is the
        # selected weapon: with depth charges or torpedoes selected the mount
        # bars do not describe a gun the player could fire.
        gun_selected = (weapon.get("status") == "observed" and weapon.get("kind") == "gun")
        ready = True if any_ready else (
            False if (known or (gun_selected and any(t["ready"] is False for t in turrets)))
            else None)
        all_ready = all(t["ready"] is True for t in turrets) if known else None
        return evidence("observed" if known or any_ready else "partial" if any(t["status"] != "unknown" for t in turrets) else "unknown",
            frame_rect((start, bar_y-27, stride*(count-1)+width, 33)), max(t["confidence"] for t in turrets), ready=ready, all_ready=all_ready,
            turrets=turrets, raw={"layout_status": "ambiguous" if ambiguous else "observed" if supported else "unresolved_four_mount_regions",
                                "amber_mount_layout_observed": len(amber_layouts) == 1,
                                "colored_mount_layout_observed": len(colored_layouts) == 1,
                                "readiness_rule": ("any_full_green_mount_is_ready; a positive countdown means reloading; "
                                  "an amber bar means the mount is traversing")})

    def _unit_number(self, rgb, box, unit, maximum):
        """Read one numeric group ending in independently observed native units.

        Ruler dashes outside the numeric group are not glyphs. Never discard
        unknown strokes inside its bounds, or accept a number without its unit.
        """
        x,y,w,h = box
        patch = rgb[y:y+h,x:x+w].astype(np.int16)
        gray = patch.min(axis=2)
        mask = ((gray>95) & (patch.max(axis=2)-gray<42) &
                (gray-cv2.GaussianBlur(gray.astype(np.float32),(0,0),3)>12)).astype(np.uint8)
        _,_,stats,_ = cv2.connectedComponentsWithStats(mask)
        parts = sorted([tuple(map(int,s)) for s in stats[1:] if s[4]>=3])
        accepted = []
        for count in (1,2):
            if len(parts)<=count:
                continue
            suffix = parts[-count:]
            sx,sy = min(p[0] for p in suffix),min(p[1] for p in suffix)
            ex,ey = max(p[0]+p[2] for p in suffix),max(p[1]+p[3] for p in suffix)
            if not 8<=ey-sy<=27:
                continue
            glyph = _normalized(mask[sy:ey,sx:ex])
            scores = {}
            for name, variants in self._unit_templates.items():
                for template, aspect in variants:
                    score = float(2*np.count_nonzero(glyph & template)/
                        (glyph.sum()+template.sum())-.18*abs(np.log(
                        ((ex-sx)/(ey-sy))/aspect)))
                    scores[name] = max(scores.get(name, -1.0), score)
            if unit not in scores or scores[unit]<.74 or scores[unit]-max(
                    (v for k,v in scores.items() if k!=unit),default=0)<.035:
                continue
            digits = [p for p in parts[:-count] if 8<=p[3]<=27 and p[0]+p[2]<=sx]
            if not digits:
                continue
            first,last = digits[0],digits[-1]
            if sx-(last[0]+last[2])>10 or abs(last[1]+last[3]-ey)>3:
                continue
            nx,ny = first[0],min(p[1] for p in digits)
            ne,nb = last[0]+last[2],max(p[1]+p[3] for p in digits)
            bounds = [x+max(0,nx-2),y+max(0,ny-3),ne-nx+4,nb-ny+6]
            number = self.numeric.read(rgb,bounds,decimal=True,maximum=maximum,match_holes=True)
            if number['status']=='observed':
                number['raw'].update(unit=unit,unit_bbox=[x+sx,y+sy,ex-sx,ey-sy],
                                     unit_score=scores[unit],grouping='native_suffix_numeric_group')
                accepted.append(number)
        return accepted[0] if len(accepted)==1 else None

    def _anchored_number(self, rgb, kind, gray_signal=None, projection=None):
        projection = projection or self.resolve(rgb)
        sx, sy = _effective(projection.x_scale), _effective(projection.y_scale)
        variants = self.scaled_match_templates(kind, projection)
        band = projection.project_rect(REFERENCE_BATTLE_ROIS["anchored_number"])
        hits = _match(rgb, self.anchors[kind], band,
                      threshold=.70 if kind == "time" else .75, scales=(1.0, 1.25, 4/3, 1.4, 1.5),
                      gray_signal=gray_signal, template_variants=variants)
        if not hits and kind == 'distance' and self.anchors[kind] is not None:
            # Background recovery is confined to these native readout anchors;
            # do not change matching for ammo, mounts or other HUD sprites.
            # Estimate on an expanded ROI so the median has real border pixels.
            x0, y0, rw, rh = band
            pad = _scale_px(7, sy)
            window = _scale_px(15, sy)
            window += 0 if window % 2 else 1
            patch = rgb[y0-pad:y0+rh+pad, x0-pad:x0+rw+pad]
            gray = patch.min(axis=2).astype(np.uint8)
            residual = gray.astype(np.float32) - cv2.medianBlur(gray, window)
            residual = residual[pad:pad+rh, pad:pad+rw]
            recovered = _match(rgb, self.anchors[kind], (x0, y0, rw, rh),
                               threshold=.75, gray_signal=residual,
                               template_variants=variants)
            original = _gray_signal(rgb, (x0, y0, rw, rh)) if gray_signal is None else gray_signal
            for hit in recovered:
                scale = hit['raw']['scale']
                shape = next(shape for s, shape, _, _ in variants if s == scale)
                bx, by, _, _ = hit['bbox']
                yy, xx = by-y0-_scale_px(2, sy), bx-x0-_scale_px(2, sx)
                native = original[yy:yy+shape.shape[0], xx:xx+shape.shape[1]]
                on = shape > shape.max()*.4
                contrast = float(native[on].mean()-native[~on].mean())
                if contrast >= 16:
                    hit['raw'].update(method='native_anchor_median_background',
                                      native_contrast=contrast, requires_unit=True)
                    hits.append(hit)
        result = evidence(seconds=None, anchor=None) if kind == "time" else evidence(km=None, anchor=None)
        result["raw"] = {"anchors": hits, "reason": "native_anchor_absent"}
        if len(hits) != 1:
            if hits: result["raw"]["reason"] = "ambiguous_native_anchors"
            return result
        anchor = hits[0]; x, y, w, h = anchor["bbox"]
        # Actual Alt HUD prints seconds LEFT of the stopwatch, distance RIGHT.
        # Preserve the right-side numeric layout, refusing two readable values.
        # The distance readout is "-> 7.7 km": the arrow sits right after the
        # anchor and the digits start ~66 px later, so the old 58 px box stopped
        # before the text (measured: 11 foreground pixels in the box, the digits
        # plainly visible at 1320-1435 on the live frame). Widen it; the caller
        # still refuses two readable values.
        boxes = [[x+w+_scale_px(2, sx), max(0, y-_scale_px(3, sy)),
                  _scale_px(58, sx), _scale_px(32, sy)]]
        if kind == "time":
            boxes.append([max(0, x-_scale_px(62, sx)), max(0, y-_scale_px(3, sy)),
                          _scale_px(62, sx), _scale_px(32, sy)])
        unit = "s" if kind == "time" else "km"
        attempts = [self.numeric.read(rgb, box, decimal=True, maximum=120 if kind == "time" else 50, unit=unit)
                    for box in boxes]
        for index,number in enumerate(attempts):
            if number['status'] != 'observed':
                grouped = self._unit_number(rgb,boxes[index],unit,120 if kind=='time' else 50)
                if grouped is not None:
                    attempts[index] = grouped
        readable = [n for n in attempts if n["status"] == "observed"
                    and ((anchor["confidence"] >= .80 and not anchor['raw'].get('requires_unit'))
                         or n["raw"].get("unit") == unit)]
        result["anchor"] = anchor
        result["raw"] = {"anchors": hits, "numbers": attempts,
                         "reason": "ambiguous_anchored_numbers" if len(readable) > 1 else "anchored_number_unresolved"}
        if len(readable) == 1:
            number = readable[0]
            result.update(status="observed", bbox=number["bbox"], confidence=min(anchor["confidence"], number["confidence"]))
            result["raw"].pop("reason")
            result["seconds" if kind == "time" else "km"] = number["value"]
        return result

    def read_aim_distance(self, rgb, reticle, _band=(44, 200), projection=None):
        """Read the "NN.N km" ballistic solution printed beside the reticle.

        The anchored-number path depends on matching a sprite that this client
        also draws elsewhere (measured: the anchor was taken 78 px left of the
        readout, so the number box landed on empty water while the digits were
        plainly readable). The readout's position relative to the reticle is
        fixed, so read that band directly. Returns the same evidence dict as the
        anchored reader; ``observed`` means one unambiguous value was read.

        The reticle is a frame point; every offset and extent around it is a
        reference length and is scaled by this frame's projection, so the band
        lands on the same glyphs on a smaller client.
        """
        if reticle is None or reticle[0] is None:
            return evidence(km=None, reason='reticle_unobserved')
        projection = projection or self.resolve(rgb)
        sx, sy = _effective(projection.x_scale), _effective(projection.y_scale)
        cx, cy = float(reticle[0]), float(reticle[1])
        attempts = []
        # Measured on four frames with known values (5.3 / 7.2 / 7.4 / 7.7):
        # the digits are 15 px tall on one band, the "km" suffix is 11 px tall on
        # a lower band, and the gap from the last digit to the "k" is 6-7 px
        # while glyphs inside the number are <=1 px apart. So isolate the number
        # by letter spacing instead of guessing a box width, and start at the
        # first digit-shaped blob (the direction arrow's 3x18 stem precedes it).
        # The readout sits just right of the reticle on some layouts and almost
        # centred under it on others (measured: "5.8 km" starts ~10 px right of
        # centre, while "7.4 km" starts ~54 px right). Starting at +44 therefore
        # clipped the near-centred layout and every read came back unknown.
        # Two layouts are in use: the tuned one starts the digits ~54 px right
        # of the reticle, this client's Alt HUD starts them ~10 px right. Try the
        # tuned offset first (a wide band alone broke it) and fall back.
        offset, band_w = _band
        x0, y0 = int(cx) + _scale_px(offset, sx), int(cy) - _scale_px(14, sy)
        band_w, band_h = _scale_px(band_w, sx), _scale_px(56, sy)
        if (x0 < 0 or y0 < 0 or x0 + band_w > rgb.shape[1] or y0 + band_h > rgb.shape[0]):
            return evidence(km=None, reason='reticle_at_frame_edge')
        patch = rgb[y0:y0+band_h, x0:x0+band_w].astype(np.int16)
        gray = np.minimum(np.minimum(patch[..., 0], patch[..., 1]), patch[..., 2])
        spread = (np.maximum(np.maximum(patch[..., 0], patch[..., 1]), patch[..., 2])
                  - gray)
        contrast = gray - cv2.GaussianBlur(gray.astype(np.float32), (0, 0), _scale_px(3, sy))
        mask = ((gray > 95) & (spread < 42) & (contrast > 12)).astype(np.uint8)
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        blobs = sorted(((int(s[0]), int(s[1]), int(s[2]), int(s[3]))
                        for s in stats[1:]
                        # h>=3 drops the 1-2 px horizontal rules that run through
                        # this band (measured: 19-21 px wide, 1-2 px tall) but not
                        # the 3 px decimal dot, which the number needs.
                        if s[4] >= _scale_area(6, sx, sy) and s[3] >= _scale_px(3, sy)
                        and s[2] < _scale_px(40, sx) and s[3] < _scale_px(24, sy)),
                       key=lambda g: g[0])
        # The direction arrow is an 18x18 blob, tall enough to look like a
        # digit and wide enough to pass the width test; starting there reads an
        # "4" out of it. Real glyphs of this font share one height, so take the
        # modal tall-blob height as the digit height and start at the first blob
        # that matches it.
        tall = [g for g in blobs if g[2] >= _scale_px(5, sx)
                and _scale_px(12, sy) <= g[3] <= _scale_px(18, sy)]
        digit_h = None
        if tall:
            counts = {}
            for g in tall:
                counts[g[3]] = counts.get(g[3], 0) + 1
            digit_h = max(counts, key=lambda h: (counts[h], h))
        origin = next((g for g in tall if digit_h is None
                       or abs(g[3] - digit_h) <= _scale_px(1, sy)), None)
        if origin is None:
            return evidence(km=None, reason='aim_distance_no_digits')
        index = blobs.index(origin)
        left, top, right, bottom = (origin[0], origin[1],
                                    origin[0] + origin[2], origin[1] + origin[3])
        for gx, gy, gw, gh in blobs[index + 1:]:
            if gx - right >= _scale_px(6, sx):
                break
            left, top = min(left, gx), min(top, gy)
            right, bottom = max(right, gx + gw), max(bottom, gy + gh)
        box = [x0 + left - _scale_px(2, sx), y0 + top - _scale_px(2, sy),
               right - left + _scale_px(4, sx), bottom - top + _scale_px(4, sy)]
        # Tuned bright-on-dark rule first (it reads the mid-tone water frames
        # reliably); only if that finds nothing, retry with either polarity for
        # the grey-on-bright case. Trying the loose rule first broke the frames
        # that already worked (measured: 7.4/7.7 -> unknown).
        found = self.numeric_distance.read(rgb, box, decimal=True, maximum=50,
                                           unit='km')
        if found['status'] != 'observed':
            retry = self.numeric_distance.read(rgb, box, decimal=True, maximum=50,
                                              unit='km', any_polarity=True)
            if retry['status'] == 'observed':
                found = retry
        if found['status'] == 'observed':
            found['km'] = found.get('value')
            found['raw'] = dict(found.get('raw') or {},
                                number_bbox=[x0 + left, y0 + top, right - left, bottom - top])
            return found
        if _band != (10, 300):
            return self.read_aim_distance(rgb, reticle, _band=(10, 300),
                                          projection=projection)
        return found

    def _aiming_view(self, rgb, weapon, reticle, battle, projection=None):
        """Recognize the supported scope ruler, never infer its zoom level.

        Vertical strokes must form regular runs on both sides of the observed
        caret. A selected gun and independent battle HUD are required too.
        Ruler presence cannot establish camera stability or target association.
        """
        projection = projection or self.resolve(rgb)
        sx, sy = _effective(projection.x_scale), _effective(projection.y_scale)
        result = evidence(mode=None, zoom_confirmed=None)
        result['raw'] = {'method': 'bilateral_regular_scope_ticks',
                         'reason': 'scope_ruler_unconfirmed'}
        ruler = scope_ruler_center(rgb, projection=projection)
        caret = reticle['center'] if reticle['status'] == 'observed' else None
        caret_source = reticle.get('raw', {}).get('search', 'green_ring') if caret is not None else None
        if caret is None and ruler is not None:
            caret, caret_source = ruler['center'], 'scope_ruler_center'
        if caret is None:
            ring = green_reticle(rgb, projection=projection)
            if ring is not None:
                caret, caret_source = ring['centre'], 'green_ring'
        result['raw']['scope_ruler_center'] = ruler
        result['raw']['caret_center'] = caret
        if (battle['active'] is not True or weapon['kind'] != 'gun' or caret is None):
            return result
        cx, cy = caret
        # The maximum-size minimap and its column labels cover the right ruler.
        # Keep only unobscured strokes; map glyphs cannot confirm a scope. The
        # minimap's left edge is measured for this frame, not read as a constant.
        # Everything below is a reference length: the band around the caret,
        # the tick's seed height, its pitch and the ruler's minimum span all
        # belong to the native ruler and are scaled by this frame's projection.
        seed = _scale_px(7, sy)
        tick_w = _scale_px(5, sx)
        near = _scale_px(50, sx)
        reach = _scale_px(820, sx)
        minimap_left = projection.from_ref(MINIMAP_ROI)[0]
        x0, x1 = max(0, int(cx)-_scale_px(860, sx)), min(minimap_left, int(cx)+_scale_px(860, sx))
        y0, y1 = max(0, int(cy)-_scale_px(45, sy)), min(rgb.shape[0], int(cy)+_scale_px(5, sy))
        patch = rgb[y0:y1, x0:x1]
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY).astype(np.float32)
        spread = np.ptp(patch.astype(np.int16), axis=2)
        contrast = gray-cv2.GaussianBlur(gray.astype(np.float32), (0, 0), _scale_px(3, sy))
        mask = ((gray > 125) & (spread < 35) & (contrast > 20)).astype(np.uint8)
        vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((seed, 1), np.uint8))
        # A sea highlight can split an otherwise real tick by one or two rows.
        # Repair only components with an existing uninterrupted seven-pixel
        # vertical seed: closing alone also turns sea texture into fake ticks.
        repaired = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                    np.ones((_scale_px(3, sy), 1), np.uint8))
        repaired = cv2.morphologyEx(repaired, cv2.MORPH_OPEN, np.ones((seed, 1), np.uint8))
        _, _, stats, _ = cv2.connectedComponentsWithStats(repaired)
        ticks = sorted((float(x0+x+w/2), int(y0+y+h), int(h))
                       for x, y, w, h, area in stats[1:]
                       if 1 <= w <= tick_w and seed <= h <= _scale_px(35, sy)
                       and area >= _scale_area(7, sx, sy)
                       and np.count_nonzero(vertical[y:y+h, x:x+w]) >= seed)
        sides = [[t for t in ticks if cx-reach <= t[0] <= cx-near],
                 [t for t in ticks if cx+near <= t[0] <= cx+reach]]
        diagnostics = []
        for side in sides:
            if len(side) < 12:
                diagnostics.append({'accepted': False, 'ticks': len(side)})
                continue
            # Establish the baseline only from existing component bottoms.
            # A tick joined to a hull stroke may extend below it; recover only
            # on-grid components with a real seven-pixel seed at the baseline.
            bottom = float(np.median([t[1] for t in side]))
            baseline_side = [t for t in side if abs(t[1]-bottom) <= _scale_px(3, sy)]
            if len(baseline_side) >= 12:
                xs = np.asarray([t[0] for t in baseline_side])
                spacing = float(np.median(np.diff(xs)))
                if _scale_px(12, sx) <= spacing <= _scale_px(40, sx):
                    distances = xs[None, :] - xs[:, None]
                    slots = np.rint(distances / spacing)
                    pairs = (distances > 0) & (slots > 0)
                    pitch = float(np.median(distances[pairs] / slots[pairs]))
                    origin = xs[0] + float(np.median(
                        xs-xs[0]-np.rint((xs-xs[0])/pitch)*pitch))

                    def on_grid(t):
                        offset = t[0]-origin
                        return abs(offset-round(offset/pitch)*pitch) <= _scale_px(2, sx)

                    for t in side:
                        if (t[1] <= bottom+_scale_px(3, sy) or
                                t[1]-t[2] > bottom-_scale_px(7, sy) or
                                not on_grid(t)):
                            continue
                        column = int(t[0]-x0-.5)
                        row = int(bottom-y0)
                        seed_patch = vertical[max(0, row-seed):row,
                                              max(0, column-1):column+2]
                        if np.count_nonzero(seed_patch) >= seed:
                            baseline_side.append((t[0], bottom, t[2]))
                    baseline_side.sort()
                    # A nearby off-grid end stroke is not an extra ruler tick.
                    baseline_side = [t for i, t in enumerate(baseline_side) if not (
                        (i == 0 and baseline_side[1][0]-t[0] < pitch-_scale_px(2, sx) or
                         i == len(baseline_side)-1 and
                         t[0]-baseline_side[-2][0] < pitch-_scale_px(2, sx)) and not on_grid(t))]
            side = baseline_side
            if len(side) < 12:
                diagnostics.append({'accepted': False, 'ticks': len(side)})
                continue
            # A shorter hull stroke can lie between two genuine ticks with the
            # same baseline. Remove it only when its two neighbours independently
            # bracket one normal spacing; missing ruler ticks remain missing.
            spacing = float(np.median(np.diff([t[0] for t in side])))
            side = [t for i, t in enumerate(side) if not (
                0 < i < len(side)-1 and
                t[2] < min(side[i-1][2], side[i+1][2]) and
                0 < t[0]-side[i-1][0] < spacing-_scale_px(2, sx) and
                0 < side[i+1][0]-t[0] < spacing-_scale_px(2, sx) and
                abs(side[i+1][0]-side[i-1][0]-spacing) <= _scale_px(2, sx))]
            if len(side) < 12:
                diagnostics.append({'accepted': False, 'ticks': len(side)})
                continue
            xs, bottoms = np.asarray([t[:2] for t in side]).T
            gaps = np.diff(xs)
            spacing = float(np.median(gaps))
            regular = float(np.mean(np.abs(gaps-spacing) <= _scale_px(2, sx)))
            baseline = float(np.mean(np.abs(bottoms-np.median(bottoms)) <= _scale_px(3, sy)))
            span = float(xs[-1]-xs[0])
            accepted = bool(_scale_px(12, sx) <= spacing <= _scale_px(40, sx) and
                            regular >= .85 and baseline >= .85 and
                            span >= _scale_px(450, sx))
            diagnostics.append(dict(accepted=accepted, ticks=len(side), spacing_px=spacing,
                                    regular_fraction=regular, baseline_fraction=baseline,
                                    span_px=span, ticks_xy=[t[:2] for t in side]))
        result['raw'].update(sides=diagnostics, ticks_xy=[t[:2] for t in ticks])
        if all(s['accepted'] for s in diagnostics):
            result.update(status='observed', mode='scope_ruler', bbox=[x0, y0, x1-x0, y1-y0],
                          confidence=min(s['regular_fraction'] for s in diagnostics))
            result['raw']['reason'] = 'gun_selected_with_bilateral_scope_ruler; zoom_unknown'
            result['raw']['caret_source'] = caret_source
        return result
    def _cooling_depth_charge_icon(self, rgb, roi, projection=None):
        if not self._depth_charge_wings:
            return False
        signal = _gray_signal(rgb, roi)
        if projection is not None and _effective(projection.x_scale) != 1.0:
            # The wing kernels are native pixels; bring the sampled window back
            # to reference size so the same template rows and columns apply.
            signal = cv2.resize(signal, None, fx=1.0/projection.x_scale,
                                fy=1.0/max(projection.y_scale, 1e-9),
                                interpolation=cv2.INTER_AREA)
        height, width = self.depth_charge.shape[:2]
        rows, cols = signal.shape[0]-height+1, signal.shape[1]-width+1
        if rows <= 0 or cols <= 0:
            return False
        agreement = None
        for x, kernel in self._depth_charge_wings:
            scores = cv2.matchTemplate(signal, kernel, cv2.TM_CCOEFF_NORMED)
            scores = scores[:rows, x:x+cols]
            scores = np.where(np.isfinite(scores), scores, -1)
            agreement = scores if agreement is None else np.minimum(agreement, scores)
        peaks = ((agreement >= .90) &
                 (agreement == cv2.dilate(agreement, np.ones((7, 7), np.uint8))))
        ys, xs = np.where(peaks)
        origins = []
        for y, x in zip(ys, xs):
            if not any(abs(x-px) < 8 and abs(y-py) < 8 for py, px in origins):
                origins.append((y, x))
        return len(origins) == 1

    def _depth_charge_selection(self, rgb, projection=None):
        """Independent battle evidence only; never gun or ASW readiness.

        Screen-native icon [1228,1410,65,65] from the unchanged frame-000 in
        state/mainview-resume-live-20261016-s6, outside its green outline.
        Require both complete vertical selection edges in the same layout.
        During cooldown only the two unoccluded upper wings may substitute
        for the full icon, at one shared native origin; readiness stays unknown.

        The layout offsets are reference space and the edge widths, the shared
        run floor and the box extent are reference pixels, so all four follow
        this frame's projection.
        """
        projection = projection or self.resolve(rgb)
        sx, sy = _effective(projection.x_scale), _effective(projection.y_scale)
        edge_w = _scale_px(12, sx)
        min_shared = 75 if sy == 1.0 else max(8, int(round(75 * sy)))
        layouts = []
        # Same bar-centring offset as the ammo sprites: the live client shifts
        # this selected auxiliary slot right by 71 px as well. The offset is a
        # reference-space shift, applied before the projection.
        for offset in (0, -44, 71):
            roi = projection.from_ref(
                (REFERENCE_DEPTH_CHARGE_ROI[0] + offset, REFERENCE_DEPTH_CHARGE_ROI[1],
                 REFERENCE_DEPTH_CHARGE_ROI[2], REFERENCE_DEPTH_CHARGE_ROI[3]))
            hits = _match(rgb, self.depth_charge, roi, threshold=.90,
                          template_variants=self.scaled_match_templates("depth_charge", projection))
            if len(hits) != 1 and not (not hits and self._cooling_depth_charge_icon(
                    rgb, roi, projection)):
                continue
            x, y, w, h = roi
            patch = rgb[y:y+h, x:x+w].astype(np.int16)
            green = ((patch[..., 1] > 125) &
                     (patch[..., 1]-patch[..., 0] > 40) & (patch[..., 2] > 60))
            edges = [green[:, :edge_w].any(axis=1), green[:, -edge_w:].any(axis=1)]
            # The native box has 82 connected rows; missing edges or disjoint
            # fragments cannot establish the selected auxiliary view.
            shared = edges[0] & edges[1]
            if np.count_nonzero(shared) >= min_shared:
                runs = np.diff(np.r_[False, shared, False].astype(np.int8))
                if max(np.flatnonzero(runs == -1)-np.flatnonzero(runs == 1)) >= min_shared:
                    layouts.append(offset)
        return layouts


    def _battle_components(self, rgb, projection=None):
        """Read the shared weapon, reticle, reload and battle evidence."""
        _validate_frame(rgb)
        projection = projection or self.resolve(rgb)
        weapon = self._weapon(rgb)
        caret_variants = self.scaled_match_templates("aim_caret", projection)
        caret = _match(rgb, self.icons["aim_caret"],
                       projection.project_rect(REFERENCE_BATTLE_ROIS["caret"]), threshold=.75,
                       template_variants=caret_variants)
        caret_search = 'native_band'
        if not caret:
            # A close-range aim sits far lower than the usual band, so the band
            # is only the first look: the caret is a native sprite and a wide
            # search keeps the same template and threshold.
            caret = _match(rgb, self.icons["aim_caret"],
                           projection.project_rect(REFERENCE_BATTLE_ROIS["caret_wide"]),
                           threshold=.75,
                           template_variants=caret_variants)
            caret_search = 'wide_search'
        reticle = evidence(center=None)
        reticle["raw"] = {"candidates": caret, "reference_source": "masked_actual_screen_aim_caret",
                          "search": caret_search, "lock_semantics": "none"}
        if len(caret) != 1:
            reticle["raw"]["reason"] = "ambiguous_carets" if caret else "caret_shape_unresolved"
        if len(caret) == 1:
            hit = caret[0]; x, y, w, h = hit["bbox"]
            reticle.update(status="observed", bbox=hit["bbox"], confidence=hit["confidence"], center=[x+w/2, y+h/2])
        if reticle['status'] != 'observed':
            rings = []
            for color in ('green', 'amber'):
                key = f'{color}_aim_ring'
                for hit in _match(rgb, self.icons[key],
                                  projection.project_rect(REFERENCE_BATTLE_ROIS["ring"]),
                                  threshold=.85,
                                  template_variants=self.scaled_match_templates(key, projection)):
                    rings.append((color, hit))
            if len(rings) == 1:
                color, hit = rings[0]; x, y, w, h = hit['bbox']
                reticle.update(status='observed', bbox=hit['bbox'], confidence=hit['confidence'],
                               center=[x+w/2, y+h/2])
                reticle['raw'].update(candidates=[hit], search=f'native_{color}_ring',
                                      reference_source=f'masked_native_{color}_ring')
                reticle['raw'].pop('reason', None)
        reload = self._reload(rgb, weapon)
        # Two native ammo sprites need an independent battle-only HUD region.
        # A single selected gun sprite instead needs BOTH a complete colored
        # mount layout and a native reticle. Selection color/slot position alone
        # cannot supply the native sprite evidence for this narrower path.
        ammo_layouts = []
        for offset in sorted({slot["layout_offset_px"] for slot in weapon["raw"]["slots"]}):
            ammo = {slot["ammo"] for slot in weapon["raw"]["slots"]
                    if slot["layout_offset_px"] == offset and slot["icon"] is not None}
            if len(ammo) >= 2:
                ammo_layouts.append(offset)
        mount_layout = reload["raw"]["layout_status"] == "observed"
        auxiliary_layouts = self._depth_charge_selection(rgb)
        auxiliary = bool(set(ammo_layouts) & set(auxiliary_layouts))
        selected_ammo_layouts = sorted({
            slot["layout_offset_px"] for slot in weapon["raw"]["slots"]
            if (weapon["status"] == "observed" and weapon["selected"]
                and weapon["kind"] == "gun" and slot["ammo"] == weapon["ammo"]
                and slot["icon"] is not None and slot["selected_outline"])
        })
        full_mount_layout = mount_layout and reload["raw"]["colored_mount_layout_observed"]
        single_ammo = (bool(selected_ammo_layouts) and full_mount_layout
                       and reticle["status"] == "observed")
        supported = bool((ammo_layouts and
                          (mount_layout or reticle["status"] == "observed" or auxiliary))
                         or single_ammo)
        battle = evidence("observed" if supported else "unknown", active=True if supported else None)
        battle["raw"] = {"ammo_layout_offsets_px": ammo_layouts,
                         "selected_native_ammo_layout_offsets_px": selected_ammo_layouts,
                         "single_ammo_battle_hud_observed": single_ammo,
                         "full_colored_mount_layout_observed": full_mount_layout,
                         "mount_layout_observed": mount_layout,
                         "reticle_observed": reticle["status"] == "observed",
                         "selected_depth_charge_layout_offsets_px": auxiliary_layouts,
                         "hud_projection": projection.report(),
                         "reason": "native_battle_hud" if supported else "battle_hud_unconfirmed"}
        return weapon, reload, reticle, battle

    def detect_battle(self, rgb):
        """Return the same battle evidence as detect, without flight or view reads.

        Takes any HxWx3 uint8 frame, just like detect: the HUD rectangles come
        from the projection of *that* frame. Weapon, caret and reload evidence
        are still computed with the full HUD policy.
        """
        return self._battle_components(rgb)[3]

    def detect(self, rgb):
        _validate_frame(rgb)
        projection = self.resolve(rgb)
        weapon, reload, reticle, battle = self._battle_components(rgb, projection)
        # Both native anchor matches use the same frame region. Colored custom
        # assets still build their own color signal inside _match.
        anchor_signal = _gray_signal(rgb, projection.project_rect(
            REFERENCE_BATTLE_ROIS["anchored_number"]))
        with ThreadPoolExecutor(max_workers=1) as worker:
            time_job = worker.submit(self._anchored_number, rgb, "time", anchor_signal,
                                     projection)
            distance = self._anchored_number(rgb, "distance", anchor_signal, projection)
            aiming_view = self._aiming_view(rgb, weapon, reticle, battle, projection)
            flight = time_job.result()
        from aim_indicators import read_aim_indicators
        indicators = read_aim_indicators(rgb)
        point = evidence(center=None, raw=dict(candidates=indicators['aim_candidates'],
                                               semantics='white_commanded_aim_point'))
        if indicators['aim_point'] is not None:
            point.update(indicators['aim_point'], status='observed', confidence=indicators['aim_point']['precision'])
        direction = indicators['gun_direction']
        direction.update(bbox=None, confidence=direction['ring']['support'] if direction['ring'] else 0.,
                         raw=dict(candidates=indicators['gun_rings'], semantics=indicators['semantics']))
        flight["distance"] = distance
        return {"weapon": weapon, "reload": reload, "reticle": reticle, "flight_time": flight,
                "battle": battle, "aiming_view": aiming_view,
                "aim_point": point, "gun_direction": direction}


# --- Low contrast glyph front end -------------------------------------------
# Storm and night frames paint the HUD's white text over grey sea (measured 255
# on a 98 grey) instead of a dark sky. The reader's own mask keys on absolute
# brightness (``gray > 95``), so the sea itself becomes foreground, every digit
# merges with its background and reads "unknown / malformed_stroke": on the
# 2026-10-20 storm frame the timer's four glyph groups are found correctly
# (widths 10, 10, 3, 10, 9) yet not one digit resolves.
#
# The front end below models the local background with a median larger than a
# stroke, keeps only the pixels that stand above it, and hands the same reader a
# white-on-black crop of exactly the resolved strokes. Threshold fraction,
# glyph topology and score margin remain the reader's own decisions; this only
# supplies contrast conditions under which those decisions can be made.
#
# Threshold choice is the one place a single number would be a guess: the
# decimal dot of a small caption merges with the "4" below 0.35 of the peak,
# while thin strokes of a timer "9" break above 0.45 and a contrast-reduced
# caption only separates at 0.15..0.20. Every read therefore runs at each
# fraction in ``LOW_CONTRAST_FRACTIONS`` and a value is accepted only when at
# least ``LOW_CONTRAST_AGREEMENT`` of them return the same observation.
LOW_CONTRAST_KERNEL = 15                 # > stroke width, < the band being read
LOW_CONTRAST_FLOOR = 12.0                # residual floor, contrast-rule units
LOW_CONTRAST_FRACTIONS = (.15, .20, .30, .35, .40)
LOW_CONTRAST_AGREEMENT = 2
LOW_CONTRAST_MIN_PEAK = 24.0             # a flat patch holds no glyph at all


def low_contrast_residual(patch, kernel=LOW_CONTRAST_KERNEL, channel="min"):
    """``(residual, peak)`` for a glyph patch, or ``(None, reason)``.

    ``residual`` is one colour channel minus the median of a ``kernel`` window:
    the local background of a stroke, whatever sea, sky or panel is behind it.
    ``peak`` is its 99.9th percentile, the height of the brightest stroke. A
    patch whose peak stays under ``LOW_CONTRAST_MIN_PEAK`` carries no glyph and
    is rejected rather than thresholded into noise.

    ``channel="min"`` is the HUD's white text, including colour fringes that
    leave one channel pale. ``channel="max"`` is the salmon range placeholder
    ("- - km"), whose minimum channel sits at the background level.
    """
    p = np.asarray(patch)
    if p.ndim == 2:
        p = np.repeat(p[:, :, None], 3, axis=2)
    if p.ndim != 3 or p.shape[0] < 3 or p.shape[1] < 3:
        return None, 'patch_outside_frame'
    p = p.astype(np.int16)
    if channel == "max":
        gray = np.maximum(np.maximum(p[..., 0], p[..., 1]), p[..., 2]).astype(np.uint8)
    else:
        gray = np.minimum(np.minimum(p[..., 0], p[..., 1]), p[..., 2]).astype(np.uint8)
    size = min(int(kernel) | 1, min(gray.shape) | 1)
    if size < 3:
        return None, 'patch_too_small'
    residual = gray.astype(np.float32) - cv2.medianBlur(gray, size).astype(np.float32)
    peak = float(np.percentile(residual, 99.9))
    if peak < LOW_CONTRAST_MIN_PEAK:
        return None, 'no_local_contrast'
    return residual, peak


def low_contrast_components(residual, peak, fraction):
    """``(labels, stats, threshold)`` for ink standing above the background."""
    threshold = max(LOW_CONTRAST_FLOOR, float(fraction) * float(peak))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (residual > threshold).astype(np.uint8))
    return labels, stats, round(threshold, 1)


def low_contrast_glyphs(residual, peak, fraction, min_height=6, max_height=27,
                        min_width=1, max_width=20, min_fill=.18):
    """Stroke-sized components, left to right, with their labels.

    ``min_height``/``max_height`` bound a glyph's own raster: the timer digits
    measure 16..19 px, the range caption 8..12 px. Punctuation is smaller still,
    so it is picked up by containment in ``canonical_ink`` instead of here.
    """
    labels, stats, threshold = low_contrast_components(residual, peak, fraction)
    glyphs = []
    for label in range(1, len(stats)):
        x, y, w, h, area = (int(v) for v in stats[label])
        if not (min_height <= h <= max_height and min_width <= w <= max_width):
            continue
        if area < max(6, min_fill * w * h):
            continue
        glyphs.append({"bbox": [x, y, w, h], "area": area, "label": label})
    glyphs.sort(key=lambda glyph: glyph["bbox"][0])
    return glyphs, labels, stats, threshold


def canonical_ink(labels, stats, box, pad=1):
    """White-on-black crop of every ink component inside ``box``, one px padded.

    Containment is what keeps a decimal dot (2x2 px, far below stroke size) and
    what keeps the panel, rain and neighbouring text out: the reader measures
    brightness and local contrast again on this crop, so it must hold exactly
    the resolved ink and a black background uniform enough to be ignored.
    """
    x, y, w, h = (int(v) for v in box)
    if w < 1 or h < 1 or x < 0 or y < 0:
        return None
    if y + h > labels.shape[0] or x + w > labels.shape[1]:
        return None
    keep = [index for index, (left, top, width, height, _) in enumerate(stats[1:], 1)
            if left >= x and top >= y and left + width <= x + w and top + height <= y + h]
    if not keep:
        return None
    ink = np.isin(labels[y:y + h, x:x + w], keep).astype(np.uint8) * 255
    canvas = np.zeros((h + 2 * pad, w + 2 * pad), np.uint8)
    canvas[pad:pad + h, pad:pad + w] = ink
    return np.repeat(canvas[:, :, None], 3, axis=2)


def _read_canonical(reader, labels, stats, box, **read):
    """One field through the front end, or None: the reader's own verdict."""
    image = canonical_ink(labels, stats, box)
    if image is None:
        return None, "glyph_box_outside_patch"
    reading = reader.read(image, [0, 0, image.shape[1], image.shape[0]], **read)
    return reading, None


def _consensus(seen):
    """The text more thresholds agreed on than any other, or None.

    Every fraction is tried, so a pair of low thresholds cannot outvote a pair
    of high ones; a field that two disjoint threshold sets read differently is
    ambiguous and stays unknown even when each pair agreed with itself.
    """
    if not seen:
        return None, "thresholds_disagree"
    ranking = sorted(seen.items(), key=lambda row: row[1], reverse=True)
    text, hits = ranking[0]
    if hits < LOW_CONTRAST_AGREEMENT:
        return None, "thresholds_disagree"
    if len(ranking) > 1 and ranking[1][1] >= hits:
        return None, "thresholds_disagree"
    return text, None


# Top-right battle timer, e.g. "17:55". Measured on a live 2560x1600 frame:
# the four digits sit at x 2277..2325, y 95..135, with a 3 px colon between the
# second and third, and the white glyph mask isolates them from the sky.
BATTLE_TIMER_BAND = (2180, 95, 260, 70)


def _timer_text(labels, stats, glyphs, reader):
    """``"MMSS"`` from one threshold's ink, or None when the run is not a timer.

    A timer is exactly four stroke-sized columns plus the colon between the
    second and third, which is smaller than a stroke and is therefore found by
    position. Four columns without a colon between the middle pair is a range,
    a coordinate or a neighbour, not a clock: it stays unknown.
    """
    columns = []
    for glyph in glyphs:
        x, y, w, h = glyph["bbox"]
        if columns and x <= columns[-1]["right"]:
            columns[-1]["right"] = max(columns[-1]["right"], x + w)
            columns[-1]["bottom"] = max(columns[-1]["bottom"], y + h)
            columns[-1]["top"] = min(columns[-1]["top"], y)
        else:
            columns.append({"left": x, "right": x + w, "top": y, "bottom": y + h})
    if len(columns) != 4:
        return None
    top = min(column["top"] for column in columns)
    bottom = max(column["bottom"] for column in columns)
    dots = [int(area) for left, dot_top, width, height, area in stats[1:]
            if left >= columns[1]["right"] - 1 and left + width <= columns[2]["left"] + 1
            and dot_top >= top - 2 and dot_top + height <= bottom + 2
            and height <= 6 and width <= 6 and area >= 2]
    if not dots or sum(dots) < 4:
        return None
    digits = []
    for column in columns:
        reading, _ = _read_canonical(reader, labels, stats,
                                     (column["left"], top,
                                      column["right"] - column["left"], bottom - top),
                                     maximum=99)
        if reading is None or reading["status"] != "observed":
            return None
        digits.append(str(int(reading["value"])))
    return "".join(digits)


def _low_contrast_timer_text(patch, reader):
    """``("MMSS", detail)`` through the front end; empty text when unresolved."""
    residual, peak = low_contrast_residual(patch)
    if residual is None:
        return None, {"reason": peak}
    seen, thresholds = {}, []
    for fraction in LOW_CONTRAST_FRACTIONS:
        glyphs, labels, stats, threshold = low_contrast_glyphs(
            residual, peak, fraction, min_height=8)
        text = _timer_text(labels, stats, glyphs, reader)
        thresholds.append(dict(fraction=fraction, threshold=threshold,
                               glyphs=len(glyphs), text=text))
        if text is not None:
            seen[text] = seen.get(text, 0) + 1
    text, reason = _consensus(seen)
    return text, {"peak": round(peak, 1), "attempts": thresholds, "reason": reason}


def battle_time_left(rgb, band=BATTLE_TIMER_BAND):
    """Seconds left in the battle, or None when the timer is not drawn.

    The absolute-white mask runs first and its digits are read exactly as
    before; only when that path yields no four-digit run is the band re-read
    through ``low_contrast_residual`` for white text over a bright sea.
    """
    x, y, w, h = (int(round(v)) for v in band)
    if y + h > rgb.shape[0] or x + w > rgb.shape[1] or x < 0 or y < 0:
        raise ValueError('timer band outside the frame')
    patch = rgb[y:y + h, x:x + w]
    white = ((patch[..., 0] > 170) & (patch[..., 1] > 170) & (patch[..., 2] > 170))
    columns = white.any(axis=0)
    groups, start = [], None
    for index, filled in enumerate(columns):
        if filled and start is None:
            start = index
        elif not filled and start is not None:
            groups.append((start, index))
            start = None
    if start is not None:
        groups.append((start, len(columns)))
    # Keep the leading run of narrow glyph groups: MM ':' SS.
    # Digit width varies with the HUD scale (measured 7-17 px on this client);
    # the old 30 px ceiling also rejected the whole timer when the band caught a
    # neighbour, which is why a plainly visible "24:39" read as None.
    glyphs = [g for g in groups if 2 <= g[1] - g[0] <= 40]
    text, reader = None, None
    if len(glyphs) >= 5:
        reader = FireHudReader()
        digits = []
        for index, (left, right) in enumerate(glyphs[:5]):
            if index == 2:            # the colon carries no value
                continue
            reading = reader.numeric.read(rgb, (x + left - 2, y, (right - left) + 4, 40),
                                          maximum=99)
            if reading['status'] != 'observed':
                digits = None
                break
            digits.append(int(reading['value']))
        if digits is not None and len(digits) == 4:
            text = "".join(str(digit) for digit in digits)
    if text is None:
        # Storm/night frames: the absolute-white rule sees the grey sea as ink
        # and no digit resolves. Retry the same band through the local
        # background front end with the same reader, and take a value only when
        # several ink thresholds agree on it.
        if reader is None:
            reader = FireHudReader()
        text, _ = _low_contrast_timer_text(patch, reader.numeric)
    if text is None or len(text) != 4 or not text.isdigit():
        return None
    digits = [int(char) for char in text]
    minutes, seconds = digits[0] * 10 + digits[1], digits[2] * 10 + digits[3]
    if minutes > 40 or seconds > 59:
        return None
    return minutes * 60 + seconds


# The nameplate range caption sits centred under the bar and reads "14.7 公里".
# The search window is the one the replay caption reader uses, so both readers
# look at the same pixels and only differ in how they find the digits.
CAPTION_BAND = (140, 34, 2)      # width, height, rows below the bar


def _caption_band(rgb, plate_bbox):
    """The caption search window and its frame origin, or ``(None, None)``."""
    x, y, w, h = (int(v) for v in plate_bbox)
    width, height, offset = CAPTION_BAND
    top = y + h + offset
    left = max(0, x + w // 2 - width // 2)
    band = rgb[top:top + height, left:left + width]
    if band.shape[0] < 8 or band.shape[1] < 8:
        return None, None
    return band, (left, top)


def _baseline_runs(glyphs, max_gap=6):
    """Runs of glyphs that share a baseline and sit ``max_gap`` apart at most."""
    runs = []
    for glyph in glyphs:
        x, y, _, h = glyph["bbox"]
        placed = False
        for run in runs:
            base = max(g["bbox"][1] + g["bbox"][3] for g in run)
            gap = x - (run[-1]["bbox"][0] + run[-1]["bbox"][2])
            if abs(y + h - base) <= 3 and -2 <= gap <= max_gap:
                run.append(glyph)
                placed = True
                break
        if not placed:
            runs.append([glyph])
    return runs


def _caption_group(glyphs, band_width, min_width=14, max_width=70):
    """The same-baseline numeric run nearest the band centre, or None.

    ``公里`` follows the digits with a wider gap and taller glyphs, so the run
    is taken by baseline and gap the way the replay caption reader does, and
    the nearest run to the centre wins when a stopwatch icon sits beside it.
    """
    centre = band_width / 2.0
    viable = []
    for group in _baseline_runs(glyphs):
        group.sort(key=lambda g: g["bbox"][0])
        left = group[0]["bbox"][0]
        right = group[-1]["bbox"][0] + group[-1]["bbox"][2]
        if not min_width <= right - left <= max_width or len(group) < 3:
            continue
        viable.append((abs((left + right) / 2 - centre), left, group))
    if not viable:
        return None
    return min(viable, key=lambda row: (row[0], row[1]))[2]


def _low_contrast_caption(rgb, plate_bbox, reader):
    """``(evidence, None)`` through the front end, or ``(None, reason)``."""
    band, origin = _caption_band(rgb, plate_bbox)
    if band is None:
        return None, 'caption_band_outside_frame'
    residual, peak = low_contrast_residual(band)
    if residual is None:
        return None, peak
    seen, details, attempts = {}, {}, []
    for fraction in LOW_CONTRAST_FRACTIONS:
        glyphs, labels, stats, threshold = low_contrast_glyphs(
            residual, peak, fraction, max_height=16, max_width=14)
        group = _caption_group(glyphs, band.shape[1])
        attempt = {"fraction": fraction, "threshold": threshold,
                   "glyphs": len(glyphs), "text": None, "reason": None}
        attempts.append(attempt)
        if group is None:
            attempt["reason"] = "numeric_group_not_located"
            continue
        left = min(g["bbox"][0] for g in group)
        top = min(g["bbox"][1] for g in group)
        right = max(g["bbox"][0] + g["bbox"][2] for g in group)
        bottom = max(g["bbox"][1] + g["bbox"][3] for g in group)
        reading, reason = _read_canonical(reader, labels, stats,
                                          (left, top, right - left, bottom - top),
                                          decimal=True, maximum=50, match_holes=True)
        if reading is None or reading["status"] != "observed":
            attempt["reason"] = reason or reading["raw"].get("reason")
            continue
        attempt["text"] = reading["text"]
        seen[reading["text"]] = seen.get(reading["text"], 0) + 1
        details[reading["text"]] = {"value": float(reading["value"]),
                                    "confidence": float(reading["confidence"]),
                                    "bbox": [origin[0] + left, origin[1] + top,
                                             right - left, bottom - top],
                                    "threshold": threshold}
    text, reason = _consensus(seen)
    if text is None:
        return None, reason
    detail = details[text]
    places = len(text.partition(".")[2])
    half = 10.0 ** -places / 2.0
    value = detail["value"]
    return {
        "status": "observed",
        "value": value,
        "interval_km": [round(max(0.0, value - half), 3), round(value + half, 3)],
        "bbox": detail["bbox"],
        "confidence": detail["confidence"],
        "text": text,
        "raw": {"reason": None, "method": "low_contrast_caption",
                "peak": round(peak, 1), "agreement": seen[text],
                "threshold": detail["threshold"], "attempts": attempts},
        "provenance": {"source": "main_view", "reader": "fire_hud.NativeNumericReader",
                       "preprocessing": "local_median_background_canonical_ink"},
    }, None


def nameplate_range_caption(rgb, plate_bbox):
    """The range caption under a nameplate, e.g. "14.7 公里", as evidence.

    The absolute-brightness reader runs first and its evidence is returned
    untouched when it resolves a caption. On storm and night frames that mask
    counts the sea as ink and the caption falls to "unknown"; the same band is
    then re-read through the local background front end. When neither path
    resolves the caption the first path's unknown evidence comes back with the
    second path's reason attached, so a caller never sees a guessed range.
    """
    bbox = [int(v) for v in plate_bbox]
    try:
        import target_range_selection
        bright = target_range_selection.read_distance_km(rgb, bbox)
    except Exception as exc:      # unavailable assets are evidence, not a guess
        bright = {"status": "unknown", "value": None, "interval_km": None,
                  "bbox": bbox, "confidence": 0.0,
                  "raw": {"reason": "bright_caption_reader_unavailable",
                          "error": type(exc).__name__},
                  "provenance": {"source": "main_view",
                                 "method": "absolute_brightness_caption"}}
    if bright.get("status") == "observed" and bright.get("value") is not None:
        bright.setdefault("raw", {})["method"] = "absolute_brightness_caption"
        return bright
    low, reason = _low_contrast_caption(rgb, bbox, FireHudReader().numeric_distance)
    if low is None:
        bright.setdefault("raw", {})["low_contrast"] = reason
        return bright
    return low


def nameplate_range_km(rgb, plate_bbox):
    """Caption distance in km, or None: the scalar form a range reader wants."""
    reading = nameplate_range_caption(rgb, plate_bbox)
    if reading.get("status") != "observed":
        return None
    return reading.get("value")


def target_range_km(rgb, nameplate_bbox):
    """``(km, source)`` for the range under a nameplate, or ``(None, reason)``.

    ``source`` is ``"absolute_brightness_caption"`` when the replay caption
    reader resolved it, ``"low_contrast_caption"`` when only the local
    background front end could, and ``"unavailable"`` otherwise.
    """
    reading = nameplate_range_caption(rgb, nameplate_bbox)
    if reading.get("status") != "observed" or reading.get("value") is None:
        return None, "unavailable"
    return reading["value"], reading["raw"].get("method", "unknown")


# The reticle prints the salvo's landing point beside the caret. On this client
# it sits to the right of the centre, within this window on both layouts (the
# tuned one at +44 px, the Alt HUD at +10 px).
IMPACT_WINDOW = (10, 140, -40, 40)      # dx0, dx1, dy0, dy1 from the reticle


def _impact_window(rgb, reticle_centre):
    """The reticle window, or ``(None, reason)`` when it falls outside.

    The centre may be the point itself or ``green_reticle``'s evidence dict.
    """
    if isinstance(reticle_centre, dict):
        reticle_centre = reticle_centre.get("centre")
    if reticle_centre is None or reticle_centre[0] is None:
        return None, 'reticle_unobserved'
    cx = int(round(float(reticle_centre[0])))
    cy = int(round(float(reticle_centre[1])))
    x0, x1 = cx + IMPACT_WINDOW[0], cx + IMPACT_WINDOW[1]
    y0, y1 = cy + IMPACT_WINDOW[2], cy + IMPACT_WINDOW[3]
    if x0 < 0 or y0 < 0 or x1 > rgb.shape[1] or y1 > rgb.shape[0]:
        return None, 'reticle_at_frame_edge'
    return rgb[y0:y1, x0:x1], None


def _dash_pair(residual, peak, fraction):
    """Two same-baseline horizontal bars, i.e. the drawn "- -" placeholder.

    The bars are 2..4 px tall and 3..13 px wide (measured 3x2 and 4x2, 8 px
    apart, on the beyond-range frame). One-pixel horizontal rules run through
    this HUD (measured 12..21 px long) and a lone bar is never a placeholder, so
    the pair, their matching size and the gap are what identify it.
    """
    labels, stats, threshold = low_contrast_components(residual, peak, fraction)
    bars = []
    for index in range(1, len(stats)):
        x, y, w, h, area = (int(v) for v in stats[index])
        if not (2 <= h <= 4 and 3 <= w <= 13 and w / h >= 1.2 and area >= 5):
            continue
        bars.append((x, y, w, h))
    bars.sort()
    for first, second in zip(bars, bars[1:]):
        ax, ay, aw, ah = first
        bx, by, bw, bh = second
        if abs((ay + ah) - (by + bh)) > 1 or abs(aw - bw) > 3:
            continue
        gap = bx - (ax + aw)
        if 5 <= gap <= 12 and 10 <= bx + bw - ax <= 32:
            return (first, second), threshold
    return None, threshold


def _km_suffix(glyphs, run):
    """True when ``km`` is drawn right after the numeric run.

    The readout's suffix is two letters, so it separates "24.6 km" from the
    "15 s" flight time that sits on the same row; a lone letter is not a unit.
    Measured: 6-7 px from the last digit to the "k", 1 px between "k" and "m".
    """
    right = max(g["bbox"][0] + g["bbox"][2] for g in run)
    base = max(g["bbox"][1] + g["bbox"][3] for g in run)
    letters = sorted((glyph["bbox"] for glyph in glyphs
                      if glyph["bbox"][0] > right + 3),
                     key=lambda box: box[0])
    if not letters:
        return False
    first = letters[0]
    if not (5 <= first[0] - right <= 9 and 8 <= first[3] <= 16 and first[2] <= 10
            and abs(first[1] + first[3] - base) <= 2):
        return False
    for second in letters[1:]:
        gap = second[0] - (first[0] + first[2])
        if 8 <= second[3] <= 14 and second[2] <= 12 and abs(second[1] + second[3] - base) <= 2 \
                and -1 <= gap <= 3:
            return True
    return False


def _low_contrast_readout(window, reader):
    """``(km, detail)`` for the printed reticle distance, or ``(None, detail)``.

    The readout always ends in ``km``, so the run must carry that two-letter
    suffix: "15 s" sits on the same row and a bare number would otherwise be
    read as a range. Hole topology is required here because this face draws a
    "6" whose counter is the only thing separating it from a "5" or an "8" at
    the 15 px size the reticle uses (measured: 6 vs 5 margin 0.027 without it).
    """
    residual, peak = low_contrast_residual(window)
    if residual is None:
        return None, {"reason": peak}
    seen, details, attempts = {}, {}, []
    for fraction in LOW_CONTRAST_FRACTIONS:
        glyphs, labels, stats, threshold = low_contrast_glyphs(
            residual, peak, fraction, min_height=8, max_height=22, max_width=16)
        attempt = {"fraction": fraction, "threshold": threshold,
                   "glyphs": len(glyphs), "text": None, "reason": None}
        attempts.append(attempt)
        heights = {}
        for glyph in glyphs:
            heights[glyph["bbox"][3]] = heights.get(glyph["bbox"][3], 0) + 1
        if not heights:
            attempt["reason"] = "no_glyphs"
            continue
        digit_height = max(heights, key=lambda row: (heights[row], row))
        runs = [run for run in _baseline_runs(
            [glyph for glyph in glyphs if abs(glyph["bbox"][3] - digit_height) <= 1])
            if len(run) >= 3]
        if not runs:
            attempt["reason"] = "no_numeric_run"
            continue
        run = min(runs, key=lambda row: row[0]["bbox"][0])
        if not _km_suffix(glyphs, run):
            attempt["reason"] = "unit_suffix_missing"
            continue
        left = min(g["bbox"][0] for g in run)
        top = min(g["bbox"][1] for g in run)
        right = max(g["bbox"][0] + g["bbox"][2] for g in run)
        bottom = max(g["bbox"][1] + g["bbox"][3] for g in run)
        reading, reason = _read_canonical(reader, labels, stats,
                                          (left, top, right - left, bottom - top),
                                          decimal=True, maximum=50, unit="km",
                                          match_holes=True)
        if reading is None or reading["status"] != "observed":
            attempt["reason"] = reason or reading["raw"].get("reason")
            continue
        attempt["text"] = reading["text"]
        seen[reading["text"]] = seen.get(reading["text"], 0) + 1
        details[reading["text"]] = {"value": float(reading["value"]),
                                    "confidence": float(reading["confidence"]),
                                    "threshold": threshold}
    text, reason = _consensus(seen)
    if text is None:
        return None, {"reason": reason, "peak": round(peak, 1), "attempts": attempts}
    detail = details[text]
    return detail["value"], {"reason": None, "text": text, "method": "low_contrast_readout",
                             "peak": round(peak, 1), "agreement": seen[text],
                             "threshold": detail["threshold"], "attempts": attempts}


def impact_readout(rgb, reticle_centre):
    """``(state, km)`` for the landing point the reticle prints.

    ``('beyond', None)`` when the readout draws the "- -" placeholder: two
    dashes where the number would be, which is how this HUD says the aim point
    is past maximum range. ``('km', value)`` when a number is drawn, read by the
    absolute-brightness rule first and by the local background front end when
    the frame is dark. ``('unknown', None)`` when neither is drawn or neither
    resolves: no estimate ever stands in for a number the HUD did not draw.
    """
    window, _ = _impact_window(rgb, reticle_centre)
    if window is None:
        return "unknown", None
    if isinstance(reticle_centre, dict):
        reticle_centre = reticle_centre["centre"]
    residual, peak = low_contrast_residual(window, channel="max")
    beyond = False
    if residual is not None:
        for fraction in LOW_CONTRAST_FRACTIONS:
            pair, _ = _dash_pair(residual, peak, fraction)
            if pair is not None:
                beyond = True
                break
    if beyond:
        return "beyond", None
    reader = FireHudReader()
    found = reader.read_aim_distance(rgb, list(reticle_centre))
    if found["status"] == "observed" and found.get("km") is not None:
        return "km", float(found["km"])
    value, _ = _low_contrast_readout(window, reader.numeric_distance)
    if value is None:
        return "unknown", None
    return "km", float(value)
