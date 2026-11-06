"""Read the enemy health text drawn on a main-view nameplate.

The HUD prints the target's health as white digits on the red plate, e.g.
``97 200 / 97 200``.  Digit ink is only ~11 px tall in the 2560x1600 capture,
so the field is cut into connected glyphs and every glyph is matched against
native renders of the game's own font (Warhelios TTF) over a small size and
offset grid.  A value is returned only when every glyph has a clear winner and
the slash separator was found; anything else is reported as ``unknown`` with a
reason.  Nothing is inferred from partial evidence.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
FONT_DIR = ROOT / "state/toalba/gamedata/data/gui/fonts"
FONT_FILES = ("Warhelios_Regular.ttf", "Warhelios_Bold.ttf")

# Binarisation is expressed as a fraction of the measured text/plate contrast so
# it survives different HUD scales and red fills.
DIGIT_THRESHOLD_FRACTION = 0.45
SEPARATOR_THRESHOLD_FRACTION = 0.18
MIN_CONTRAST = 45.0
MAX_GLYPH_HEIGHT = 64

# Digit acceptance gates.  Measured on live frames: correct digits score
# 0.77..0.98 and their runner-up trails by at least 0.02.
MIN_DIGIT_SCORE = 0.60
MIN_DIGIT_MARGIN = 0.02


def _unknown(reason: str, **extra: Any) -> dict[str, Any]:
    result = {"status": "unknown", "current": None, "maximum": None, "reason": reason}
    result.update(extra)
    return result


def _ok(current: int, maximum: int, text: str, confidence: float) -> dict[str, Any]:
    return {"status": "ok", "current": int(current), "maximum": int(maximum),
            "reason": "ok", "text": text, "confidence": round(float(confidence), 4)}


@lru_cache(maxsize=None)
def _glyph_templates(size: int) -> dict[str, tuple[np.ndarray, ...]]:
    """Return alpha masks of ``0``..``9`` rendered at ``size`` px in every HUD font."""
    templates: dict[str, list[np.ndarray]] = {}
    for name in FONT_FILES:
        path = FONT_DIR / name
        if not path.exists():
            continue
        font = ImageFont.truetype(str(path), int(size))
        for char in "0123456789":
            canvas = Image.new("L", (size * 3 + 8, size * 3 + 8))
            ImageDraw.Draw(canvas).text((size, size // 2), char, font=font, fill=255)
            alpha = np.asarray(canvas).astype(np.float32) / 255.0
            ys, xs = np.nonzero(alpha > 0.06)
            if len(ys) == 0:
                continue
            templates.setdefault(char, []).append(
                alpha[ys.min():ys.max() + 1, xs.min():xs.max() + 1])
    return {char: tuple(t) for char, t in templates.items()}


def _fonts_available() -> bool:
    return any((FONT_DIR / name).exists() for name in FONT_FILES)


def _match(observation: np.ndarray, template: np.ndarray) -> float:
    if template.shape[0] > observation.shape[0] or template.shape[1] > observation.shape[1]:
        return -2.0
    response = cv2.matchTemplate(observation.astype(np.float32), template.astype(np.float32),
                                 cv2.TM_CCORR_NORMED)
    value = float(np.nanmax(response))
    return value if np.isfinite(value) else -2.0


def _classify_digit(observation: np.ndarray, height: int) -> tuple[str, float, float]:
    """Best digit, its score, and the score of the runner-up digit."""
    scores: dict[str, float] = {}
    # Measured on live plates: an 11 px ink height matches a 15 px Warhelios
    # render.  Wider windows let tiny chunky renders win on any glyph, so the
    # scale is pinned to the measured height.
    for size in range(max(6, height + 2), min(MAX_GLYPH_HEIGHT, height + 6) + 1):
        for char, templates in _glyph_templates(size).items():
            best = max(_match(observation, template) for template in templates)
            if best > scores.get(char, -3.0):
                scores[char] = best
    if not scores:
        return "", -2.0, -2.0
    ordered = sorted(scores.items(), key=lambda item: -item[1])
    if len(ordered) == 1:
        return ordered[0][0], ordered[0][1], -2.0
    return ordered[0][0], ordered[0][1], ordered[1][1]


def _merge_fragments(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """Join pieces that share columns (antialiased glyphs split by the threshold)."""
    merged: list[list[int]] = []
    for x0, y0, x1, y1 in sorted(boxes):
        for target in merged:
            if not (x0 >= target[2] or x1 <= target[0]):
                target[0] = min(target[0], x0)
                target[1] = min(target[1], y0)
                target[2] = max(target[2], x1)
                target[3] = max(target[3], y1)
                break
        else:
            merged.append([x0, y0, x1, y1])
    return [tuple(b) for b in merged]


def _split_touching(mask: np.ndarray, box: tuple[int, int, int, int],
                    median_width: float) -> list[tuple[int, int, int, int]]:
    """Split a component that spans several touching glyphs at its emptiest columns."""
    x0, y0, x1, y1 = box
    width = x1 - x0
    if median_width <= 0 or width <= 1.55 * median_width:
        return [box]
    parts = max(2, int(round(width / median_width)))
    columns = mask[y0:y1, x0:x1].sum(axis=0)
    cuts = [0]
    for index in range(1, parts):
        target = int(round(index * width / parts))
        low = max(cuts[-1] + 2, target - 1)
        high = min(width - 2, target + 1)
        if low >= high:
            low, high = max(1, target - 1), min(width - 1, target + 1)
        segment = columns[low:high + 1]
        cuts.append(low + int(np.argmin(segment)))
    cuts.append(width)
    return [(x0 + cuts[i], y0, x0 + cuts[i + 1], y1) for i in range(len(cuts) - 1)]


def _components(mask: np.ndarray, width: int, height: int) -> list[tuple[int, int, int, int]]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes: list[tuple[int, int, int, int]] = []
    for index in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[index])
        if area < 2 or h < 3 or w < 1:
            continue
        touches = x <= 1 or y <= 1 or x + w >= width - 1 or y + h >= height - 1
        if touches and (w >= 0.55 * width or h >= 0.7 * height):
            continue  # plate border / fill, not text
        if w >= 0.9 * width:
            continue
        boxes.append((x, y, x + w, y + h))
    return boxes


def _separator(mask: np.ndarray, text_top: int, text_bottom: int) -> tuple[list[tuple[int, int, int, int]], str]:
    """Find the diagonal ``/`` between the two numbers on a low-threshold mask."""
    height = mask.shape[0]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    found: list[tuple[int, int, int, int]] = []
    for index in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[index])
        if area < 3 or h < 4:
            continue
        if x <= 1 or y <= 1 or x + w >= mask.shape[1] - 1 or y + h >= height - 1:
            continue
        if w > 0.8 * mask.shape[1] or h >= 0.75 * height:
            continue
        if not (y + h > text_top and y < text_bottom):
            continue
        rows = np.nonzero(labels[y:y + h, x:x + w] == index)[0]
        if len(rows) < 4:
            continue
        centres = []
        widths = []
        for row in rows:
            columns = np.nonzero((labels[y + row, x:x + w] == index))[0]
            centres.append(float(columns.mean()))
            widths.append(len(columns))
        slope = float(np.polyfit(rows, centres, 1)[0]) if len(rows) > 2 else 0.0
        text_height = max(1, text_bottom - text_top)
        # The slash is a thin backslash-ish stroke: narrow, sparse, drifting left.
        if w <= 0.7 * h and max(widths) <= 0.4 * text_height and slope <= -0.22:
            found.append((x, y, x + w, y + h))
    if not found:
        return [], "separator_missing"
    if len(found) > 1:
        return found, "separator_ambiguous"
    return found, "ok"


def read_hp(frame_rgb: np.ndarray, nameplate_bbox: tuple[int, int, int, int]) -> dict[str, Any]:
    """Read ``current/maximum`` health from the digits inside a nameplate box.

    ``nameplate_bbox`` is the ``(x, y, w, h)`` red plate rectangle as returned by
    ``main_view_perception.detect_nameplates``.  The returned dict always has
    ``status`` (``"ok"`` or ``"unknown"``), ``current``, ``maximum`` and
    ``reason``.  When anything is uncertain the value is ``None``; the reader
    never guesses a digit or a missing glyph.
    """
    if not isinstance(frame_rgb, np.ndarray) or frame_rgb.ndim != 3 or frame_rgb.shape[2] < 3:
        return _unknown("frame_shape")
    if not _fonts_available():
        return _unknown("fonts_missing")
    try:
        x, y, w, h = (int(round(float(v))) for v in nameplate_bbox)
    except (TypeError, ValueError):
        return _unknown("invalid_bbox")
    frame_h, frame_w = frame_rgb.shape[:2]
    if w < 8 or h < 6:
        return _unknown("invalid_bbox")
    if x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
        return _unknown("bbox_outside_frame")

    crop = np.asarray(frame_rgb[y:y + h, x:x + w, :3], dtype=np.float32)
    # Whiteness = weakest colour channel; white ink survives, red fill does not,
    # regardless of the capture's channel order.
    white = crop.min(axis=2)
    background = float(np.median(white))
    peak = float(np.percentile(white, 99))
    contrast = peak - background
    if contrast < MIN_CONTRAST:
        return _unknown("low_contrast", contrast=round(contrast, 1))

    digit_threshold = background + max(24.0, DIGIT_THRESHOLD_FRACTION * contrast)
    mask = (white > digit_threshold).astype(np.uint8)
    # The glyphs must sit on the red plate; a box over blue water or a
    # non-plate HUD element is rejected rather than read as health.
    fill = crop[white <= digit_threshold]
    if fill.shape[0] < 0.15 * crop.shape[0] * crop.shape[1]:
        return _unknown("no_background")
    fill_mean = fill.mean(axis=0)
    redness = float(fill_mean[0] - max(fill_mean[1], fill_mean[2]))
    if redness < 20.0:
        return _unknown("not_red_plate", redness=round(redness, 1))
    boxes = _merge_fragments(_components(mask, w, h))
    if not boxes:
        return _unknown("no_glyphs")
    median_width = float(np.median([b[2] - b[0] for b in boxes]))

    pieces: list[tuple[int, int, int, int]] = []
    for box in boxes:
        pieces.extend(_split_touching(mask, box, median_width))
    heights = [b[3] - b[1] for b in pieces]
    median_height = float(np.median(heights))
    band_centre = 0.5 * (min(b[1] for b in pieces) + max(b[3] for b in pieces))

    for x0, y0, x1, y1 in boxes:
        if not (x0 <= 1 or x1 >= w - 1):
            continue
        # A text-height fragment hugging a side *at the text's own baseline*
        # means the supplied box cut a glyph in half; the visible remainder
        # must not be read as the value.  Marks drawn above the plate (damage
        # numbers) sit clearly off that centre line and are ignored.
        if (0.35 * median_height <= y1 - y0 <= 1.3 * median_height
                and x1 - x0 <= 2.0 * median_width
                and abs(0.5 * (y0 + y1) - band_centre) <= 0.35 * median_height):
            return _unknown("text_clipped")
    pieces = [b for b in pieces
              if b[3] - b[1] >= 0.55 * median_height and b[2] - b[0] >= 2]
    if not pieces:
        return _unknown("no_glyphs")
    pieces.sort()
    text_top = min(b[1] for b in pieces)
    text_bottom = max(b[3] for b in pieces)

    low_threshold = background + max(16.0, SEPARATOR_THRESHOLD_FRACTION * contrast)
    low_mask = (white > low_threshold).astype(np.uint8)
    separators, separator_reason = _separator(low_mask, text_top, text_bottom)
    if separator_reason != "ok":
        return _unknown(separator_reason)
    slash_x = (separators[0][0] + separators[0][2]) / 2.0
    sx0, sy0, sx1, sy1 = separators[0]
    # The slash sometimes survives the digit threshold as its own narrow
    # component; it is the separator, never a digit.
    pieces = [b for b in pieces
              if not (b[0] >= sx0 - 1 and b[2] <= sx1 + 1 and b[1] >= sy0 - 1 and b[3] <= sy1 + 1)]
    if not pieces:
        return _unknown("no_glyphs")
    piece_widths = [b[2] - b[0] for b in pieces]
    median_piece_width = float(np.median(piece_widths))
    if max(piece_widths) > 1.6 * median_piece_width:
        # A glyph this wide still hides two touching digits: never split the
        # widest one by guesswork.
        return _unknown("glyph_width", width=max(piece_widths),
                        median=round(median_piece_width, 1))

    glyphs: list[tuple[tuple[int, int, int, int], str, float]] = []
    for box in pieces:
        x0, y0, x1, y1 = box
        centre = (x0 + x1) / 2.0
        if x0 <= slash_x <= x1:
            return _unknown("glyph_overlaps_separator")
        # Normalised patch with two pixels of background so the matcher can slide.
        left, right = max(0, x0 - 2), min(w, x1 + 2)
        top, bottom = max(0, y0 - 2), min(h, y1 + 2)
        patch = np.clip((white[top:bottom, left:right] - background) / contrast, 0.0, 1.0)
        char, score, runner_up = _classify_digit(patch, y1 - y0)
        if not char:
            return _unknown("no_glyphs")
        if score < MIN_DIGIT_SCORE:
            return _unknown("low_score", score=round(score, 3))
        if score - runner_up < MIN_DIGIT_MARGIN:
            return _unknown("ambiguous_digit", score=round(score, 3),
                            runner_up=round(runner_up, 3))
        glyphs.append((box, char, score))

    left_side = [g for g in glyphs if (g[0][0] + g[0][2]) / 2.0 < slash_x]
    right_side = [g for g in glyphs if (g[0][0] + g[0][2]) / 2.0 > slash_x]
    if not left_side or not right_side:
        return _unknown("missing_side")
    left_side.sort(key=lambda g: g[0][0])
    right_side.sort(key=lambda g: g[0][0])
    # A digit lost between the numbers leaves a double-width hole; the slash
    # must sit one normal advance away from each side.
    if sx0 - left_side[-1][0][2] > median_height:
        return _unknown("glyph_gap", gap=sx0 - left_side[-1][0][2])
    if right_side[0][0][0] - sx1 > median_height:
        return _unknown("glyph_gap", gap=right_side[0][0][0] - sx1)

    values: list[int] = []
    confidences: list[float] = []
    for side in (left_side, right_side):
        for previous, current in zip(side, side[1:]):
            gap = current[0][0] - previous[0][2]
            if gap > median_height:
                return _unknown("glyph_gap", gap=gap)
        digits = "".join(g[1] for g in side)
        if len(digits) > 7:
            return _unknown("too_many_digits")
        values.append(int(digits))
        confidences.extend(g[2] for g in side)

    current, maximum = values
    if maximum <= 0 or current < 0:
        return _unknown("nonsense_value")
    if current > maximum:
        return _unknown("current_exceeds_maximum")
    return _ok(current, maximum, f"{current}/{maximum}", float(np.mean(confidences)))
