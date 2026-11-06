#!/usr/bin/env python3
"""Shot accounting from the enemy nameplate health readout.

A shot is credited only from the client's own numbers: the enemy plate prints
``<current> / <maximum>`` inside its red health bar, and the same ship is read
in the frame before and after the salvo.  ``damage = before - after`` is the
only damage evidence here; damage popups, hit ribbons and the camera are not
consulted, so a script's ``fired`` flag can never turn into a hit claim.

The readout is tiny (7 px per digit, 11-12 px tall) and drawn with the in-game
font at a fixed pixel size.  Glyphs are segmented by connected components of
the bright text (blue channel, which separates the pale digits from the red
bar), merged components are split on the 7 px advance, and every glyph is
matched against aspect-preserving renderings of the bundled Warhelios and
source-Han faces plus the game's numeric atlases.  Stretching the glyphs to a
fixed 20x32 box (as the HUD reader does) is deliberately avoided: it makes a
4 px wide ``1`` look like a ``4``.

Every threshold in a small band is tried and a value is only reported when the
same ``(current, maximum)`` wins at least twice.  Readings that disagree are
reported as unknown rather than guessed, because a guessed 10 000 HP drop is a
false hit.
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _path in (HERE.parent / "replay", HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from fire_hud import _glyph_holes  # noqa: E402
from main_view_perception import detect_nameplates  # noqa: E402

FONT_ROOT = REPO / "state/toalba/gamedata/data/gui/fonts"
FONT_FACES = (
    "Warhelios_Bold.ttf",
    "Warhelios_Regular.ttf",
    "WarheliosKO_Bold.ttf",
    "Source_Han_Sans_CN_Bold_WH.ttf",
)
FONT_SIZES = range(11, 20)
# Feature box shared by glyphs and templates: height is fixed, width is kept
# proportional inside it so narrow digits stay narrow.
FEATURE_HEIGHT, FEATURE_WIDTH = 32, 24
# The plate prints one digit every 7 px inside a number group.
DIGIT_ADVANCE = 7
# Threshold probes relative to the plate's Otsu split of the blue channel.
THRESHOLD_OFFSETS = (-6, 0, 4, 8, 12, 16, 22, 30)
MIN_GLYPH_SCORE = 0.70
MIN_READ_SUPPORT = 2
# A followed target can move this far between the two frames and still be one
# ship; beyond it the plate is treated as a different enemy.
MAX_TRACK_DISTANCE = 500.0

_TEMPLATES: list[tuple[str, np.ndarray, int]] | None = None


def _feature(mask: np.ndarray) -> np.ndarray:
    """Aspect-preserving 32 px high silhouette in a 24 px wide box."""
    mask = np.asarray(mask, dtype=np.uint8)
    height, width = mask.shape
    target_w = max(1, int(round(width * FEATURE_HEIGHT / height)))
    resized = cv2.resize(mask * 255, (min(target_w, FEATURE_WIDTH - 2), FEATURE_HEIGHT),
                         interpolation=cv2.INTER_AREA) > 127
    canvas = np.zeros((FEATURE_HEIGHT, FEATURE_WIDTH), dtype=bool)
    left = (FEATURE_WIDTH - resized.shape[1]) // 2
    canvas[:, left:left + resized.shape[1]] = resized
    return canvas


def _hole_count(mask: np.ndarray) -> int:
    stretched = cv2.resize(np.asarray(mask, dtype=np.uint8), (20, 32),
                           interpolation=cv2.INTER_NEAREST) > 0
    return len(_glyph_holes(stretched))


def _trim(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    return mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _templates() -> list[tuple[str, np.ndarray, int]]:
    """Digit templates: game faces plus the numeric atlases, built once."""
    global _TEMPLATES
    if _TEMPLATES is not None:
        return _TEMPLATES
    templates: list[tuple[str, np.ndarray, int]] = []
    if FONT_ROOT.is_dir():
        for face in FONT_FACES:
            path = FONT_ROOT / face
            if not path.exists():
                continue
            for size in FONT_SIZES:
                font = ImageFont.truetype(str(path), size)
                for stroke in (0, 1):
                    for char in "0123456789":
                        canvas = Image.new("L", (50, 50))
                        ImageDraw.Draw(canvas).text((6, 0), char, font=font, fill=255,
                                                    stroke_width=stroke, stroke_fill=255)
                        mask = _trim(np.asarray(canvas) > 128)
                        if mask is None:
                            continue
                        templates.append((char, _feature(mask), _hole_count(mask)))
        for size in ("medium", "big"):
            meta = FONT_ROOT / size / "WarHeliosCond_Textile.json"
            atlas_path = meta.with_suffix(".png")
            if not meta.exists() or not atlas_path.exists():
                continue
            frames = json.loads(meta.read_text(encoding="utf-8-sig"))["frames"]
            atlas = np.asarray(Image.open(atlas_path).convert("RGBA"))
            for char in "0123456789":
                rect = frames[char]["frame"]
                x, y, w, h = (rect[k] for k in ("x", "y", "w", "h"))
                mask = _trim(atlas[y:y + h, x:x + w, 3] > 110)
                if mask is None:
                    continue
                templates.append((char, _feature(mask), _hole_count(mask)))
    _TEMPLATES = templates
    return templates


def _classify(mask: np.ndarray) -> tuple[str, float] | None:
    """Best digit and its agreement score for a single glyph mask."""
    templates = _templates()
    trimmed = _trim(mask)
    if trimmed is None or trimmed.shape[0] < 5 or trimmed.shape[1] < 3:
        return None
    feature = _feature(trimmed)
    foreground = int(feature.sum())
    if foreground < 6:
        return None
    holes = _hole_count(trimmed)
    scores: dict[str, float] = {}
    for char, template, template_holes in templates:
        overlap = 2 * np.count_nonzero(feature & template) / (foreground + int(template.sum()))
        if template_holes == holes:
            overlap += 0.03
        scores[char] = max(scores.get(char, 0.0), overlap)
    if not scores:
        return None
    char, score = max(scores.items(), key=lambda item: item[1])
    return char, float(score)


def _split_component(mask: np.ndarray, count: int):
    """Split a merged glyph component into `count` digits on the 7 px advance."""
    width = mask.shape[1]
    if count == 1:
        result = _classify(mask)
        return ([result], None) if result else (None, None)
    base = [int(round(i * width / count)) for i in range(1, count)]
    best = None
    for deltas in itertools.product((-1, 0, 1), repeat=count - 1):
        cuts = [b + d for b, d in zip(base, deltas)]
        if len(set(cuts)) < len(cuts) or any(not 0 < c < width for c in cuts):
            continue
        edges = [0] + cuts + [width]
        if any(not 3 <= b - a <= 10 for a, b in zip(edges, edges[1:])):
            continue
        glyphs = []
        total = 0.0
        for a, b in zip(edges, edges[1:]):
            result = _classify(mask[:, a:b])
            if result is None or result[1] < MIN_GLYPH_SCORE:
                break
            glyphs.append(result)
            total += result[1]
        if len(glyphs) == count and (best is None or total > best[0]):
            best = (total, glyphs, cuts)
    return (best[1], best[2]) if best else (None, None)


def _runs(mask: np.ndarray):
    """Connected glyph components grouped into runs separated by wide gaps.

    Returns the runs (each a list of ``(x, y, w, h, area, label)``) and the
    label image, so callers do not repeat the connected-component pass.
    """
    height, width = mask.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    components = []
    for index in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[index])
        if area < 5 or h < max(4, 0.4 * height):
            continue
        components.append((x, y, w, h, area, index))
    components.sort()
    runs: list[list[tuple[int, int, int, int, int, int]]] = []
    for comp in components:
        if runs and comp[0] - (runs[-1][-1][0] + runs[-1][-1][2]) <= 3:
            runs[-1].append(comp)
        else:
            runs.append([comp])
    return runs, labels


def _parse_band(frame: np.ndarray, bbox, threshold: float):
    """Read `current / maximum` at one binarisation threshold.

    Returns ``((current, maximum, mean_score), None)`` or ``(None, reason)``.
    """
    x, y, w, h = (int(round(v)) for v in bbox)
    if w <= 6 or h <= 6:
        return None, "nameplate_too_small"
    band = frame[y + 1:y + h - 1, x + 2:x + w - 2]
    if band.size == 0:
        return None, "empty_nameplate"
    mask = band[:, :, 2] > float(threshold)
    height, width = mask.shape
    # Drop the bright plate frame that survives the crop.
    mask[mask.sum(axis=1) >= width, :] = False
    mask[:, mask.sum(axis=0) >= height] = False
    if int(mask.sum()) < 10:
        return None, "no_hp_text"
    runs, labels = _runs(mask)
    if not runs:
        return None, "no_hp_text"
    groups = []
    for run in runs:
        left = min(comp[0] for comp in run)
        right = max(comp[0] + comp[2] for comp in run)
        run_width = right - left
        if run_width < 3 or run_width > 26:
            return None, "unsupported_glyph_run"
        glyphs = []
        for cx, cy, cw, ch, _area, index in run:
            sub = labels[cy:cy + ch, cx:cx + cw] == index
            digits, _cuts = _split_component(sub, max(1, int(round(cw / DIGIT_ADVANCE))))
            if digits is None:
                return None, "unreadable_glyph"
            glyphs.extend(digits)
        groups.append((left, run_width, glyphs))
    # The '/' separator is the one narrow single-glyph group no digit explains.
    separators = [i for i, (_left, run_width, glyphs) in enumerate(groups)
                  if len(glyphs) == 1 and run_width <= 6 and glyphs[0][1] < 0.80]
    if len(separators) != 1:
        return None, "slash_not_found"
    split = separators[0]
    current = "".join(g[0] for i, group in enumerate(groups) if i < split for g in group[2])
    maximum = "".join(g[0] for i, group in enumerate(groups) if i > split for g in group[2])
    if not current.isdigit() or not maximum.isdigit():
        return None, "hp_not_digits"
    current, maximum = int(current), int(maximum)
    if not (0 < current <= maximum) or maximum < 100:
        return None, "hp_implausible"
    scores = [g[1] for i, group in enumerate(groups) if i != split for g in group[2]]
    return (current, maximum, float(np.mean(scores))), None


def _thresholds(frame: np.ndarray, bbox) -> list[float]:
    x, y, w, h = (int(round(v)) for v in bbox)
    band = frame[y + 1:y + h - 1, x + 2:x + w - 2, 2]
    if band.size == 0:
        return []
    otsu, _ = cv2.threshold(band.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return [float(np.clip(otsu + offset, 90, 175)) for offset in THRESHOLD_OFFSETS]


def read_nameplate_hp(frame: np.ndarray, nameplate_bbox) -> dict:
    """Read the plate's `current / maximum` HP with repeated-threshold support."""
    candidates, reasons = [], []
    for threshold in _thresholds(frame, nameplate_bbox):
        parsed, reason = _parse_band(frame, nameplate_bbox, threshold)
        if parsed is None:
            reasons.append(reason)
        else:
            candidates.append((parsed[0], parsed[1], parsed[2], threshold))
    if not candidates:
        reason = max(set(reasons), key=reasons.count) if reasons else "no_hp_candidates"
        return {"status": "unknown", "hp": None, "maximum": None, "confidence": 0.0,
                "support": 0, "reason": reason, "candidates": []}
    grouped: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for current, maximum, score, threshold in candidates:
        grouped.setdefault((current, maximum), []).append((score, threshold))
    ranked = sorted(grouped.items(),
                    key=lambda item: (len(item[1]), max(s for s, _ in item[1])), reverse=True)
    (current, maximum), support = ranked[0]
    if len(ranked) > 1:
        second_support = len(ranked[1][1])
        first_score = max(s for s, _ in ranked[0][1])
        second_score = max(s for s, _ in ranked[1][1])
        if second_support == len(support) and abs(first_score - second_score) <= 0.02:
            return {"status": "unknown", "hp": None, "maximum": None, "confidence": 0.0,
                    "support": len(support), "reason": "hp_ambiguous",
                    "candidates": [(c, m, round(max(s for s, _ in rows), 3))
                                   for (c, m), rows in ranked]}
    confidence = float(np.mean([s for s, _ in support]))
    status = "observed" if len(support) >= MIN_READ_SUPPORT else "weak"
    return {"status": status, "hp": current, "maximum": maximum,
            "confidence": round(confidence, 3), "support": len(support),
            "reason": "nameplate_hp_text" if status == "observed" else "single_threshold_read",
            "candidates": [(c, m, round(max(s for s, _ in rows), 3))
                           for (c, m), rows in ranked]}


def _plates(frame: np.ndarray) -> list[list[int]]:
    try:
        found = detect_nameplates(frame)
    except (ValueError, cv2.error):
        return []
    return [list(map(int, row["nameplate_bbox"])) for row in found
            if row.get("nameplate_bbox")]


def _centre(bbox) -> tuple[float, float]:
    x, y, w, h = bbox
    return x + w / 2.0, y + h / 2.0


def _distance(first, second) -> float:
    ax, ay = _centre(first)
    bx, by = _centre(second)
    return math.hypot(ax - bx, ay - by)


def _resolve_before(frame: np.ndarray, hint):
    plates = _plates(frame)
    if hint is not None:
        hint = list(map(int, hint))
        if not plates:
            return hint, "hint"
        best = min(plates, key=lambda plate: _distance(plate, hint))
        return (best if _distance(best, hint) <= 80 else hint), "hint"
    if len(plates) == 1:
        return plates[0], "detected"
    if not plates:
        return None, "no_nameplate"
    return None, "ambiguous_nameplates"


def _resolve_after(frame: np.ndarray, before_bbox):
    plates = _plates(frame)
    if not plates:
        return None, "no_nameplate_after"
    best = min(plates, key=lambda plate: _distance(plate, before_bbox))
    if len(plates) > 1:
        ordered = sorted(plates, key=lambda plate: _distance(plate, before_bbox))
        if abs(_distance(ordered[0], before_bbox) - _distance(ordered[1], before_bbox)) < 8:
            return None, "ambiguous_nameplates_after"
    if _distance(best, before_bbox) > MAX_TRACK_DISTANCE:
        return None, "nameplate_lost_after"
    return best, "detected"


def classify_shot(before_frame: np.ndarray, after_frame: np.ndarray,
                  nameplate_bbox_hint=None) -> dict:
    """Classify one salvo from the same enemy plate's HP in both frames.

    ``hit`` is True only when the plate's current HP strictly drops.  Equal HP
    is a miss; a rise, a mismatched maximum or an unreadable read is unknown.
    """
    result = {"hit": None, "hp_before": None, "hp_after": None, "damage": None,
              "reason": "", "before": None, "after": None}
    if before_frame is None or after_frame is None:
        result["reason"] = "frame_missing"
        return result
    before_frame = np.asarray(before_frame)
    after_frame = np.asarray(after_frame)
    before_bbox, why = _resolve_before(before_frame, nameplate_bbox_hint)
    result["before"] = {"bbox": before_bbox, "source": why}
    if before_bbox is None:
        result["reason"] = why
        return result
    after_bbox, why = _resolve_after(after_frame, before_bbox)
    result["after"] = {"bbox": after_bbox, "source": why}
    if after_bbox is None:
        result["reason"] = why
        return result
    before = read_nameplate_hp(before_frame, before_bbox)
    after = read_nameplate_hp(after_frame, after_bbox)
    result["before"].update({k: before[k] for k in ("status", "hp", "maximum", "support", "reason")})
    result["after"].update({k: after[k] for k in ("status", "hp", "maximum", "support", "reason")})
    if before["status"] == "observed":
        result["hp_before"] = before["hp"]
    if after["status"] == "observed":
        result["hp_after"] = after["hp"]
    if before["status"] != "observed" or after["status"] != "observed":
        result["reason"] = "hp_unreadable:%s/%s" % (before["reason"], after["reason"])
        return result
    if before["maximum"] != after["maximum"]:
        result["reason"] = "max_hp_mismatch"
        return result
    damage = int(before["hp"]) - int(after["hp"])
    result["damage"] = damage
    if damage > 0:
        result["hit"] = True
        result["reason"] = "nameplate_hp_drop"
    elif damage == 0:
        result["hit"] = False
        result["reason"] = "nameplate_hp_unchanged"
    else:
        result["hit"] = False
        result["reason"] = "nameplate_hp_increased"
    return result


def main(argv=None) -> int:
    """CLI: account one shot from two saved frames."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("--nameplate", nargs=4, type=int, metavar=("X", "Y", "W", "H"))
    args = parser.parse_args(argv)
    frames = []
    for path in (args.before, args.after):
        image = cv2.imread(path)
        if image is None:
            raise SystemExit("cannot read %s" % path)
        frames.append(np.ascontiguousarray(image[:, :, ::-1]))
    print(json.dumps(classify_shot(frames[0], frames[1], args.nameplate), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
