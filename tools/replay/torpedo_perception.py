"""Conservative enemy torpedo HUD-marker recognition in native full-view RGB.

Evidence is a rendered warning sprite, never a torpedo body, wake, world position,
range, heading, or trajectory. Empty output means no resolved marker evidence.
Native sprites are from the extracted battle_hud/markers/torpedoes assets. Native
1x and capture resampling at 1.25x/1.5x/2x are searched without resizing the frame.
The normal marker in state/tracking-mainview-236374.png supports the 1.5x case;
other families currently have asset-composite, not real-positive, coverage.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, lru_cache
import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

FRAME_SHAPE = (1600, 2560, 3)
ASSET_ROOT = Path(__file__).resolve().parent / "torpedo_assets"
SCALES = (1.0, 1.25, 1.5, 2.0)
FAMILIES = ("normal", "deepwater", "acoustic", "magnetic", "not_dangerous")
# Shape agreement is necessary but insufficient: color, complete surroundings,
# and support/background contrast must also agree with the rendered sprite.
MIN_SCORE = 0.91
SUBTYPE_MARGIN = 0.025
PAD = 3


@dataclass(frozen=True)
class _Template:
    family: str
    path: str
    sha256: str
    scale: float
    shape: np.ndarray
    support: np.ndarray
    background: np.ndarray
    width: int
    height: int
    top_padding: int

    @cached_property
    def color_support(self) -> tuple[int, int, np.ndarray, int]:
        x, y, width, height = cv2.boundingRect(self.support.view(np.uint8))
        core = self.support[y:y+height, x:x+width].astype(np.uint8)
        # ceil(.80 * N), without a floating-point boundary decision.
        required = (4 * cv2.countNonZero(core) + 4) // 5
        return x, y, core, required


@lru_cache(maxsize=1)
def _templates() -> tuple[_Template, ...]:
    result = []
    for family in FAMILIES:
        path = ASSET_ROOT / f"icon_torpedo_{family}_enemy.png"
        if not path.is_file():
            continue  # Missing native evidence must never enable a contour fallback.
        with Image.open(path) as image:
            rgba = np.asarray(image.convert("RGBA"))
        alpha = rgba[..., 3].astype(np.float32) / 255.0
        color = rgba[..., :3].astype(np.float32)
        native = np.maximum(0, color[..., 0] - np.maximum(color[..., 1], color[..., 2])) * alpha
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        for scale in SCALES:
            width, height = round(rgba.shape[1] * scale), round(rgba.shape[0] * scale)
            shape = cv2.resize(native, (width, height), interpolation=cv2.INTER_LINEAR)
            # The circular not-dangerous sprite also matches the dot of a
            # spotted-alert exclamation mark. Its evidence must include the
            # upper glyph context, not just the dot's immediate border.
            top_padding = PAD + 3 * height if family == "not_dangerous" else PAD
            shape = np.pad(shape, ((top_padding, PAD), (PAD, PAD)))
            support = shape >= shape.max() * 0.55
            background = shape <= shape.max() * 0.10
            result.append(_Template(family, path.name, digest, scale, shape,
                                    support, background, width, height, top_padding))
    return tuple(result)


def asset_manifest() -> list[dict]:
    """Hashes and native sizes of the available recognition evidence."""
    return [{"family": t.family, "asset": t.path, "sha256": t.sha256,
             "native_size": [t.width, t.height]} for t in _templates() if t.scale == 1.0]


def _excluded(extra: np.ndarray | None) -> np.ndarray:
    h, w = FRAME_SHAPE[:2]
    mask = np.zeros((h, w), dtype=bool)
    for x, y, rw, rh in ((0, 0, w, 80), (0, h-90, w, 90),
                        (0, 75, 560, 190), (0, h-360, 360, 360),
                        (1980, 65, 580, 690), (940, h-280, 1020, 280),
                        (1836, 796, 720, 723), (1000, 80, 540, 115)):
        mask[y:y+rh, x:x+rw] = True
    if extra is not None:
        if not isinstance(extra, np.ndarray) or extra.shape != mask.shape or extra.dtype != np.bool_:
            raise ValueError("excluded_mask must be a 1600x2560 boolean array")
        mask |= extra
    return mask


def _overlaps(a: list[int], b: list[int]) -> bool:
    return a[0] < b[0]+b[2] and b[0] < a[0]+a[2] and a[1] < b[1]+b[3] and b[1] < a[1]+a[3]


def _color_feasible(colored: np.ndarray, integral: np.ndarray, total: int,
                    template: _Template) -> bool:
    """Necessary color gate only; never crop the normalized score domain.

    The support rectangle first bounds the number of orange support pixels.
    Only its feasible translations need the tighter, binary-support count.
    The entire padded template still determines valid translations, including
    the not-dangerous sprite's full upper negative context.
    """
    sx, sy, core, required = template.color_support
    if total < required:
        return False
    th, tw = template.shape.shape
    sh, sw = core.shape
    nh, nw = colored.shape[0]-th+1, colored.shape[1]-tw+1
    region = integral[sy:sy+nh+sh, sx:sx+nw+sw]
    counts = region[sh:, sw:] - region[:-sh, sw:]
    counts -= region[sh:, :-sw]
    counts += region[:-sh, :-sw]
    x, y, width, height = cv2.boundingRect((counts >= required).view(np.uint8))
    if not width:
        return False
    counts = cv2.matchTemplate(
        colored[y+sy:y+height+sy+sh-1, x+sx:x+width+sx+sw-1],
        core, cv2.TM_CCORR)
    # Binary correlation has integer mathematical results. Leave a generous
    # sub-integer margin for OpenCV's floating-point correlation roundoff;
    # false positives merely retain the unchanged full NCC computation.
    return float(counts.max()) >= required - 0.5


def detect(rgb: np.ndarray, excluded_mask: np.ndarray | None = None) -> list[dict]:
    """Return spatial sprite evidence; caller exclusions add to mandatory HUD masks.

    Coordinates refer to the unchanged 2560x1600 RGB uint8 capture, including
    letterbox bands. A complete padded template must be visible and unexcluded.
    Competing sprite families retain an unknown subtype and their alternatives.
    """
    if not isinstance(rgb, np.ndarray) or rgb.shape != FRAME_SHAPE or rgb.dtype != np.uint8:
        raise ValueError("rgb must be a 1600x2560x3 uint8 RGB array")
    excluded = _excluded(excluded_mask)
    # Saturating uint8 subtraction is exactly max(0, r-max(g,b)). Keep
    # full-frame work single-channel; widen only accepted candidate patches.
    red = cv2.extractChannel(rgb, 0)
    other = cv2.max(cv2.extractChannel(rgb, 1), cv2.extractChannel(rgb, 2))
    excess = cv2.subtract(red, other)
    signal = excess.astype(np.float32)
    green = cv2.extractChannel(rgb, 1)
    colored = ((cv2.subtract(red, green) > 60) &
               (cv2.subtract(green, cv2.extractChannel(rgb, 2)) > 15)).astype(np.uint8)
    # Components only nominate local search windows. Their contour is never
    # evidence for an object class; every accepted candidate requires a sprite.
    bright = ((excess >= 70) & (red >= 145) & ~excluded).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    candidates = []
    templates = _templates()
    for x, y, width, height, area in stats[1:]:
        if not (3 <= area <= 300 and width <= 30 and height <= 36):
            continue
        x0, y0 = max(0, int(x)-32), max(0, int(y)-36)
        x1, y1 = min(2560, int(x+width)+32), min(1600, int(y+height)+36)
        src = signal[y0:y1, x0:x1]
        local_excluded = excluded[y0:y1, x0:x1].astype(np.uint8)
        integral = cv2.integral(local_excluded)
        local_color = colored[y0:y1, x0:x1]
        color_integral = cv2.integral(local_color)
        color_total = cv2.countNonZero(local_color)
        for template in templates:
            th, tw = template.shape.shape
            if th > src.shape[0] or tw > src.shape[1]:
                continue
            if not _color_feasible(local_color, color_integral, color_total, template):
                continue
            scores = cv2.matchTemplate(src, template.shape, cv2.TM_CCOEFF_NORMED)
            if not np.any(scores >= MIN_SCORE-SUBTYPE_MARGIN):
                continue
            masked = integral[th:, tw:] - integral[:-th, tw:] - integral[th:, :-tw] + integral[:-th, :-tw]
            scores[(masked > 0) | ~np.isfinite(scores)] = -1
            peaks = (scores >= MIN_SCORE-SUBTYPE_MARGIN) & (scores >= cv2.dilate(scores, np.ones((3, 3), np.uint8)))
            for yy, xx in zip(*np.where(peaks)):
                score = float(scores[yy, xx])
                patch = src[yy:yy+th, xx:xx+tw]
                support_mean = float(patch[template.support].mean())
                background_mean = float(patch[template.background].mean())
                if support_mean < 75 or support_mean-background_mean < 55:
                    continue
                # An opaque/bright border or adjacent colored HUD text is not
                # the transparent margin surrounding a standalone warning.
                if float(np.mean(patch[template.background] > 55)) > 0.12:
                    continue
                pixels = rgb[y0+yy:y0+yy+th, x0+xx:x0+xx+tw][template.support].astype(np.int16)
                orange = (pixels[:, 0] > pixels[:, 1]+60) & (pixels[:, 1] > pixels[:, 2]+15)
                if float(orange.mean()) < 0.80:
                    continue
                bbox = [x0+int(xx)+PAD, y0+int(yy)+template.top_padding,
                        template.width, template.height]
                candidates.append({"bbox": bbox, "score": score, "template": template,
                                   "support_mean": support_mean,
                                   "background_mean": background_mean})
    # One screen marker can match several scales or families. Preserve close
    # family alternatives instead of inventing a precise weapon type.
    candidates.sort(key=lambda row: (-row["score"], row["bbox"][1], row["bbox"][0]))
    groups: list[list[dict]] = []
    for candidate in candidates:
        for group in groups:
            if _overlaps(candidate["bbox"], group[0]["bbox"]):
                group.append(candidate)
                break
        else:
            groups.append([candidate])
    result = []
    for group in groups:
        best = group[0]
        if best["score"] < MIN_SCORE:
            continue
        template = best["template"]
        alternatives = sorted({c["template"].family for c in group
                               if best["score"]-c["score"] <= SUBTYPE_MARGIN})
        result.append({"class": "torpedo", "bbox": best["bbox"], "score": best["score"],
                       "evidence": "native_enemy_torpedo_warning_sprite",
                       "marker_type": alternatives[0] if len(alternatives) == 1 else "unknown",
                       "marker_type_candidates": alternatives,
                       "asset": template.path, "asset_sha256": template.sha256,
                       "template_scale": template.scale,
                       "screen_evidence": {"red_excess_support_mean": best["support_mean"],
                                           "red_excess_background_mean": best["background_mean"]},
                       "geometry_status": "marker_only", "world_position": None,
                       "wake_geometry": None, "world_trajectory": None})
    return sorted(result, key=lambda row: (row["bbox"][1], row["bbox"][0]))
