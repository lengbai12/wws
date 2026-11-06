#!/usr/bin/env python3
"""Conservative main-view range evidence and target selection.

Public API
==========
``inspect_candidates(rgb, candidates, *, max_range_km, previous=None,
min_range_km=4.0)`` consumes rows from ``main_view_perception`` (or rows with
its compatible fields) and returns a JSON-serialisable mapping::

    {
      "schema": "target_range_selection.v1",
      "selected": <final hull/BowNet-confirmed candidate or null>,
      "range_selected": <distance/track-qualified candidate or null>,
      "candidates": [<enriched candidate>, ...],
      "provenance": { ... }
    }

``range_selected`` is the hand-off to BowNet: it has passed distance,
tracking, explicit occlusion, and ambiguity gates but may still have
``eligible == false`` because hull confirmation is a later stage. ``selected``
independently selects the nearest fully eligible candidate, retaining a unique
prior identity where possible; a nearer unconfirmed hull cannot hide it.

A candidate's ``distance_eligibility`` is deliberately separate from final
``eligible`` selection.  A readable range only proves that a nameplate's HUD
caption is in bounds; it does not prove that the dark pixels below it are a
ship.  Final selection additionally requires a trackable, unambiguous target
and either explicit current hull confirmation (``body_association ==
"confirmed"``/``hull_confirmed``) or a current BowNet confirmation
(``bow_confirmation.confirmed``).  ``visible_silhouette_candidate`` and
``body_visible`` alone are intentionally insufficient.

Distance qualification uses the centre of the observed evidence interval within
the inclusive ``min_range_km`` .. ``max_range_km`` bounds. A scalar is an exact
interval; OCR captions retain their displayed quantisation interval (half of
the last displayed decimal place) for control uncertainty and ambiguity checks.
The default operating bounds are 4–15 km; callers must explicitly supply their
maximum range. Unknown or malformed evidence is never sorted as nearest.

Provenance is retained in every distance evidence object. The OCR reader uses
the extracted WarHelios glyphs from ``fire_hud.NativeNumericReader``. Caption
localisation excludes unit-sized components; minimum-channel grayscale keeps
anti-aliased white strokes against bright backgrounds without relaxing native
glyph score/margin gates. No replay truth, roster, or inferred distance is used.

Unresolved/occluded hulls are excluded from both selection stages unless a
current BowNet confirmation explicitly exists. Unverified nameplate detections
may enter coarse acquisition, but never final selection. Bow confirmations must
contain both ``confirmed: true`` and ``current: true``. Retention requires one
matching stable identity; overlapping nearest range intervals remain ambiguous.
Real frame regressions use ``main_view_fixtures/hull-aim-2p7km.png`` and
``hull-aim-8p4km.png``; the latter's native plate is ``[1221, 642, 123, 17]``.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np

SCHEMA = "target_range_selection.v1"
DEFAULT_MIN_RANGE_KM = 4.0
DEFAULT_MAX_RANGE_KM = 15.0


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _rgb_frame(rgb: Any) -> Optional[np.ndarray]:
    if rgb is None:
        return None
    frame = np.asarray(rgb)
    if frame.ndim != 3 or frame.shape[2] not in (3, 4):
        raise ValueError("rgb must be HxWx3 or HxWx4")
    if frame.shape[2] == 4:
        frame = frame[..., :3]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def _unknown_distance(reason: str, *, bbox: Optional[list[int]] = None,
                      provenance: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "unknown", "value": None, "interval_km": None,
        "bbox": bbox, "confidence": 0.0,
        "raw": {"reason": reason},
        "provenance": provenance or {"source": "main_view", "method": "unknown"},
    }
    return result


def _caption_groups(frame: np.ndarray, plate_bbox: Iterable[int]) -> tuple[list[tuple[int, int, int, int]], int, int] | None:
    """Locate a numeric group while excluding the adjacent ``公里`` fragments."""
    x, y, w, h = [int(v) for v in plate_bbox]
    top = y + h + 2
    x0 = max(0, x + w // 2 - 70)
    band = frame[top:top + 34, x0:x0 + 140]
    if band.size == 0:
        return None
    p = band.astype(np.int16)
    gray = np.minimum(np.minimum(p[..., 0], p[..., 1]), p[..., 2])
    contrast = gray - cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 3)
    mask = ((gray > 95) & (contrast > 12)).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    glyphs = [(int(xx), int(yy), int(ww), int(hh))
              for xx, yy, ww, hh, area in stats[1:]
              if 9 <= hh <= 15 and 2 <= ww <= 12 and ww <= hh and area >= 8]
    if len(glyphs) < 2:
        return None
    glyphs.sort()
    groups: list[list[tuple[int, int, int, int]]] = []
    for glyph in glyphs:
        baseline = glyph[1] + glyph[3]
        placed = False
        for group in groups:
            group_base = max(g[1] + g[3] for g in group)
            gap = glyph[0] - (group[-1][0] + group[-1][2])
            if abs(baseline - group_base) <= 3 and -2 <= gap <= 6:
                group.append(glyph)
                placed = True
                break
        if not placed:
            groups.append([glyph])
    viable = []
    center = x + w / 2 - x0
    for group in groups:
        group.sort()
        gx = group[0][0]
        right = group[-1][0] + group[-1][2]
        gw = right - gx
        if 14 <= gw <= 70 and len(group) >= 2:
            viable.append((abs((gx + right) / 2 - center), group, gx, right))
    if not viable:
        return None
    _, group, gx, right = min(viable, key=lambda row: (row[0], -len(row[1]), row[2]))
    gy = min(g[1] for g in group)
    gh = max(g[1] + g[3] for g in group) - gy
    return group, x0 + gx, top + gy


def read_distance_km(rgb: Any, plate_bbox: Iterable[int]) -> dict[str, Any]:
    """Read a nameplate's current distance caption with native glyph evidence.

    The returned ``interval_km`` is the displayed-caption uncertainty interval,
    not a physical ranging error.  It is half an LCD/HUD last-place unit for a
    decimal caption and is retained for control uncertainty and ambiguity checks.
    """
    frame = _rgb_frame(rgb)
    bbox = [int(v) for v in plate_bbox]
    provenance = {
        "source": "main_view",
        "coordinate_system": "native_frame_xy_pixels",
        "method": "native_font_stroke_ocr",
        "reader": "fire_hud.NativeNumericReader",
    }
    if frame is None:
        return _unknown_distance("frame_missing", bbox=bbox, provenance=provenance)
    located = _caption_groups(frame, bbox)
    if located is None:
        return _unknown_distance("numeric_caption_not_located", bbox=bbox, provenance=provenance)
    group, gx, gy = located
    x, y, w, h = bbox
    gw = group[-1][0] + group[-1][2] - group[0][0]
    gh = max(g[1] + g[3] for g in group) - min(g[1] for g in group)
    gx, gy = max(0, gx - 1), max(0, gy - 1)
    try:
        from fire_hud import NativeNumericReader
        reader = NativeNumericReader(distance_fonts=True)
        # Colour fringes split thin white captions against bright sky. Keep
        # minimum-channel intensity and native glyph thresholds unchanged.
        crop = frame[gy:gy + gh + 3, gx:gx + gw + 4]
        neutral = np.repeat(crop.min(axis=2)[..., None], 3, axis=2)
        read = reader.read(neutral, [0, 0, neutral.shape[1], neutral.shape[0]],
                           decimal=True, maximum=50, match_holes=True)
        read["bbox"] = [gx, gy, neutral.shape[1], neutral.shape[0]]
        for component in read.get("raw", {}).get("components", []):
            component["bbox"][0] += gx
            component["bbox"][1] += gy
        provenance["preprocessing"] = "minimum_channel_neutral_caption"
        provenance['glyph_geometry'] = 'enclosed_counter_count_and_position; shape_score_and_margin_retained'
    except Exception as exc:  # unavailable assets are an evidence failure, not a guess
        return _unknown_distance("native_reader_unavailable", bbox=[gx, gy, gw + 2, gh + 2],
                                 provenance={**provenance, "error": type(exc).__name__})
    raw = copy.deepcopy(read.get("raw") or {})
    evidence = {
        "status": read.get("status", "unknown"),
        "value": read.get("value") if _finite(read.get("value")) else None,
        "interval_km": None,
        "bbox": list(read.get("bbox") or [gx, gy, gw + 2, gh + 2]),
        "confidence": float(read.get("confidence") or 0.0),
        "raw": raw,
        "provenance": provenance,
    }
    value = evidence["value"]
    if evidence["status"] != "observed" or value is None or value < 0 or value > 50:
        evidence["status"] = "unknown"
        evidence["value"] = None
        evidence["raw"].setdefault("reason", "native_reader_unresolved")
        return evidence
    text = str(raw.get("candidate_text") or read.get("text") or "")
    numeric = text.strip()
    if (not numeric or any(ch not in '0123456789.' for ch in numeric) or
            numeric.count('.') > 1 or numeric.startswith('.') or numeric.endswith('.') or
            not math.isclose(float(numeric), value, rel_tol=0, abs_tol=1e-9)):
        evidence['status'] = 'unknown'
        evidence['value'] = None
        evidence['raw']['reason'] = 'caption_precision_unverified'
        return evidence
    decimal_places = len(numeric.partition('.')[2]) if '.' in numeric else 0
    step = 10.0 ** -decimal_places
    half = step / 2.0
    evidence["interval_km"] = [max(0.0, value - half), value + half]
    return evidence


def _coerce_interval(value: Any) -> Optional[list[float]]:
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(_finite(v) for v in value):
        lo, hi = float(value[0]), float(value[1])
        if lo <= hi and lo >= 0:
            return [lo, hi]
        return None
    if _finite(value) and float(value) >= 0:
        number = float(value)
        return [number, number]
    return None


def _distance_input(candidate: dict[str, Any], frame: Optional[np.ndarray]) -> dict[str, Any]:
    """Normalize supported caller evidence without turning absence into zero."""
    for key in ("distance_interval_km", "range_interval_km"):
        if key in candidate:
            interval = _coerce_interval(candidate.get(key))
            if interval is None:
                return _unknown_distance("malformed_distance_interval")
            return {"status": "observed", "value": candidate.get("distance_km"),
                    "interval_km": interval, "confidence": 1.0,
                    "raw": {"source_field": key},
                    "provenance": {"source": "caller", "method": "provided_interval"}}
    for key in ("distance_evidence", "range_evidence", "distance"):
        if key not in candidate:
            continue
        supplied = candidate[key]
        if isinstance(supplied, dict):
            status = supplied.get("status", "unknown")
            value = supplied.get("value", supplied.get("km", supplied.get("distance_km")))
            interval = _coerce_interval(supplied.get("interval_km", supplied.get("distance_interval_km", value)))
            if status != "observed" or interval is None:
                return _unknown_distance(str((supplied.get("raw") or {}).get("reason", "distance_unknown")),
                                         provenance={"source": "caller", "method": "provided_evidence"})
            result = copy.deepcopy(supplied)
            result.update(status="observed", value=float(value) if _finite(value) else interval[0], interval_km=interval)
            result.setdefault("provenance", {"source": "caller", "method": "provided_evidence"})
            return result
        interval = _coerce_interval(supplied)
        if interval is not None:
            return {"status": "observed", "value": interval[0], "interval_km": interval,
                    "confidence": 1.0, "raw": {"source_field": key},
                    "provenance": {"source": "caller", "method": "provided_scalar"}}
        return _unknown_distance("malformed_distance")
    for key in ("distance_km", "range_km"):
        if key in candidate:
            interval = _coerce_interval(candidate[key])
            if interval is None:
                return _unknown_distance("malformed_distance")
            return {"status": "observed", "value": interval[0], "interval_km": interval,
                    "confidence": 1.0, "raw": {"source_field": key},
                    "provenance": {"source": "caller", "method": "provided_scalar"}}
    bbox = candidate.get("nameplate_bbox")
    if frame is not None and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        return read_distance_km(frame, bbox)
    return _unknown_distance("distance_not_provided")


def _stable_key(candidate: Optional[dict[str, Any]]) -> Optional[tuple[str, str]]:
    if not isinstance(candidate, dict):
        return None
    for field in ("target_id", "track_id", "association_id", "identity"):
        value = candidate.get(field)
        if value is not None and value != "" and value != "unknown":
            return field, str(value)
    return None


def _bool_true(candidate: dict[str, Any], *fields: str) -> bool:
    return any(candidate.get(field) is True for field in fields)


def _ambiguous(candidate: dict[str, Any]) -> bool:
    if candidate.get("ambiguous") is True or candidate.get("association_ambiguous") is True:
        return True
    values = (candidate.get("track_state"), candidate.get("body_status"),
              candidate.get("body_association"), candidate.get("identity_association"),
              candidate.get("association_status"))
    return any(str(value).lower().startswith("ambiguous") or
               str(value).lower() == "unknown_ambiguous" for value in values)


def _trackable(candidate: dict[str, Any]) -> bool:
    if candidate.get("trackable") is True:
        return True
    return candidate.get("track_state") in {"observed", "tracked"}


def _hull_confirmed(candidate: dict[str, Any]) -> bool:
    bow = candidate.get("bow_confirmation")
    if isinstance(bow, dict) and bow.get("confirmed") is True and bow.get("current") is True:
        return True
    if candidate.get("body_status") in {"occluded_or_unresolved", "ambiguous_silhouettes"}:
        return False
    return (_bool_true(candidate, "hull_confirmed") or
            candidate.get("body_association") == "confirmed" or
            candidate.get("body_status") == "confirmed")


def _enrich(candidate: dict[str, Any], distance: dict[str, Any], *, min_range: float,
            max_range: float) -> dict[str, Any]:
    row = copy.deepcopy(candidate)
    row["distance_evidence"] = distance
    interval = distance.get("interval_km")
    center = sum(interval) / 2 if isinstance(interval, list) and len(interval) == 2 else None
    reasons: list[str] = []
    distance_ok = False
    if distance.get("status") != "observed" or not isinstance(interval, list) or len(interval) != 2:
        reasons.append("distance_unknown")
    elif center < min_range:
        reasons.append("range_below_min")
    elif center > max_range:
        reasons.append("range_above_max")
    else:
        distance_ok = True
    row["distance_eligibility"] = {
        "eligible": distance_ok,
        "interval_km": interval,
        "min_range_km": min_range,
        "max_range_km": max_range,
        "strict_lower_bound": False,
        "boundary_semantics": "inclusive_interval_center",
        "center_km": center,
    }
    if not _trackable(row):
        reasons.append("not_trackable")
    if _ambiguous(row):
        reasons.append("ambiguous_target")
    occluded = row.get("body_status") == "occluded_or_unresolved" and not _hull_confirmed(row)
    if occluded:
        reasons.append("hull_occluded_or_unresolved")
    range_eligible = distance_ok and _trackable(row) and not _ambiguous(row) and not occluded
    if not _hull_confirmed(row):
        reasons.append("hull_unconfirmed")
    row["range_eligible"] = range_eligible
    row["eligible"] = range_eligible and _hull_confirmed(row)
    row["reasons"] = reasons
    row["identity_key"] = _stable_key(row)
    return row


def _select(rows: list[dict[str, Any]], previous_key: Optional[tuple[str, str]],
            empty_reason: str) -> tuple[Optional[dict[str, Any]], str]:
    retained = [row for row in rows if previous_key is not None and row["identity_key"] == previous_key]
    if len(retained) == 1:
        return retained[0], "retained_unambiguous_identity"
    if len(retained) > 1:
        return None, "previous_identity_ambiguous"
    if not rows:
        return None, empty_reason
    ordered = sorted(rows, key=lambda row: sum(row["distance_evidence"]["interval_km"]) / 2)
    if len(ordered) > 1:
        first = ordered[0]["distance_evidence"]["interval_km"]
        second = ordered[1]["distance_evidence"]["interval_km"]
        if first[1] >= second[0]:
            return None, "nearest_distance_ambiguous"
    return ordered[0], "nearest_eligible"


def inspect_candidates(rgb: Any, candidates: Iterable[dict[str, Any]], *, max_range_km: float,
                       previous: Optional[dict[str, Any]] = None,
                       min_range_km: float = DEFAULT_MIN_RANGE_KM) -> dict[str, Any]:
    """Enrich and conservatively select one current main-view target."""
    if not _finite(max_range_km) or not _finite(min_range_km):
        raise ValueError("range limits must be finite")
    min_range, max_range = float(min_range_km), float(max_range_km)
    if max_range <= min_range:
        raise ValueError("max_range_km must be greater than min_range_km")
    frame = _rgb_frame(rgb)
    rows = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            rows.append(_enrich({}, _unknown_distance("candidate_not_mapping"), min_range=min_range, max_range=max_range))
            continue
        rows.append(_enrich(candidate, _distance_input(candidate, frame), min_range=min_range, max_range=max_range))
    range_eligible = [row for row in rows if row["range_eligible"]]
    previous_key = _stable_key(previous)
    range_selected, selection_reason = _select(range_eligible, previous_key, "none_range_eligible")
    selected, final_reason = _select([row for row in rows if row["eligible"]],
                                     previous_key, "none_eligible")
    replacement_required = (previous_key is not None and range_selected is not None
                            and range_selected["identity_key"] != previous_key)
    provenance = {
        "source": "main_view",
        "coordinate_system": "native_frame_xy_pixels",
        "distance_policy": "inclusive_interval_center",
        "selection_policy": selection_reason,
        "final_selection_policy": final_reason,
        "ocr_method": "native_font_stroke_ocr_when_caption_not_supplied",
        "replay_truth_used": False,
        "previous_identity_key": previous_key,
        "replacement_required": replacement_required,
    }
    return {"schema": SCHEMA, "selected": selected,
            "range_selected": range_selected, "candidates": rows,
            "provenance": provenance}


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Select a bounded main-view target from JSON candidates")
    parser.add_argument("--candidates", required=True, type=Path, help="JSON list or {candidates: [...]} file")
    parser.add_argument("--image", type=Path, help="optional RGB/RGBA main-view image")
    parser.add_argument("--max-range-km", required=True, type=float)
    parser.add_argument("--min-range-km", type=float, default=DEFAULT_MIN_RANGE_KM)
    parser.add_argument("--previous", type=Path, help="optional JSON previous selected row")
    args = parser.parse_args()
    payload = json.loads(args.candidates.read_text(encoding="utf-8"))
    candidates = payload.get("candidates", []) if isinstance(payload, dict) else payload
    previous = json.loads(args.previous.read_text(encoding="utf-8")) if args.previous else None
    rgb = None
    if args.image:
        from PIL import Image
        with Image.open(args.image) as image:
            rgb = np.asarray(image.convert("RGB"))
    print(json.dumps(inspect_candidates(rgb, candidates, max_range_km=args.max_range_km,
                                        min_range_km=args.min_range_km, previous=previous),
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())


__all__ = ["SCHEMA", "DEFAULT_MIN_RANGE_KM", "DEFAULT_MAX_RANGE_KM",
           "read_distance_km", "inspect_candidates"]
