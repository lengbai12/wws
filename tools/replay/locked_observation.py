"""Current-frame, same-target locked aiming observations; no fire authority.

Ruler geometry supplies an observed layout epoch, never a numeric zoom estimate.
A zoom change that leaves every supported ruler glyph unchanged is not observable
here; upstream camera epochs, when supplied, also reset the local model. Capture
age covers pipeline acquisition, not unknown upstream rendering latency.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time

import numpy as np

from locked_aim_control import LockedAimControl
from locked_target_motion import LockedTargetMotion
from main_view_perception import estimate_camera_shift
from target_confirmation import confirm_target

CAPTURE_SEMANTICS = "gst_segment_running_time;upstream_render_latency_unknown"
MAX_AGE_NS = 750_000_000


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _bbox(candidate):
    box = candidate.get("nameplate_bbox") if isinstance(candidate, dict) else None
    if (not isinstance(box, (list, tuple)) or len(box) != 4
            or not all(_finite(v) for v in box)):
        return None
    x, y, w, h = box
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > 2560 or y + h > 1600:
        return None
    return list(box)


def _aim_interval(distance):
    """Infer precision only from the unique accepted top-level value's glyphs."""
    value = distance["km"]
    matching = []
    for number in (distance.get("raw") or {}).get("numbers", []):
        if not isinstance(number, dict) or number.get("status") != "observed" or number.get("value") != value:
            continue
        text = str((number.get("raw") or {}).get("candidate_text", ""))
        match = re.fullmatch(r"(\d{1,3})(?:\.(\d{1,2}))?(?:km)?", text)
        if match:
            step = 10.0 ** -len(match.group(2)) if match.group(2) else 1.0
            matching.append(step)
    # Native reader supports up to two decimals. Missing glyph evidence remains
    # conservative rather than treating binary float trailing zeroes as precision.
    step = matching[0] if len(matching) == 1 else 0.1
    return [max(0.0, value - step / 2), value + step / 2]


def _ruler_geometry(view, centre):
    if view.get("status") != "observed" or view.get("mode") != "scope_ruler":
        return None
    raw = view.get("raw") or {}
    sides = raw.get("sides")
    ticks = raw.get("ticks_xy")
    if (not isinstance(sides, list) or len(sides) != 2 or
            not isinstance(ticks, list) or len(ticks) < 24):
        return None
    geometry = [float(centre[0]), float(centre[1])]
    for side in sides:
        if not isinstance(side, dict) or side.get("accepted") is not True:
            return None
        spacing, span = side.get("spacing_px"), side.get("span_px")
        if not _finite(spacing) or not _finite(span) or not 12 <= spacing <= 40 or span < 450:
            return None
        geometry.extend([float(spacing), float(span)])
    if any(not isinstance(t, (tuple, list)) or len(t) != 2 or
           not all(_finite(v) for v in t) for t in ticks):
        return None
    geometry.append(float(np.median([t[1] for t in ticks])))
    return geometry


class LockedObservation:
    """Consume native RGB frames and current producer evidence as pure data.

    Nonzero moves are single-use proposals. The caller executes exactly once,
    or resets after any skipped, failed, or unrelated mouse input.
    """

    def __init__(self):
        self._epoch = -1
        self.motion = LockedTargetMotion()
        self.control = LockedAimControl()
        self.reset()

    def reset(self):
        self.motion.reset()
        self.control.reset()
        self._previous_frame = None
        self._previous_boxes = []
        self._epoch += 1
        self._stamp = self._hash = self._identity = self._geometry = self._target_id = None

    def input_completed(self, delivered_ns):
        self.control.input_completed(delivered_ns)

    def _reject(self, reason, lock=None, locked=False):
        self.reset()
        return dict(status="rejected", reason=reason, move=[0, 0], solution=None,
                    control=None, lock=lock, locked=locked, fire_authorized=False)

    def observe(self, frame, hud, candidates, target, centre, meta):
        frame = np.asarray(frame)
        if (frame.shape != (1600, 2560, 3) or frame.dtype != np.uint8
                or not frame.flags.c_contiguous):
            return self._reject("native_contiguous_rgb_frame_required")
        if (not isinstance(hud, dict) or not isinstance(target, dict) or
                not isinstance(meta, dict) or not isinstance(candidates, list)):
            return self._reject("observation_evidence_missing")
        if (not isinstance(centre, (tuple, list)) or len(centre) != 2 or
                not all(_finite(v) for v in centre)):
            return self._reject("reticle_centre_missing")
        stamp = meta.get("captured_monotonic_ns")
        now = time.monotonic_ns()
        if (type(stamp) is not int or stamp <= 0 or not 0 <= now - stamp <= MAX_AGE_NS
                or meta.get("capture_age_semantics") != CAPTURE_SEMANTICS):
            return self._reject("capture_time_unconfirmed_or_expired")
        digest = meta.get("raw_pixel_sha256")
        if (not isinstance(digest, str) or not digest or
                hashlib.sha256(memoryview(frame)).hexdigest() != digest):
            return self._reject("capture_pixels_unconfirmed")
        if self._stamp is not None and (stamp <= self._stamp or digest == self._hash):
            return self._reject("capture_repeated_or_out_of_order")
        if not meta.get("source") or not meta.get("identity") or not meta.get("window"):
            return self._reject("capture_view_identity_missing")
        try:
            identity = json.dumps([meta["source"], meta["identity"], meta["window"],
                                   meta.get("camera_epoch")], sort_keys=True, allow_nan=False)
        except (TypeError, ValueError):
            return self._reject("capture_view_identity_invalid")
        boxes = [_bbox(c) for c in candidates]
        if any(b is None for b in boxes):
            return self._reject("candidate_nameplate_geometry_missing")
        lock = confirm_target(frame, candidates)
        index = lock.get("detection_index")
        target_id = target.get("target_id")
        locked = (lock.get("confirmed") is True and type(index) is int and
                  0 <= index < len(candidates) and target_id is not None and
                  candidates[index].get("target_id") == target_id and
                  sum(c.get("target_id") == target_id for c in candidates) == 1)
        if not locked:
            return self._reject("same_target_ui_lock_unconfirmed", lock)
        view = hud.get("aiming_view") or {}
        geometry = _ruler_geometry(view, centre)
        if geometry is None:
            return self._reject("current_scope_ruler_geometry_unconfirmed", lock, locked)
        flight = hud.get("flight_time") or {}
        distance = flight.get("distance") or {}
        if (flight.get("status") != "observed" or distance.get("status") != "observed" or
                not _finite(flight.get("seconds")) or not 0 < flight["seconds"] <= 120 or
                not _finite(distance.get("km")) or distance["km"] <= 0):
            return self._reject("current_aim_range_or_flight_time_unconfirmed", lock, locked)
        interval = target.get("distance_interval_km")
        if interval is None:
            interval = (target.get("distance_evidence") or {}).get("interval_km")
        if (not isinstance(interval, (list, tuple)) or len(interval) != 2 or
                not all(_finite(v) for v in interval) or not 0 < interval[0] <= interval[1]):
            return self._reject("target_range_interval_unconfirmed", lock, locked)
        changed = self._identity is not None and self._identity != identity
        if self._geometry is not None:
            tolerances = [3, 3, 1, 4, 1, 4, 3]
            changed |= any(abs(a - b) > tol for a, b, tol in
                           zip(geometry, self._geometry, tolerances))
        if self._stamp is not None and stamp - self._stamp > MAX_AGE_NS:
            changed = True
        if self._target_id is not None and self._target_id != target_id:
            changed = True
        if changed:
            self.motion.reset()
            self.control.reset()
            self._previous_frame = None
            self._previous_boxes = []
            self._epoch += 1
        if self._geometry is None or changed:
            self._geometry = geometry
        view_id = json.dumps([identity, self._epoch], sort_keys=True)
        shift = None
        if self._previous_frame is not None:
            excluded = np.zeros(frame.shape[:2], dtype=bool)
            for x, y, w, h in self._previous_boxes + boxes:
                left, right = max(0, int(x - w)), min(2560, int(x + 2 * w))
                top, bottom = max(0, int(y - 175)), min(1600, int(y + 6 * w))
                excluded[top:bottom, left:right] = True
            shift = estimate_camera_shift(self._previous_frame, frame, exclude=excluded)
        anchor_x = float(boxes[index][0] + boxes[index][2] / 2)
        aim_interval = _aim_interval(distance)
        solution = self.motion.observe(
            target_id=target_id, view_id=view_id, captured_ns=stamp, now_ns=now,
            range_interval_km=interval, aim_range_km=distance["km"],
            flight_s=flight["seconds"], locked=True, image_shift=shift, target_x=anchor_x)
        control = self.control.update(
            solution=solution, target_x=anchor_x, reticle_x=float(centre[0]),
            aim_range_km=distance["km"], aim_range_interval_km=aim_interval,
            captured_ns=stamp, now_ns=now, view_id=view_id, pixel_hash=digest,
            image_shift=shift)
        if control["status"] == "rejected" or solution["status"] == "rejected":
            reason = control["reason"] if control["status"] == "rejected" else solution["reason"]
            result = self._reject(reason, lock, locked)
            result.update(solution=solution, control=control,
                          aim_range_interval_km=aim_interval,
                          target_range_interval_km=list(interval))
            return result
        self._previous_frame = frame.copy()
        self._previous_boxes = boxes
        self._stamp, self._hash, self._identity = stamp, digest, identity
        self._target_id = target_id
        status = "ready" if control["status"] == "correcting" else control["status"]
        return dict(status=status, reason=control["reason"], move=control["move"],
                    solution=solution, control=control, lock=lock, locked=locked,
                    image_shift=list(shift) if shift is not None else None,
                    target_anchor_x=anchor_x, aim_range_km=distance["km"],
                    aim_range_interval_km=aim_interval, flight_s=flight["seconds"],
                    target_range_interval_km=list(interval), view_epoch=self._epoch,
                    view_id=view_id,
                    view_geometry=geometry, zoom=None,
                    view_identity_semantics="observed_ruler_layout_plus_camera_epoch; numeric_zoom_unknown; unchanged_glyph_zoom_unobservable",
                    acquisition_semantics=CAPTURE_SEMANTICS, fire_authorized=False)


def main(argv=None):
    """Inspect a saved native frame without manufacturing capture provenance."""
    import argparse
    from pathlib import Path
    from PIL import Image
    from fire_hud import FireHudReader
    from main_view_perception import detect_nameplates
    from target_range_selection import inspect_candidates

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", required=True, type=Path)
    parser.add_argument("--meta", type=Path, help="original capture JSON; never restamped")
    parser.add_argument("--max-range-km", required=True, type=float)
    args = parser.parse_args(argv)
    frame = np.asarray(Image.open(args.frame).convert("RGB"))
    hud = FireHudReader().detect(frame)
    candidates = [dict(candidate, target_id="saved-frame-plate-" + str(index), trackable=True)
                  for index, candidate in enumerate(detect_nameplates(frame, scope_observed=True))]
    selection = inspect_candidates(frame, candidates, max_range_km=args.max_range_km)
    selected = selection.get("range_selected")
    candidates = selection["candidates"]
    lock = confirm_target(frame, candidates)
    meta = json.loads(args.meta.read_text()) if args.meta else {}
    centre = (hud.get("reticle") or {}).get("center")
    if centre is None:
        centre = ((hud.get("aiming_view") or {}).get("raw") or {}).get("scope_ruler_center") or {}
        centre = centre.get("center")
    observation = LockedObservation().observe(frame, hud, candidates, selected or {}, centre, meta)
    print(json.dumps(dict(schema="locked_observation_saved_frame.v1", frame=str(args.frame),
                          identity_semantics="single_saved_frame_indices; no_cross_frame_identity_claim",
                          hud=hud, lock=lock, selection=selection, observation=observation,
                          fire_authorized=False), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
