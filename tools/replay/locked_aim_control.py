"""Bounded single-axis feedback for one image-confirmed locked target.

Only consecutive no-input background flow establishes the natural client-lock
baseline. Input responses subtract its elapsed-time projection; target motion
never calibrates the actuator. All velocities remain local screen quantities.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _interval(value, centre):
    if value is None:
        return [centre, centre]
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    low, high = map(_number, value)
    if low is None or high is None or low <= 0 or high < low or not low <= centre <= high:
        return None
    return [low, high]


@dataclass
class _Pending:
    axis: int
    units: int
    captured_ns: int
    aim_range_km: float
    aim_interval: list
    baseline_rate: float
    baseline_uncertainty: float
    new_frames: int = 0
    shift_x: float = 0.0
    shifts_reliable: bool = True
    delivered_ns: int | None = None
    shift_y: float = 0.0
    pixel_drift_y: float = 0.0
    pixel_uncertainty_y: float = 0.0


class LockedAimControl:
    """Propose a move once, then require its isolated measured response.

    The caller must execute each nonzero move exactly once before the next
    observation, or reset the controller. A skipped or externally mixed input
    invalidates response attribution. This class never authorizes firing.
    """

    X_TOLERANCE_PX = 30.0
    RANGE_TOLERANCE_KM = 0.5
    X_RESPONSE_RESOLUTION_PX = 1.0
    Y_RESPONSE_RESOLUTION_KM = 0.005
    MAX_PENDING_NS = 750_000_000
    MAX_RESPONSE_FRAMES = 3
    MAX_VERTICAL_PROBE_STEPS = 4
    MAX_VERTICAL_PROBE_NS = 4_000_000_000

    def __init__(self):
        self.reset()

    def reset(self):
        self.gain_x_px_per_unit = None
        self.gain_y_km_per_unit = None
        self._target_id = self._view_id = None
        self._captured_ns = self._pixel_hash = None
        self._aim_range = self._aim_interval = None
        self._pending = None
        self._baseline = []
        self._alignment = None
        self._horizontal_error = None
        self._response = None
        self._range_baseline = []
        self._pixel_y_baseline = []
        self._vertical_series = None
        self.gain_y_interval_km_per_unit = None

    def _result(self, status, reason, move=(0, 0)):
        return {
            "status": status, "reason": reason, "move": list(move),
            "gain_x_px_per_unit": self.gain_x_px_per_unit,
            "gain_y_km_per_unit": self.gain_y_km_per_unit,
            "gain_y_interval_km_per_unit": self.gain_y_interval_km_per_unit,
            "horizontal_error_px": self._horizontal_error,
            "horizontal_tolerance_px": self.X_TOLERANCE_PX,
            "range_alignment": self._alignment,
            "no_input_baseline_intervals": len(self._baseline),
            "response": self._response,
            "vertical_probe": None if self._vertical_series is None else dict(self._vertical_series),
            "calibration_semantics": "isolated_input_minus_preinput_no_input_drift_projection",
            "fire_authorized": False,
        }

    def _reject(self, reason):
        self.reset()
        return self._result("rejected", reason)

    @staticmethod
    def _shift(image_shift):
        if isinstance(image_shift, Mapping):
            values = [image_shift.get("dx"), image_shift.get("dy"),
                      image_shift.get("score", image_shift.get("confidence"))]
        elif isinstance(image_shift, (tuple, list)) and len(image_shift) == 3:
            values = image_shift
        else:
            return None
        values = [_number(value) for value in values]
        if any(value is None for value in values) or values[2] < 0.5:
            return None
        return values

    def _drift(self, axis):
        if len(self._baseline) < 2:
            return None
        rates = [row[axis] for row in self._baseline]
        rate = sum(rates) / len(rates)
        # Measured variation and OCR endpoint widths bound extrapolation.
        uncertainty = max(abs(value - rate) + row[2 + axis]
                          for value, row in zip(rates, self._baseline))
        return rate, uncertainty

    def _vertical_drift(self):
        """Endpoint intervals bound range drift; repeated bins do not gain precision."""
        if len(self._range_baseline) < 3 or len(self._pixel_y_baseline) < 2:
            return None
        first, last = self._range_baseline[0], self._range_baseline[-1]
        span = (last[0]-first[0])/1e9
        if span < 1.0:
            return None
        low = (last[1][0]-first[1][1])/span
        high = (last[1][1]-first[1][0])/span
        # Each intermediate cell must admit the same drift interval.
        for stamp, interval in self._range_baseline[1:-1]:
            dt = (stamp-first[0])/1e9
            low = max(low, (interval[0]-first[1][1])/dt)
            high = min(high, (interval[1]-first[1][0])/dt)
        if low > high:
            return None
        y_rate = sum(self._pixel_y_baseline)/len(self._pixel_y_baseline)
        y_uncertainty = max(abs(v-y_rate) for v in self._pixel_y_baseline)
        return (low+high)/2, (high-low)/2, y_rate, y_uncertainty

    def input_completed(self, delivered_ns):
        """Bind the pending proposal to actual delivery, not pre-input capture."""
        pending = self._pending
        if (pending is None or pending.delivered_ns is not None or
                type(delivered_ns) is not int or delivered_ns < pending.captured_ns or
                delivered_ns-pending.captured_ns > self.MAX_PENDING_NS):
            self.reset()
            raise ValueError('invalid pending input delivery timestamp')
        pending.delivered_ns = delivered_ns

    def update(self, *, solution, target_x, reticle_x, aim_range_km,
               captured_ns, view_id, pixel_hash, image_shift=None,
               aim_range_interval_km=None, now_ns=None):
        if not isinstance(solution, Mapping) or solution.get("status") not in ("ready", "waiting"):
            return self._reject("motion_solution_not_usable")
        target_id = solution.get("target_id")
        if (not isinstance(target_id, (str, int)) or isinstance(target_id, bool)
                or target_id == ""):
            return self._reject("missing_target_identity")
        if (not isinstance(view_id, str) or not view_id
                or not isinstance(pixel_hash, str) or not pixel_hash
                or type(captured_ns) is not int or captured_ns <= 0):
            return self._reject("invalid_frame_identity")
        now_ns = captured_ns if now_ns is None else now_ns
        if type(now_ns) is not int or not 0 <= now_ns - captured_ns <= self.MAX_PENDING_NS:
            return self._reject("observation_expired_or_future")
        target_x, reticle_x, aim_range_km = map(_number, (target_x, reticle_x, aim_range_km))
        if any(v is None for v in (target_x, reticle_x, aim_range_km)) or aim_range_km <= 0:
            return self._reject("invalid_sensor_values")
        aim_interval = _interval(aim_range_interval_km, aim_range_km)
        if aim_interval is None:
            return self._reject("invalid_aim_range_interval")
        quantization = _number(solution.get("range_quantization_floor_km"))
        if (quantization is not None and quantization >= 0 and
                quantization + aim_interval[1]-aim_interval[0] > 2*self.RANGE_TOLERANCE_KM+1e-12):
            return self._reject("range_quantization_infeasible")
        if self._target_id is not None and target_id != self._target_id:
            return self._reject("target_changed")
        if self._view_id is not None and view_id != self._view_id:
            return self._reject("view_changed")
        previous_ns = self._captured_ns
        if previous_ns is not None:
            if captured_ns <= previous_ns:
                return self._reject("capture_time_reversed_or_reused")
            if captured_ns - previous_ns > self.MAX_PENDING_NS:
                return self._reject("sample_gap_exceeded")
            if pixel_hash == self._pixel_hash:
                return self._reject("duplicate_pixels")
        pending = self._pending
        if pending is not None:
            response_start = pending.delivered_ns if pending.delivered_ns is not None else pending.captured_ns
            if captured_ns <= response_start:
                self._captured_ns, self._pixel_hash = captured_ns, pixel_hash
                return self._result('waiting', 'capture_precedes_input_completion')
            if captured_ns - response_start > self.MAX_PENDING_NS:
                return self._reject('response_timed_out')
        shift = self._shift(image_shift)
        if pending is None:
            self._range_baseline.append((captured_ns, aim_interval))
            self._range_baseline = self._range_baseline[-16:]
        if pending is None and previous_ns is not None:
            dt = (captured_ns - previous_ns) / 1e9
            if shift is None:
                self._baseline.clear()
                self._range_baseline = [(captured_ns, aim_interval)]
                self._pixel_y_baseline.clear()
            else:
                range_half = (aim_interval[1] - aim_interval[0] +
                              self._aim_interval[1] - self._aim_interval[0]) / 2
                self._baseline.append((shift[0] / dt,
                                       (aim_range_km - self._aim_range) / dt,
                                       0.0, range_half / dt))
                self._pixel_y_baseline.append(shift[1]/dt)
                self._pixel_y_baseline = self._pixel_y_baseline[-3:]
                self._baseline = self._baseline[-3:]
        self._target_id, self._view_id = target_id, view_id
        self._captured_ns, self._pixel_hash = captured_ns, pixel_hash
        self._aim_range, self._aim_interval = aim_range_km, aim_interval
        self._response = None
        # Publish current interval diagnostics even when the motion producer
        # withholds readiness (for example, excessive fitted rate uncertainty).
        # Never retain an earlier frame's apparent alignment during warmup.
        self._alignment = None
        self._horizontal_error = None
        interval = solution.get("predicted_range_interval_km")
        if isinstance(interval, (tuple, list)) and len(interval) == 2:
            low, high = map(_number, interval)
            if low is not None and high is not None and 0 < low <= high:
                error_low = low - aim_interval[1]
                error_high = high - aim_interval[0]
                contained = (error_low >= -self.RANGE_TOLERANCE_KM - 1e-12 and
                             error_high <= self.RANGE_TOLERANCE_KM + 1e-12)
                self._alignment = dict(predicted_interval_km=[low, high],
                                       aim_interval_km=aim_interval,
                                       error_interval_km=[error_low, error_high],
                                       tolerance_km=self.RANGE_TOLERANCE_KM,
                                       contained=contained, uncertain=not contained)
        if pending is not None:
            pending.new_frames += 1
            elapsed = (captured_ns - pending.captured_ns) / 1e9
            if pending.axis == 0:
                if shift is None:
                    pending.shifts_reliable = False
                elif pending.shifts_reliable:
                    pending.shift_x += shift[0]
                    pending.shift_y += shift[1]
                measured = pending.shift_x
                resolution = self.X_RESPONSE_RESOLUTION_PX
                uncertainty = pending.baseline_uncertainty * elapsed
                reliable = pending.shifts_reliable
            else:
                if shift is None:
                    pending.shifts_reliable = False
                elif pending.shifts_reliable:
                    pending.shift_y += shift[1]
                    pending.shift_x += shift[0]
                series = self._vertical_series
                start_range = pending.aim_range_km if series is None else series['start_range_km']
                start_interval = pending.aim_interval if series is None else series['start_interval_km']
                elapsed = (captured_ns-(pending.captured_ns if series is None else series['start_ns']))/1e9
                measured = aim_range_km - start_range
                resolution = self.Y_RESPONSE_RESOLUTION_KM
                uncertainty = (pending.baseline_uncertainty * elapsed +
                               (aim_interval[1] - aim_interval[0] +
                                start_interval[1] - start_interval[0]) / 2)
                reliable = True
            natural = pending.baseline_rate * elapsed
            response = measured - natural
            self._response = dict(axis=pending.axis, measured=measured,
                                  natural_drift_projection=natural, isolated=response,
                                  uncertainty=uncertainty, elapsed_s=elapsed)
            if pending.axis == 1 and self._vertical_series is not None:
                series = self._vertical_series
                local_elapsed = (captured_ns-pending.captured_ns)/1e9
                pixel_response = pending.shift_y-pending.pixel_drift_y*local_elapsed
                pixel_uncertainty = 1.0+pending.pixel_uncertainty_y*local_elapsed
                self._response.update(pixel_response_y=pixel_response,
                    pixel_uncertainty_y=pixel_uncertainty, total_units=series['total_units'],
                    range_response_interval_km=[response-uncertainty,response+uncertainty])
                if captured_ns-series['start_ns'] > self.MAX_VERTICAL_PROBE_NS:
                    return self._reject('vertical_probe_total_deadline')
                if abs(measured) > .5:
                    return self._reject('vertical_probe_range_travel_exceeded')
                visual = (pending.shifts_reliable and abs(pixel_response) > pixel_uncertainty and
                          abs(pending.shift_x) <= max(3.0, abs(pending.shift_y)*.5))
                if visual and series.get('pixel_sign') is not None and pixel_response*series['pixel_sign'] <= 0:
                    return self._reject('vertical_probe_pixel_direction_changed')
                if abs(response) < resolution+uncertainty and visual:
                    if series['steps'] >= self.MAX_VERTICAL_PROBE_STEPS:
                        return self._reject('vertical_probe_quantization_unresolved')
                    if (solution.get('status') != 'ready' or self._alignment is None or
                            self._alignment['contained'] or
                            self._alignment['error_interval_km'][0] <= 0 <= self._alignment['error_interval_km'][1]):
                        return self._reject('vertical_probe_target_margin_lost')
                    series['steps'] += 1
                    series['total_units'] += pending.units
                    series['pixel_sign'] = 1 if pixel_response > 0 else -1
                    self._pending = _Pending(1, pending.units, captured_ns, aim_range_km,
                        aim_interval, pending.baseline_rate, pending.baseline_uncertainty,
                        pixel_drift_y=pending.pixel_drift_y,
                        pixel_uncertainty_y=pending.pixel_uncertainty_y)
                    return self._result('correcting', 'vertical_probe_pixel_confirmed_next_step', [0,pending.units])
                # Range endpoints already carry quantization and drift bounds.
                # A resolved interval is independent evidence even when the
                # water-dominated global flow cannot see a vertical response.
                # Pixel evidence is required only to extend an unresolved probe.
                if not visual and abs(response) < resolution + uncertainty:
                    reliable = False
            if not reliable or abs(response) < resolution + uncertainty:
                if pending.new_frames >= self.MAX_RESPONSE_FRAMES:
                    return self._reject("response_unresolved")
                return self._result("waiting", "awaiting_measurable_isolated_response")
            total_units = pending.units if self._vertical_series is None or pending.axis == 0 else self._vertical_series['total_units']
            gain = response / total_units
            if pending.axis == 0:
                self.gain_x_px_per_unit = gain
            else:
                self.gain_y_km_per_unit = gain
                self.gain_y_interval_km_per_unit = sorted(
                    [(response-uncertainty)/total_units, (response+uncertainty)/total_units])
                self._vertical_series = None
            self._pending = None
            self._baseline.clear()
            self._range_baseline.clear()
            self._pixel_y_baseline.clear()
            return self._result("waiting", "response_measured_rebuilding_no_input_baseline")
        if solution.get("status") != "ready":
            return self._result("waiting", "motion_history_warming")
        lead = _number(solution.get("horizontal_lead_px"))
        range_error = _number(solution.get("range_error_km"))
        horizon = _number(solution.get("horizon_s"))
        interval = solution.get("predicted_range_interval_km")
        if (any(value is None for value in (lead, range_error, horizon)) or horizon <= 0
                or not isinstance(interval, (tuple, list)) or len(interval) != 2):
            return self._reject("incomplete_motion_solution")
        low, high = map(_number, interval)
        if low is None or high is None or low <= 0 or high < low:
            return self._reject("invalid_range_interval")
        self._horizontal_error = target_x + lead - reticle_x
        horizontal_ok = abs(self._horizontal_error) <= self.X_TOLERANCE_PX
        # Readiness is still required before diagnostics may authorize a move.
        error_low, error_high = self._alignment["error_interval_km"]
        range_ok = self._alignment["contained"]
        if horizontal_ok and range_ok:
            return self._result("aligned", "lead_and_range_interval_aligned")
        if not horizontal_ok:
            axis, limit, gain = 0, 150, self.gain_x_px_per_unit
            requested = 20 if gain is None else -self._horizontal_error / gain
        else:
            if error_low <= 0 <= error_high:
                return self._result("waiting", "range_uncertainty_overlaps_aim")
            axis, limit, gain = 1, 3, self.gain_y_km_per_unit
            if gain is not None and self.gain_y_interval_km_per_unit is not None:
                gain = max(self.gain_y_interval_km_per_unit, key=abs)
            requested = 3 if gain is None else range_error / gain
        baseline = self._drift(axis)
        if baseline is None:
            return self._result("waiting", "awaiting_no_input_drift_baseline")
        pixel_drift_y = pixel_uncertainty_y = 0.0
        quantized_y = axis == 1 and aim_interval[1]-aim_interval[0] >= .09
        if quantized_y:
            vertical_drift = self._vertical_drift()
            if vertical_drift is None:
                return self._result('waiting', 'awaiting_quantized_range_drift_baseline')
            baseline = vertical_drift[:2]
            pixel_drift_y, pixel_uncertainty_y = vertical_drift[2:]
        units = round(max(-limit, min(limit, requested)))
        if units == 0:
            return self._result("waiting", "correction_below_control_resolution")
        move = [0, 0]
        move[axis] = units
        self._pending = _Pending(axis, units, captured_ns, aim_range_km,
                                 aim_interval, *baseline, pixel_drift_y=pixel_drift_y,
                                 pixel_uncertainty_y=pixel_uncertainty_y)
        if quantized_y:
            self._vertical_series = dict(start_ns=captured_ns, start_range_km=aim_range_km,
                start_interval_km=list(aim_interval), steps=1, total_units=units)
        self._baseline.clear()
        return self._result("correcting", "probing_isolated_response" if gain is None
                            else "correcting_error", move)
