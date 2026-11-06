#!/usr/bin/env python3
"""Offline target-motion estimation for a locked target.

This module deliberately estimates only two local quantities: radial range rate
from nameplate range intervals, and a horizontal screen displacement after
subtracting a measured global image shift.  The latter is a local projection,
not own-ship or world velocity.  It never authorizes firing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

SCHEMA = "locked_target_motion.v1"
CLI_SCHEMA = "locked_target_motion_cli.v1"
MAX_POINTS = 64
WINDOW_POINTS = 32
MIN_POINTS = 3
MIN_SPAN_S = 1.0
MAX_GAP_S = 0.75
MAX_AGE_S = 0.75
MIN_SHIFT_CONFIDENCE = 0.5
MAX_RATE_UNCERTAINTY = 1.0
MOTION_SEMANTICS = (
    "radial range OLS plus local screen horizontal projection with measured "
    "background-shift subtraction; horizontal velocity is not own-ship or "
    "world velocity and is never a complete motion model"
)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _number(value: Any) -> float | None:
    return float(value) if _finite(value) else None


def _interval(value: Any) -> tuple[float, float] | None:
    """Retain supplied OCR intervals; scalars conservatively assume 0.1 km."""
    if isinstance(value, dict):
        for key in ("interval_km", "range_interval_km", "interval"):
            if key in value:
                value = value[key]
                break
        else:
            value = value.get("value")
    if _finite(value):
        center = float(value)
        if center <= 0.0:
            return None
        return (max(0.0, center - 0.05), center + 0.05)
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(_finite(v) for v in value):
        lo, hi = float(value[0]), float(value[1])
        if lo <= 0.0 or hi < lo:
            return None
        center = (lo + hi) * 0.5
        half_width = (hi - lo) * 0.5
        return (max(0.0, center - half_width), center + half_width)
    return None


def _valid_flight(value: Any) -> bool:
    return _finite(value) and 0.0 < float(value) <= 120.0


def _valid_aim_range(value: Any) -> bool:
    return _finite(value) and float(value) > 0.0


def _valid_identity(value: Any, *, allow_tuple: bool = False) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, int) and not isinstance(value, bool):
        return True
    return (allow_tuple and isinstance(value, tuple) and bool(value)
            and all(_valid_identity(part) for part in value))
def _valid_timestamp(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)



def _ols(points: list[tuple[float, float, float, float]]) -> dict[str, float] | None:
    """Fit center values and return a conservative parameter/prediction model."""
    if len(points) < MIN_POINTS:
        return None
    times = [p[0] for p in points]
    values = [p[1] for p in points]
    lo = min(times)
    hi = max(times)
    span = hi - lo
    if span < MIN_SPAN_S:
        return None
    mean_t = sum(times) / len(times)
    mean_v = sum(values) / len(values)
    denominator = sum((t - mean_t) ** 2 for t in times)
    if denominator <= 1e-12:
        return None
    slope = sum((t - mean_t) * (v - mean_v) for t, v in zip(times, values)) / denominator
    intercept = mean_v - slope * mean_t
    residuals = [v - (intercept + slope * t) for t, v in zip(times, values)]
    if len(points) > 2:
        residual_scale = math.sqrt(sum(r * r for r in residuals) / (len(points) - 2))
    else:
        residual_scale = 0.0
    # Interval endpoints bound the fitted slope.  This prevents the 0.1 km
    # OCR quantization from being reported as spuriously precise motion.
    slope_interval = sum(abs(t - mean_t) * (p[3] - p[2]) * 0.5
                         for t, p in zip(times, points)) / denominator
    rate_uncertainty = slope_interval + residual_scale / max(span, 1e-9)
    return {
        "intercept": intercept,
        "slope": slope,
        "rate_uncertainty": max(0.0, slope_interval, rate_uncertainty),
        "mean_t": mean_t,
        "denominator": denominator,
        "residual_scale": residual_scale,
    }


def _prediction(points: list[tuple[float, float, float, float]], model: dict[str, float], horizon: float) -> tuple[float, list[float]]:
    future_t = max(p[0] for p in points) + horizon
    predicted = model["intercept"] + model["slope"] * future_t
    denominator = model["denominator"]
    mean_t = model["mean_t"]
    # Linear-fit leverage gives an interval bound from every OCR interval.
    parameter_uncertainty = 0.0
    n = len(points)
    for t, _v, lo, hi in points:
        coefficient = (1.0 / n) + (future_t - mean_t) * (t - mean_t) / denominator
        parameter_uncertainty += abs(coefficient) * (hi - lo) * 0.5
    uncertainty = parameter_uncertainty + model["residual_scale"]
    uncertainty = max(uncertainty, model["rate_uncertainty"] * max(horizon, 0.0))
    return predicted, [max(0.0, predicted - uncertainty), predicted + uncertainty]


class LockedTargetMotion:
    """Bounded, reset-on-epoch motion estimator for one currently locked target."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._target_id: Any = None
        self._view_id: Any = None
        self._range_points: list[tuple[float, float, float, float]] = []
        self._horizontal_points: list[tuple[float, float]] = []
        self._horizontal_baseline: tuple[float, float] | None = None
        self._last_captured_ns: int | None = None
        self._last_now_ns: int | None = None
        self._last_target_x: float | None = None

    def _result(
        self,
        *,
        status: str,
        reason: str,
        target_id: Any,
        aim_range_km: float | None,
        horizon_s: float | None = None,
        range_rate: float | None = None,
        range_uncertainty: float | None = None,
        predicted: float | None = None,
        predicted_interval: list[float] | None = None,
        horizontal_velocity: float | None = None,
        horizontal_lead: float | None = None,
    ) -> dict[str, Any]:
        error = None
        if predicted is not None and aim_range_km is not None:
            error = predicted - aim_range_km
        return {
            "schema": SCHEMA,
            "status": status,
            "reason": reason,
            "target_id": target_id,
            "samples": len(self._range_points),
            "range_rate_km_s": range_rate,
            "range_rate_uncertainty_km_s": range_uncertainty,
            "predicted_range_km": predicted,
            "predicted_range_interval_km": predicted_interval,
            "range_quantization_floor_km": min((p[3]-p[2] for p in self._range_points), default=None),
            "aim_range_km": aim_range_km,
            "range_error_km": error,
            "horizontal_velocity_px_s": horizontal_velocity,
            "horizontal_lead_px": horizontal_lead,
            "horizon_s": horizon_s,
            "fire_authorized": False,
            "motion_semantics": MOTION_SEMANTICS,
        }

    def observe(
        self,
        *,
        target_id: Any,
        view_id: Any,
        captured_ns: Any,
        now_ns: Any,
        range_interval_km: Any,
        aim_range_km: Any,
        flight_s: Any,
        locked: Any,
        image_shift: Any = None,
        target_x: Any = None,
        input_delay_s: Any = 0.0,
    ) -> dict[str, Any]:
        """Consume one producer observation and return a JSON-serializable result."""
        aim = _number(aim_range_km) if _valid_aim_range(aim_range_km) else None
        if (not _valid_identity(target_id) or not _valid_identity(view_id, allow_tuple=True)):
            self.reset()
            return self._result(status="rejected", reason="invalid_identity", target_id=target_id, aim_range_km=aim)
        if (not _valid_timestamp(captured_ns) or not _valid_timestamp(now_ns)
                or not _finite(input_delay_s)):
            self.reset()
            return self._result(status="rejected", reason="invalid_time", target_id=target_id, aim_range_km=aim)
        captured = captured_ns
        now = now_ns
        delay = float(input_delay_s)
        if captured <= 0 or now < captured or delay < 0.0:
            self.reset()
            return self._result(status="rejected", reason="invalid_time", target_id=target_id, aim_range_km=aim)
        if not isinstance(locked, bool) or not locked:
            self.reset()
            return self._result(status="rejected", reason="lock_lost", target_id=target_id, aim_range_km=aim)
        if self._last_now_ns is not None and now < self._last_now_ns:
            self.reset()
            return self._result(status="rejected", reason="now_out_of_order", target_id=target_id, aim_range_km=aim)
        if self._last_captured_ns is not None:
            if captured == self._last_captured_ns:
                self.reset()
                return self._result(status="rejected", reason="duplicate_timestamp", target_id=target_id, aim_range_km=aim)
            if captured < self._last_captured_ns:
                self.reset()
                return self._result(status="rejected", reason="timestamp_out_of_order", target_id=target_id, aim_range_km=aim)
        age = (now - captured) / 1e9
        if age > MAX_AGE_S:
            self.reset()
            return self._result(status="rejected", reason="observation_expired", target_id=target_id, aim_range_km=aim)

        epoch_reason = None
        if self._target_id is not None and (target_id != self._target_id or view_id != self._view_id):
            self.reset()
            epoch_reason = "target_or_view_changed"
        elif self._last_captured_ns is not None and (captured - self._last_captured_ns) / 1e9 > MAX_GAP_S:
            self.reset()
            epoch_reason = "sample_gap_exceeded"
        self._target_id = target_id
        self._view_id = view_id
        self._last_captured_ns = captured
        self._last_now_ns = now

        interval = _interval(range_interval_km)
        if interval is None:
            self._range_points.clear()
            range_reason = "range_invalid"
        else:
            t = captured / 1e9
            self._range_points.append((t, (interval[0] + interval[1]) * 0.5, interval[0], interval[1]))
            if len(self._range_points) > WINDOW_POINTS:
                del self._range_points[:-WINDOW_POINTS]
            range_reason = None

        # A reliable global shift is required for each horizontal displacement.
        # Missing/weak shifts deliberately rebuild history, so a static screen
        # point cannot be mistaken for zero target velocity.
        x = _number(target_x)
        shift = None
        if isinstance(image_shift, (list, tuple)) and len(image_shift) >= 3 and all(_finite(v) for v in image_shift[:3]):
            if float(image_shift[2]) >= MIN_SHIFT_CONFIDENCE:
                shift = (float(image_shift[0]), float(image_shift[1]))
        elif isinstance(image_shift, dict):
            vals = (image_shift.get("dx"), image_shift.get("dy"), image_shift.get("confidence", image_shift.get("score")))
            if all(_finite(v) for v in vals) and float(vals[2]) >= MIN_SHIFT_CONFIDENCE:
                shift = (float(vals[0]), float(vals[1]))
        if x is None or shift is None:
            self._horizontal_points.clear()
            self._horizontal_baseline = None if x is None else (captured / 1e9, x)
            self._last_target_x = x
            horizontal_reason = "horizontal_shift_missing_or_low_confidence"
        elif self._horizontal_baseline is None or self._last_target_x is None:
            self._horizontal_baseline = (captured / 1e9, x)
            self._last_target_x = x
            horizontal_reason = "horizontal_seed"
        else:
            dt = (captured / 1e9) - self._horizontal_baseline[0]
            relative = (x - self._last_target_x) - shift[0]
            previous_displacement = self._horizontal_points[-1][1] if self._horizontal_points else 0.0
            displacement = previous_displacement + relative
            self._horizontal_points.append((dt, displacement))
            if len(self._horizontal_points) > WINDOW_POINTS:
                del self._horizontal_points[:-WINDOW_POINTS]
            self._last_target_x = x
            horizontal_reason = None

        flight_valid = _valid_flight(flight_s)
        flight = float(flight_s) if flight_valid else None
        if flight is not None:
            horizon = flight + age + delay
        else:
            horizon = None
        model = _ols(self._range_points)
        predicted = predicted_interval = range_rate = rate_uncertainty = None
        if model is not None and horizon is not None:
            range_rate = model["slope"]
            rate_uncertainty = model["rate_uncertainty"]
            predicted, predicted_interval = _prediction(self._range_points, model, horizon)
        horizontal_velocity = horizontal_lead = None
        if len(self._horizontal_points) >= MIN_POINTS:
            hp = self._horizontal_points
            t0 = hp[0][0]
            shifted = [(t - t0, d, d, d) for t, d in hp]
            hm = _ols(shifted)
            if hm is not None:
                horizontal_velocity = hm["slope"]
                if horizon is not None:
                    horizontal_lead = horizontal_velocity * horizon

        if model is None:
            reason = range_reason or "insufficient_range_samples"
            status = "waiting"
        elif horizon is None:
            reason = "invalid_aim_range_or_flight_time" if aim is None or flight is None else "invalid_flight_time"
            status = "waiting"
        elif aim is None:
            reason, status = "aim_range_invalid", "waiting"
        elif horizontal_velocity is None:
            reason = horizontal_reason or "insufficient_horizontal_samples"
            status = "waiting"
        elif rate_uncertainty is not None and rate_uncertainty > MAX_RATE_UNCERTAINTY:
            reason, status = "range_rate_uncertainty_high", "waiting"
        else:
            reason, status = "ready", "ready"
        if epoch_reason and status == "waiting":
            reason = epoch_reason
        return self._result(
            status=status,
            reason=reason,
            target_id=target_id,
            aim_range_km=aim,
            horizon_s=horizon,
            range_rate=range_rate,
            range_uncertainty=rate_uncertainty,
            predicted=predicted,
            predicted_interval=predicted_interval,
            horizontal_velocity=horizontal_velocity,
            horizontal_lead=horizontal_lead,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", required=True, help="JSONL observations, or - for stdin")
    parser.add_argument("--out", required=True, type=Path, help="new JSON report path")
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error(f"refusing to overwrite existing output: {args.out}")
    source = sys.stdin.buffer if args.observations == "-" else Path(args.observations).open("rb")
    raw = bytearray()
    solutions = []
    estimator = LockedTargetMotion()
    try:
        for line in source:
            raw.extend(line)
            if not line.strip():
                continue
            observation = json.loads(line)
            if not isinstance(observation, dict):
                raise ValueError("each observation must be a JSON object")
            solutions.append(estimator.observe(**observation))
    finally:
        if args.observations != "-":
            source.close()
    implementation = Path(__file__).resolve()
    report = {
        "schema": CLI_SCHEMA,
        "input_sha256": hashlib.sha256(bytes(raw)).hexdigest(),
        "implementation_sha256": _sha256(implementation),
        "fire_authorized": False,
        "solutions": solutions,
    }
    args.out.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
