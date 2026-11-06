#!/usr/bin/env python3
"""Bounded visual-servo aim: move the camera until a commanded screen point reaches the reticle.

The commanded point is a capture-space pixel on the scene (for example an enemy
nameplate). The loop tracks that patch between steps, probes one axis at a time
to learn the mouse gain from the response the game actually produced, then
iterates until the error is inside tolerance. Firing is opt-in (--fire) and only
happens after two consecutive converged steps on re-read frames.

Default is a dry-run: it verifies nothing live and prints the plan. Execute is
bounded by --seconds, re-verifies game identity and window focus before every
action, and always releases the button it holds. Tracking loss, an unreadable
reticle, or a response too small to steer aborts instead of guessing. This tool
does not identify friend or foe and makes no hit claim.
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'tools/replay'))

import aim_lead
from main_view_perception import estimate_camera_shift, hud_mask
import fire_hud
from fire_hud import green_reticle
import fire_gate

import hud_projection
from probe_aim_input import Probe, ProbeError, game_identity, pointer_state
from autopilot_course import chunk_deltas


_PROJECTION_CACHE = {}

# Relative mouse transports this servo can drive. XTest moves the seat pointer,
# so a chunk larger than the remaining pointer travel is silently clipped and
# every "reposition" fallback below exists for that. libei (gamescope's
# gamescope-0-ei socket) delivers relative motion to the client directly and is
# unaffected by where the pointer sits, so those fallbacks must not fire and a
# single event may carry much more than the pointer-safe 400 units.
RELATIVE_TRANSPORTS = ('xtest', 'ei')
TRANSPORT_ENV = 'WW_RELATIVE_TRANSPORT'
XTEST_TURN_CHUNK_MAX = 400.0
EI_TURN_CHUNK_MAX = 4000.0


def relative_transport_name(probe=None, default='xtest'):
    """Transport actually in use: the probe's own field, else the env handoff."""
    name = getattr(probe, 'relative_transport', None) if probe is not None else None
    if not name:
        name = os.environ.get(TRANSPORT_ENV) or default
    name = str(name).strip().lower()
    return name or default


def uses_ei_transport(probe=None):
    return relative_transport_name(probe) == 'ei'


def apply_relative_transport(probe, transport):
    """Hand the CLI transport to the move layer.

    ``Probe.move`` reads ``relative_transport`` (seeded from the environment at
    construction), so both are set: the env var for a probe created later, the
    attribute for the one already running.
    """
    name = str(transport or 'xtest').strip().lower()
    if name not in RELATIVE_TRANSPORTS:
        raise ValueError(f'unknown relative transport {transport!r}')
    os.environ[TRANSPORT_ENV] = name
    if probe is not None:
        probe.relative_transport = name
    return name


def turn_chunk_limit(args, probe=None):
    """Largest single yaw event allowed for this transport.

    XTest is clamped to the pointer-safe 400 units no matter what the operator
    asked for; libei has no pointer travel to run out of, so ``--turn-chunk``
    (up to 4000) is used as given.
    """
    requested = abs(float(getattr(args, 'turn_chunk', XTEST_TURN_CHUNK_MAX)
                          or XTEST_TURN_CHUNK_MAX))
    if uses_ei_transport(probe):
        return max(1.0, min(requested, EI_TURN_CHUNK_MAX))
    return max(1.0, min(requested, XTEST_TURN_CHUNK_MAX))


def note_transport_skip(report, probe, reason, **extra):
    """Record that a pointer-position fallback was skipped as unnecessary."""
    report.setdefault('transport_skips', []).append(
        dict(reason=reason, transport=relative_transport_name(probe), **extra))


def stamp_transport(report, transport):
    """Tag every step/input row with the transport, leaving probe rows intact."""
    name = str(transport or 'xtest')
    report['relative_transport'] = name
    for section in ('inputs', 'steps'):
        for row in report.get(section) or ():
            if isinstance(row, dict):
                row.setdefault('transport', name)
    return name


def frame_projection(frame):
    """HUD projection for this frame: measured anchors, else declared content.

    Every HUD window in this servo (reticle band, scope ruler, ammo/turret row,
    nameplate mask) is written in reference space and comes through here, so a
    client that renders a smaller surface does not get its HUD read at the
    reference coordinates. A measurement costs ~20 ms, and one frame is read by
    several helpers per step, so the result is memoised against the frame's
    object id plus a cheap fingerprint of the array itself.
    """
    key = (id(frame), frame.shape)
    fingerprint = (frame[64, 64].tobytes() + frame[-64, -64].tobytes()
                   + frame[frame.shape[0] // 2, frame.shape[1] // 2].tobytes())
    cached = _PROJECTION_CACHE.get(key)
    if cached is not None and cached[0] == fingerprint:
        return cached[1]
    projection = hud_projection.resolve(frame)
    if len(_PROJECTION_CACHE) >= 4:
        _PROJECTION_CACHE.clear()
    _PROJECTION_CACHE[key] = (fingerprint, projection)
    return projection

SCOPE_REACQUIRE_UNITS = 6
SCOPE_ZOOM_EST = 4.0
SCOPE_GAIN_EST = 7.0
MIN_GAIN = .15
# A camera gain outside this band (px of screen movement per mouse unit) is not
# a camera: measured values cluster at 1.2-1.7 third person and 4-8 in the scope.
GAIN_SANE_MIN = 0.6
GAIN_SANE_MAX = 9.0
FRAME_MAX_AGE_NS = 750_000_000
# Hard cap on any single vertical camera action.  30-160 unit steps threw the
# view at the sky and back; 6 units is ~7-40 px depending on the view, which the
# fine loop can repeat smoothly.
# The camera-motion limits live in one module so every tool obeys the same
# numbers: no vertical action may jump the view, and a sweep may not walk far
# enough to park the pointer on the screen edge.
from camera_limits import (MAX_VERTICAL_STEP_UNITS, SCAN_BUDGET_UNITS,  # noqa: E402
                           set_vertical_cap)
_VERTICAL_SIGNS = []


def reset_vertical_guard():
    """Start a new aim cycle: forget the previous reversal history."""
    del _VERTICAL_SIGNS[:]


class FrameTimeError(ProbeError):
    def __init__(self, evidence):
        self.evidence = evidence
        super().__init__(f"live frame rejected: {evidence['reason']} "
                         f"at {evidence['checkpoint']} (age_ns={evidence['age_ns']})")


def frame_capture_ns(meta, now_ns, previous_ns=None, *, checkpoint='unspecified', frame=None):
    """Reject unknown, repeated, future or expired live frames before control."""
    captured = (meta or {}).get('captured_monotonic_ns')
    age = now_ns - captured if type(captured) is int else None
    reason = ('missing_or_invalid_timestamp' if age is None else
              'future_timestamp' if age < 0 else
              'expired' if age > FRAME_MAX_AGE_NS else
              'nonadvancing_timestamp' if previous_ns is not None and captured <= previous_ns else None)
    if reason is not None:
        raise FrameTimeError(dict(reason=reason, checkpoint=checkpoint, frame=frame,
                                  captured_monotonic_ns=captured, checked_monotonic_ns=now_ns,
                                  age_ns=age, max_age_ns=FRAME_MAX_AGE_NS,
                                  previous_captured_monotonic_ns=previous_ns))
    return captured


def observed_response(error_before, error_after, move):
    """Screen pixels of error change per mouse unit, or None when unmeasurable."""
    if move == 0:
        return None
    return (error_after - error_before) / move


def plan_move(error, gain, probe=60, reach=400, damping=1.0, *, probe_y=3, reach_y=3):
    """Return (move_x, move_y, reason) for one servo step.

    gain is [gx, gy]; a missing component is probed with a positive pulse on
    that axis, so no sign is assumed before the game has shown one. damping
    below one under-corrects on purpose, because the response to a mouse move
    is not a constant and an over-correcting loop oscillates.
    """
    if gain is None or gain[0] is None:
        return probe, 0, 'probe_x'
    if gain[1] is None:
        return 0, min(probe_y, reach_y), 'probe_y'
    if abs(gain[0]) < MIN_GAIN or abs(gain[1]) < MIN_GAIN:
        return 0, 0, 'gain_too_small'
    move_x = max(-reach, min(reach, -error[0] / gain[0] * damping))
    move_y = max(-reach_y, min(reach_y, -error[1] / gain[1] * damping))
    return int(round(move_x)), int(round(move_y)), 'correct'


def mounts_traversing(battery):
    """True/False when the HUD says whether a turret is still swinging, else None.

    Entering the scope while the guns are still traversing shows where they
    point *now*, not where the reticle is: measured 2026-10-29 the target jumped
    300-900 px the moment the zoom opened, and the run then chased it.
    """
    reload_view = (battery or {}).get('reload') or {}
    turrets = reload_view.get('turrets') or reload_view.get('raw', {}).get('turrets')
    if not turrets:
        status = reload_view.get('status')
        return None if status is None else False
    states = [t.get('status') for t in turrets if isinstance(t, dict)]
    if not states:
        return None
    if any(state == 'traversing' for state in states):
        return True
    if any(state in ('ready', 'reloading') for state in states):
        return False
    return None


def normalise_presses(precheck):
    """Whether the camera must be toggled back before coarse aiming, and why.

    The corner-brightness test can read "binocular" in a dark sky. Pressing
    Shift on that false positive *openes* the scope, and the later "enter scope"
    press then closed it: the run spent its steps in the wrong view (measured
    2026-10-29: -53 px error jumped to 842 px right after the scope step).
    So only a strongly evidenced binocular reading may drive a press, and the
    caller verifies the result instead of trusting the decision.
    """
    if precheck.get('binocular') is not True:
        return 0, 'third_person'
    if precheck.get('reason') != 'corner_darkened_by_binocular_view':
        return 0, 'weak_evidence'
    if float(precheck.get('corner_drop_px') or 0) < 40.0:
        return 0, 'weak_evidence'
    return 1, 'binocular_confirmed'


def plausible_response(response, current, ratio=3.0, ceiling=8.0):
    """Is this measured response a gain, or contamination?

    The plate also moves because the ship itself moves, because a turret is
    still traversing, or because another ship's plate took over. A response that
    is more than `ratio` away from the current estimate (or larger than any real
    camera gain) is not evidence about the gain: blending it destroyed a working
    solution (measured: -3 px/unit jumped to -17.5 and then +37.9, and the error
    walked -15 -> +307 px after the aim had been inside 90 px).
    """
    if response is None:
        return False
    if abs(response) > ceiling:
        return False
    if current is None or current == 0:
        return True
    low, high = sorted((abs(response), abs(current)))
    return low > 0 and (high / low) <= ratio


def update_gain(gain, previous_error, error, move, blend=.5, state=None):
    """Blend the gain with the response this move actually produced.

    Returns the new gain list and the measured responses per axis (None where
    the axis was not moved enough to measure). The response is not constant
    across a battle, so a single probe value must be revised every step.

    A single sign-flipped response is still dropped as contamination, but the
    estimate can be seeded from a contaminated step: a large horizontal sweep
    also lifts or drops the target, and a 3-unit vertical nudge in that same
    step inherits the whole vertical change. When two consecutive responses
    agree with each other and contradict the stored gain, the stored value is
    the wrong one, so it is replaced. Without this the loop moves the wrong way
    on that axis for the rest of the run (measured: y gain seeded +1.57 from a
    150-unit yaw, true response -1.5, error grew -37 -> -80 px).
    """
    measured = [None, None]
    counters = None if state is None else state.setdefault('contradictions', {})
    for axis in (0, 1):
        if move is None or abs(move[axis]) < (5 if axis == 0 else 1) or previous_error is None or error is None:
            continue
        response = observed_response(previous_error[axis], error[axis], move[axis])
        measured[axis] = response
        if response is None or abs(response) < MIN_GAIN:
            continue
        if not plausible_response(response, gain[axis]):
            if state is not None:
                state.setdefault('rejected_gain_jumps', []).append(
                    dict(axis=axis, response=round(response, 3),
                         gain=None if gain[axis] is None else round(gain[axis], 3)))
            measured[axis] = None
            continue
        if gain[axis] is not None and response * gain[axis] < 0:
            if counters is None:
                measured[axis] = None
                continue
            previous = counters.get(axis)
            same_sign = previous is not None and previous[1] * response > 0
            count = (previous[0] + 1) if same_sign else 1
            if count >= 2 and plausible_response(response, gain[axis]):
                # Two agreeing responses against the stored sign: adopt them,
                # but only if the magnitude is in a range a camera can have.
                gain[axis] = response
                counters.pop(axis, None)
                measured[axis] = response
                continue
            counters[axis] = (count, response)
            measured[axis] = None
            continue
        if counters is not None:
            counters.pop(axis, None)
        gain[axis] = (response if gain[axis] is None
                      else gain[axis] * (1 - blend) + response * blend)
    return gain, measured




def _signature(frame):
    """Cheap content signature, for telling two grabbed frames apart."""
    import hashlib
    return hashlib.sha1(np.ascontiguousarray(frame[::16, ::16]).tobytes()).hexdigest()


FAST_VISION = True
# An operator can assert that the game is in battle when this client's HUD layout
# is not in the supported table (measured: 临海 matches only one ammo sprite, so
# the gate never opens). Everything else still has to hold: reticle, nameplate,
# minimap reading. The override is recorded in the report.
ASSUME_BATTLE = False


def reticle_centre(rgb, projection=None):
    from fire_hud import FireHudReader
    projection = projection or frame_projection(rgb)
    reader = FireHudReader(projection=projection)
    if FAST_VISION:
        # Full detect() also reads flight time, aiming view, aim point and the
        # gun-direction rings: ~110 ms per frame, all of it unused while
        # converging. The battle components alone cost ~23 ms and still give the
        # native caret, so accuracy is unchanged; measured step time drops by
        # about 80 ms per step.
        _, _, reticle, battle_ev = reader._battle_components(rgb, projection)
        reticle_centre.last_bbox = reticle.get('bbox')
        battle, weapon = battle_ev.get('active'), None
        if reticle.get('status') == 'observed' and reticle.get('center'):
            return ([float(v) for v in reticle['center']],
                    True if ASSUME_BATTLE else battle, weapon,
                    reticle['raw'].get('search'))
        # The fast path only re-uses the native caret. When that caret is not
        # drawn (measured: guns pointed away from the screen centre, the HUD
        # then shows the white aim chevron and a green direction ring instead)
        # paying for one full read is far cheaper than reporting a healthy
        # frame as "no reticle": the fire gate refuses on that, so the whole
        # engagement dies with the gun loaded.
        hud = reader.detect(rgb)
        reticle = hud['reticle']
        reticle_centre.last_bbox = reticle.get('bbox')
        battle, weapon = (True if ASSUME_BATTLE else hud['battle']['active'],
                          hud['weapon'].get('status'))
        if reticle.get('status') == 'observed' and reticle.get('center'):
            return ([float(v) for v in reticle['center']], battle, weapon,
                    reticle['raw'].get('search'))
        point = hud.get('aim_point') or {}
        if point.get('status') == 'observed' and point.get('raw', {}).get('candidates'):
            centre = point['raw']['candidates'][0].get('center')
            if centre:
                return [float(v) for v in centre], battle, weapon, 'aim_point_chevron'
        green = green_reticle(rgb)
        if green is not None:
            return green['centre'], battle, weapon, 'green_ring'
        from fire_hud import scope_ruler_center
        ruler = scope_ruler_center(rgb)
        if ruler is not None:
            return ruler['center'], True if ASSUME_BATTLE else battle, weapon, 'scope_ruler_center'
        return None, True if ASSUME_BATTLE else battle, weapon, None
    hud = reader.detect(rgb)
    reticle = hud['reticle']
    reticle_centre.last_bbox = reticle.get('bbox')
    battle, weapon = hud['battle']['active'], hud['weapon'].get('status')
    if reticle.get('status') == 'observed' and reticle.get('center'):
        return [float(v) for v in reticle['center']], battle, weapon, reticle['raw'].get('search')
    green = green_reticle(rgb)
    if green is not None:
        return green['centre'], battle, weapon, 'green_ring'
    # The binocular view does not draw the green ring; it draws a ruler whose
    # centre is the aim point (measured 1279 against a 1280 frame centre, while
    # the third-person ruler is asymmetric and refused).
    from fire_hud import scope_ruler_center
    ruler = scope_ruler_center(rgb)
    if ruler is not None:
        return ruler['center'], battle, weapon, 'scope_ruler_center'
    return None, battle, weapon, None


def clamped_delivery(requested, delivered, tolerance=.15):
    """True when the pointer moved less than asked, i.e. an edge ate the delta.

    A clamped relative move rotates nothing, so the step must not be measured
    and its delta must be re-sent from the window centre.
    """
    if delivered is None:
        return False
    return any(abs(d) < abs(r) * (1 - tolerance)
               for d, r in zip(delivered, requested) if r)


def pan_compensated(history, drifts):
    """Remove camera-induced motion from the error history before fitting.

    The error is the target's screen position relative to a fixed reticle, so a
    camera pan moves it exactly as a moving target would. ``drifts`` holds the
    measured scene shift for each consecutive pair (None where unmeasurable).
    A stationary target under a pan then fits ~0 px/s instead of hundreds.
    """
    out = []
    accum_x = accum_y = 0.0
    for index, (stamp, ex, ey) in enumerate(history):
        if index:
            drift = drifts[index-1] if index-1 < len(drifts) else None
            if drift:
                accum_x += float(drift[0])
                accum_y += float(drift[1])
        out.append((stamp, ex - accum_x, ey - accum_y))
    return out


def fit_velocity(samples, minimum=3, min_span_s=1.5):
    """Least-squares slope of error versus time, or None for a short baseline.

    Consecutive-frame differences carry the full per-frame box noise, which is
    what made an earlier lead estimate swing by tens of pixels per second. A fit
    over a longer baseline averages that noise down.
    """
    if len(samples) < minimum:
        return None
    times = [sample[0] for sample in samples]
    span = times[-1] - times[0]
    if span < min_span_s:
        return None
    mean_t = sum(times) / len(times)
    variance = sum((t - mean_t) ** 2 for t in times)
    if variance <= 1e-9:
        return None
    slopes = []
    for axis in (1, 2):
        values = [sample[axis] for sample in samples]
        mean_v = sum(values) / len(values)
        slopes.append(sum((t - mean_t) * (v - mean_v)
                          for t, v in zip(times, values)) / variance)
    return slopes


def blend_velocity(previous, sample, blend=.4):
    """Blend a screen-velocity sample, resetting on a sign flip.

    Frame-to-frame velocity from tracked boxes is noisy, so the lead is only
    applied once several samples agree and the blend resets rather than
    averaging across a sign change.
    """
    if sample is None:
        return previous
    if previous is None:
        return list(sample)
    if any(p * v < 0 for p, v in zip(previous, sample)):
        return list(sample)
    return [p * (1 - blend) + v * blend for p, v in zip(previous, sample)]


def concurrent_input(probe, keys=('w', 'a', 's', 'd', 'Shift_L', 'Control_L', 'x'),
                     settle_s=.5):
    """Signals that somebody else is driving while this tool runs.

    Every live claim this project makes has to be attributable to the tool, so a
    run refuses to start when the pointer moves without an instruction from us
    or when a control key is already held. Returns the list of signals found.
    """
    before = pointer_state(probe.identity['display'])
    time.sleep(settle_s)
    after = pointer_state(probe.identity['display'])
    signals = []
    if (before['x'], before['y']) != (after['x'], after['y']):
        signals.append(dict(kind='pointer_moved_without_input',
                            before=[before['x'], before['y']], after=[after['x'], after['y']]))
    if before['mask'] or after['mask']:
        signals.append(dict(kind='button_held', mask=after['mask']))
    for key in keys:
        try:
            if probe.key_is_down(key):
                signals.append(dict(kind='key_held', key=key))
        except Exception:
            continue
    return signals


def own_ship_drift(previous_frame, frame, floor=.35):
    """Camera-only scene shift between two frames, or None when unmeasurable.

    While the ship turns, the camera follows it, so the target's screen error
    changes without any mouse input. That drift is what flipped a measured gain
    to the wrong sign, so it is measured and removed from the response.
    """
    if previous_frame is None:
        return None
    shift = estimate_camera_shift(previous_frame, frame,
                                  hud_mask(frame.shape[:2], frame_projection(frame)))
    if not shift or len(shift) < 3 or shift[2] < floor:
        return None
    return [float(shift[0]), float(shift[1])]


def usable_axis_shift(shift, axis, minimum=20.0):
    """True when a probe moved the target along the axis it was probing.

    A tracking lock that slides sideways while the vertical axis is probed is
    not a gain measurement, so it is refused instead of poisoning the solve.
    """
    if shift is None:
        return False
    other = 1 - axis
    return (abs(shift[axis]) >= minimum and abs(shift[axis]) > abs(shift[other]))


_flight_reader = None


def bounded_move(probe, dx, dy, vertical_limit, chunk=0):
    """Bound the entire vertical action; never split a large tilt into a burst.

    Two independent guards, both measured as necessary on 2026-10-21:

    * a hard cap of MAX_VERTICAL_STEP_UNITS on any single vertical action.  The
      range/beyond/coarse paths defaulted to 30-160 units, and one such step
      throws the view at the sky and then slams it onto the sea;
    * a flip guard: three direction reversals inside the last eight vertical
      moves is oscillation, not aiming, so further vertical motion is refused
      instead of shaking the camera in place.
    """
    global _VERTICAL_SIGNS
    from camera_limits import clamp_vertical as _clamp_vertical
    limit = max(1, _clamp_vertical(400, vertical_limit))
    actual_y = max(-limit, min(limit, dy))
    if actual_y:
        sign = 1 if actual_y > 0 else -1
        _VERTICAL_SIGNS.append(sign)
        del _VERTICAL_SIGNS[:-8]
        # Six reversals in the last eight moves is near-perfect alternation:
        # that is shaking, not aiming.  A legitimate correction may reverse
        # twice (overshoot, then come back).
        flips = sum(1 for a, b in zip(_VERTICAL_SIGNS, _VERTICAL_SIGNS[1:]) if a != b)
        if flips >= 6:
            probe.report.setdefault('vertical_flips_refused', 0)
            probe.report['vertical_flips_refused'] += 1
            probe.report['vertical_oscillation_stopped'] = True
            actual_y = 0
    if chunk and abs(dx) > chunk:
        # Wine accelerates large single deltas (measured 2026-10-23: the same
        # sign and size of correction produced 0.82 px/unit in one round and
        # then made the residual *worse* by 479 px in the next). Sending the
        # yaw as a run of small pieces keeps the per-event response linear, so
        # one gain stays valid across steps.
        remaining = float(dx)
        first = True
        while abs(remaining) > .5:
            step = max(-chunk, min(chunk, remaining))
            probe.move(step, actual_y if first else 0)
            if first:
                first = False
            probe.report['inputs'][-1]['chunked_move'] = round(step, 2)
            remaining -= step
            time.sleep(.012)
        if actual_y != dy:
            probe.report['inputs'][-1]['requested_relative'] = [dx, dy]
            probe.report['inputs'][-1]['vertical_limited'] = True
        return dx, actual_y
    probe.move(dx, actual_y)
    if actual_y != dy:
        probe.report['inputs'][-1]['requested_relative'] = [dx, dy]
        probe.report['inputs'][-1]['vertical_limited'] = True
    return dx, actual_y


def measure_gain(probe, frame_getter, settle=.45, units=20, min_shift=3.0, *, units_y=3):
    """Measure the camera's mouse gain in this very session and view.

    The gain changes with the camera mode and the zoom, so a value carried over
    from another state is wrong by a factor of three or more. Each axis is
    pulsed forward and back; an axis that does not move the scene is refused
    rather than guessed. Returns gains in screen pixels per mouse unit along
    with the raw responses, so the caller can record and re-check them.
    """
    reset_vertical_guard()   # the probe pulses alternate by design
    base = frame_getter('gain-base')
    shifts = []
    gains = [None, None]
    axis_units = [units, units_y]
    for axis, (dx, dy) in enumerate(((units, 0), (0, units_y))):
        probe.move(dx, dy)
        time.sleep(settle)
        forward = estimate_camera_shift(base, frame_getter(f'gain-{axis}-plus'))
        probe.move(-dx, -dy)
        time.sleep(settle)
        back = estimate_camera_shift(base, frame_getter(f'gain-{axis}-back'))
        response = None if forward is None else float(forward[axis])
        shifts.append(dict(axis=axis, forward=None if forward is None else [round(float(v), 2) for v in forward[:2]],
                           back=None if back is None else [round(float(v), 2) for v in back[:2]],
                           response_px=response, units=axis_units[axis]))
        if response is not None and abs(response) >= min_shift:
            gains[axis] = response / float(axis_units[axis])
    ok = all(g is not None for g in gains)
    return dict(ok=ok, gain=gains, units=units, axis_units=axis_units, min_shift_px=min_shift, axes=shifts,
                reason=None if ok else 'an axis did not move the scene')


def hud_flight_seconds(frame):
    """The client's own shell flight time for the current aim point.

    The HUD prints it next to the reticle ("11s"); the reader decodes it with
    the native-font OCR. It is the real ballistics number, unlike a nominal
    speed, so the lead prefers it whenever it is readable.
    """
    global _flight_reader
    if _flight_reader is None:
        from fire_hud import FireHudReader
        _flight_reader = FireHudReader()
    view = _flight_reader.detect(frame)
    flight = view.get('flight_time') or {}
    # The decoded number lives inside raw.numbers; the top-level entry only
    # carries the shape it matched, and its status may be unknown while a
    # nested number is readable.
    numbers = [row for row in ((flight.get('raw') or {}).get('numbers') or [])
               if row.get('status') == 'observed' and row.get('value') is not None]
    if not numbers:
        return None
    best = max(numbers, key=lambda row: float(row.get('confidence') or 0))
    value = float(best['value'])
    if not 0 < value <= 25:
        return None
    return dict(value=value, text=best.get('text'),
                confidence=round(float(best.get('confidence') or 0), 3),
                source='hud_flight_time')


def scope_empty_action(empty_reads, grace, exits, retries):
    """What an enemy-free scoped frame means: hold, hand back, or give up.

    ``empty_reads`` counts consecutive scoped frames without a trackable enemy,
    ``grace`` absorbs the zoom transition, ``exits`` is how many times the scope
    already handed back to coarse aim in this run.
    """
    if empty_reads < max(1, grace):
        return 'hold'
    return 'give_up' if exits + 1 > max(0, retries) else 'return'


AIM_BUDGET_S = 15.0


def view_jump(drift_px, commanded_px, gain, limit=520.0):
    """True when the scene moved in a way the delivered move cannot explain.

    ``drift_px`` is the measured inter-frame scene shift, ``commanded_px`` the
    delivery actually made and ``gain`` the current pixels-per-unit. A camera
    cut (mode change, cinematic, focus loss) shows up as hundreds of pixels of
    unexplained motion; ship-follow drift and the move itself do not.
    """
    if drift_px is None:
        return False
    for axis in (0, 1):
        expected = 0.0
        if commanded_px:
            factor = abs(gain[axis]) if gain and gain[axis] else 0.0
            expected = abs(commanded_px[axis]) * factor
        if abs(drift_px[axis]) - expected > limit:
            return True
    return False


def vertical_reversals(errors, amplitude, minimum=3):
    """Count large vertical direction flips; a shaking sight never closes.

    Only errors at least ``amplitude`` px count, so ordinary dither around the
    tolerance does not trip it.
    """
    flips, previous = 0, None
    for value in errors:
        if value is None or abs(value) < amplitude:
            previous = None
            continue
        if previous is not None and (previous > 0) != (value > 0):
            flips += 1
        previous = value
    return flips


def moved_but_unchanged(previous_error, error, last_move, min_units=20, epsilon_px=1.0):
    """True when a real move produced no change, i.e. the frame is stale.

    The capture pipeline has latency, so a frame read right after a move can
    still show the pre-move scene. Acting on it would apply the same correction
    twice, which is the oscillation the reference bot avoids by subtracting the
    still-unobserved command. This is the cheap equivalent: notice it and read
    again instead of trusting it.
    """
    if previous_error is None or error is None or last_move is None:
        return False
    if max(abs(last_move[0]), abs(last_move[1])) < min_units:
        return False
    return (abs(error[0] - previous_error[0]) < epsilon_px
            and abs(error[1] - previous_error[1]) < epsilon_px)


def lead_with_horizon(flight_s, configured_s, step_period_s, cap_s=.5):
    """Total lead time = ballistic flight time + the latency horizon.

    The reference bot leads by the aim latency alone (its bullets are hitscan);
    WoWS adds the client's printed flight time on top. The horizon is half the
    observed step period, which is the measured stand-in for the image age plus
    the half input path the reference sums.
    """
    horizon = 0.0
    if step_period_s and step_period_s > 0:
        horizon = min(max(cap_s, 0.0), step_period_s / 2.0)
    base = flight_s if flight_s is not None else (configured_s or 0.0)
    return base + horizon, horizon


def choose_target(candidates, previous_bbox=None):
    """Keep the target we were aiming at instead of the nearest to the reticle.

    A sweep sees several ships; hopping to whichever plate is closest to the
    crosshair makes successive moves fight each other, which is how an earlier
    live run lost the track. Continuity of the plate wins, proximity is only
    the tie-break.
    """
    if not candidates:
        return None
    if previous_bbox is not None:
        cx = previous_bbox[0] + previous_bbox[2] / 2.0
        cy = previous_bbox[1] + previous_bbox[3] / 2.0

        def continuity(row):
            box = row.get('nameplate_bbox') or row.get('body_roi') or [0, 0, 0, 0]
            px = box[0] + box[2] / 2.0
            py = box[1] + box[3] / 2.0
            return math.hypot(px - cx, py - cy)

        return min(candidates, key=lambda row: (continuity(row),
                                                row.get('distance_from_reticle_px', 1e9)))
    return min(candidates, key=lambda row: row.get('distance_from_reticle_px', 1e9))


def scope_blocks_fire(scope_requested, scope_evidenced):
    """True when the caller asked for the scope and the HUD does not show it.

    The verdict comes from ``fire_gate`` so the scope precondition has one
    implementation; the boolean pair is kept as this tool's public signature.
    """
    return fire_gate.scope_blocks(scope_requested, scope_evidenced)


SCOPE_CORNER_ROI = (60, 100, 260, 180)
SCOPE_CENTRE_ROI = (1150, 700, 1410, 900)


def camera_mode_evidence(frame, baseline=None):
    """Tell the binocular view from third person using the frame itself.

    The scope darkens the top corners far below the frame centre; a third-person
    frame has bright sky there (measured on this client: 195 vs 136 third person,
    108 vs 165 scoped). The ratio is used so the test does not depend on how
    bright the map is, and it never claims a zoom level.
    """
    import numpy as np
    if frame is None:
        return dict(status='unknown', reason='no_frame', binocular=None)
    pixels = np.asarray(frame)
    x0, y0, x1, y1 = SCOPE_CORNER_ROI
    cx0, cy0, cx1, cy1 = SCOPE_CENTRE_ROI
    if pixels.shape[0] < max(y1, cy1) or pixels.shape[1] < max(x1, cx1):
        return dict(status='unknown', reason='frame_too_small', binocular=None)
    def roi_mean(box):
        xa, ya, xb, yb = box
        patch = np.asarray(pixels[ya:yb, xa:xb], dtype=np.float32)
        if patch.ndim == 3:
            patch = patch.mean(axis=2)
        return float(patch.mean())
    corner = roi_mean(SCOPE_CORNER_ROI)
    centre = roi_mean(SCOPE_CENTRE_ROI)
    if centre <= 1e-6:
        return dict(status='unknown', reason='blank_frame', binocular=None)
    ratio = corner / centre
    # A dark scene (measured: corner 54, centre 44 in the scope) defeats the
    # ratio alone, so the third-person corner brightness measured in this very
    # session is the reference: the scope darkens those corners far more than
    # any lighting change seen between adjacent frames.
    reference = (baseline or {}).get('corner_mean') if isinstance(baseline, dict) else None
    dropped = reference is not None and corner < reference - 40
    binocular = bool(ratio < .9 or dropped)
    return dict(status='observed', corner_mean=round(corner, 2),
                centre_mean=round(centre, 2), ratio=round(ratio, 3),
                reference_corner_mean=reference,
                corner_drop_px=(None if reference is None else round(reference - corner, 2)),
                binocular=binocular,
                reason='corner_darkened_by_binocular_view' if binocular
                else 'bright_corners_third_person')


def scope_confirmed(hud_evidence, camera_evidence):
    """Scope evidence accepted from either source, never from intent alone."""
    return bool((hud_evidence or {}).get('observed')
                or (camera_evidence or {}).get('binocular'))


def main_battery_fire_evidence(frame):
    """Read selected weapon and reload from the final frame before firing."""
    global _flight_reader
    if _flight_reader is None:
        from fire_hud import FireHudReader
        _flight_reader = FireHudReader()
    hud = _flight_reader.detect(frame)
    return {key: hud.get(key) or {} for key in ('weapon', 'reload')}


def hud_scope_evidence(frame):
    """Whether the client is actually showing the main-gun scope.

    Asking for the scope and having it are different things: the key press can
    be swallowed (an earlier probe measured no visual change). This reads the
    HUD's own evidence, which needs battle HUD, selected main guns, reticle and
    both ruler tick columns.
    """
    global _flight_reader
    if _flight_reader is None:
        from fire_hud import FireHudReader
        _flight_reader = FireHudReader()
    hud = _flight_reader.detect(frame) or {}
    view = hud.get('aiming_view') or {}
    raw = view.get('raw') or {}
    battle_active = (hud.get('battle') or {}).get('active')
    # The binocular vignette is image evidence that the main-gun scope is up,
    # measured from the frame itself. The HUD ruler reading needs both tick
    # columns and has never been observed on this client, so requiring only that
    # reading blocked a scope the player could plainly see.
    # The vignette also darkens synthetic or non-battle frames, so it only counts
    # when the battle HUD itself is confirmed.
    try:
        binocular = bool(camera_mode_evidence(frame).get('binocular'))
    except Exception:
        binocular = False
    return dict(status=view.get('status'), mode=view.get('mode'),
                zoom_confirmed=view.get('zoom_confirmed'),
                reason=raw.get('reason'), binocular=binocular,
                battle_active=battle_active,
                observed=view.get('status') == 'observed')


HULL_OFFSET_PX = 30.0
HULL_OFFSET_PX_KM = 1415.0    # legacy distance scaling, kept for the caption path
HULL_AIM_FRACTION = 0.6


MINIMAP_RECT = (1836, 796, 720, 723)
MINIMAP_KM_PER_PX = .0501   # measured 2026-10-20; matches the printed 11.0/12.6 rings
RANGE_KM_PER_UNIT = .015    # measured 2026-10-19 at the scope zoom: 20 units = 0.3 km
RANGE_TOLERANCE_KM = .15
RANGE_STEP_UNITS = 30


def range_correction(aim_km, target_km, tolerance_km=RANGE_TOLERANCE_KM,
                     km_per_unit=RANGE_KM_PER_UNIT, max_units=RANGE_STEP_UNITS):
    """Vertical units that bring the reticle's ballistic range onto the target.

    The game prints the ballistic solution next to the reticle ("3.6 s / 7.7 km")
    and the target's own range under its nameplate. Comparing those two is the
    only self-consistent way to close elevation: hull-pixel heuristics aimed
    0.3 km short and every shell fell in the water. Aiming further is a negative
    mouse-y move (measured), so the sign is inverted here.

    Returns (units, reason). ``reason`` is 'closed' when the two ranges already
    agree, 'move' when one bounded correction is due, and 'unavailable' when a
    reading is missing -- an unreadable range must never be treated as closed.
    """
    if aim_km is None or target_km is None:
        return 0, 'unavailable'
    if not math.isfinite(float(aim_km)) or not math.isfinite(float(target_km)):
        return 0, 'unavailable'
    delta = float(target_km) - float(aim_km)
    if abs(delta) <= tolerance_km:
        return 0, 'closed'
    if km_per_unit <= 0:
        return 0, 'unavailable'
    units = int(round(-delta / km_per_unit))
    if units == 0:
        units = -1 if delta > 0 else 1
    units = max(-max_units, min(max_units, units))
    return units, 'move'


def target_range(caption_km, operator_km):
    """Target range plus where it came from.

    A read caption always wins: it is the game's own number for the ship being
    aimed at. An operator-supplied value is only a fallback for this HUD, whose
    caption is 8 px tall and unreadable by the font templates, and it is tagged
    so a report can never present it as a measurement.
    """
    if caption_km is not None:
        return float(caption_km), 'nameplate_caption'
    if operator_km is not None:
        return float(operator_km), 'operator'
    return None, 'unavailable'


def hud_aim_km(frame, reticle):
    """Ballistic range printed beside the reticle, or None."""
    if reticle is None or reticle[0] is None:
        return None
    from fire_hud import FireHudReader
    reader = hud_aim_km.reader
    if reader is None:
        reader = hud_aim_km.reader = FireHudReader()
    try:
        found = reader.read_aim_distance(frame, reticle)
    except (ValueError, OSError):
        return None
    if found.get('status') != 'observed':
        return None
    value = found.get('km')
    return None if value is None else float(value)


hud_aim_km.reader = None


def measure_range_units(probe, frame_getter, centre_getter, report, units=40, min_km=0.3):
    """Kilometres of reticle range per vertical mouse unit, measured in place.

    RANGE_KM_PER_UNIT was calibrated at the scope zoom only (20 units = 0.3 km);
    third person responds ~6x less, so the closure either stalled or overshot.
    This moves the aim once, reads the mode-based range, and puts it back.
    """
    def reading(tag):
        frame = frame_getter(tag)
        centre = centre_getter(frame)
        return None if centre is None else aim_distance_km(frame, centre)

    before = reading('range-gain-base')
    if before is None:
        return None, 'readout_missing'
    probe.move(0, units)
    time.sleep(.4)
    after = reading('range-gain-after')
    probe.move(0, -units)
    time.sleep(.3)
    report.setdefault('range_gain_probe', []).append(
        dict(units=units, before_km=before, after_km=after))
    if after is None:
        return None, 'readout_missing_after'
    delta = after - before
    if abs(delta) < min_km:
        return None, 'range_did_not_move'
    return round(delta / units, 5), None


def impact_readout(frame, centre):
    """The shell-impact field beside the reticle.

    Measured layout on a live frame: under the locked nameplate a row reads
    "13.3 公里" (the target's range) and immediately right of it "- - km" when the
    aim is beyond maximum range. Returns ``('km', value)``, ``('beyond', None)``
    or ``('unknown', None)``; "beyond" is the operator's "- -" and must be acted
    on by lowering the aim, never read as an estimate.
    """
    if centre is None or centre[0] is None:
        return 'unknown', None
    cx, cy = float(centre[0]), float(centre[1])
    value = aim_distance_km(frame, centre)
    if value is not None:
        return 'km', value
    # Look for the dash pair in the band right of the target-distance text.
    x0, y0 = int(cx) + 26, int(cy) - 30
    x1, y1 = int(cx) + 150, int(cy) + 12
    if x0 < 0 or y0 < 0 or x1 > frame.shape[1] or y1 > frame.shape[0]:
        return 'unknown', None
    band = frame[y0:y1, x0:x1]
    light = band.min(axis=2) > 150
    runs, start = [], None
    for row in light:
        columns = row.any()
        if not columns:
            continue
    # Count horizontal segments in the row that carries the most ink.
    rows = light.sum(axis=1)
    if rows.size == 0 or rows.max() < 4:
        return 'unknown', None
    row = light[int(rows.argmax())]
    segments, start = [], None
    for index, filled in enumerate(row):
        if filled and start is None:
            start = index
        elif not filled and start is not None:
            segments.append((start, index))
            start = None
    if start is not None:
        segments.append((start, len(row)))
    wide = [s for s in segments if 4 <= s[1] - s[0] <= 26]
    if len(wide) >= 2:
        lengths = [s[1] - s[0] for s in wide[:2]]
        if abs(lengths[0] - lengths[1]) <= 4:
            report_free = True
            return ('beyond', None) if report_free else ('unknown', None)
    return 'unknown', None


def target_range_km(frame, plate_bbox):
    """The enemy's own range, from the nameplate caption only.

    The minimap is not consulted any more: the shot chain reads the main view,
    and a range that comes from a different rendering pipeline cannot be
    reconciled with what the reticle shows.
    """
    value = hud_distance_km(frame, plate_bbox) if plate_bbox else None
    if value:
        return value, 'caption'
    return None, 'unavailable'


def impact_range_km(frame, shell_speed=800.0):
    """Where the salvo would land, from the HUD's own flight time.

    The "NN.N km" text beside the reticle is unreliable on this client (one
    frame reads 24.4 / 4.4 / 1.3 at neighbouring offsets -- 24.4 is the maximum
    range label). The flight time is read reliably, and impact range is
    ``flight_s * shell_speed``: measured 15.0 s -> 12.0 km against an 11.8 km
    target (0.2 km gap), and 8.1 s -> 6.5 km against a 16.2 km target, which is
    exactly the 9.7 km shortfall the missed salvos showed.
    """
    reading = hud_flight_seconds(frame) or {}
    value = reading.get('value')
    if not value:
        return None
    if not 0.5 <= float(value) <= 30.0:
        return None
    return round(float(value) * shell_speed / 1000.0, 2)


def aim_distance_km(frame, centre):
    """Shell-impact distance printed beside the reticle, tried over offsets.

    The readout's x placement differs per layout, and a single probe point
    measured 22 px off missed it entirely (None on frames where the digits are
    plainly drawn). Try a small fan of candidate points and take the first
    unambiguous reading; the value is what the operator compares against the
    target's own range before opening the scope.
    """
    if centre is None or centre[0] is None:
        return None
    cx, cy = float(centre[0]), float(centre[1])
    # Several neighbouring offsets land on different numbers (measured on one
    # frame: 24.4 at four offsets, 4.4 at one). The operator reads 24.4 km off
    # the screen, so the honest estimate is the *mode* of the readings, and a
    # single outlier must not decide it.
    values = []
    for dx in (0, -22, 12, -34, 24, -46, 36):
        for dy in (9, 18, 0, -9):
            value = hud_aim_km(frame, [cx + dx, cy + dy])
            if value is not None:
                values.append(round(float(value), 1))
    if not values:
        return None
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    best, seen = max(counts.items(), key=lambda kv: (kv[1], -abs(kv[0] - 10.0)))
    # One frame produced 22 x 24.4 against 4 x 4.0: a bare majority is not enough
    # when a neighbouring number can be read at a few offsets, and that flip
    # corrupted a 40-unit calibration into -0.5 km/unit. Require a clear
    # supermajority or report nothing.
    if seen < 2 or seen < 0.7 * len(values):
        return None
    aim_distance_km.readings = values
    return best


def minimap_target_km(frame, km_per_px=MINIMAP_KM_PER_PX, package=None):
    """Target range from the minimap, which is the only machine-readable one.

    The nameplate caption is 8 px tall and no font template matches it, so the
    static `--range-target-km` had to be supplied by hand. The minimap gives the
    same number every frame: own arrow and enemy marker are both detected and
    the panel is linear (measured 0.0501 km/px, cross-checked against the
    printed 11.0 / 12.6 range rings).
    """
    # The minimap recogniser runs a small classifier and costs ~1.2 s per call,
    # which dominated every closure step. The target's range changes about
    # 0.1 km/s, so a short cache is far cheaper than the error it can introduce
    # (the closure tolerance is 0.7 km).
    ttl = float(os.environ.get('WW_MINIMAP_TTL', '2.5'))
    now = time.monotonic()
    cached = minimap_target_km.cache
    if cached is not None and now - cached[0] <= ttl:
        return cached[1]
    recogniser = minimap_target_km.recogniser
    if recogniser is None:
        from minimap_recognizer import MinimapRecognizer
        path = package or (Path(__file__).resolve().parents[2]
                           / 'replays/parsed/artifacts/enemy-marker-22s-v1/package.json')
        recogniser = minimap_target_km.recogniser = MinimapRecognizer(str(path))
    x, y, w, h = MINIMAP_RECT
    found = recogniser.detect(frame[y:y + h, x:x + w])
    own, enemies = found.get('own'), found.get('enemies') or []
    if not own or not enemies:
        return None
    ex, ey = enemies[0]['u'] * w, enemies[0]['v'] * h
    value = float(np.hypot(ex - own['pixel_x'], ey - own['pixel_y'])) * km_per_px
    minimap_target_km.cache = (now, value)
    return value


minimap_target_km.recogniser = None
minimap_target_km.cache = None

_RANGE_READER = None


def step_range_evidence(frame, plate_bbox, reticle=None):
    """Range readings for one step, computed once and reused by every gate.

    The caption under the plate is the game's own number for that ship and
    costs ~58 ms; the minimap recogniser costs ~4.5 s uncached, so it is only
    read when the caption is unreadable, and either reading is memoised against
    the frame so the step's gates cannot each pay for it again.
    """
    global _RANGE_READER
    if _RANGE_READER is None:
        _RANGE_READER = fire_gate.RangeReader(
            caption_reader=lambda rgb, box: hud_distance_km(rgb, box),
            minimap_reader=None)
    reading = _RANGE_READER.read(frame, plate_bbox)
    aim_km = hud_aim_km(frame, reticle)
    target_km = reading['target_km']
    return dict(target_km=target_km, aim_km=aim_km,
                range_source=reading['source'],
                range_cached=reading['cached'],
                range_gap_km=(None if target_km is None or aim_km is None
                              else round(target_km - aim_km, 2)))


_CAPTION_FONT_DIR = ROOT / 'state/toalba/gamedata/data/gui/fonts'
# The caption is drawn in the Korean-capable Warhelios face, not the plain one:
# matching 86 labelled digits from the saved captures (state/now4-20261020.png,
# state/after-burst-20261020.png, state/final-20261020.png, state/cadence-*/)
# with Warhelios scored 78/86 (every miss a 6 read as 5), WarheliosKO 86/86.
_CAPTION_FACES = ('WarheliosKO_Medium.ttf', 'WarheliosKO_Bold.ttf')
# The HUD is laid out at a smaller internal height and scaled up to 2560x1600,
# so the sprite is a ~1.15x expansion of a 10-13 px rasterization. Rendering at
# the final size instead leaves crisp strokes that no real caption resembles.
_CAPTION_SIZES = (10, 11, 12, 13)
_CAPTION_SCALE = 1.15
# Minimum normalised correlation for a glyph to count as read. Measured on 38
# saved frames: correct glyphs score 0.75-0.96, so this only rejects garbage.
_CAPTION_MIN_SCORE = 0.60


def _caption_templates():
    """Digit masks for the caption face, rendered then scaled like the HUD.

    Kept at 2x, the scale the correlation runs at, so a frame costs no rendering
    and no upscaling.
    """
    if _caption_templates.cache is not None:
        return _caption_templates.cache
    from PIL import Image, ImageDraw, ImageFont
    rendered = []
    for name in _CAPTION_FACES:
        path = _CAPTION_FONT_DIR / name
        if not path.exists():
            continue
        for size in _CAPTION_SIZES:
            font = ImageFont.truetype(str(path), size)
            for char in '0123456789':
                canvas = Image.new('L', (80, 80))
                ImageDraw.Draw(canvas).text((20, 20), char, font=font, fill=255)
                ink = np.asarray(canvas).astype(np.float32) / 255.0
                ink = cv2.resize(ink, None, fx=_CAPTION_SCALE, fy=_CAPTION_SCALE,
                                 interpolation=cv2.INTER_LINEAR)
                ys, xs = np.where(ink > 0.10)
                if not len(ys):
                    continue
                glyph = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
                rendered.append((char, cv2.resize(glyph, None, fx=2, fy=2,
                                                  interpolation=cv2.INTER_CUBIC)))
    _caption_templates.cache = rendered
    return rendered


_caption_templates.cache = None


def _caption_glyph_scores(patch):
    """Per-digit correlation for one 11 px caption glyph.

    The strokes are 1 px wide and anti-aliased, so connected-component OCR
    rejects them (no template matches a fragment). Correlation is run over ±1 px
    of shift at 2x, which absorbs the sub-pixel phase of the HUD placement.
    """
    if _caption_templates.cache is None:
        _caption_templates()
    templates = _caption_templates()
    patch = np.clip(patch, 0, None)
    if not templates or patch.size == 0 or patch.max() <= 0:
        return {}
    up = cv2.resize(patch / patch.max(), None, fx=2, fy=2,
                    interpolation=cv2.INTER_CUBIC)
    scores = {}
    for char, up_t in templates:
        if (abs(up_t.shape[0] - up.shape[0]) > 10
                or abs(up_t.shape[1] - up.shape[1]) > 10):
            continue
        height = max(up_t.shape[0], up.shape[0]) + 4
        width = max(up_t.shape[1], up.shape[1]) + 4
        image = np.zeros((height, width), np.float32)
        image[2:2 + up.shape[0], 2:2 + up.shape[1]] = up
        best = -9.0
        for dy in (-2, 0, 2):
            for dx in (-2, 0, 2):
                target = np.zeros((height, width), np.float32)
                target[2 + dy:2 + dy + up_t.shape[0],
                       2 + dx:2 + dx + up_t.shape[1]] = up_t
                a = (image - image.mean()) / (image.std() + 1e-6)
                b = (target - target.mean()) / (target.std() + 1e-6)
                best = max(best, float((a * b).mean()))
        scores[char] = max(scores.get(char, -9.0), best)
    return scores


def _caption_ink(frame, plate_bbox):
    """Local contrast of the caption band, on the caption's own row window."""
    x, y, w, h = [int(v) for v in plate_bbox]
    top = y + h + 2
    x0 = max(0, x + w // 2 - 70)
    band = frame[top:top + 34, x0:x0 + 140]
    if band.size == 0 or band.shape[0] < 19 or band.shape[1] < 40:
        # A plate near the frame edge leaves a band too small for the blur
        # kernel; OpenCV raises on an empty source and that used to crash the
        # whole aiming run mid-convergence.
        return None
    p = band.astype(np.int16)
    gray = np.minimum(np.minimum(p[..., 0], p[..., 1]), p[..., 2])
    spread = np.maximum(np.maximum(p[..., 0], p[..., 1]), p[..., 2]) - gray
    background = cv2.medianBlur(gray.astype(np.uint8), 11).astype(np.int16)
    # Signed contrast: the digits are always brighter than the water, but the
    # decimal point is drawn dimmer than bright water and only shows as a dark
    # blob (measured at 9.4 km, where the point reads 13-30 over water at 50-100).
    con = (gray - background).astype(np.float32)
    # Measured: the caption's own strokes have a channel spread of 17-55 while
    # the sea around them is 85-95. Erasing coloured HUD instead of trusting
    # brightness is what lets the same rule work on bright water, where the
    # whole band is above 150 and every brightness threshold saturates.
    con[spread >= 75] = 0
    return con


def _caption_split(comp, window):
    """Split a component that holds two touching glyphs at its emptiest column."""
    x, y, w, h, _ = comp
    if w <= 7:
        return [comp]
    columns = window[y:y + h, x:x + w].sum(axis=0)
    lo, hi = max(1, w // 3), min(w - 1, (2 * w) // 3)
    if hi <= lo:
        return [comp]
    cut = x + lo + int(np.argmin(columns[lo:hi]))
    left_ink = columns[:cut - x].sum()
    right_ink = columns[cut - x:].sum()
    if left_ink <= 0 or right_ink <= 0:
        return [comp]
    parts = []
    for x0, x1 in ((x, cut), (cut, x + w)):
        piece = window[y:y + h, x0:x1]
        rows, cols = np.where(piece > 0)
        if not len(rows):
            continue
        # A part that only carries baseline ink is the decimal point caught by
        # the split, not a glyph: it is dropped here and found again as ink.
        if rows.max() - rows.min() < 4:
            continue
        parts.append([x0 + int(cols.min()), y + int(rows.min()),
                      int(cols.max() - cols.min()) + 1,
                      int(rows.max() - rows.min()) + 1, comp[4]])
    return parts or [comp]


def _caption_peak(con):
    window = np.clip(con[3:19], 0, None)
    return float(window.max()) if window.size else 0.0


def _caption_dot_between(con, left, right):
    """X of a decimal point in the gap between two glyph boxes, or None.

    The point is never its own component at this size (its halo touches the
    neighbours), so it is ink that appears low in the glyph box inside a gap
    while the same columns are empty at glyph height. Measured on the live
    caption: the point is 2-3 px wide in the bottom third of the digits, while a
    real glyph gap has no ink at all and the strips beside the glyphs only carry
    their anti-aliased halo.
    """
    start = left[0] + left[2]
    stop = right[0]
    if stop - start < 3:
        return None
    peak = _caption_peak(con)
    if peak <= 0:
        return None
    floor = 0.45 * peak
    top = min(left[1], right[1])
    bottom = max(left[1] + left[3], right[1] + right[3])
    upper = con[top:top + 5, start + 1:stop - 1]
    lower = con[min(top + 5, bottom):bottom, start + 1:stop - 1]
    if lower.size == 0 or lower.max() < floor:
        return None
    if upper.size and upper.max() >= floor:
        return None
    return start + (stop - start) / 2.0


def _caption_digits(con):
    """Digit boxes of the range caption, left to right, or None.

    Measured over 38 saved frames (state/now4-20261020.png and the cadence
    bursts): the caption occupies rows 3..18 of the band with 11 px digits,
    neighbours are 1-2 px apart, and the unit "公里" starts 6-9 px to their
    right. A vertical close joins the dim upper and lower halves of a glyph (a
    "9" at 10.7 km breaks apart) without bridging the horizontal gaps that
    separate the digits and the decimal point.
    """
    window = np.clip(con[3:19], 0, None)
    peak = _caption_peak(con)
    if peak < 25:
        return None
    ink = ((window / peak) > 0.30).astype(np.uint8)
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 1), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(ink)
    comps = [[int(v) for v in stats[i]] for i in range(1, count)
             if stats[i][4] >= 4 and stats[i][3] >= 4]
    if not comps:
        return None
    comps.sort()
    band = [[c[0], c[1] + 3, c[2], c[3], c[4]] for c in comps]
    runs, current = [], [0]
    for index in range(1, len(comps)):
        gap = comps[index][0] - (comps[index - 1][0] + comps[index - 1][2])
        # The gap to the unit is 6-9 px too, so a wider jump is only the same
        # number when a decimal point sits in it.
        if gap <= 6 or (gap <= 9 and _caption_dot_between(
                con, band[index - 1], band[index]) is not None):
            current.append(index)
        else:
            runs.append(current)
            current = [index]
    runs.append(current)
    for run in runs:
        digits = []
        for index in run:
            digits.extend(_caption_split(comps[index], window))
        digits = sorted([c for c in digits if c[3] >= 7 and c[2] >= 2],
                        key=lambda c: c[0])
        # A number is two or three glyphs wide (one or two integer digits plus
        # the decimal digit), or four when the point was joined to a digit and
        # the split had to separate them; a longer run is the unit or other HUD.
        if 2 <= len(digits) <= 3 or (len(digits) == 4
                                     and min(c[2] for c in digits) <= 4):
            # The window starts three rows into the band; callers crop the band.
            return [[c[0], c[1] + 3, c[2], c[3], c[4]] for c in digits]
    return None


def _caption_dots(con, digits):
    """Candidate X positions of the decimal point, widest run first.

    The point is not reliably a separate component (its halo touches the
    neighbours) and it does not always sit in a gap either: at 10.5 km it is
    inside the last digit's box. What holds on every measured frame is that its
    columns carry ink in the lower part of the glyphs and none at glyph height,
    so every run of such columns is offered and the reader keeps the one that
    splits the digits into a valid number.
    """
    top = min(c[1] for c in digits)
    # A caption glyph is 10-12 rows tall; the component box can run longer when
    # a shadow under the number is joined to it (measured at 12.1 km), so the
    # search window is taken from the glyph top rather than from the boxes.
    bottom = min(con.shape[0], top + 13)
    left = min(c[0] for c in digits) + 2
    right = max(c[0] + c[2] for c in digits) - 2
    if right - left < 3:
        return []
    upper = con[top:top + 6, left:right]
    lower = con[top + 7:bottom, left:right]
    if upper.size == 0 or lower.size == 0:
        return []
    # Measured: the point's own contrast is 16-117 while every empty column in
    # the same rows stays under 10, on both the dark and the bright-water frames.
    dot = (lower.max(axis=0) >= 10) & (upper.max(axis=0) < 10)
    found = []
    index = 0
    while index < len(dot):
        if not dot[index]:
            index += 1
            continue
        stop = index
        while stop < len(dot) and dot[stop]:
            stop += 1
        found.append((left + index + (stop - index) / 2.0, stop - index))
        index = stop
    found.sort(key=lambda item: -item[1])
    return [item[0] for item in found]


def _caption_layouts(con, digits):
    """Ways the detected glyphs can split into a number, best first."""
    found = _caption_dots(con, digits)
    for dot in found:
        yield dot, digits
    # Four boxes under one bar is a number whose point was joined to a digit by
    # the segmentation (measured at 13.1 km), so one box is the point: drop the
    # one nearest a detected point first, then any other.
    if len(digits) == 4:
        order = sorted(digits, key=lambda c: min(
            [abs(c[0] + c[2] / 2.0 - dot) for dot in found] or [99]))
        for cell in order:
            if cell[2] <= 5:
                yield (cell[0] + cell[2] / 2.0,
                       [c for c in digits if c is not cell])


def _caption_value(con, digits, dot):
    """Read the number a layout describes, or None when it does not fit."""
    top = min(c[1] for c in digits)
    bottom = max(c[1] + c[3] for c in digits)
    integer = [c for c in digits if c[0] + c[2] / 2.0 < dot]
    fraction = [c for c in digits if c[0] + c[2] / 2.0 > dot]
    if not 1 <= len(integer) <= 2 or len(fraction) != 1:
        return None
    chars = []
    for cell in integer + fraction:
        scores = _caption_glyph_scores(con[top - 1:bottom + 1,
                                          max(0, cell[0] - 1):cell[0] + cell[2] + 1])
        if not scores:
            return None
        char, score = max(scores.items(), key=lambda kv: kv[1])
        if score < _CAPTION_MIN_SCORE:
            return None
        chars.append(char)
    try:
        value = float(''.join(chars[:len(integer)]) + '.' + chars[-1])
    except ValueError:
        return None
    if not 0.2 <= value <= 40:
        return None
    return value


def hud_distance_km(frame, plate_bbox):
    if plate_bbox is None:
        return None
    """Range caption under a nameplate, as the game itself draws it.

    The caption sits centred under the bar and reads like "7.0 km" or "7.0 公里".
    A fixed crop fails because the glyph group is narrower than the bar, so the
    digits are located first: separate same-baseline components below the bar,
    split by the decimal point into "D.D" or "DD.D", then read by correlation
    against the caption face (see _caption_glyph_scores for why component OCR
    cannot be used). Anything that does not fit that shape returns None rather
    than a guessed range.
    """
    con = _caption_ink(frame, plate_bbox)
    if con is None:
        return None
    digits = _caption_digits(con)
    if not digits:
        return None
    for dot, cells in _caption_layouts(con, digits):
        value = _caption_value(con, cells, dot)
        if value is not None:
            return value
    return None


def scope_ship_blob(frame, bar_bbox, drop=40, min_area=2000):
    """The scoped hull silhouette under a nameplate bar.

    Measured on a live 10.3 km scoped frame: hull 440x74 px, brightness 70..135
    against sea 157, sitting 100..180 px below the bar bottom. Aiming at the
    bar plus a fixed multiple of its width left the reticle ~30 px low, while
    the silhouette itself is unambiguous at this size.
    """
    import numpy as np
    x, y, w, h = [int(v) for v in bar_bbox]
    x0 = max(0, x + w // 2 - 520)
    x1 = min(frame.shape[1], x + w // 2 + 520)
    y0 = min(frame.shape[0] - 1, y + h + 10)
    y1 = min(frame.shape[0], y + h + 460)
    patch = frame[y0:y1, x0:x1].astype(np.int16)
    if patch.size == 0:
        return None
    gray = patch.mean(axis=2)
    sea = float(np.median(gray))
    mask = (gray < sea - drop).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 15), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    best = None
    for index in range(1, count):
        bx, by, bw, bh, area = stats[index]
        if area < min_area or bw < 60 or bh < 15:
            continue
        if not 1.5 <= bw / bh <= 20:
            continue
        # A hull is the tallest wide dark object here; the waterline band is
        # wide but flat, and ranking by height keeps the ship over its wake.
        if best is None or bh > best[0]:
            best = (bh, bx, by, bw, bh, area)
    if best is None:
        return None
    _, bx, by, bw, bh, area = best
    return dict(aim=[x0 + bx + bw / 2, y0 + by + bh / 2],
                bbox=[x0 + bx, y0 + by, bw, bh], area=int(area), sea=round(sea, 1))


def horizon_hull_aim(frame, plate_bbox, span=250, above=20, below=70, quantile=0.5):
    """Aim at the ship silhouette on the horizon, not at the bar's centre.

    Measured on a live 6.5 km frame: the hull is a small dark blob at
    (1355, 800) while the bar centre sat at x=1346 and the servo's reticle had
    drifted to x=1280, i.e. a 75 px horizontal miss that no bar-derived rule can
    see. The hull is the darkest few pixels in the band just under the plate.
    """
    import numpy as np
    x, y, w, h = [int(v) for v in plate_bbox]
    cx = x + w // 2
    x0, x1 = max(0, cx - span), min(frame.shape[1], cx + span)
    y0 = max(0, y + h + 8)          # never include the plate's own dark edge
    y1 = min(frame.shape[0], y + h + below)
    band = frame[y0:y1, x0:x1]
    if band.size == 0:
        return None
    gray = band.astype(np.int16).mean(axis=2)
    threshold = float(np.percentile(gray, quantile))
    if threshold > 150:          # nothing dark enough to be a hull
        return None
    mask = gray <= threshold
    ys, xs = np.nonzero(mask)
    if len(xs) < 8 or len(xs) > .2 * mask.size or threshold < 20:
        return None              # blank or uniformly dark: no hull evidence
    weight = (threshold - gray[ys, xs] + 1.0)
    px = float((xs * weight).sum() / weight.sum()) + x0
    py = float((ys * weight).sum() / weight.sum()) + y0
    return dict(aim=[px, py], threshold=round(threshold, 1), pixels=int(len(xs)),
                bbox=[x0, y0, x1 - x0, y1 - y0])


#: Vertical aim offset below the plate, measured on live frames:
#: 38 px at 8.4-9.1 km and 61-63 px at 12-13.5 km. A linear fit through those
#: two anchors is offset = a + b*km; the old 1415/km rule predicted 101 px at
#: 8.4 km and was therefore disabled rather than fixed.
#: How far above the measured waterline the scope aim sits. The operator asked
#: for "slightly above the waterline"; 8 px at the scope zoom is a hair of deck.
AIM_ABOVE_WATERLINE_PX = 8


def waterline_row(frame, plate_bbox, lo=0.5, hi=2.6, min_gain=1.6):
    """Find the ship's waterline as the strongest horizontal edge below the bar.

    The scope magnifies the hull about fourfold, so the third-person offset rule
    landed the aim on the superstructure (measured 2026-10-23: aim 703 while the
    hull met the sea at row 800, edge 39.8 against a 25.4 edge at the body-box
    bottom). Returns ``dict(row, gain, peak)`` or None when no edge stands out.
    """
    import cv2
    x, y, w, h = [int(v) for v in plate_bbox]
    r0 = max(0, y + h + int(lo * w))
    r1 = min(frame.shape[0], y + h + int(hi * w))
    x0 = max(0, int(x - 1.6 * w))
    x1 = min(frame.shape[1], int(x + 2.6 * w))
    if r1 - r0 < 12 or x1 - x0 < 20:
        return None
    grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
    band = grey[r0:r1, x0:x1]
    edges = np.abs(np.diff(band, axis=0)).mean(axis=1)
    if edges.size < 8:
        return None
    k = int(edges.argmax())
    peak = float(edges[k])
    # A quantised image makes half the row differences exactly zero, so the
    # median is a useless scale (measured: 0.0 on an 8-level synthetic sea).
    # The 75th percentile is robust to the single peak and still zero on a flat
    # frame, which is the case the guard must reject.
    scale = float(np.percentile(edges, 75))
    if scale <= 0.05 or peak < min_gain * scale:
        return None
    return dict(row=r0 + k, gain=round(peak / scale, 2), peak=round(peak, 1))


AIM_OFFSET_CALIBRATION = (-14.5, 6.0)
AIM_OFFSET_BOUNDS = (28.0, 95.0)
AIM_OFFSET_MIN_KM = 6.0


def capture_aim_offset_km(km, calibration=None, bounds=None):
    """Vertical offset below the health bar for a target at ``km``.

    Third-person only: it was calibrated by the operator's manual aim in that
    view. Below ``AIM_OFFSET_MIN_KM`` the horizon/hull rule is better and this
    one is not consulted.
    """
    a, b = calibration or AIM_OFFSET_CALIBRATION
    low, high = bounds or AIM_OFFSET_BOUNDS
    value = a + b * float(km)
    return max(low, min(high, value))


def hull_aim_point(frame, plate_bbox):
    """Aim inside the hull, from the bar plus the game's own range caption.

    Measured on two live frames: the bar bottom sits 524 px above the waterline
    at 2.7 km and 203 px at 7.0 km, i.e. the gap scales as 1415 px*km / range.
    Aiming 60% of that gap below the bar lands on the belt (predicted y 616 in
    a hull spanning 301..825 at 2.7 km, and 470 in 442..552 at 7.0 km).
    Without a readable range the bar-height rule is used instead.
    """
    x, y, w, h = [int(v) for v in plate_bbox]
    scoped = bool(camera_mode_evidence(frame).get('binocular'))
    if scoped:
        # The scope draws the hull large; the horizon-blob rule reads the plate
        # text there and lands above the ship (measured), so use the big blob.
        blob = scope_ship_blob(frame, plate_bbox)
        if blob is not None:
            aim = [float(v) for v in blob['aim']]
            meta = dict(source='scope_hull_blob', bbox=list(blob['bbox']), area=blob['area'])
            # The blob's aim is a fraction of a box that includes sky and water,
            # so it sits high on the hull. The operator's rule is "just above the
            # waterline": when a confident waterline edge is found lower than the
            # blob aim, move down to it -- but never past the hull's own bottom.
            line = waterline_row(frame, plate_bbox)
            if line is not None:
                bottom = float(blob['bbox'][1] + blob['bbox'][3])
                candidate = min(float(line['row'] - AIM_ABOVE_WATERLINE_PX), bottom)
                if candidate > aim[1]:
                    meta.update(source='scope_waterline', waterline_row=line['row'],
                                edge_gain=line['gain'], above_px=AIM_ABOVE_WATERLINE_PX,
                                blob_aim_y=round(aim[1], 1))
                    aim = [aim[0], candidate]
            return aim, meta
    else:
        hull = horizon_hull_aim(frame, plate_bbox)
        if hull is not None:
            # A real hull is a compact blob. On a live 10+ km frame the darkest
            # pixels scattered across a 500 px band (199 px, 0.6% dense) and the
            # servo fired five salvos into the water; require a compact, dense
            # patch before trusting this rule over the plate-anchored one.
            bx, by, bw, bh = hull['bbox']
            dense = hull['pixels'] / max(1.0, float(bw * bh)) >= 0.02
            if bw <= 250 and dense:
                return hull['aim'], dict(source='horizon_hull', threshold=hull['threshold'],
                                         pixels=hull['pixels'], bbox=hull['bbox'])
    # The caption now reads reliably (2026-10-20), but it must NOT drive the
    # aim height: measured truth is 38 px at 8.4 km and 61-63 px at 12-13.5 km,
    # while 0.6*1415/km would say 101 px at 8.4 km. The distance is an input to
    # range closure, not to this offset; bar_geometry stays the calibrated rule.
    # The scope aims at the measured waterline: the distance rule below was
    # calibrated in third person and lands on the superstructure here (aim 703
    # against a waterline at 800 on a 13 km scoped frame). The operator's rule is
    # "slightly above the waterline", so the aim sits AIM_ABOVE_WATERLINE_PX over it.
    if scoped:
        line = waterline_row(frame, plate_bbox)
        if line is not None:
            aim_y = float(line['row'] - AIM_ABOVE_WATERLINE_PX)
            # Never below the hull's own bottom: on a huge close ship the
            # strongest edge can sit on the deck, so the blob (when there is one)
            # bounds how far this may go.
            return [x + w / 2.0, aim_y], dict(
                source='scope_waterline', waterline_row=line['row'],
                edge_gain=line['gain'], above_px=AIM_ABOVE_WATERLINE_PX)
    # Third person: the ship is 20x7 px at 10 km, so an edge detector cannot find
    # its waterline (measured: the strongest row in the band was sea texture at
    # 900 while the hull met the sea at 804). The bar-to-sea gap was measured
    # directly instead -- 34 px at 8.4 km, 37 px at 10.9 km -- i.e. ~0.30 bar
    # widths, which is what bar_geometry already used; the distance-scaled rule
    # overshot it (51 px at 10.9 km) and aimed into the water. The aim sits the
    # same AIM_ABOVE_WATERLINE_PX above that measured line, matching the scope.
    import os as _os
    # Off by default: the distance slope (-14.5 + 6.0*km) was fitted on two points
    # that were not comparable and overshot the measured bar-to-sea gap (51 px at
    # 10.9 km against 37 measured). Kept behind the variable for experiments.
    km = (hud_distance_km(frame, plate_bbox)
          if (not scoped and _os.environ.get('WW_CAPTION_AIM', '0') == '1') else None)
    if km and km >= AIM_OFFSET_MIN_KM:
        offset = capture_aim_offset_km(km)
        return [x + w / 2.0, y + h + offset], dict(source='range_caption_linear', km=km,
                                                   offset_px=round(offset, 1),
                                                   calibration=list(AIM_OFFSET_CALIBRATION))
    # The nameplate caption is often unreadable (glare, "公里" glyphs mixed in),
    # but the reticle's own distance readout decodes reliably and is the same
    # quantity: at 9.7 km this rule is the one that produced the verified hits.
    # Measured 2026-10-20 (state/aim-offset-calibration-20261020): the reticle
    # readout is the water point under the crosshair, not the target range, and
    # it misreads badly (2.3 km for an 11.5 km target). Off unless asked for.
    import os as _os
    aim_km = (_os.environ.get('WW_USE_RETICLE_RANGE', '0') == '1'
              and hud_aim_km(frame, getattr(reticle_centre, 'last_bbox', None)))
    if aim_km:
        offset = HULL_AIM_FRACTION * HULL_OFFSET_PX_KM / aim_km
        return [x + w / 2.0, y + h + offset], dict(source='reticle_range', km=aim_km,
                                                   offset_px=round(offset, 1))
    # Calibrated against the operator's own manual aim, which is the reference:
    # scope frame bar [1211,609,121,17] with the reticle parked on the waterline
    # at y=811 -> 1.53 bar-widths below the bar bottom; third-person 8.4 km frame
    # bar [1221,642,123,17] with the ship at y~693 -> 0.28. The scope magnifies
    # that gap about fivefold, so one constant cannot serve both views.
    if not scoped:
        from fire_hud import scope_ruler_center
        scoped = scope_ruler_center(frame) is not None
    # Scope: mid-to-lower hull. Measured on a live 10.3 km scoped frame the hull
    # spans 650..758 px with the bar bottom at 589, so 1.55 bar widths put the
    # reticle 17 px below the waterline and every salvo landed short; 1.15 lands
    # just above the waterline.
    # Scope: the operator judged the 1.15 result "slightly high" and the model's
    # own residual agreed (aim point 30 px below the reticle), so the scope aim
    # now sits ~0.25 bar widths lower, on the hull centre.
    # The two constants are calibrated (0.30 third person, 1.40 scope); the
    # environment overrides exist so one session can A/B the height on the
    # same target instead of guessing.
    import os
    third = float(os.environ.get('WW_THIRD_OFFSET_BARS', '0.30'))
    scope = float(os.environ.get('WW_SCOPE_OFFSET_BARS', '1.40'))
    offset = (scope if scoped else third) * w
    if not scoped:
        # bar_geometry's 0.30 w is the measured bar-to-sea gap; sit just above it.
        offset -= AIM_ABOVE_WATERLINE_PX
    return [x + w / 2.0, y + h + offset], dict(source='bar_geometry',
                                               view='scope' if scoped else 'third_person',
                                               offset_px=round(offset, 1),
                                               offset_bar_widths=round(scope if scoped else third, 2))


def red_marker_blobs(crop, min_area=15):
    """Cluster saturated red pixels; returns [(x, y)] in panel coordinates.

    The minimap draws enemy contacts as a small red ship glyph. When the marker
    classifier misses it (it did on 2026-10-23 while the glyph was plainly
    visible) the geometric reading still works, and a bearing from it is far
    better than no bearing at all.
    """
    import cv2

    r = crop[:, :, 0].astype(int)
    g = crop[:, :, 1].astype(int)
    b = crop[:, :, 2].astype(int)
    mask = ((r > 140) & (g < 90) & (b < 90)).astype('uint8')
    count, labels, stats, centres = cv2.connectedComponentsWithStats(mask, 8)
    blobs = []
    for index in range(1, count):
        area = int(stats[index][4])
        if area < min_area:
            continue
        blobs.append((float(centres[index][0]), float(centres[index][1]), area))
    if not blobs:
        return []
    # Group glyph fragments that sit within one marker's footprint.
    blobs.sort(key=lambda item: -item[2])
    groups = []
    for x, y, area in blobs:
        for group in groups:
            if abs(group['x'] / group['n'] - x) < 18 and abs(group['y'] / group['n'] - y) < 18:
                group['x'] += x
                group['y'] += y
                group['n'] += 1
                break
        else:
            groups.append(dict(x=x, y=y, n=1))
    return [(group['x'] / group['n'], group['y'] / group['n']) for group in groups]


def minimap_contacts_from_frame(frame):
    """Own pixel, heading and enemy pixels from the trained minimap model."""
    from minimap_recognizer import MinimapRecognizer
    x, y, w, h = 1836, 796, 720, 723
    # read_frame already hands back RGB; flipping here would hide every colour
    # cue and left only the white own arrow readable.
    crop = np.ascontiguousarray(frame[y:y + h, x:x + w])
    package = ROOT / 'replays/parsed/artifacts/enemy-marker-22s-v1/package.json'
    seen = MinimapRecognizer(str(package)).detect(crop)
    own = seen.get('own')
    if not own:
        return None, seen
    contacts = []
    for row in seen.get('enemies') or []:
        # The marker classifier reports normalised u/v (no pixel box) plus the
        # species it read from the icon.
        u, v = row.get('u'), row.get('v')
        if u is None or v is None:
            continue
        contacts.append(dict(pixel=[u * (w - 1), v * (h - 1)], species=row.get('species'),
                             icon=row.get('icon_candidate'), score=row.get('score')))
    if not contacts:
        # Measured 2026-10-23: the marker classifier returned zero enemies while
        # a red ship marker sat at panel (339, 490) -- the flow then had no
        # bearing and aimed at nothing. A plain red-blob cluster is a reliable
        # fallback: the enemy marker is the only saturated red thing on the map
        # that is not the panel's own damage tint.
        blobs = red_marker_blobs(crop)
        for centre in blobs:
            contacts.append(dict(pixel=[float(centre[0]), float(centre[1])],
                                 species='unknown', icon=None, score=None,
                                 source='red_blob_fallback'))
    own_pixel = [own['pixel_x'], own['pixel_y']]
    best = None
    if contacts:
        import minimap_contacts as MC
        best = MC.nearest_contact(own_pixel, contacts, 1.0)
        best['relative_bearing_deg'] = MC.relative_bearing_deg(
            own.get('angle_deg'), best['bearing_deg'])
    return dict(own=own, own_pixel=own_pixel, contacts=contacts, nearest=best), seen


TERRAIN_CACHE_S = None


def refine_limit_y(needed_units, reach_y, coarse_reach_y):
    """How much vertical one correction round may ask for.

    Measured 2026-10-23: a 13.8 km shot fired with the aim point 45 px above the
    reticle. The safe 3-unit step can express that error only over a dozen rounds,
    so the loop gave up and the gate accepted a 45 px offset. The rule here is
    "exactly what the error needs, never more than the coarse grant": the safe
    step still applies to small errors, and no round can exceed the coarse budget
    that the operator already allowed.
    """
    safe = abs(float(reach_y or 0.0))
    needed = abs(float(needed_units or 0.0))
    if needed <= safe:
        return safe
    return max(safe, min(abs(float(coarse_reach_y or 0.0)), needed))


def budget_exceeded(aim_started, budget_s, now=None):
    """Has the shot budget run out?

    The one-shot path used to fire without ever asking, so a 12.9 s engagement
    reported success under a 10 s rule. The budget is the operator's rule for
    "scan to shot", so it is checked at the same place the trigger is.
    """
    if not budget_s:
        return False
    now = time.monotonic() if now is None else now
    return (now - aim_started) > budget_s


def terrain_evidence(frame, plate_bbox=None, contacts_fn=None, fov_deg=68.2,
                     ttl=None):
    """Is land on the minimap line between us and the target?

    The fire gate already treats ``terrain_clear is False`` as a blocker, but the
    servo never filled it, so a shot at a ship behind an island was decided as if
    the sea were open. Returns ``dict(clear, ...)``; ``clear`` is ``None`` when
    the question cannot be answered, which the gate treats as unknown (no veto).

    The contact used is the minimap mark whose bearing best matches the plate in
    front of us, so a second contact on the other side cannot decide this shot.
    """
    import minimap_terrain as MT
    ttl = TERRAIN_CACHE_S if ttl is None else ttl
    provider = contacts_fn or minimap_contacts_from_frame
    now = time.monotonic()
    cached = terrain_evidence.contacts
    if cached is not None and ttl is not None and now - cached[0] <= ttl:
        seen, raw = cached[1], cached[2]
    else:
        seen, raw = provider(frame)
        terrain_evidence.contacts = (now, seen, raw)
    if seen is None or not seen.get('contacts'):
        return dict(clear=None, reason='no_minimap_contact',
                    contacts=len(raw.get('enemies') or []) if raw else 0)
    own = seen['own_pixel']
    contacts = seen['contacts']
    chosen, match = None, None
    if plate_bbox is not None:
        cx = float(plate_bbox[0]) + float(plate_bbox[2]) / 2
        height, width = frame.shape[0], frame.shape[1]
        bearing = (cx - width / 2) / (width / 2) * (fov_deg / 2)
        def axis(row):
            dx = row['pixel'][0] - own[0]
            dy = row['pixel'][1] - own[1]
            # Minimap x grows east, y grows south; screen bearing grows right.
            return (math.degrees(math.atan2(dx, -dy)) + 360.0) % 360.0
        heading = float(seen['own'].get('angle_deg') or 0.0)
        best = None
        for row in contacts:
            delta = abs(((axis(row) - heading - bearing + 180.0) % 360.0) - 180.0)
            if best is None or delta < best[0]:
                best = (delta, row)
        match = round(best[0], 1)
        chosen = best[1] if best[0] <= 25.0 else None
    if chosen is None:
        chosen = seen.get('nearest') or contacts[0]
        match = None
    panel = np.ascontiguousarray(frame[MINIMAP_RECT[1]:MINIMAP_RECT[1] + MINIMAP_RECT[3],
                                       MINIMAP_RECT[0]:MINIMAP_RECT[0] + MINIMAP_RECT[2]])
    result = MT.shot_terrain(panel, own, chosen['pixel'])
    result.update(source='minimap_line_of_fire', plate_bearing_match_deg=match,
                  contact=dict(pixel=[round(float(v), 1) for v in chosen['pixel']],
                               origin=chosen.get('source', 'model')))
    return result


terrain_evidence.contacts = None
terrain_evidence.result = None
terrain_evidence.thread = None


def terrain_prefetch(frame, plate_bbox=None, contacts_fn=None, ttl=8.0):
    """Measure the line of fire in the background, without spending the budget.

    The minimap recogniser costs ~2 s per call (it runs a small classifier), and
    the fire confirm sits inside the 5-10 s shot budget. Paying that cost inline
    pushed measured shots from 8.4 s to 13.5 s, i.e. over budget. The terrain
    question changes on the scale of islands, not frames, so it is answered once
    per engagement in a thread and read at the confirm. A result that is not
    ready yet is reported as ``terrain_pending`` and never vetoes the shot.
    """
    import threading
    running = terrain_evidence.thread
    if running is not None and running.is_alive():
        return False

    def work():
        try:
            answer = terrain_evidence(frame, plate_bbox, contacts_fn, ttl=ttl)
        except Exception as error:  # a probe must never kill the engagement
            answer = dict(clear=None, reason=f'terrain_probe_failed: {error}')
        terrain_evidence.result = (time.monotonic(), answer)

    thread = threading.Thread(target=work, name='terrain-prefetch', daemon=True)
    terrain_evidence.thread = thread
    thread.start()
    return True


def terrain_latest(max_age_s=25.0):
    """The prefetched answer, or an explicit "not ready" record."""
    stored = terrain_evidence.result
    if stored is None:
        return dict(clear=None, reason='terrain_pending')
    age = time.monotonic() - stored[0]
    if age > max_age_s:
        return dict(clear=None, reason='terrain_stale',
                    age_s=round(age, 1), previous=stored[1])
    answer = dict(stored[1])
    answer['age_s'] = round(age, 2)
    return answer




def turn_pieces(units, chunk=400.0):
    """Split a yaw request into equal signed pieces no larger than ``chunk``."""
    chunk = abs(float(chunk)) or 400.0
    remaining = float(units)
    pieces = []
    while abs(remaining) > 1e-9:
        step = max(-chunk, min(chunk, remaining))
        pieces.append(round(step, 1))
        remaining -= step
    return pieces


def calibrate_axis(probe, args, report, frame, body_bbox, axis, settle, frame_getter=None):
    """Measure one axis' screen px per mouse unit with a reversible probe.

    Returns (gain, shift) or (None, None) when the probe produced no usable
    motion; the caller then falls back to the configured gain.
    """
    template = crop_template(frame, body_bbox)
    if template is None:
        return None, None
    units = args.autogain_units if axis == 0 else min(args.probe_y, args.reach_y)
    move = (units, 0) if axis == 0 else (0, units)
    probe.move(move[0], move[1])
    report['inputs'][-1].update(reason='autogain_probe', axis=axis)
    time.sleep(settle)
    get_frame = frame_getter or (lambda name: read_frame(probe.snapshot(name)))
    after = get_frame(f'autogain-{axis}')
    found = track_patch(template, body_bbox, after, search=args.autogain_search)
    # Put the aim back where it was before returning.
    probe.move(-move[0], -move[1])
    report['inputs'][-1].update(reason='autogain_revert', axis=axis)
    if found is None:
        return None, None
    shift = [found['bbox'][0]-body_bbox[0], found['bbox'][1]-body_bbox[1]]
    if not usable_axis_shift(shift, axis, args.autogain_min_shift):
        return None, shift
    gain = shift[axis]/units if units else None
    return (None if gain is None else gain), shift


def open_loop_delta(target_px, reticle_px, gain):
    """Mouse units that should bring the reticle onto the target, no feedback.

    Used for a one-shot aim in the middle of a moving battle: the closed loop
    needs several settled measurements, which a live engagement may not allow.
    """
    if not gain or not gain[0] or not gain[1]:
        raise ValueError('open-loop gain must be a non-zero pair')
    # gain is d(error)/d(mouse), so the correction is -error/gain, exactly as
    # plan_move does it. Dividing without the minus sent the live fast shot the
    # wrong way (residual -85 px became -178 px).
    return [-(target_px[0]-reticle_px[0])/gain[0], -(target_px[1]-reticle_px[1])/gain[1]]


SCOPE_GAIN_CACHE = Path(__file__).resolve().parents[2] / "state" / "aim-scope-gain.json"


def load_scope_gain(identity):
    """Reuse a scope gain measured in this very game process."""
    try:
        data = json.loads(SCOPE_GAIN_CACHE.read_text())
    except (OSError, ValueError):
        return None
    if (identity or {}).get("start_ticks") != data.get("start_ticks"):
        return None
    if (identity or {}).get("pid") != data.get("pid"):
        return None
    gain = data.get("gain")
    if not gain or not gain[0] or not gain[1]:
        return None
    return gain


def save_scope_gain(identity, gain):
    try:
        SCOPE_GAIN_CACHE.parent.mkdir(parents=True, exist_ok=True)
        SCOPE_GAIN_CACHE.write_text(json.dumps(dict(pid=(identity or {}).get("pid"),
                                                   start_ticks=(identity or {}).get("start_ticks"),
                                                   gain=[round(g, 3) for g in gain],
                                                   at=time.strftime('%Y-%m-%dT%H:%M:%S')), indent=1))
    except OSError:
        pass


def measure_scope_gain(probe, frame_getter, find_plate, report, units=(24, 9), min_shift=4.0):
    """Learn the binocular view's mouse gain, sign included, right now.

    A carried-over value was the reason the scope phase walked the camera down:
    the sign was never verified in this view. Each axis is moved and put back;
    an axis that does not move the plate by min_shift is refused.
    """
    base = find_plate(frame_getter('scope-gain-base'))
    if base is None:
        return None, 'plate_missing'
    gains = [None, None]
    for axis, delta in ((0, units[0]), (1, units[1])):
        move = [0, 0]
        move[axis] = delta
        probe.move(move[0], move[1])
        time.sleep(.35 + (units[axis] / 400.0))
        after = find_plate(frame_getter(f'scope-gain-{axis}'))
        probe.move(-move[0], -move[1])
        time.sleep(.3)
        if after is None:
            continue
        shift = after[axis] - base[axis]
        report.setdefault('scope_gain_probe', []).append(
            dict(axis=axis, units=delta, shift_px=round(shift, 2)))
        if abs(shift) < min_shift:
            continue
        gains[axis] = shift / delta
    if gains[0] is None or gains[1] is None:
        return None, 'axis_did_not_move'
    return gains, None


def learn_gain(current, observed, floor=0.15, low=0.3, high=4.0):
    """Accept one gain sample only when it is plausible and not a sign flip.

    Measured 2026-10-23: a nameplate anchor that drifted between rounds produced
    a single opposite-signed sample, the scope gain flipped, and the next
    correction overshot by 534 px. A flip needs two agreeing samples elsewhere;
    one sample may only refine a gain of the same sign.
    """
    if observed is None or abs(observed) < floor or not (low <= abs(observed) <= high):
        return current
    if current is None or (current > 0) == (observed > 0):
        return observed
    return current


def descend_to_readable(measure, move, max_km=20.0, rounds=6,
                        step=MAX_VERTICAL_STEP_UNITS, stall_rounds=2):
    """Pitch down until the ballistic readout is inside ``max_km``.

    Measured 2026-10-23: the camera sat at 24-25 km (sky) while the target was
    8 km away. A horizontal search cannot find a contact that far below the
    band, and the km-based closure refuses to act without a target reading, so
    nothing ever moved. ``move`` applies ``(0, +step)``: positive y pitches down.
    """
    trace = []
    previous = object()
    stalls = 0
    for round_index in range(max(0, rounds)):
        aim_km = measure(round_index)
        trace.append(dict(round=round_index, aim_km=aim_km))
        if aim_km is not None and aim_km <= max_km:
            return trace, 'readable'
        if aim_km is not None and aim_km == previous:
            stalls += 1
            if stalls >= max(1, stall_rounds):
                # A drawn number that never changes: the action does nothing, so
                # stop instead of grinding. An *absent* readout is different --
                # that is exactly the state a descent is meant to leave.
                return trace, 'not_responding'
        else:
            stalls = 0
        previous = aim_km
        move(0, step)
    return trace, 'not_reached'


def close_range_steps(measure, move, tolerance_km, km_per_unit, step_units,
                      rounds=4, max_units=MAX_VERTICAL_STEP_UNITS):
    """Iterate the ballistic-range closure, one bounded vertical action at a time.

    ``measure`` returns ``(aim_km, target_km)`` from a *fresh* frame and ``move``
    applies ``(0, units)``. The vertical action limit is per action, never per
    round, so a 3 km gap still closes in a few bounded steps. Returns the trace
    so the report can show why it stopped ('closed', 'unavailable' or 'rounds').
    """
    trace = []
    for round_index in range(max(0, rounds)):
        aim_km, target_km = measure(round_index)
        units, why = range_correction(aim_km, target_km, tolerance_km, km_per_unit, step_units)
        trace.append(dict(round=round_index, aim_km=aim_km, target_km=target_km,
                          units=units, why=why))
        if why != 'move':
            return trace
        remaining = int(units)
        guard = 0
        while remaining and guard < 8:
            step = max(-max_units, min(max_units, remaining))
            move(0, step)
            remaining -= step
            guard += 1
    return trace


def recentre_pointer(probe, report, key, settle=1.0, reason='recentre'):
    """Recover the whole pointer travel by toggling the tactical map twice.

    Measured 2026-10-23 in a training battle: with the pointer parked at x=1919
    every further rightward chunk was delivered as 0 px (pointer_delta [0, 0]),
    so the correction was silently lost and the next residual grew. Pressing M
    twice put the pointer back at (961, 1005) while the *camera stayed put*, and
    rotation continued past the old limit (one nameplate walked from x=2292 to
    x=451 across two 1200-unit legs). This is the practical replacement for the
    300-unit retreat, which costs rotation and only returns 300 units.
    """
    probe.key(key)
    time.sleep(settle)
    probe.key(key)
    time.sleep(settle)
    pointer = pointer_state(probe.identity['display'])
    report['inputs'].append(dict(key=key, reason=f'{reason}_centre', pointer=pointer))
    return pointer


def ensure_room(probe, report, dx, width, reason, margin=60, recentre_key=None):
    """Reposition when a horizontal leg has no pointer travel left.

    Measured 2026-10-23: with the seat pointer parked at the window edge a
    rightward chunk is delivered as ~0 px of camera rotation (learned gain
    0.006 px/unit), so the solve looks broken while the transport is simply out
    of room. A 300-unit retreat costs one input and frees the leg.

    libei delivers relative motion without pointer travel, so the retreat would
    only cost an input and add unrequested yaw: it is skipped entirely.
    """
    if uses_ei_transport(probe):
        note_transport_skip(report, probe, f'{reason}_reposition', dx=dx)
        return False
    if abs(dx) < 40:
        return False
    pointer = pointer_state(probe.identity['display'])
    room = (width - margin - pointer['x']) if dx > 0 else (pointer['x'] - margin)
    if room >= 60:
        return False
    if recentre_key:
        recentre_pointer(probe, report, recentre_key, reason=reason)
        return True
    back = -300 if dx > 0 else 300
    probe.move(back, 0)
    report['inputs'].append(dict(relative=[back, 0], reason=f'{reason}_reposition'))
    time.sleep(.05)
    return True


def fast_shot_plan(target_px, reticle_px, gain, limit=(400., 300.)):
    """One open-loop move for the whole aim error, clamped per axis.

    A live engagement does not allow the closed loop: measured end to end the
    scan/probe/converge path takes 15-50 s while a useful shot needs under 5 s.
    The static-target case only needs one frame, one solve and one move; the
    gain is a per-view constant measured offline.
    """
    dx, dy = open_loop_delta(target_px, reticle_px, gain)
    return [max(-limit[0], min(limit[0], dx)), max(-limit[1], min(limit[1], dy))]


def ladder_units(hint_px, gain_y, max_units):
    """Vertical units toward a plate that is off-frame after the zoom.

    ``hint_px`` is the plate's wide-view offset from the reticle (negative =
    above it). The zoom multiplies that offset by the magnification, so only its
    sign is trusted; the magnitude is capped and damped. Without a measured
    vertical gain there is no honest direction to send, so this returns 0 and
    the caller falls back to closing the scope.
    """
    if hint_px is None or not gain_y:
        return 0.0
    raw = -float(hint_px) / float(gain_y) * 0.6
    limit = abs(float(max_units))
    if abs(raw) < 1.0:
        raw = math.copysign(1.0, raw)
    return max(-limit, min(limit, raw))


def resolve_fire_tolerance(plate_bbox, ceiling, floor=18.0):
    """Fire tolerance from the drawn plate's own width, or None without a plate.

    A fixed 50 px is a third of a 150 px plate but twice a 25 px one: against a
    close, wide plate it refuses shots that are already inside the drawn ship,
    and against a distant, narrow plate it fires while the reticle is still off
    the hull. 0.30 of the plate width keeps the reticle inside the drawn target
    at every range (measured plates: 123 px at 13 km and 124 px at 2.7 km; the
    drawn hull is 1.2-1.6 bars wide). The ceiling stays the operator's value so
    this can never loosen the gate beyond what was asked for.
    """
    if not plate_bbox or len(plate_bbox) < 4:
        return None, 'no_plate'
    try:
        width = float(plate_bbox[2])
    except (TypeError, ValueError):
        return None, 'no_plate'
    if not width > 0:
        return None, 'no_plate'
    return max(float(floor), min(float(ceiling), 0.30 * width)), 'plate_width'


def fire_ready(distance_px, converged_steps, step, tolerance_px, min_steps=4):
    """Require two current tight solutions and the observation warmup."""
    if distance_px <= tolerance_px and converged_steps >= 2 and step >= min_steps:
        return True, 'fresh_confirmed_solution'
    return False, None


def judged_residual(residual_px, lead_px):
    """The error that decides the shot: distance to the PREDICTED point.

    With lead the reticle sits ahead of the ship on purpose, so the distance to
    the ship itself would veto every correctly-led shot.
    """
    if residual_px is None:
        return None
    lead = lead_px or [0.0, 0.0]
    return [residual_px[0] + lead[0], residual_px[1] + lead[1]]


SCREEN_STILL_PX_S = 2.0
RANGE_MOVING_KM_S = 0.003


def lead_plan(screen, ranged, screen_speed_px_s=None):
    """Which lead geometry the observed target calls for.

    The game's target lock makes the camera follow the ship's bearing, so a
    locked ship does not move on screen while its range still changes. Using the
    screen offset there would aim at nothing (measured 2026-10-23: a locked run
    reported 19 px/s from stale frames and the lead never converged). The choice
    is made from the observation, not from whether we pressed the lock key:

    * the ship slides on screen            -> screen lead;
    * the ship is still on screen but the range moves -> range lead (lock);
    * neither signal is usable             -> no lead, with the refusals named.
    """
    speed = abs(float(screen_speed_px_s)) if screen_speed_px_s is not None else None
    screen_ok = bool(screen and screen.get('applied'))
    range_ok = bool(ranged and ranged.get('applied'))
    if speed is not None and speed >= SCREEN_STILL_PX_S:
        if screen_ok:
            return dict(mode='screen', **screen)
        return dict(mode='screen', applied=False, reason=(screen or {}).get('reason', 'unusable'),
                    screen_speed_px_s=round(speed, 2))
    if range_ok and abs(float(ranged.get('rate_km_s') or 0)) >= RANGE_MOVING_KM_S:
        plan = dict(ranged)
        plan['mode'] = 'range'
        plan['screen_speed_px_s'] = None if speed is None else round(speed, 2)
        return plan
    if speed is None:
        # No screen series at all: fall back to the range evidence if it exists.
        if range_ok:
            plan = dict(ranged)
            plan['mode'] = 'range'
            return plan
        return dict(mode='none', applied=False, reason='no_lead_evidence')
    return dict(mode='none', applied=False, reason='target_effectively_still',
                screen_speed_px_s=round(speed, 2),
                range_rate_km_s=(ranged or {}).get('rate_km_s'))


def predicted_range_km(samples, flight_s, minimum=3, min_span_s=1.5,
                       max_rate_km_s=0.08, max_shift_km=3.0):
    """Where the locked target will be in range terms when the shells arrive.

    Under the game's target lock the camera follows the ship's bearing, so the
    target does not move on screen and a screen-space lead is meaningless. What
    still changes is its range, and the shells have to be aimed at the range the
    ship will have after ``flight_s``: ``predicted = last + rate * flight``.

    ``samples`` are ``(t, km)`` pairs; the rate is a least-squares slope so a
    single noisy caption cannot define it. Everything is refused by name.
    """
    if flight_s is None:
        return dict(applied=False, reason='no_flight_time', samples=len(samples))
    if not samples or len(samples) < minimum:
        return dict(applied=False, reason='insufficient_samples', samples=len(samples))
    times = [s[0] for s in samples]
    span = times[-1] - times[0]
    if span < min_span_s:
        return dict(applied=False, reason='baseline_too_short',
                    samples=len(samples), span_s=round(span, 2))
    mean_t = sum(times) / len(times)
    variance = sum((t - mean_t) ** 2 for t in times)
    if variance <= 1e-9:
        return dict(applied=False, reason='degenerate_times', samples=len(samples))
    mean_km = sum(s[1] for s in samples) / len(samples)
    rate = (sum((t - mean_t) * (km - mean_km) for t, km in samples) / variance)
    if abs(rate) > max_rate_km_s:
        return dict(applied=False, reason='rate_out_of_range',
                    rate_km_s=round(rate, 4), samples=len(samples))
    shift = rate * float(flight_s)
    clamped = max(-max_shift_km, min(max_shift_km, shift))
    return dict(applied=True, rate_km_s=round(rate, 4),
                shift_km=round(clamped, 2), predicted_km=round(samples[-1][1] + clamped, 2),
                last_km=round(samples[-1][1], 2), flight_s=round(float(flight_s), 2),
                samples=len(samples), span_s=round(span, 2),
                clamped=bool(abs(shift) > max_shift_km))


def unmoved_samples(samples, width=None):
    """Longest run of samples that share one input count.

    A velocity fitted across one of our own corrections measures the correction,
    not the target: a run measured 26.7 px/s of "target motion" that was the aim
    move itself and threw a 125 px lead, which then blocked the shot. Samples
    carry the number of inputs issued when they were taken, so a run with a
    constant count is by construction a run the mouse did not disturb.
    """
    best, current, count = [], [], object()
    for sample in samples:
        # The count is whatever follows the sample's own fields, so it is read
        # from `width` rather than a fixed index: screen samples are
        # (t, x, y, count) and range samples are (t, km, count).
        n = sample[width] if width and len(sample) > width else None
        if current and n == count:
            current.append(sample)
        else:
            current = [sample]
            count = n
        if len(current) > len(best):
            best = list(current)
    # width says how many leading fields are the sample itself (screen
    # samples are (t, x, y, count), range samples are (t, km, count)); the
    # trailing count is dropped so the fitter sees exactly what it expects.
    return [tuple(s[:width]) if width else tuple(s) for s in best]


def moving_target_lead(samples, flight_s, ceiling=150.0, minimum=3, min_span_s=1.2):
    """Preliminary lead for a moving target, or a named refusal.

    ``samples`` are ``(t, x, y)`` observations of the target's screen point taken
    **without moving the mouse** in between: a mouse move rotates the camera and
    would be read as the target's own motion. With a fitted velocity and the
    game's own flight time, the shells have to be aimed ``v * t_flight`` ahead.
    """
    if flight_s is None:
        return dict(applied=False, reason='no_flight_time', samples=len(samples))
    if not samples or len(samples) < minimum:
        return dict(applied=False, reason='insufficient_samples', samples=len(samples))
    velocity = fit_velocity(samples, minimum=minimum, min_span_s=min_span_s)
    if velocity is None:
        span = round(samples[-1][0] - samples[0][0], 2)
        return dict(applied=False, reason='baseline_too_short',
                    samples=len(samples), span_s=span)
    lead = lead_offset(velocity, flight_s, ceiling=ceiling)
    if abs(lead[0]) < 1.0 and abs(lead[1]) < 1.0:
        return dict(applied=False, reason='target_effectively_still',
                    velocity_px_s=[round(v, 1) for v in velocity],
                    lead_px=[round(v, 1) for v in lead], samples=len(samples))
    return dict(applied=True, velocity_px_s=[round(v, 1) for v in velocity],
                lead_px=[round(v, 1) for v in lead], flight_s=round(float(flight_s), 2),
                samples=len(samples),
                span_s=round(samples[-1][0] - samples[0][0], 2))


def lead_offset(velocity_px_s, seconds, ceiling=400.0):
    """Screen offset to aim ahead of a moving target, bounded per axis.

    velocity is the target's motion relative to the reticle, which is the frame
    the aim is expressed in, so the offset is what the shells have to lead. The
    arithmetic is aim_lead's; only the refuse-unknown and clamp rules are here.
    """
    if velocity_px_s is None or not seconds:
        return [0.0, 0.0]
    raw = aim_lead.lead_offset(velocity_px_s, seconds)
    return [max(-ceiling, min(ceiling, v)) for v in raw]


def reposition_move(pointer, width, height, move, margin=40, step=400):
    """The relative retreat needed so `move` can be delivered in full.

    The X pointer stops at a screen edge, and an absolute warp is ignored by the
    game (measured), so room has to be made with relative motion. The retreat
    rotates the camera the wrong way; the caller re-measures afterwards instead
    of learning a gain from that step.
    """
    retreat = [0, 0]
    for axis, delta in ((0, move[0]), (1, move[1])):
        if not delta:
            continue
        extent = width if axis == 0 else height
        room = extent - 1 - pointer[axis] if delta > 0 else pointer[axis]
        if room >= abs(delta) + margin:
            continue
        # Retreat by exactly the missing room: a larger retreat rotates the
        # camera the wrong way for longer and shows up as oscillation.
        shortfall = abs(delta) + margin - room
        if delta > 0:
            retreat[axis] = -min(shortfall, max(0, pointer[axis]))
        else:
            retreat[axis] = min(shortfall, max(0, extent - 1 - pointer[axis]))
    return retreat if any(retreat) else None


def recentre_move(pointer, width, height):
    """Relative move that puts the pointer mid-screen on both axes.

    Applied before the first measurement so the rotation it causes is absorbed
    into the initial error instead of biasing a correction.
    """
    return [int(width // 2 - pointer[0]), int(height // 2 - pointer[1])]


def window_fits(frame, bbox, margin=80):
    """True when an 80 px tracking window around the point stays in the scene."""
    x, y = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
    return (margin <= x < frame.shape[1] - margin and
            margin <= y - margin and y + margin < frame.shape[0] - 80)


def track_search(last_move, gain, floor=160, ceiling=900):
    """Search radius for the next track, sized by the move just sent.

    A short radius loses a target after a large yaw, so the radius follows the
    displacement the last move was expected to produce.
    """
    if last_move is None:
        return floor
    expect_x = abs(last_move[0]) * abs(gain[0] or 1.2)
    expect_y = abs(last_move[1]) * abs(gain[1] or 1.2)
    return int(max(floor, min(ceiling, 2.0 * max(expect_x, expect_y) + 40)))


def crop_template(frame, bbox, min_texture=5.0):
    """Crop a trackable template, refusing a flat or out-of-frame window."""
    x, y, w, h = [int(v) for v in bbox]
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x+w > frame.shape[1] or y+h > frame.shape[0]:
        return None
    template = frame[y:y+h, x:x+w].copy()
    if template.shape[0] != h or template.shape[1] != w or float(template.std()) < min_texture:
        return None
    return template


def predict_patch(bbox, velocity_px_s, dt_s, max_shift=160.0):
    """Where the patch should be next step, from the velocity already fitted.

    The reference bot predicts with a Kalman filter before it matches; this is
    the light equivalent, and it matters because the search window otherwise has
    to cover the camera move and the target's motion at once. A missing or
    implausible velocity leaves the box alone rather than inventing motion.
    """
    if not velocity_px_s or not dt_s or dt_s <= 0:
        return list(bbox)
    dx = float(velocity_px_s[0]) * float(dt_s)
    dy = float(velocity_px_s[1]) * float(dt_s)
    scale = max(abs(dx), abs(dy))
    if scale > max_shift and scale > 0:
        shrink = max_shift / scale
        dx *= shrink
        dy *= shrink
    return [bbox[0] + dx, bbox[1] + dy, bbox[2], bbox[3]]


def track_patch(template, previous, frame, search=180, floor=.45):
    """Find the stored template near its previous centre in a new frame.

    The template comes from the frame the point was commanded on, so a moving
    scene produces a real displacement instead of matching the new frame
    against itself. Returns None when the window is unusable or the best score
    is below floor, rather than guessing a position.
    """
    if template is None:
        return None
    x, y, w, h = [int(v) for v in previous]
    x0, y0 = max(0, x-search), max(0, y-search)
    x1, y1 = min(frame.shape[1], x+w+search), min(frame.shape[0], y+h+search)
    if x1-x0 < w or y1-y0 < h:
        return None
    window = frame[y0:y1, x0:x1]
    scores = cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED)
    _, best, _, loc = cv2.minMaxLoc(scores)
    if not math.isfinite(best) or best < floor:
        return None
    return dict(bbox=[x0+loc[0], y0+loc[1], w, h], score=float(best))


def patch_centre(bbox):
    return bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2


def nameplate_patch(frame, candidate):
    """Track the detected red bar; keep the commanded aim offset separate.

    A hull search window can overlap the fixed minimap and lock onto its edge.
    The bar is the observed anchor, not a verified hull or waterline.
    """
    bbox = list(candidate['nameplate_bbox'])
    template = crop_template(frame, bbox)
    if template is None:
        raise ProbeError('enemy nameplate has no trackable texture')
    centre = patch_centre(bbox)
    offset = [candidate['aim_px'][axis] - centre[axis] for axis in (0, 1)]
    return dict(bbox=bbox, score=float(candidate['confidence'])), template, offset



def find_enemy(frame, reticle=None, aim_fraction=None, *, scope_observed=False):
    """Find red enemy plates and legacy geometric aim candidates, not learned bows.

    The nameplate search masks HUD regions through the frame's own projection
    (measured in ``main_view_perception``), so this client's minimap and ammo
    bar are excluded where they actually are.
    """
    from main_view_perception import detect_nameplates, verify_ship_bodies
    candidates = []
    plates = (detect_nameplates(frame, scope_observed=True)
              if scope_observed
              else detect_nameplates(frame))
    # A plate can be drawn for a ship whose hull is not rendered at all (smoke,
    # island, radar-only contact). Measured 2026-10-20: every salvo at such a
    # plate missed while the nameplate looked perfect, so the shot gate may
    # require a rendered hull.
    import os as _os
    require_body = _os.environ.get('WW_REQUIRE_BODY', '0') == '1'
    hidden = 0
    for plate in verify_ship_bodies(frame, plates):
        if require_body and not plate.get('body_visible'):
            hidden += 1
            continue
        body = plate.get('body_roi') or plate['nameplate_bbox']
        bar = plate['nameplate_bbox']
        if aim_fraction is None:
            aim, aim_meta = hull_aim_point(frame, bar)
        else:
            fraction = min(1.0, max(0.0, aim_fraction))
            aim = [body[0] + body[2] / 2, body[1] + body[3] * fraction]
            aim_meta = dict(source='body_window_fraction', fraction=fraction)
        distance = None
        if reticle is not None:
            distance = math.hypot(aim[0] - reticle[0], aim[1] - reticle[1])
        candidates.append(dict(nameplate_bbox=list(bar), body_roi=list(body),
                               body_visible=plate.get('body_visible'),
                               red_px=plate.get('red_pixels'),
                               aim_px=[round(v, 1) for v in aim], aim_source=aim_meta,
                               trackable=window_fits(frame, body),
                               confidence=plate.get('confidence'),
                               distance_from_reticle_px=None if distance is None else round(distance, 1)))
    if reticle is not None:
        candidates.sort(key=lambda c: (not c['trackable'], c['distance_from_reticle_px']))
    else:
        candidates.sort(key=lambda c: (not c['trackable'], -float(c.get('confidence') or 0)))
    find_enemy.last_hidden = hidden
    return candidates


def yaw_units_for_degrees(degrees, gain_px_per_unit, fov_deg=68.2, screen_px=2560.0):
    """Mouse units that turn the camera by ``degrees`` at the measured gain.

    Measured 2026-10-23: the gain on this client is ~0.12 px/unit, so a fixed
    300-unit search step moved the view by about one degree and a 20-step sweep
    covered ~20 degrees of a 360-degree search.
    """
    gain = abs(float(gain_px_per_unit or 0.0))
    if gain <= 0 or degrees <= 0:
        return 0.0
    return degrees * (screen_px / max(fov_deg, 1e-6)) / gain


def scan_coverage_deg(steps_per_row, step, gain_px_per_unit=1.5, fov_deg=68.0,
                       screen_w=2560.0):
    """Degrees of yaw one horizontal sweep actually covers.

    The operator asks for a 360-degree look before the first shot; a sweep that
    only spans 38 degrees (12 steps x 80 units, the old default) is not one. The
    conversion uses the measured third-person gain and the client FOV.
    """
    units = max(0, steps_per_row) * abs(step)
    return (units * abs(gain_px_per_unit) / float(screen_w)) * fov_deg


def scan_path(steps_per_row, rows, step, tilt, *, reach_y=3, budget=None):
    """Horizontal sweeps with a tilt between rows.

    A single horizontal ring never finds a contact that sits above or below it,
    so each new row tilts the camera and sweeps back the other way. Bounded:
    the caller walks this list once.
    """
    moves = []
    position = 0
    direction = 1
    for row in range(max(1, rows)):
        for _ in range(max(0, steps_per_row - 1)):
            if budget is not None and abs(position + direction * step) > budget:
                # Optional cap for callers that must not leave a small arc.  It
                # is OFF for a real search: with the old 700-unit default the
                # "360-degree" sweep reversed every ~1.75 steps and only ever
                # covered about +-28 degrees, so a contact outside that arc was
                # never seen (measured 2026-10-21: 35 scan frames, no plate).
                # Clipping is handled by repositioning the pointer, not by never
                # turning.
                direction = -direction
            position += direction * step
            moves.append((direction * step, 0))
        if row + 1 < max(1, rows):
            moves.append((0, max(-reach_y, min(reach_y, tilt))))
            direction = -direction   # documented row alternation
    return moves


def read_frame(path):
    image = cv2.imread(str(path))
    if image is None or image.shape != (1600, 2560, 3):
        raise ValueError('snapshot must be a native 2560x1600 RGB frame')
    return np.ascontiguousarray(image[:, :, ::-1])


def run(args):
    global HULL_AIM_FRACTION
    HULL_AIM_FRACTION = float(args.hull_gap_fraction)
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(action='visual_servo_aim', execute=args.execute,
                  relative_transport=getattr(args, 'relative_transport', 'xtest'),
                  target=None if args.target is None else list(args.target),
                  tolerance_px=args.tolerance, max_steps=args.max_steps, fire=args.fire,
                  lock=args.lock, post_fire_frames=args.post_fire_frames,
                  live_source=args.live_source,
                  scope=args.scope,
                  aim_fraction=args.aim_fraction, aim_nudge=args.aim_nudge,
                  fire_tolerance_px=args.fire_tolerance, fire_min_steps=args.fire_min_steps,
                  compensate_drift=args.compensate_drift, scope_after_px=args.scope_after,
                  lead_seconds=args.lead_seconds,
                  lead_min_samples=args.lead_min_samples, lead_ceiling=args.lead_ceiling,
                  settle_s=args.settle, reach=args.reach, reach_y=args.reach_y, damping=args.damping,
                  out=str(args.out.resolve()), steps=[], inputs=[], scan=[], status='dry-run',
                  snapshots=[], focus_checks=[], warnings=[],
                  limits=dict(seconds=args.seconds, capture_space='native 2560x1600 frame',
                              probe_mouse=args.probe, probe_mouse_y=min(args.probe_y, args.reach_y),
                              scan_tilt_mouse=args.scan_tilt),
                  semantics='visual servo on one commanded scene point; no friend/foe identity, no hit claim')
    (args.out/'plan.json').write_text(json.dumps(report, indent=2)+'\n')
    if not args.execute:
        print(json.dumps(report, indent=2))
        return 0
    probe = Probe(args.seconds, report)
    # The probe reads the transport from its environment at construction; push
    # the CLI choice onto the instance too in case a probe was built earlier.
    apply_relative_transport(probe, getattr(args, 'relative_transport', 'xtest'))
    pointer_probe = probe
    def move_pointer(dx, dy, vertical_limit=None):
        # The coarse elevation phase is allowed a larger bounded vertical
        # action; every other path keeps the small fine-step limit.
        return bounded_move(probe, dx, dy,
                            args.reach_y if vertical_limit is None else vertical_limit,
                            chunk=getattr(args, 'move_chunk', 0))

    held_button = None
    handlers = {}
    timer = (0., 0.)
    old_pointer = None
    try:
        def stop(signum, frame):
            raise ProbeError(f'interrupted by signal {signum}')
        scope_entered = False
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM):
            handlers[sig] = signal.signal(sig, stop)
        timer = signal.setitimer(signal.ITIMER_REAL, max(.001, probe.action_deadline-time.monotonic()))
        probe.identity = game_identity()
        report['identity'] = probe.identity
        windows = []
        for window in probe.x('search', '--onlyvisible', '--name', '.').splitlines():
            try:
                probe.verify_window(window); windows.append(window)
            except ProbeError:
                continue
        if len(windows) != 1:
            raise ProbeError('expected exactly one verified game window')
        probe.window = windows[0]
        geometry = dict(line.split('=', 1) for line in
                        probe.x('getwindowgeometry', '--shell', probe.window).splitlines())
        width, height = int(geometry['WIDTH']), int(geometry['HEIGHT'])
        report['window_geometry'] = [width, height]
        old_pointer = pointer_state(probe.identity['display'])
        if old_pointer['mask'] & (256 | 512 | 1024 | 2048 | 4096):
            raise ProbeError('pointer button already held; refusing ownership')
        last_frame_ns = None
        scope_empty_reads = 0
        def check_scoped_target(frame):
            """In the scope: a frame with no enemy closes it and returns to coarse aim.

            The user's procedure moves back a stage instead of ending the run, so
            the scope is exited, every scope-specific belief (gain, patch, speed,
            convergence) is dropped, and the wide-view loop resumes. Bounded by
            --scope-retries so a target that never appears cannot loop forever.
            """
            nonlocal scope_entered, scope_empty_reads, scope_for_aim, tracking_candidate
            nonlocal gain, gain_state, weak_moves, history, pan_log
            nonlocal velocity_ema, velocity_samples, previous_error, previous_captured
            nonlocal last_move, previous_frame, template, next_search, converged
            if not scope_entered:
                return frame
            found = find_enemy(frame, scope_observed=True)
            if any(c['trackable'] for c in found):
                scope_empty_reads = 0
                return frame
            scope_empty_reads += 1
            action = scope_empty_action(scope_empty_reads, args.scope_empty_grace,
                                        report.get('scope_returns_to_coarse', 0),
                                        args.scope_retries)
            if action == 'hold':
                # The zoom transition takes a few frames; do not exit on the
                # first one, and never sweep blind (that turned the camera away
                # from a target that was in fact present).
                report.setdefault('scope_empty_grace', []).append(scope_empty_reads)
                return frame
            steps_used = int(report.get('scope_ladder_moves') or 0)
            ladder_cap = int(getattr(args, 'scope_ladder_steps', 0) or 0)
            if action != 'hold' and steps_used < ladder_cap:
                units = ladder_units(report.get('scope_vertical_hint_px'),
                                     (gain or [None, None])[1],
                                     getattr(args, 'scope_ladder_units', 12))
                if units:
                    move_pointer(0, units, vertical_limit=max(1, abs(int(round(units)))))
                    report['inputs'][-1].update(reason='scope_ladder')
                    report['scope_ladder_moves'] = steps_used + 1
                    report.setdefault('scope_ladder', []).append(
                        dict(hint_px=report.get('scope_vertical_hint_px'),
                             units=round(units, 2),
                             gain_y=(gain or [None, None])[1]))
                    if args.settle:
                        time.sleep(args.settle)
                    return next_frame('scope-ladder')
            probe.key(args.scope_key)
            scope_entered = False
            report['inputs'].append(dict(key=args.scope_key, reason='exit_empty_scope'))
            report['scope_rejection'] = 'enemy_healthbar_missing'
            exits = report.get('scope_returns_to_coarse', 0) + 1
            report['scope_returns_to_coarse'] = exits
            scope_for_aim = False
            gain, gain_state = [None, None], {}
            weak_moves = {0: 0, 1: 0}
            history, pan_log = [], []
            velocity_ema, velocity_samples = None, 0
            previous_error = previous_captured = last_move = previous_frame = None
            template, next_search, converged = None, 160, 0
            scope_empty_reads = 0
            # The patch belongs to the zoomed view; keeping it after the exit
            # means the loop aims at a stale box with third-person limits while
            # the target sits somewhere else entirely (measured 2026-10-29:
            # scope_for_aim False, reach_y 3, error 743 px at step 0). Drop the
            # acquisition so the next step re-detects in the wide view.
            tracking_candidate = None
            report['scope_exit_cleared_target'] = True
            if action == 'give_up':
                raise ProbeError(f'scope exited without a target {exits} times; '
                                 'the wide view never put one in the zoom')
            return frame

        def next_frame(name, reuse=False):
            """Newest frame through the fast persistent-pipeline grab, or a PNG snapshot."""
            nonlocal last_frame_ns
            next_frame.name = name
            if args.live_source:
                path = Path(args.out) / f'{name}.ppm'
                probe.command(['/usr/bin/python3', str(HERE / 'capture_host.py'),
                               'grab', str(path)])
                frame = read_frame(path)
                sidecar = Path(str(path) + '.json')
                if sidecar.exists():
                    try:
                        next_frame.meta = json.loads(sidecar.read_text())
                    except (OSError, ValueError):
                        next_frame.meta = None
                else:
                    next_frame.meta = None
                last_frame_ns = frame_capture_ns(next_frame.meta, time.monotonic_ns(), last_frame_ns,
                                                 checkpoint='after_frame_read', frame=name)
                return check_scoped_target(frame)
            next_frame.meta = None
            # Only the aiming read asks for reuse: stubs keep the plain call,
            # and a confirmation always grabs a fresh picture.
            # grab() returns a raw PPM: measured 52-77 ms against 1314-1346 ms
            # for the PNG snapshot (PNG encoding dominates), i.e. ~1.3 s saved on
            # every frame -- two frames per step, which is exactly the stutter
            # felt while aiming. PPM decodes in 15 ms against 126 ms.
            getter = getattr(probe, 'grab', None) or probe.snapshot
            if reuse:
                try:
                    path = getter(name, allow_reuse=True)
                except TypeError:
                    path = getter(name)
            else:
                path = getter(name)
            return check_scoped_target(read_frame(path))

        next_frame.meta = None
        next_frame.name = None

        def check_frame(checkpoint):
            return frame_capture_ns(next_frame.meta, time.monotonic_ns(),
                                    checkpoint=checkpoint, frame=next_frame.name)

        def confirm_fire(name, expected=None, lead=(0., 0.), fallback=None,
                         allow_missing=False, plate_fallback=None):
            # A lock wait or a scope change invalidates the earlier solution.
            # Confirm on a new frame, then recheck age after the focus query.
            use_plate_fallback = False
            current = next_frame(name)
            centre, active, weapon, source = reticle_centre(current)
            # An assumed frame centre may steer the coarse phase, but it must
            # never authorise a shot in third person: a missing reticle means the
            # gun is pointed at something nobody measured. Inside the scope the
            # centre is the aim point by construction (ruler centre 1279 against
            # a 1280 frame centre), so that case still fires.
            # A shot needs an actually observed reticle. The assumed frame centre
            # may steer, never authorise -- including inside the scope, where the
            # ruler centre is observed and the fallback only exists for dark
            # frames.
            # An assumed centre may steer the aim, never authorise the shot --
            # including inside the scope, where only the ruled centre is observed.
            if source in (None, 'scope_screen_center', 'frame_center_assumed'):
                raise ProbeError('fire revoked: reticle_not_observed')
            if active is not True or centre is None:
                # The green ring is only painted after mouse motion, so an idle
                # frame can read "missing" while the reticle is merely undrawn.
                # Wiggle to redraw it, but do NOT authorise the shot from the
                # reread in this same call: measured while a stale centre was
                # cached, that let a shot leave with no observed reticle at all.
                # The wiggle still helps, because the caller's confirmation loop
                # reads a fresh frame next iteration.
                probe.move(6, 0)
                report['inputs'].append(dict(relative=[6, 0], reason='confirm_reticle_redraw'))
                time.sleep(.05)
                probe.move(-6, 0)
                report['inputs'].append(dict(relative=[-6, 0],
                                             reason='confirm_reticle_redraw_back'))
                time.sleep(.15)
                if source in (None, 'scope_screen_center', 'frame_center_assumed'):
                    raise ProbeError('fire revoked: reticle_not_observed')
                current = next_frame(f'{name}-reread')
                centre, active, weapon, source = reticle_centre(current)
            evidence = dict(battle_active=active, reticle=centre,
                            reticle_source=source, weapon_status=weapon,
                            confirmed=False)
            report['fire_confirmation'] = evidence
            if centre is None and active is True and camera_mode_evidence(current).get('binocular'):
                # In the binocular view the aim point is the camera centre by
                # construction; when neither the ring nor the ruler is readable
                # the centre prior is used and labelled as such.
                centre = [1280.5, 805.0]
                source = 'scope_center_prior'
                evidence['reticle_prior'] = True
            if active is not True or centre is None:
                raise ProbeError('fire revoked: current battle HUD or reticle missing')
            battery = main_battery_fire_evidence(current)
            scoped_frame = bool(camera_mode_evidence(current).get('binocular'))
            if (fallback and scoped_frame
                    and battery['weapon'].get('kind') != 'gun'
                    and battery['weapon'].get('kind') != 'torpedo'):
                # Nothing in this run changes the selection between the coarse
                # frame and the scoped one; if the scope hid the sprite, the
                # read from ~1.5 s earlier is the current truth and is named.
                battery = dict(battery)
                battery['weapon'] = fallback['weapon']
                evidence['weapon_evidence'] = 'pre_scope_frame'
            if (fallback and scoped_frame
                    and battery['reload'].get('status') not in ('observed', 'partial')):
                # The scope darkens the turret bars, so the reload read can fail
                # on the confirmation frame while the pre-scope frame a second
                # earlier proved it. Use that, and say so.
                battery = dict(battery)
                battery['reload'] = fallback['reload']
                evidence['reload_evidence'] = 'pre_scope_frame'
            evidence['main_battery'] = battery
            outlined = [slot for slot in (battery['weapon'].get('raw') or {}).get('slots', [])
                        if slot.get('selected_outline') and slot.get('ammo') in ('HE', 'AP')]
            gun_confirmed = (battery['weapon'].get('status') == 'observed'
                             and battery['weapon'].get('kind') == 'gun')
            if not gun_confirmed and scoped_frame and outlined:
                # The scope dims the sprite match, but a highlighted HE/AP slot
                # is still the main battery selection (measured on a live scoped
                # frame: icon null, edge rows 59/73).
                gun_confirmed = True
                evidence['weapon_evidence'] = 'scope_outlined_shell_slot'
            ready = battery['reload'].get('ready')
            if (ready is not True and fallback
                    and (fallback.get('reload') or {}).get('ready') is True):
                # Reload state never regresses without a shot, so a True read
                # from the pre-scope frame is still current. The scope darkens
                # the bars, and the gates want a real True, not a guess.
                battery = dict(battery)
                battery['reload'] = fallback['reload']
                ready = True
                evidence['reload_evidence'] = 'pre_scope_frame'
            # Reloading is a wait, not a refusal: the aim is already where it
            # needs to be, so closing the whole engagement on reload_not_ready
            # throws away a good solution and spends the outer loop's attempts.
            # Poll the HUD for a bounded time and re-decide with the new value.
            if (ready is False and getattr(args, 'confirm_reload_wait_s', 0) > 0
                    and battery['reload'].get('status') in ('observed', 'partial')):
                waited = 0.0
                poll_s = max(0.2, float(getattr(args, 'poll_reload_s', 0.5)))
                while waited < args.confirm_reload_wait_s:
                    time.sleep(min(poll_s, max(0.05, args.confirm_reload_wait_s - waited)))
                    waited += poll_s
                    reload_frame = next_frame(f'{name}-reload')
                    view_now = fire_hud.FireHudReader().detect(reload_frame)
                    reload_now = (view_now.get('reload') or {})
                    if reload_now.get('status') not in ('observed', 'partial'):
                        continue
                    if reload_now.get('ready') is True:
                        battery = dict(battery)
                        battery['reload'] = reload_now
                        ready = True
                        report.setdefault('confirm_reload_wait', []).append(
                            dict(waited_s=round(waited, 2), ready=True))
                        break
                else:
                    report.setdefault('confirm_reload_wait', []).append(
                        dict(waited_s=round(waited, 2), ready=ready))
            scope = hud_scope_evidence(current) if (args.scope or args.scope_after) else None
            camera_view = camera_mode_evidence(current) if (args.scope or args.scope_after) else None
            evidence['scope_evidence'] = scope
            evidence['scope_camera_evidence'] = camera_view
            candidates = [c for c in find_enemy(current, centre, args.aim_fraction)
                          if c['trackable']
                          and (expected is None or math.dist(c['aim_px'], expected) <= args.fire_tolerance)
                          and math.hypot(c['aim_px'][0] + lead[0] - centre[0],
                                         c['aim_px'][1] + lead[1] - centre[1]) <= args.tolerance]
            evidence['candidates'] = candidates
            evidence['captured_s'] = round(time.monotonic_ns() / 1e9, 3)
            evidence['distance_px'] = (None if len(candidates) != 1 else math.hypot(
                candidates[0]['aim_px'][0] + lead[0] - centre[0],
                candidates[0]['aim_px'][1] + lead[1] - centre[1]))
            if len(candidates) != 1:
                residuals = (report.get('fast') or {}).get('residuals') or []
                last = residuals[-1] if residuals else None
                # A residual inside the firing tolerance is a solved shot: at
                # 10 km 60 px is about one ship length, and the last round did
                # measure the plate. It is recorded as the previous frame's
                # evidence; whether that is enough to fire is the gate's call,
                # and the gate requires a plate from this frame.
                solved = last is not None and math.hypot(*last) <= args.fire_tolerance
                route_steps = report.get('steps') or []
                route_last = route_steps[-1].get('error_px') if route_steps else None
                route_solved = (route_last is not None
                                and math.hypot(*route_last) <= args.fire_tolerance)
                if not solved and route_solved:
                    evidence['residual_source'] = 'previous_route_step'
                use_plate_fallback = bool(allow_missing and not candidates
                                          and plate_fallback is not None)
                if allow_missing and not candidates and expected is not None:
                    evidence['target_evidence'] = 'previous_frame'
            try:
                probe.check_identity()
                identity_ok = True
            except ProbeError:
                identity_ok = False
            try:
                probe.focus()
                focus_ok = True
            except ProbeError:
                focus_ok = False
            if args.live_source:
                stamp = check_frame('after_final_focus_check')
                age_s = (time.monotonic_ns() - stamp) / 1e9
                evidence.update(captured_monotonic_ns=stamp, frame_age_s=age_s)
                evidence['captured_s'] = round(stamp / 1e9, 3)
                frame_fresh = age_s <= FRAME_MAX_AGE_NS / 1e9
            else:
                # next_frame took this snapshot for this decision.
                frame_fresh = True
            plate = candidates[0] if len(candidates) == 1 else None
            if plate is None and scope_entered and report.get('pre_scope_plate'):
                pre = report['pre_scope_plate']
                plate = dict(nameplate_bbox=pre['bbox'], hp=pre.get('hp'),
                             red_px=pre.get('red_px'),
                             body_visible=pre.get('body_visible'))
                evidence['plate_source'] = 'pre_scope'
            hp_reading = {}
            if plate is not None:
                if plate.get('hp') is not None:
                    # A caller that already measured the plate supplies it; the
                    # detection candidate carries red_px for the same reason.
                    hp_reading = dict(status='supplied', current=plate['hp'],
                                      reason='caller_measured')
                else:
                    from nameplate_hp import read_hp
                    try:
                        hp_reading = read_hp(current, plate['nameplate_bbox'])
                    except (ValueError, OSError):
                        hp_reading = dict(status='unknown', current=None,
                                          reason='reader_failed')
            if use_plate_fallback and plate is None:
                # Operator rule: once the binocular view is open, a solution that
                # was already inside the fire tolerance is enough. The zoom can
                # push a 100 px third-person error out of frame, so the plate
                # measured during the solve is handed over explicitly, named as
                # such, and the gate decides with that knowledge.
                plate = dict(plate_fallback)
                evidence['plate_source'] = 'solve_frame'
                evidence['plate_evidence'] = 'previous_frame_solution'
            evidence['plate_hp'] = hp_reading
            if getattr(args, 'no_terrain_gate', False):
                terrain = dict(clear=None, reason='terrain_probe_disabled')
            else:
                # Ask for a fresh measurement now (idempotent while one runs) and
                # use whatever the background probe already produced.
                terrain_prefetch(current, plate['nameplate_bbox'] if plate else None,
                                 ttl=getattr(args, 'terrain_ttl_s', 8.0))
                terrain = terrain_latest()
            evidence['terrain'] = terrain
            report['fire_terrain'] = terrain
            evidence['own_alive_evidence'] = 'battle_hud_active'
            request_scope = bool(args.scope or args.scope_after)
            steps_so_far = report.get('steps') or []
            decision = fire_gate.fire_decision(fire_gate.Evidence(
                battle_active=active,
                own_alive=active,
                weapon_kind=('gun' if gun_confirmed else
                             (battery['weapon'].get('kind')
                              if battery['weapon'].get('status') == 'observed' else None)),
                reload_ready=(ready if battery['reload'].get('status')
                              in ('observed', 'partial') else None),
                enemy_plate=(None if plate is None else dict(
                    bbox=plate['nameplate_bbox'],
                    hp=hp_reading.get('current'),
                    red_px=plate.get('red_px'))),
                frame_fresh=frame_fresh,
                aim_error_px=evidence['distance_px'],
                fire_tolerance_px=args.fire_tolerance,
                identity_ok=identity_ok,
                focus_ok=focus_ok,
                scope_requested=request_scope,
                scope_evidenced=scope_confirmed(scope, camera_view),
                scope_target_visible=(bool(candidates)
                                      or evidence.get('plate_source') == 'pre_scope'),
                range_gap_km=(steps_so_far[-1].get('range_gap_km') if steps_so_far else None),
                range_tolerance_km=getattr(args, 'range_tolerance_km', None),
                body_visible=(None if plate is None else plate.get('body_visible')),
                terrain_clear=terrain.get('clear'),
                terrain_detail=terrain))
            evidence['fire_decision'] = decision
            report['fire_decision'] = decision
            if not decision['allow']:
                raise ProbeError('fire revoked: ' + ', '.join(decision['blockers']))
            evidence['confirmed'] = True
            return evidence


        # The operator's rule is 15 s for the whole procedure, so the clock
        # starts before the minimap read and the bearing turn, not at the first
        # sweep frame: otherwise those two seconds were free.
        aim_started = time.monotonic()
        report['budget_includes_preposition'] = True
        target_provided = args.target is not None
        report['target_provided'] = target_provided
        if not args.allow_concurrent_input:
            signals = concurrent_input(probe)
            report['concurrent_input'] = signals
            if signals:
                raise ProbeError(f'concurrent input detected before starting: {signals}')
        scope_entered = False
        scope_camera = None
        scope_reference = None
        if not args.stay_in_scope:
            # Third person is the resting camera: coarse aiming happens there.
            # If an earlier session left the binocular view on, the servo owns
            # that state from here, so it puts the camera back and records why.
            precheck = camera_mode_evidence(next_frame('scope-precheck'))
            report['scope_precheck'] = precheck
            scope_reference = precheck if precheck.get('binocular') is False else None
            wanted, why = normalise_presses(precheck)
            report['normalise_reason'] = why
            presses = 0
            while wanted and presses < 2:
                probe.key(args.scope_key)
                presses += 1
                report['inputs'].append(dict(key=args.scope_key,
                                             reason='normalise_to_third_person',
                                             attempt=presses))
                if args.settle:
                    time.sleep(args.settle)
                again = camera_mode_evidence(next_frame(f'scope-normalised-{presses}'))
                report[f'scope_normalised_{presses}'] = again
                scope_camera = again
                if again.get('binocular') is False:
                    break
                # Either the first press did nothing, or the precheck was a
                # false positive and that press opened the scope. One more
                # toggle puts it back; the bound stops a toggling loop.
                wanted = 1
            report['scope_presses_at_start'] = presses
            if presses:
                report['scope_after_normalise'] = scope_camera
        if args.aim_nudge:
            # The aim reticle is only drawn after the mouse moves, so a nudge
            # is sent before the first frame; it is a single unit of motion.
            move_pointer(args.aim_nudge, 0)
            report['inputs'][-1].update(reason='activate_aim_reticle')
        if args.recenter_margin > 0 and not uses_ei_transport(probe):
            # The pointer can sit on a screen edge, where one direction has no
            # room at all. Centre it before the first frame so both directions
            # are available; the rotation this causes is absorbed into the
            # initial error rather than biasing a correction. libei does not
            # use pointer travel, so it is left where it is.
            pointer = pointer_state(probe.identity['display'])
            move = recentre_move((pointer['x'], pointer['y']), width, height)
            if move != [0, 0]:
                move_pointer(move[0], move[1])
                report['inputs'][-1].update(reason='initial_pointer_recentre')
                report['initial_recentre'] = move
        if args.scope and not args.oneshot:
            # A requested view change still needs image evidence before firing.
            probe.key(args.scope_key)
            report['inputs'].append(dict(key=args.scope_key, reason='enter_scope'))
            scope_entered = True
            # `scope_for_aim` selects the scoped limits (60-unit coarse vertical,
            # scoped reach, scoped gain resets). Leaving it False while the
            # camera really is zoomed means every correction runs on
            # third-person budgets: measured 2026-10-29 the vertical sat at
            # -120 px and the moves quantised to [0,0].
            scope_for_aim = True
            if args.settle:
                time.sleep(args.settle)
            scope_camera = camera_mode_evidence(next_frame('scope-entered'),
                                                baseline=scope_reference)
            report['scope_camera_evidence'] = scope_camera
        if args.oneshot:
            report.update(mode='oneshot_open_loop')
            if getattr(args, 'lead', False) and args.aim_budget_s:
                # The lead sampling costs ~2.5-4 s on top of the aim, and the scan
                # that precedes it spends the same budget, so the raise has to
                # happen before the first scan step, not at the lead block.
                raised = max(args.aim_budget_s, args.lead_budget_s)
                if raised != args.aim_budget_s:
                    report['budget_raised_for_lead'] = dict(was=args.aim_budget_s, now=raised)
                    args.aim_budget_s = raised
            frame = next_frame('oneshot-0')
            centre, battle, weapon, source = reticle_centre(frame)
            candidates = [c for c in find_enemy(frame, centre, args.aim_fraction) if c['trackable']]
            if not candidates and args.scan:
                # Wide search: sweep the view until a red marker is in it, then
                # solve from a fresh frame rather than a remembered point.
                # The camera drifts back to the bow when left alone, and the
                # ship is sailing toward the contacts, so that is where the
                # sweep should start rather than wherever a previous run left it.
                if args.scan_idle:
                    time.sleep(args.scan_idle)
                reset_vertical_guard()
                path = scan_path(args.scan_steps, args.scan_rows, args.scan_step, args.scan_tilt,
                                 reach_y=args.reach_y)
                for index, move in enumerate(path, start=1):
                    if battle is not True:
                        raise ProbeError('scan revoked: current battle HUD unconfirmed')
                    if args.live_source:
                        check_frame('before_oneshot_scan_move')
                    move_pointer(move[0], move[1])
                    report['inputs'][-1].update(step=f'oneshot-scan-{index}',
                                                reason='scan_for_enemy')
                    if args.scan_settle:
                        time.sleep(args.scan_settle)
                    frame = next_frame(f'oneshot-scan-{index}')
                    centre, battle, weapon, source = reticle_centre(frame)
                    candidates = [c for c in find_enemy(frame, centre, args.aim_fraction)
                                  if c['trackable']]
                    report['steps'].append(dict(mode='oneshot_scan', step=index, move=list(move),
                                                reticle=centre, reticle_source=source,
                                                battle_active=battle,
                                                candidates=len(candidates)))
                    if candidates:
                        break
            row = dict(mode='oneshot', reticle=centre, reticle_source=source,
                       battle_active=battle, weapon_status=weapon,
                       candidates=len(candidates),
                       target=None if not candidates else candidates[0]['aim_px'])
            report['steps'].append(row)
            if battle is not True or centre is None or not candidates:
                raise ProbeError('one-shot aim needs current battle HUD, reticle and enemy')
            target = candidates[0]['aim_px']
            if not getattr(args, 'no_terrain_gate', False):
                # Start measuring the line of fire now, in the background, so the
                # confirm does not pay ~2 s inside the shot budget.
                terrain_prefetch(frame, candidates[0].get('nameplate_bbox'),
                                 ttl=getattr(args, 'terrain_ttl_s', 8.0))
            gain = list(args.open_loop_gain)
            if args.autogain:
                # Solve with a coefficient measured in this very session: the
                # gain changes with the camera mode and the ship's state.
                body = candidates[0].get('body_roi') or candidates[0]['nameplate_bbox']
                measured = []
                for axis in (0, 1):
                    value, shift = calibrate_axis(probe, args, report, frame, body, axis,
                                                  args.settle, frame_getter=next_frame)
                    measured.append(dict(axis=axis, gain=None if value is None else round(value, 3),
                                         shift=None if shift is None else list(shift)))
                    if value is not None and abs(value) >= .2:
                        gain[axis] = value
                row['autogain'] = measured
                # The probes were reverted, so the aim is where it was: the
                # solved point from the original frame is still valid. One
                # tracking check confirms the target is still there without
                # requiring the detector to fire again.
                if args.settle:
                    time.sleep(args.settle)
                check = next_frame('after-autogain')
                template = crop_template(check, body)
                if template is not None and args.autogain_verify:
                    back = track_patch(template, body, check, search=args.autogain_search)
                    row['after_autogain'] = dict(tracked=None if back is None else back['bbox'],
                                                 gain=gain)
                    if back is None:
                        raise ProbeError('the target left the view during gain calibration')
            delta = open_loop_delta(target, centre, gain)
            row['requested_open_loop_mouse'] = [round(v, 1) for v in delta]
            delta[1] = max(-args.reach_y, min(args.reach_y, delta[1]))
            row['open_loop_mouse'] = [round(v, 1) for v in delta]
            if args.live_source:
                check_frame('before_oneshot_move')
            for chunk in chunk_deltas(int(round(delta[0])), int(round(delta[1]))):
                move_pointer(chunk[0], chunk[1])
                report['inputs'][-1].update(step='oneshot', reason='open_loop_move')
            if args.settle:
                time.sleep(args.settle)
            after = next_frame('oneshot-1')
            centre2, _, _, source2 = reticle_centre(after)
            again = [c for c in find_enemy(after, centre2, args.aim_fraction) if c['trackable']]
            row['after'] = dict(reticle=centre2, reticle_source=source2,
                                target=None if not again else again[0]['aim_px'])
            if centre2 is not None and again:
                row['residual_px'] = [round(again[0]['aim_px'][0]-centre2[0], 1),
                                      round(again[0]['aim_px'][1]-centre2[1], 1)]
            # Measured correction rounds before firing. The open-loop solve
            # leaves a residual (measured 2026-10-23: 20-140 px, occasionally
            # 500+), and the two axes need different budgets: x is fast, y is
            # limited to the safe 3-unit steps unless a coarse budget was
            # granted. Each round corrects both axes, re-reads, and stops as soon
            # as the residual is inside the tight tolerance.
            series = []
            screen_samples = []
            km_samples = []
            if row.get('after', {}).get('target'):
                screen_samples.append((time.monotonic(), float(row['after']['target'][0]),
                                       float(row['after']['target'][1]),
                                       len(report['inputs'])))
            if row.get('residual_px') is not None:
                series.append([round(v, 1) for v in row['residual_px']])
            rounds = []
            if not args.no_oneshot_refine:
                for round_index in range(max(0, args.oneshot_refine_rounds)):
                    residual = series[-1] if series else None
                    if residual is None:
                        break
                    if max(abs(residual[0]), abs(residual[1])) <= args.oneshot_refine_px:
                        break
                    if args.aim_budget_s:
                        left = args.aim_budget_s - (time.monotonic() - aim_started)
                        if left < args.refine_min_headroom_s:
                            rounds.append(dict(round=round_index + 1,
                                               skipped='budget',
                                               left_s=round(left, 2)))
                            break
                    want_x = (-residual[0] / gain[0] * args.settle_damping
                              if gain[0] else 0.0)
                    want_y = (-residual[1] / gain[1] * args.settle_damping
                              if gain[1] else 0.0)
                    limit_y = refine_limit_y(want_y, args.reach_y, args.coarse_reach_y)
                    want_x = max(-args.fast_reach, min(args.fast_reach, want_x))
                    want_y = max(-limit_y, min(limit_y, want_y))
                    if abs(want_x) < 2 and abs(want_y) < 1:
                        break
                    for chunk in chunk_deltas(int(round(want_x)), int(round(want_y))):
                        move_pointer(chunk[0], chunk[1])
                        report['inputs'][-1].update(step='oneshot_refine',
                                                    reason='oneshot_refine')
                    if args.settle:
                        time.sleep(args.settle)
                    refined_frame = next_frame(f'oneshot-refine-{round_index + 1}')
                    r_centre, _, _, _ = reticle_centre(refined_frame)
                    r_again = [c for c in find_enemy(refined_frame, r_centre,
                                                     args.aim_fraction)
                               if c['trackable']]
                    entry = dict(round=round_index + 1,
                                 moved_units=[round(want_x, 1), round(want_y, 1)],
                                 residual_before=residual)
                    if r_centre is None or not r_again:
                        entry['lost'] = True
                        rounds.append(entry)
                        break
                    after_residual = [round(r_again[0]['aim_px'][0] - r_centre[0], 1),
                                      round(r_again[0]['aim_px'][1] - r_centre[1], 1)]
                    # The frames are already here, so the target's motion is
                    # measured from them instead of paying for extra captures.
                    screen_samples.append((time.monotonic(),
                                           float(r_again[0]['aim_px'][0]),
                                           float(r_again[0]['aim_px'][1]),
                                           len(report['inputs'])))
                    if r_again[0].get('nameplate_bbox'):
                        km_now = hud_distance_km(refined_frame, r_again[0]['nameplate_bbox'])
                        if km_now is not None:
                            km_samples.append((time.monotonic(), float(km_now),
                                               len(report['inputs'])))
                    entry['residual_after'] = after_residual
                    series.append(after_residual)
                    rounds.append(entry)
                    row['refined_residual_px'] = after_residual
                    row['residual_px'] = after_residual
                    row['after']['target'] = r_again[0]['aim_px']
                    row['after']['reticle'] = r_centre
                    row['oneshot_refined'] = dict(
                        moved_units=entry['moved_units'],
                        residual_before=entry['residual_before'],
                        residual_after=after_residual)
                    # A target that drifts can defeat a correction; the loop is
                    # bounded by the round budget, and the fire gate still has
                    # the final say on the last residual.
                    if (args.oneshot_refine_px and
                            max(abs(after_residual[0]), abs(after_residual[1]))
                            <= args.oneshot_refine_px):
                        break
            if rounds:
                row['oneshot_refine_rounds'] = rounds
                # The effective gain of this shot, measured from real motion:
                # moving u units changed the residual by (before - after), so
                # the axis response is that difference per unit. It is the best
                # seed for the next shot in the same game process.
                for entry in rounds:
                    before, after = entry.get('residual_before'), entry.get('residual_after')
                    moved = entry.get('moved_units') or [0, 0]
                    if before is None or after is None:
                        continue
                    measured = []
                    for axis in (0, 1):
                        if abs(moved[axis] or 0) >= 2:
                            measured.append(round((before[axis] - after[axis]) / moved[axis], 3))
                        else:
                            measured.append(None)
                    if any(v is not None for v in measured):
                        row['gain_effective'] = measured
                        break
            row['gain_used'] = [None if g is None else round(float(g), 3) for g in gain]
            if series:
                row['error_px'] = series[-1]
                row['error_series'] = series
                report['oneshot_errors'] = series

            # Moving-target support. Two geometries, because the game gives two:
            #  * unlocked: the ship slides on screen, so lead is a screen offset;
            #  * locked: the camera follows the ship's bearing, the ship does not
            #    move on screen, and the only thing that changes is its RANGE --
            #    so the lead is "aim at the range it will have when the shells
            #    arrive", applied through the range closure.
            lead_px = [0.0, 0.0]
            range_plan = None
            plan = dict(mode='none', applied=False, reason='lead_not_run')
            # Opt-in: the extra frames cost ~2.5 s, worth it against a moving ship
            # and wasted on a still one.
            if getattr(args, 'lead', False) and not args.no_lead and gain[0]:
                sample_frame = frame
                # Start from the frames already captured (refine rounds). Extra
                # captures are only paid for when those do not span long enough.
                samples = unmoved_samples(screen_samples, width=3)
                need = args.lead_samples if len(samples) < 3 else 0
                for sample_index in range(max(0, need)):
                    if args.settle:
                        time.sleep(max(args.lead_sample_s - args.settle, 0.05))
                    time.sleep(args.settle if args.settle else 0)
                    sample_frame = next_frame(f'lead-sample-{sample_index + 1}')
                    s_centre, _, _, _ = reticle_centre(sample_frame)
                    found_again = [c for c in find_enemy(sample_frame, s_centre, args.aim_fraction)
                                   if c['trackable']]
                    if not found_again:
                        break
                    samples.append((time.monotonic(), found_again[0]['aim_px'][0],
                                    found_again[0]['aim_px'][1], len(report['inputs'])))
                flight_read = hud_flight_seconds(sample_frame)
                flight_now = float(flight_read['value']) if flight_read else None
                if flight_now is None and args.lead_seconds:
                    # The flight-time readout is drawn near the reticle in the
                    # scope; in third person it is often absent, so an explicitly
                    # supplied value is the only honest fallback.
                    flight_now = float(args.lead_seconds)
                    row['lead_flight_source'] = 'cli'
                else:
                    row['lead_flight_source'] = 'hud' if flight_now is not None else None
                flight_value = float(flight_now) if flight_now is not None else None
                range_samples = unmoved_samples(km_samples, width=2)
                need_km = args.lead_samples if len(range_samples) < 3 else 0
                for sample_index in range(max(0, need_km)):
                    if args.settle:
                        time.sleep(max(args.lead_sample_s - args.settle, 0.05))
                    range_frame = next_frame(f'lead-range-{sample_index + 1}')
                    plate_now = None
                    if range_frame is not None:
                        refined = [c for c in find_enemy(range_frame, s_centre, args.aim_fraction)
                                   if c['trackable']]
                        if refined:
                            plate_now = refined[0]['nameplate_bbox']
                    km = hud_distance_km(range_frame, plate_now) if plate_now else None
                    if km is None:
                        break
                    range_samples.append((time.monotonic(), float(km),
                                          len(report['inputs'])))
                range_plan = predicted_range_km(range_samples, flight_value,
                                                min_span_s=args.lead_min_span)
                row['range_lead'] = range_plan
                if (plan.get('mode') == 'range' and plan.get('applied')
                        and not args.no_lead):
                    # Locked-target lead: close the elevation onto the PREDICTED
                    # range. The impact-range readout next to the reticle is the
                    # only honest input; when this client does not draw it, say so
                    # instead of inventing a vertical nudge.
                    impact = aim_distance_km(sample_frame,
                                             getattr(reticle_centre, 'last_bbox', None))
                    if impact is None:
                        range_plan['closure'] = 'impact_range_unreadable'
                    else:
                        units, why = range_correction(impact, range_plan['predicted_km'],
                                                      args.range_tolerance_km,
                                                      getattr(args, 'range_km_per_unit', None)
                                                      or RANGE_KM_PER_UNIT,
                                                      args.range_step_units)
                        range_plan['closure'] = why
                        range_plan['impact_km'] = round(float(impact), 2)
                        if why == 'move' and units:
                            for chunk in chunk_deltas(0, int(round(units))):
                                move_pointer(chunk[0], chunk[1],
                                             vertical_limit=max(abs(units), args.reach_y))
                                report['inputs'][-1].update(step='range_lead',
                                                            reason='range_lead')
                            if args.settle:
                                time.sleep(args.settle)
                decision = moving_target_lead(samples, flight_now,
                                              ceiling=args.lead_ceiling,
                                              min_span_s=args.lead_min_span)
                row['lead'] = decision
                speed = None
                if decision.get('velocity_px_s'):
                    vx, vy = decision['velocity_px_s']
                    speed = (vx * vx + vy * vy) ** 0.5
                plan = lead_plan(decision, range_plan, speed)
                row['lead_plan'] = plan
                if plan.get('mode') == 'screen' and plan.get('applied'):
                    lead_px = [float(v) for v in decision['lead_px']]
                    # residual = target - reticle and a mouse move m changes the
                    # residual by gain*m, so putting the reticle ahead by `lead`
                    # needs m = -lead/gain. The sign was inverted (measured
                    # 2026-10-27: a 320 px lead left a +391 px residual, i.e. the
                    # reticle ended up behind the ship instead of ahead of it).
                    move_units = [-lead_px[0] / gain[0],
                                  -lead_px[1] / (gain[1] or gain[0])]
                    reach_x = max(float(args.fast_reach), float(args.lead_reach))
                    move_units[0] = max(-reach_x, min(reach_x, move_units[0]))
                    limit_y = refine_limit_y(move_units[1], args.reach_y, args.coarse_reach_y)
                    move_units[1] = max(-limit_y, min(limit_y, move_units[1]))
                    delivered_y = 0.0
                    remaining_y = move_units[1]
                    for packet in range(3):
                        cap = min(args.lead_reach, 60.0)
                        step_y = max(-cap, min(cap, remaining_y))
                        for chunk in chunk_deltas(int(round(move_units[0])), int(round(step_y))):
                            move_pointer(chunk[0], chunk[1],
                                         vertical_limit=max(abs(step_y), args.reach_y))
                            report['inputs'][-1].update(step='lead_move', reason='lead_move')
                        delivered_y += step_y
                        remaining_y -= step_y
                        move_units[0] = 0.0
                        if abs(remaining_y) < 1 or packet == 2:
                            break
                        if args.settle:
                            time.sleep(max(0.15, args.settle / 2))
                    move_units[1] = delivered_y
                    if args.settle:
                        time.sleep(args.settle)
                    lead_frame = next_frame('lead-applied')
                    l_centre, _, _, _ = reticle_centre(lead_frame)
                    l_found = [c for c in find_enemy(lead_frame, l_centre, args.aim_fraction)
                               if c['trackable']]
                    decision['moved_units'] = [round(v, 1) for v in move_units]
                    # The gain's sign flips with the camera state (measured
                    # 2026-10-27: an assumed -1.5 was really +0.84 in that frame,
                    # so the lead moved the reticle away). The response of this
                    # very step is the only trustworthy coefficient: measure it
                    # and, if the reticle is not yet at the predicted point, walk
                    # the remainder with the measured gain.
                    if (l_centre is not None and l_found
                            and abs(move_units[0]) >= 20 and residual is not None):
                        residual_before_x = float(residual[0])
                        residual_after_x = float(l_found[0]['aim_px'][0] - l_centre[0])
                        measured_gain = ((residual_after_x - residual_before_x)
                                         / float(move_units[0]))
                        decision['lead_gain_measured'] = round(measured_gain, 3)
                        goal = -lead_px[0]
                        shortfall = goal - residual_after_x
                        if abs(measured_gain) >= 0.3 and abs(shortfall) > max(8.0, args.tolerance):
                            extra_units = shortfall / measured_gain
                            cap = max(float(args.lead_reach), float(args.fast_reach))
                            extra_units = max(-cap, min(cap, extra_units))
                            for chunk in chunk_deltas(int(round(extra_units)), 0):
                                move_pointer(chunk[0], chunk[1])
                                report['inputs'][-1].update(step='lead_trim',
                                                            reason='lead_trim')
                            decision['lead_trim_units'] = [round(extra_units, 1)]
                            if args.settle:
                                time.sleep(args.settle)
                            trim_frame = next_frame('lead-trim')
                            t_centre, _, _, _ = reticle_centre(trim_frame)
                            t_found = [c for c in find_enemy(trim_frame, t_centre, args.aim_fraction)
                                       if c['trackable']]
                            if t_centre is not None and t_found:
                                l_centre = t_centre
                                l_found = t_found
                                residual_after_x = float(t_found[0]['aim_px'][0] - t_centre[0])
                    if l_centre is not None and l_found:
                        point_residual = [round(l_found[0]['aim_px'][0] + lead_px[0] - l_centre[0], 1),
                                          round(l_found[0]['aim_px'][1] + lead_px[1] - l_centre[1], 1)]
                        row['lead_residual_px'] = point_residual
                        row['residual_px'] = [round(l_found[0]['aim_px'][0] - l_centre[0], 1),
                                              round(l_found[0]['aim_px'][1] - l_centre[1], 1)]
                    else:
                        decision['lost_after_lead'] = True

            if args.scope or args.scope_after:
                probe.key(args.scope_key)
                report['inputs'].append(dict(key=args.scope_key, reason='scope_for_oneshot'))
                # Mark it so the finally block toggles the binocular view back
                # off: leaving it open would change the player's own view.
                scope_entered = True
                if args.settle:
                    time.sleep(args.settle)
            if args.lock:
                probe.key(args.lock_key)
                report['inputs'].append(dict(key=args.lock_key, reason='lock_target'))
                if args.settle:
                    time.sleep(max(.4, args.settle/2))
            residual = row.get('residual_px')
            # With lead the reticle deliberately sits ahead of the ship, so the
            # error that matters is the distance to the predicted point.
            judged = judged_residual(residual, lead_px)
            within = (judged is not None
                      and max(abs(judged[0]), abs(judged[1])) <= args.fire_tolerance)
            row['fire_residual_px'] = judged
            if args.fire and not within:
                # An open-loop move that the re-read did not confirm must never
                # fire: the earlier version did, and put a shell 570 px off.
                report['fired'] = False
                report['fire_reason'] = 'oneshot_residual_outside_tolerance'
                report['fire_blocked'] = dict(residual_px=residual,
                                              tolerance=args.fire_tolerance,
                                              observed_at='after_oneshot_move')
            if args.fire and within:
                over = budget_exceeded(aim_started, args.aim_budget_s)
                report['aim_budget_s'] = args.aim_budget_s
                report['aim_ms'] = round((time.monotonic() - aim_started) * 1000)
                if over:
                    # Refuse rather than report a late shot as a success.
                    report['fired'] = False
                    report['fire_reason'] = 'aim_budget_exceeded'
                    report['fire_blocked'] = dict(
                        reason='aim_budget_exceeded', residual_px=residual,
                        aim_ms=report['aim_ms'], budget_s=args.aim_budget_s)
                    report['status'] = 'completed'
                    print(json.dumps(report, indent=2))
                    return 0
                converged = 0
                report['fired'] = False
                for step in range(args.max_steps):
                    confirmation = confirm_fire(
                        f'pre-fire-oneshot-{step}',
                        expected=None if (args.scope or args.scope_after) else row['after']['target'],
                        lead=tuple(lead_px))
                    converged += 1
                    report['steps'].append(dict(
                        step=step, state='converged', distance_px=confirmation['distance_px'],
                        captured_s=confirmation['captured_s'],
                        frame_age_s=confirmation.get('frame_age_s')))
                    ready, reason = fire_ready(confirmation['distance_px'], converged, step,
                                               args.tolerance, args.fire_min_steps)
                    if ready:
                        held_button = 1
                        probe.x('mousedown', 1)
                        probe.x('mouseup', 1)
                        held_button = None
                        report['inputs'].append(dict(button=1, pulse=True, step=step, reason='fire'))
                        report['fired'] = True
                        report['fire_reason'] = reason
                        report['fire_frames'] = [str(probe.snapshot(f'fire-{index}'))
                                                 for index in range(args.post_fire_frames)]
                        break
                    if args.settle:
                        time.sleep(args.settle)
            report['status'] = 'completed'
            print(json.dumps(report, indent=2))
            return 0
        patch = None
        template = None
        tracking_candidate = None
        acquisition_frame = None
        aim_offset = [0., 0.]
        # A known gain saves the two probe steps (each costs a full capture
        # round-trip, ~1.9 s here). It is only a seed: update_gain still
        # corrects it, and a wrong sign is flipped after two consistent
        # opposite responses.
        gain = list(args.seed_gain) if args.seed_gain else [None, None]
        gain_state = {}
        stale_reads = 0
        if args.range_closure and args.measure_range_units:
            def _centre(frame):
                found, _, _, _ = reticle_centre(frame)
                return found
            measured_units, why = measure_range_units(
                probe, next_frame, _centre, report, units=args.measure_range_units)
            if measured_units is not None:
                args.range_km_per_unit = abs(measured_units)
                report['range_km_per_unit'] = measured_units
            else:
                report['range_km_per_unit'] = None
                report['range_km_per_unit_reason'] = why
        if args.measure_gain:
            # The third-person vertical responds with 0.1-0.4 px/unit, so the old
            # 3-unit probe (a ~1 px shift) could never clear the threshold and
            # the vertical gain stayed at whatever the seed claimed (measured
            # overshoot 8x: -54 -> -467 in one step).
            measured = measure_gain(probe, next_frame, settle=max(.3, args.settle or .45),
                                    units=args.measure_gain_units,
                                    units_y=args.measure_gain_units_y)
            report['measured_gain'] = measured
            if not measured['ok']:
                raise ProbeError('gain measurement refused: ' + str(measured['reason']))
            gain = list(measured['gain'])
            report['gain_source'] = 'measured_in_session'
        error = None
        previous_error = None
        last_move = None
        vertical_errors = []
        beyond_moves = 0
        closure_gaps = 0
        weak_moves = {0: 0, 1: 0}
        probe_rounds = {}
        previous_captured = None
        velocity_ema = None
        velocity_samples = 0
        pan_log = []
        history = []
        previous_frame = None
        last_source = None
        next_search = 160
        converged = 0
        pre_scope_ok = 0
        locked = False
        # `--scope` opens the binocular view before aiming, so the loop must run
        # on the scoped budgets from the first step. This assignment used to
        # overwrite the flag set by the --scope branch with "scope_after == 0",
        # which is false when an --scope-after default is present: every
        # correction then ran on third-person limits (measured 2026-10-29:
        # scope_for_aim False, reach_y 3/6, vertical error stuck at -190 px).
        scope_for_aim = bool(args.scope) or args.scope_after == 0
        if args.fast:
            # Under-five-second shot: one frame, one open-loop solve, one move,
            # one confirmation frame, fire. No probe, no convergence loop.
            started = time.monotonic()
            centre = battle = source = None
            frame = None
            for look in range(3):
                frame = next_frame(f'fast-0-{look}')
                centre, battle, _, source = reticle_centre(frame)
                if battle is True and centre is not None:
                    break
                # The ring is only painted after motion, and one bad frame is
                # not evidence that the battle HUD is gone. A small out-and-back
                # wakes the cursor without netting any camera rotation.
                probe.move(6, 0)
                report['inputs'].append(dict(relative=[6, 0], reason='fast_reticle_redraw'))
                time.sleep(.05)
                probe.move(-6, 0)
                report['inputs'].append(dict(relative=[-6, 0], reason='fast_reticle_redraw_back'))
                time.sleep(.15)
            if battle is not True or centre is None:
                raise ProbeError('fast shot revoked: battle HUD or reticle missing')

            def _aim_only(index):
                frame_d = next_frame(f'fast-descend-{index}')
                centre_d, battle_d, _, _ = reticle_centre(frame_d)
                if battle_d is not True or centre_d is None:
                    return None
                _state, km_d = fire_hud.impact_readout(frame_d, centre_d)
                return km_d

            descent_step = max(MAX_VERTICAL_STEP_UNITS, args.descent_step_units)
            descent, descent_state = descend_to_readable(
                _aim_only,
                lambda dx, dy: (probe.move(dx, dy),
                                report['inputs'].append(dict(relative=[dx, dy],
                                                             reason='fast_descent'))),
                max_km=args.fast_max_aim_km, step=descent_step,
                rounds=max(6, args.descent_rounds))
            report['fast_descent'] = descent
            report['fast_descent_state'] = descent_state
            if descent_state == 'readable':
                frame = next_frame('fast-0-after-descent')
                centre, battle, _, source = reticle_centre(frame)
                if battle is not True or centre is None:
                    raise ProbeError('fast shot revoked: reticle missing after descent')
            cands = [c for c in find_enemy(frame, centre) if c['trackable']]
            pick = choose_target(cands, None)
            # The target may be a little off axis; a bounded sweep is still the
            # model acting on its own frames. Each look costs about 0.35 s.
            search_step = float(args.fast_search_step)
            if args.fast_search_deg > 0:
                search_step = yaw_units_for_degrees(
                    args.fast_search_deg, abs(args.open_loop_gain[0]),
                    args.fov_deg) or search_step
                report['fast_search_step_units'] = round(search_step, 1)
            search_limit = turn_chunk_limit(args, probe)
            report['fast_search_chunk'] = search_limit
            for look in range(args.fast_search_steps):
                if pick is not None:
                    break
                remaining = search_step
                while abs(remaining) > 1:
                    # The pointer hits a window edge and then the move delivers
                    # nothing, so the sweep silently repeats one view. Make room
                    # first when the requested direction has no travel left;
                    # libei has no such edge, so a reposition there would only
                    # add unrequested yaw.
                    if not uses_ei_transport(probe):
                        pointer = pointer_state(probe.identity['display'])
                        room = (width - 60 - pointer['x']) if remaining > 0 else (pointer['x'] - 60)
                        if room < 40:
                            back = -300 if remaining > 0 else 300
                            probe.move(back, 0)
                            report['inputs'].append(dict(relative=[back, 0],
                                                         reason='fast_search_reposition',
                                                         transport=relative_transport_name(probe)))
                            time.sleep(.05)
                    step = max(-search_limit, min(search_limit, remaining))
                    probe.move(step, 0)
                    report['inputs'].append(dict(relative=[step, 0], reason='fast_search'))
                    remaining -= step
                    time.sleep(.03)
                time.sleep(.25)
                frame = next_frame(f'fast-search-{look}')
                centre, battle, _, source = reticle_centre(frame)
                if battle is not True or centre is None:
                    continue
                cands = [c for c in find_enemy(frame, centre) if c['trackable']]
                pick = choose_target(cands, None)
            if pick is None:
                raise ProbeError('fast shot revoked: no enemy nameplate in view')
            if camera_mode_evidence(frame).get('binocular'):
                # Start from third person: the coarse solve and the battery read
                # both depend on it, and a run started in the scope had no valid
                # pre-scope evidence to fall back on.
                probe.key(args.scope_key)
                report['inputs'].append(dict(key=args.scope_key,
                                             reason='fast_leave_scope_first'))
                time.sleep(.4)
                frame = next_frame('fast-0-thirdperson')
                centre, battle, _, source = reticle_centre(frame)
                if battle is not True or centre is None:
                    raise ProbeError('fast shot revoked: reticle missing after leaving scope')
                cands = [c for c in find_enemy(frame, centre) if c['trackable']]
                if cands:
                    pick = choose_target(cands, pick['nameplate_bbox'] if pick else None)
            pre_battery = main_battery_fire_evidence(frame)
            report['fast_battery'] = pre_battery
            report['fast'] = dict(aim_px=pick['aim_px'], nameplate=pick['nameplate_bbox'],
                                  reticle=centre, reticle_source=source,
                                  aim_source=pick.get('aim_source'))
            gain = [args.open_loop_gain[0], args.open_loop_gain[1]]
            # The open-loop solve may have to cross most of the screen; the plan
            # is allowed the whole error and the sender already splits it into
            # 400-unit pieces (measured: a 1078 px error truncated to 400 units
            # still left the target 500 px off and the shot was refused).
            move = fast_shot_plan(pick['aim_px'], centre, gain,
                                  limit=(args.fast_reach, args.fast_reach_y))
            report['fast']['move'] = [round(move[0], 1), round(move[1], 1)]
            # Fast mode is not bound by the per-action aiming cap: the whole
            # error goes out as bounded chunks of relative motion instead.
            remaining = [move[0], move[1]]
            first = True
            while abs(remaining[0]) > 1 or abs(remaining[1]) > 1:
                ensure_room(probe, report, remaining[0], width, 'fast_open_loop_aim',
                            recentre_key=getattr(args, 'recentre_key', None))
                step = [max(-400., min(400., remaining[0])), max(-300., min(300., remaining[1]))]
                probe.move(step[0], step[1])
                report['inputs'].append(dict(relative=step,
                                             reason='fast_open_loop_aim' if first else 'fast_aim_chunk'))
                first = False
                remaining = [remaining[0] - step[0], remaining[1] - step[1]]
                time.sleep(.03)
                if len(report['inputs']) > 40:
                    break
            report['fast']['aim_ms'] = round((time.monotonic() - started) * 1000)
            # Up to two short corrections inside the same budget: the first
            # solve is open loop, so one measured residual keeps the shot on the
            # hull without the closed loop's scan/probe rounds.
            deadline = started + args.fast_deadline_s
            residual = None
            previous_residual = None
            previous_move = None
            detected_centre = None
            for attempt in range(3):
                if args.settle:
                    time.sleep(args.settle if attempt == 0 else .5)
                frame = next_frame(f'fast-check-{attempt}')
                centre, battle, _, _ = reticle_centre(frame)
                if battle is not True:
                    break
                if centre is None:
                    # The reticle is redrawn by mouse motion; right after a large
                    # correction the read can miss for a frame. Detection may use
                    # the last observed centre -- firing still requires an
                    # observed reticle in the confirmation frame.
                    centre = detected_centre
                if centre is None:
                    # Right after the bearing preset the reticle may not be drawn
                    # yet: give the next attempt a chance instead of ending with
                    # no measurement at all.
                    time.sleep(.35)
                    continue
                detected_centre = centre
                cands = [c for c in find_enemy(frame, centre) if c['trackable']]
                chosen = choose_target(cands, pick['nameplate_bbox'])
                if chosen is None:
                    # The sweep that found it may have ended at the frame edge, or
                    # the contact blinked out for one frame: one bounded re-search
                    # keeps the run alive instead of ending at 338 px.
                    for _retry in range(2):
                        ensure_room(probe, report, search_step, width, 'fast_research',
                            recentre_key=getattr(args, 'recentre_key', None))
                        remaining = float(search_step)
                        while abs(remaining) > 1:
                            step_r = max(-400., min(400., remaining))
                            probe.move(step_r, 0)
                            report['inputs'].append(dict(relative=[step_r, 0],
                                                         reason='fast_research'))
                            remaining -= step_r
                            time.sleep(.03)
                        time.sleep(.25)
                        frame = next_frame(f'fast-research-{attempt}-{_retry}')
                        centre, battle, _, _ = reticle_centre(frame)
                        if battle is not True:
                            continue
                        if centre is None:
                            centre = detected_centre
                        if centre is None:
                            continue
                        detected_centre = centre
                        cands = [c for c in find_enemy(frame, centre) if c['trackable']]
                        chosen = choose_target(cands, pick['nameplate_bbox'])
                        if chosen is not None:
                            break
                if chosen is None:
                    break
                pick = chosen
                dx = pick['aim_px'][0] - centre[0]
                dy = pick['aim_px'][1] - centre[1]
                residual = [round(dx, 1), round(dy, 1)]
                report['fast'].setdefault('residuals', []).append(residual)
                if previous_residual is not None and previous_move is not None:
                    for axis in (0, 1):
                        if abs(previous_move[axis]) >= 2:
                            observed = (dx if axis == 0 else dy) - previous_residual[axis]
                            observed /= previous_move[axis]
                            learned = learn_gain(gain[axis], observed, low=0.1, high=6.0)
                            if learned != gain[axis]:
                                gain[axis] = learned
                                report['fast'].setdefault('gain_learned', []).append(
                                    dict(axis=axis, observed=round(observed, 3)))
                if (previous_residual is not None
                        and math.hypot(dx, dy) > math.hypot(*previous_residual) * 1.1):
                    # The correction grew the error: the assumed screen gain has
                    # the wrong sign for this client/zoom. Flip it once and try
                    # again instead of repeating the same wrong move (measured:
                    # residual 1078 -> 1154 px with a positive solve gain).
                    gain = [-value for value in gain]
                    report['fast'].setdefault('gain_sign_flipped', []).append(attempt)
                previous_residual = [dx, dy]
                # Stop at the *fire* tolerance, not the tight servo tolerance:
                # measured 2026-10-23 a 55 px residual (inside the relaxed 120 px
                # rule) was corrected again and the 0.12 px/unit gain overshot to
                # 1078 px. Once the shot is allowed, further correcting can only
                # make it worse.
                accept_px = max(args.tolerance, args.fire_tolerance)
                if math.hypot(dx, dy) <= accept_px or time.monotonic() > deadline:
                    break
                move = fast_shot_plan(pick['aim_px'], centre, gain, limit=(args.fast_reach, args.fast_reach_y))
                # Damping: with the gain learned from one noisy sample the plan
                # overshoots and the residual oscillates (measured -290 -> -404
                # -> -594 px). Under-correct and let the next round finish.
                move = [move[0] * args.fast_damping, move[1] * args.fast_damping]
                if args.fast_axis_split:
                    # A large pitch move also swings the yaw on this client
                    # (measured: dy -349 corrected, dx then drifted -302 ->
                    # -417). Correct one axis per round, the bigger error first.
                    if abs(dy) >= abs(dx):
                        move[0] = 0.0
                    else:
                        move[1] = 0.0
                previous_move = [move[0], move[1]]
                move_pointer(move[0], move[1])
                report['inputs'][-1].update(reason='fast_correction')
                delta = report['inputs'][-1].get('pointer_delta') or [1, 1]
                if (getattr(args, 'recentre_key', None) and abs(move[0]) >= 40
                        and delta[0] == 0):
                    # Measured 2026-10-23: a +212 unit correction at the window
                    # edge reported pointer_delta [0, 0] and the residual grew
                    # from -530 to -898 px. The M toggle restores the travel
                    # without turning the camera, so the same move is re-sent.
                    recentre_pointer(probe, report, args.recentre_key,
                                     reason='fast_correction')
                    move_pointer(move[0], move[1])
                    report['inputs'][-1].update(reason='fast_correction_after_recentre')
            entry_tolerance = args.fire_tolerance
            if args.fast_scope:
                # A roughly-right solution must still reach the scope, otherwise
                # the relaxed post-scope rule could never apply.
                entry_tolerance = max(entry_tolerance, args.scope_fire_tolerance_px)
            if residual is None:
                # No measured solution at all is not "close enough": without a
                # residual there is nothing that says the gun points at the ship.
                report['status'] = 'completed'
                report['fired'] = False
                report['fire_reason'] = 'fast_no_measurement'
                report['fast']['total_ms'] = round((time.monotonic() - started) * 1000)
                print(json.dumps(report, indent=2))
                return 1
            if math.hypot(*residual) > entry_tolerance:
                report['status'] = 'completed'
                report['fired'] = False
                report['fire_reason'] = 'fast_residual_outside_tolerance'
                report['fast_entry_tolerance_px'] = entry_tolerance
                report['fast']['total_ms'] = round((time.monotonic() - started) * 1000)
                report['fast']['within_budget'] = report['fast']['total_ms'] <= 5000
                print(json.dumps(report, indent=2))
                return 1
            if args.range_closure:
                # The operator's rule is a 2 km agreement between the target's
                # range and the reticle's ballistic range. In fast mode the
                # vertical is otherwise a bar-geometry pixel guess bounded to 6
                # units per action, which cannot close a kilometre gap
                # (measured 2026-10-23: 104 px of vertical error never moved).
                def _measure(index, _pick=pick):
                    frame_r = next_frame(f'fast-range-{index}')
                    centre_r, battle_r, _, _ = reticle_centre(frame_r)
                    if battle_r is not True or centre_r is None:
                        return None, None
                    state, km = fire_hud.impact_readout(frame_r, centre_r)
                    taken = [c for c in find_enemy(frame_r, centre_r) if c['trackable']]
                    chosen_r = choose_target(taken, _pick['nameplate_bbox'])
                    tgt, _src = (target_range_km(frame_r, chosen_r['nameplate_bbox'])
                                 if chosen_r else (None, 'no_plate'))
                    return km, tgt

                trace = close_range_steps(_measure,
                                          lambda dx, dy: move_pointer(dx, dy),
                                          args.range_tolerance_km,
                                          args.range_km_per_unit,
                                          args.range_step_units)
                report['fast']['range_closure'] = trace
                if trace and trace[-1]['why'] == 'unavailable':
                    report['fast']['range_closure_note'] = 'a reading was missing'
            if args.fast_scope:
                relaxed = max(args.fire_tolerance, args.scope_fire_tolerance_px)
                if relaxed != args.fire_tolerance:
                    report['fire_tolerance_relaxed'] = dict(before=args.fire_tolerance,
                                                            after=relaxed,
                                                            reason='scope_open')
                    args.fire_tolerance = relaxed
                # Scope phase: bounded rounds that learn the gain from their own
                # response. A dedicated probe measured +7.18 and then -11.04
                # px/unit minutes apart (capture lag), so the sign is taken from
                # what the last move actually did instead of from a constant.
                probe.key(args.scope_key)
                report['inputs'].append(dict(key=args.scope_key, reason='fast_enter_scope'))
                scope_entered = True
                time.sleep(.3)
                gain = [1.4, -1.0]
                previous = None
                previous_move = None
                for round_index in range(3):
                    frame = next_frame(f'fast-scope-{round_index}')
                    centre, battle, _, _ = reticle_centre(frame)
                    if battle is not True or centre is None:
                        break
                    cands = [c for c in find_enemy(frame, centre) if c['trackable']]
                    scoped_pick = choose_target(cands, pick['nameplate_bbox'])
                    if scoped_pick is not None:
                        # Anchors jump: two close-range frames named plates 80 px
                        # apart while the ship was large and the camera moving,
                        # which threw the solve 150 px. Reject an anchor that
                        # moves further than a ship can between rounds.
                        px, py = pick['nameplate_bbox'][:2]
                        nx, ny = scoped_pick['nameplate_bbox'][:2]
                        moved_px = math.hypot(nx - px, ny - py)
                        width_changed = abs(scoped_pick['nameplate_bbox'][2]
                                            - pick['nameplate_bbox'][2]) > 8
                        # A bar that drifted 60+ px between rounds made the servo
                        # report a 3 px residual while the reticle sat 66 px off
                        # the hull (measured). Consecutive rounds must agree.
                        if moved_px > 25 or width_changed:
                            report['fast'].setdefault('anchor_rejected', []).append(
                                dict(round=round_index, from_=[px, py], to=[nx, ny]))
                            scoped_pick = None
                    if scoped_pick is None:
                        report['fast'].setdefault('scope_plate_missing', []).append(round_index)
                        break
                    pick = scoped_pick
                    blob = scope_ship_blob(frame, pick['nameplate_bbox'])
                    if blob is not None:
                        # In the scope the hull is drawn large: aim at it直接
                        # instead of at a bar-derived point.
                        pick['aim_px'] = blob['aim']
                        report['fast']['scope_blob'] = [int(v) for v in blob['bbox']]
                    err = [pick['aim_px'][0] - centre[0], pick['aim_px'][1] - centre[1]]
                    report['fast'].setdefault('residuals', []).append([round(x, 1) for x in err])
                    if previous is not None and previous_move is not None:
                        for axis in (0, 1):
                            if abs(previous_move[axis]) >= 2:
                                observed = (err[axis] - previous[axis]) / previous_move[axis]
                                learned = learn_gain(gain[axis], observed)
                                if learned != gain[axis]:
                                    gain[axis] = learned
                                elif observed is not None and abs(observed) >= .15:
                                    report['fast'].setdefault('gain_rejected', []).append(
                                        dict(axis=axis, observed=round(observed, 3),
                                             kept=gain[axis]))
                        report['fast']['scope_gain_learned'] = [round(g, 3) for g in gain]
                    if math.hypot(*err) <= args.tolerance:
                        break
                    move = fast_shot_plan(pick['aim_px'], centre, gain, limit=(args.reach, 300.))
                    move = [move[0] * .7, move[1] * .7]
                    remaining = [move[0], move[1]]
                    while abs(remaining[0]) > 1 or abs(remaining[1]) > 1:
                        ensure_room(probe, report, remaining[0], width, 'fast_scope_aim',
                            recentre_key=getattr(args, 'recentre_key', None))
                        step = [max(-400., min(400., remaining[0])),
                                max(-300., min(300., remaining[1]))]
                        probe.move(step[0], step[1])
                        report['inputs'].append(dict(relative=step, reason='fast_scope_aim'))
                        remaining = [remaining[0] - step[0], remaining[1] - step[1]]
                        time.sleep(.03)
                    previous = err
                    previous_move = move
                    if args.settle:
                        time.sleep(args.settle)
            if args.fast_scope:
                # Fire only on a stationary picture: the range drifted 6.7 -> 8.1
                # km while a salvo was in the air and every shell fell short.
                time.sleep(.45)          # let the last aim move settle
                stable = None
                for probe_index in range(2):
                    f2 = next_frame(f'fast-stability-{probe_index}')
                    c2, b2, _, _ = reticle_centre(f2)
                    if b2 is not True or c2 is None:
                        continue
                    cands2 = [c for c in find_enemy(f2, c2) if c['trackable']]
                    p2 = choose_target(cands2, pick['nameplate_bbox'])
                    if p2 is None:
                        continue
                    if stable is None:
                        stable = p2['nameplate_bbox'][:2]
                    else:
                        moved = math.hypot(p2['nameplate_bbox'][0] - stable[0],
                                           p2['nameplate_bbox'][1] - stable[1])
                        report['fast']['stability_px'] = round(moved, 1)
                        if moved > 6:
                            report['status'] = 'completed'
                            report['fired'] = False
                            report['fire_reason'] = 'target_moving_between_frames'
                            report['fast']['total_ms'] = round((time.monotonic() - started) * 1000)
                            print(json.dumps(report, indent=2))
                            return 1
                    time.sleep(.35)
            # The enemy is auto-locked on entering the battle; pressing X would
            # toggle that lock off, so fast mode never sends it unless asked.
            if args.lock:
                probe.key(args.lock_key)
                report['inputs'].append(dict(key=args.lock_key, reason='fast_lock'))
                if args.settle:
                    time.sleep(.4)
            confirm_fire('fast-confirm', expected=pick['aim_px'],
                         fallback=pre_battery if args.fast_scope else None,
                         allow_missing=args.fast_scope,
                         plate_fallback=dict(nameplate_bbox=list(pick['nameplate_bbox']),
                                             red_px=pick.get('red_px'),
                                             body_visible=pick.get('body_visible'))
                         if (args.fast_scope and pick) else None)
            held_button = 1
            probe.x('mousedown', 1)
            probe.x('mouseup', 1)
            held_button = None
            report['inputs'].append(dict(button=1, pulse=True, reason='fast_fire'))
            report['fired'] = True
            report['fire_reason'] = 'fast_open_loop'
            report['fire_frames'] = [str(probe.snapshot(f'fast-fire-{index}'))
                                     for index in range(args.post_fire_frames)]
            report['fast']['total_ms'] = round((time.monotonic() - started) * 1000)
            report['fast']['budget_ms'] = 5000
            report['fast']['within_budget'] = report['fast']['total_ms'] <= 5000
            report['status'] = 'completed'
            print(json.dumps(report, indent=2))
            return 0
        if args.scan and args.scan_confirm_steps > 0:
            # When the minimap already gave a bearing and the camera was turned
            # onto it, the sweep is only a confirmation: a full arc costs ~6 s
            # of the 15 s budget (12 frames x ~0.5 s) and the coarse phase then
            # cannot finish (measured 2026-10-23: stopped at 197 px).
            bearing = report.get('minimap_bearing') or {}
            if bearing.get('action') == 'turned_to_bearing':
                args.scan_steps = min(args.scan_steps, args.scan_confirm_steps)
                report['scan_mode'] = 'bearing_confirmation'
            else:
                report['scan_mode'] = 'full_search'
        if args.scan:
            target = None
            reset_vertical_guard()
            covered = scan_coverage_deg(args.scan_steps, args.scan_step,
                                        args.scan_gain_px_per_unit, args.scan_fov_deg)
            report['scan_coverage_deg'] = round(covered, 1)
            report['scan_units_total'] = int(args.scan_steps - 1) * abs(args.scan_step)
            if args.scan_min_deg and covered < args.scan_min_deg:
                raise ProbeError(
                    f'one sweep covers {covered:.0f} deg but {args.scan_min_deg:.0f} deg is '
                    'required; raise --scan-steps or --scan-step')
            path = scan_path(args.scan_steps, args.scan_rows, args.scan_step, args.scan_tilt,
                             reach_y=args.reach_y)
            report['scan_path'] = [list(move) for move in path]
            if args.scan_start_units:
                move_pointer(args.scan_start_units, 0)
                report['inputs'][-1].update(step='scan-start', reason='scan_start_offset')
                report['scan_start_units'] = args.scan_start_units
                if args.scan_settle:
                    time.sleep(args.scan_settle)
            for attempt in range(len(path) + 1):
                # The aim budget starts at the search, so the search itself must
                # honour it: measured 2026-10-20 a target-less sweep spent the
                # whole 25 s budget (33 scan inputs) before the loop ever ran.
                # Search has its own clock: the operator's 10 s applies from the
                # moment a qualified target is accepted, not from process start,
                # and letting the sweep spend the aim budget is how every round
                # ended in aim_budget_exceeded without a shot.
                search_budget = float(getattr(args, 'search_budget_s', 0) or 0) or args.aim_budget_s
                if search_budget and (time.monotonic() - aim_started) > search_budget:
                    report['error'] = 'search_budget_exceeded'
                    report['elapsed_ms'] = round((time.monotonic() - aim_started) * 1000.0)
                    report['scan_budget_exceeded_at'] = attempt
                    raise SystemExit(1)
                for hud_try in range(max(0, args.hud_retries) + 1):
                    frame = next_frame(f'scan-{attempt}' if hud_try == 0
                                       else f'scan-{attempt}-hud{hud_try}')
                    centre, battle, _, _ = reticle_centre(frame)
                    if battle is True:
                        break
                    report.setdefault('hud_misses', []).append(
                        dict(stage='scan', attempt=attempt, retry=hud_try))
                else:
                    # One unconfirmed frame is not a lost battle (measured: the
                    # gate flickers on a dark sea), so it is retried, not fatal.
                    raise ProbeError('scan revoked: current battle HUD unconfirmed')
                found = find_enemy(frame, centre, args.aim_fraction)
                report['scan'].append(dict(attempt=attempt, reticle=centre, battle_active=battle,
                                           candidates=found))
                usable = [c for c in found if c['trackable']]
                if usable:
                    target = usable[0]['aim_px']
                    report['scan_target'] = usable[0]
                    tracking_candidate = usable[0]
                    if report.get('target_accepted_s') is None:
                        report['target_accepted_s'] = round(
                            time.monotonic() - aim_started, 3)
                        report['aim_clock'] = 'target_accepted'
                    # Keep the frame the plate was actually seen in: the tracking
                    # template has to come from a frame that contains it, or the
                    # next (fresh) frame's re-detection can miss and abandon an
                    # acquisition that was already at confidence 0.95.
                    acquisition_frame = frame
                    break
                if attempt < len(path):
                    move = path[attempt]
                    if args.live_source:
                        check_frame('before_scan_move')
                    move_pointer(move[0], move[1])
                    report['inputs'][-1].update(step=f'scan-{attempt}', reason='scan_for_enemy')
                    delivered = report['inputs'][-1].get('pointer_delta')
                    if getattr(args, 'recentre_key', None) and clamped_delivery(move, delivered):
                        # A clamped sweep step rotates nothing, so the "360 degree
                        # scan" silently repeats one view. The M toggle restores
                        # the travel without turning the camera, so the same step
                        # is re-sent and the sweep keeps covering new ground.
                        recentre_pointer(probe, report, args.recentre_key, reason='scan')
                        move_pointer(move[0], move[1])
                        report['inputs'][-1].update(step=f'scan-{attempt}',
                                                    reason='scan_resend_after_recentre')
                    elif (args.recenter_margin > 0 and not uses_ei_transport(probe)
                            and clamped_delivery(move, delivered)):
                        # A clamped scan move turns the camera by nothing, so the
                        # sweep would silently repeat the same view. libei has no
                        # pointer room to run out of, so it never lands here.
                        pointer = pointer_state(probe.identity['display'])
                        retreat = reposition_move((pointer['x'], pointer['y']), width, height,
                                                  move, args.recenter_margin,
                                                  args.reposition_step)
                        if retreat is not None:
                            recovery = move_pointer(retreat[0], retreat[1])
                            report['inputs'][-1].update(step=f'scan-{attempt}',
                                                        reason='scan_reposition_retreat')
                            if recovery == tuple(retreat):
                                short = [move[0] - (delivered or [0, 0])[0],
                                         move[1] - (delivered or [0, 0])[1]]
                                move_pointer(short[0], short[1])
                                report['inputs'][-1].update(step=f'scan-{attempt}',
                                                            reason='scan_resend_after_clamp')
                            else:
                                report['inputs'][-1]['reread_before_resend'] = True
                    if args.scan_settle:
                        time.sleep(args.scan_settle)
            if target is None:
                raise ProbeError(f'no enemy healthbar found in {args.scan_steps} scan steps')
            args.target = (target[0], target[1])
            report['target'] = list(args.target)
            if args.settle:
                # The last scan step can still be traversing; let it finish so
                # the first measured error is not the scan's own motion.
                time.sleep(args.settle)
        for step in range(args.max_steps):
            # The operator budget runs from the search start, not from process
            # start: imports, preflight and the scan's own settle are excluded
            # only up to the moment the search began.
            # The aim budget starts when a qualified target was accepted; the
            # search has its own clock (--search-budget-s). Charging the sweep to
            # the aim budget is how every round ended without a shot.
            aim_ref = aim_started + float(report.get('target_accepted_s') or 0.0)
            if args.aim_budget_s and (time.monotonic() - aim_ref) > args.aim_budget_s:
                report['error'] = 'aim_budget_exceeded'
                report['aim_ms'] = round((time.monotonic() - aim_started) * 1000.0)
                break
            # A single unreadable frame is not evidence that the reticle is
            # gone: re-read bounded times before giving up on this step.
            centre = battle = weapon = None
            for attempt in range(args.reticle_retries):
                frame = next_frame(f'step-{step}-{attempt}', reuse=True)
                centre, battle, weapon, source = reticle_centre(frame)
                for hud_try in range(max(0, args.hud_retries)):
                    if battle is True:
                        break
                    frame = next_frame(f'step-{step}-hud{hud_try}', reuse=True)
                    centre, battle, weapon, source = reticle_centre(frame)
                    report.setdefault('hud_misses', []).append(
                        dict(stage='step', step=step, retry=hud_try))
                if battle is not True:
                    raise ProbeError('action revoked: current battle HUD unconfirmed')
                if centre is not None:
                    last_source = source
                    break
                if attempt == 1:
                    # The game only draws the green ring after the mouse moves,
                    # so an idle loop can read "unreadable" while the reticle is
                    # simply not painted. One unit is far below any aiming step.
                    move_pointer(args.aim_nudge or 1, 0)
                    report['inputs'][-1].update(step=step, reason='reticle_redraw_nudge')
            else:
                # In the binocular view this client draws neither the green ring
                # nor (in dark scenes) a readable ruler, but the scope picture is
                # centred by construction: measured ruler centre 1279 against a
                # 1280 frame centre. Accepting the frame centre is only allowed
                # once we have actually entered the scope in this run.
                # Both views put the aim caret on the frame centre (measured
                # 1280.5,805 on repeated live frames); this dark client simply
                # stops painting the green ring, so fall back to the centre and
                # say so instead of aborting the run.
                # The aim caret sits on the frame centre in both views (measured
                # 1280.5,805 across live frames); this client simply stops
                # painting the green ring in dark scenes. Refusing to aim there
                # aborted whole runs before the scope was even reached, so the
                # centre is accepted in third person too -- it can only arm the
                # coarse phase, because a shot still requires the scope evidence.
                # Inside the scope the centre is the aim point by construction,
                # so a missing ring no longer matters. In third person we cannot
                # invent one -- but we can stop guessing and go to the scope,
                # which is where the operator's procedure aims anyway.
                # Keep the third-person plate as the shot's evidence: the zoomed
                # view draws no nameplate at all, so the gate would otherwise
                # report enemy_plate_missing for a target that is plainly there.
                if tracking_candidate:
                    pre_hp = {}
                    try:
                        from nameplate_hp import read_hp
                        pre_hp = read_hp(frame, tracking_candidate['nameplate_bbox'])
                    except (ValueError, OSError):
                        pre_hp = {}
                    report['pre_scope_plate'] = dict(
                        bbox=list(tracking_candidate['nameplate_bbox']),
                        hp=pre_hp.get('current'),
                        red_px=tracking_candidate.get('red_px'),
                        body_visible=tracking_candidate.get('body_visible'))
                    time.sleep(args.scope_settle)
                    frame = next_frame(f'step-{step}-centre', reuse=True)
                    centre = [frame.shape[1] / 2.0, frame.shape[0] / 2.0]
                    source = 'scope_screen_center'
                    report.setdefault('reticle_centre_fallbacks', []).append(
                        dict(step=step, centre=centre, via='scope_on_unreadable'))
                else:
                    # No plate either: the caret still sits on the frame centre
                    # (measured 1280.5,805 in both views), so use it and record
                    # why. Aborting here ended whole runs (measured 2026-10-23:
                    # step 2 inside the scope); the shot still needs the scope
                    # and plate evidence, so a centre fallback cannot arm one.
                    frame = next_frame(f'step-{step}-centre', reuse=True)
                    centre = [frame.shape[1] / 2.0, frame.shape[0] / 2.0]
                    source = 'scope_screen_center'
                    report.setdefault('reticle_centre_fallbacks', []).append(
                        dict(step=step, centre=centre, via='unreadable_reticle_no_plate'))
            # The radius must cover the displacement of the move that was just
            # sent, so it is computed when that move is planned.
            radius = next_search
            captured_ns = (check_frame('after_reticle_detection')
                           if args.live_source else time.monotonic_ns())
            period_s = (None if previous_captured is None
                        else (captured_ns - previous_captured) / 1e9)
            if template is None and patch is not None:
                # The binocular zoom animates: for a few frames the plate is not
                # drawn at its final place, so one read is not evidence of a lost
                # target. Re-read a bounded number of fresh frames with the
                # scoped detector before failing.
                candidates = [c for c in find_enemy(frame, centre, args.aim_fraction,
                                                    scope_observed=bool(scope_for_aim))
                              if c['trackable']]
                scope_settle_tries = 0
                while len(candidates) != 1 and scope_settle_tries < 3:
                    scope_settle_tries += 1
                    time.sleep(0.25)
                    # The grabber lags the zoom animation and can hand back the
                    # same picture: measured, all six re-reads of a failed
                    # transition were one frame (identical mean, no plate) even
                    # though the settled frame did contain the enemy. Wait for a
                    # frame that actually differs before judging.
                    previous = _signature(frame)
                    for _ in range(8):
                        frame = next_frame(f'scope-settle-{step}-{scope_settle_tries}')
                        if _signature(frame) != previous:
                            break
                        time.sleep(0.2)
                    moved = reticle_centre(frame)
                    if moved and moved[0] is not None:
                        centre = moved[0]
                    candidates = [c for c in find_enemy(frame, centre, args.aim_fraction,
                                                        scope_observed=bool(scope_for_aim))
                                  if c['trackable']]
                # Still nothing: the third-person camera cannot pitch (measured
                # 0.1-0.4 px per 3 units), so a plate 200+ px off centre lands
                # ~4x further out once the binocular view magnifies and leaves
                # the frame entirely. The wide frame does tell us the direction,
                # so reacquire with a bounded scoped sweep sized by the measured
                # magnification, re-detecting after every move.
                sweep = 0
                while len(candidates) != 1 and sweep < 3:
                    sweep += 1
                    wide_y = error[1]
                    # Size the sweep from the physics instead of the wide gain:
                    # the binocular view magnifies ~4x and its vertical response
                    # is ~7 px/unit (both measured), so the off-screen distance
                    # is |wide_y| * 4 scoped px, i.e. |wide_y| * 4 / 7 units.
                    # The old /6 on the wide-gain solution delivered only 26
                    # units per sweep (182 px) and never reached a 616 px offset.
                    magnitude = abs(wide_y) * SCOPE_ZOOM_EST / SCOPE_GAIN_EST
                    magnitude = max(12, min(SCOPE_REACQUIRE_UNITS, int(round(magnitude))))
                    direction = -1 if (wide_y >= 0) == ((gain[1] or -1) >= 0) else 1
                    units = direction * magnitude
                    move_pointer(0, units, vertical_limit=SCOPE_REACQUIRE_UNITS)
                    report['inputs'].append(dict(relative=[0, units], reason='scope_reacquire',
                                                sweep=sweep, pointer_delta=[0, units]))
                    row.setdefault('scope_reacquire', []).append(units)
                    time.sleep(0.25)
                    frame = next_frame(f'scope-reacquire-{step}-{sweep}')
                    moved = reticle_centre(frame)
                    if moved and moved[0] is not None:
                        centre = moved[0]
                    candidates = [c for c in find_enemy(frame, centre, args.aim_fraction,
                                                        scope_observed=True)
                                  if c['trackable']]
                # Three enemies in the zoom is normal in a training room, and
                # demanding exactly one plate there ended the run with
                # "scope transition target missing or ambiguous". Pick by
                # identity first (the plate we were already tracking), else the
                # nearest to the reticle, and record which rule decided.
                if not candidates:
                    raise ProbeError('scope transition lost the target')
                if len(candidates) == 1:
                    pick, why = candidates[0], 'only_candidate'
                else:
                    previous_bbox = (tracking_candidate or {}).get('nameplate_bbox')
                    pick = choose_target(candidates, previous_bbox)
                    why = 'same_ship' if (previous_bbox and pick is not None
                                          and pick.get('nameplate_bbox') == previous_bbox) \
                        else 'nearest_to_reticle'
                if pick is None:
                    raise ProbeError('scope transition lost the target')
                report['scope_transition_pick'] = why
                report['scope_transition_candidates'] = len(candidates)
                args.target = pick['aim_px']
                tracking_candidate = pick
                patch = None
            if patch is None:
                if tracking_candidate is not None:
                    # First try the plate we already acquired, cropped from the
                    # frame that showed it. Measured 2026-10-29: a 0.95-confidence
                    # acquisition was dropped because the *next* frame's
                    # re-detection returned nothing, and every round ended before
                    # a single aiming step.
                    picked = None
                    if acquisition_frame is not None:
                        try:
                            patch, template, aim_offset = nameplate_patch(
                                acquisition_frame, tracking_candidate)
                            picked = tracking_candidate
                        except (ValueError, ProbeError):
                            patch = None
                    if picked is None:
                        candidates = [c for c in find_enemy(frame, centre, args.aim_fraction,
                                                            scope_observed=bool(scope_for_aim))
                                      if c['trackable']]
                        pick = choose_target(candidates, tracking_candidate['nameplate_bbox'])
                        if pick is None:
                            # One unreadable frame is not a lost target: the plate
                            # flickers while the camera settles. Retry once on a
                            # fresh frame before giving up on the acquisition.
                            for retry in range(2):
                                if args.settle:
                                    time.sleep(args.settle)
                                frame = next_frame(f'tracking-retry-{retry}')
                                centre, _, _, _ = reticle_centre(frame)
                                again = [c for c in find_enemy(
                                    frame, centre, args.aim_fraction,
                                    scope_observed=bool(scope_for_aim))
                                    if c['trackable']]
                                pick = choose_target(again, tracking_candidate['nameplate_bbox'])
                                if pick is not None:
                                    report.setdefault('tracking_retries', 0)
                                    report['tracking_retries'] += 1
                                    break
                        if pick is None:
                            raise ProbeError('enemy nameplate missing before tracking')
                        patch, template, aim_offset = nameplate_patch(frame, pick)
                elif target_provided:
                    x, y = int(round(args.target[0])), int(round(args.target[1]))
                    if not (80 <= x and x+80 < 2560 and 80 <= y-80 and y+80 < 1520):
                        raise ProbeError('commanded point has no trackable window inside the scene')
                    patch = dict(bbox=[x-40, y-40, 80, 80], score=1.0)
                    template = crop_template(frame, patch['bbox'])
                    if template is None:
                        raise ProbeError('commanded point has no trackable texture')
                else:
                    # The point came from our own scan, so losing the plate is a
                    # detection problem, not a commanded aim: re-detect instead
                    # of cropping an 80 px box of empty sea (measured: that box
                    # has no texture and aborted the run at 13 s).
                    again = [c for c in find_enemy(frame, centre, args.aim_fraction,
                                                   scope_observed=bool(scope_for_aim))
                             if c['trackable']]
                    pick = choose_target(again, tracking_candidate['nameplate_bbox']
                                         if tracking_candidate else None)
                    if pick is None:
                        raise ProbeError('enemy nameplate missing before tracking')
                    patch, template, aim_offset = nameplate_patch(frame, pick)
                    report.setdefault('reacquired_from_scan', 0)
                    report['reacquired_from_scan'] += 1
                expected = list(patch['bbox'])
            else:
                expected = predict_patch(patch['bbox'], velocity_ema, period_s)
                found = track_patch(template, expected, frame, search=radius)
                if found is None and args.reacquire:
                    # The patch track can fail on a moving hull; re-detect the
                    # nameplate and keep aiming instead of giving up. The fire
                    # gate is unchanged: two converged steps are still required.
                    again = [c for c in find_enemy(frame, centre, args.aim_fraction)
                             if c['trackable']]
                    if again:
                        pick = choose_target(again, patch.get('bbox'))
                        target = list(pick['aim_px'])
                        patch, template, aim_offset = nameplate_patch(frame, pick)
                        row_reacquire = dict(step=step, reacquired=True,
                                             target=[round(v, 1) for v in target],
                                             candidates=len(again))
                        report.setdefault('reacquired', []).append(row_reacquire)
                        found = patch
                if found is None:
                    raise ProbeError(f'tracking lost at step {step} (search {radius} px)')
                patch = found
                # Follow the target with a fresh crop so a turning hull does not
                # slowly break the match; keep the old template if the new
                # window has no texture.
                refreshed = crop_template(frame, found['bbox'])
                if refreshed is not None:
                    template = refreshed
            px, py = patch_centre(patch['bbox'])
            # Before the scope opens the reticle must sit on the *plate*, not on
            # the hull: the binocular view magnifies ~4x, and a hull-aim offset
            # of ~1.5 bar widths pushes the plate out of the top of the frame
            # (measured: every scoped frame showed open water). Centring the
            # plate first keeps it findable, and the scoped geometry then scales
            # the hull offset by the magnification on its own.
            pre_scope = bool(args.scope_after and not scope_for_aim)
            if pre_scope:
                aim_offset = [0.0, 0.0]
            px, py = px + aim_offset[0], py + aim_offset[1]
            error = [px - centre[0], py - centre[1]]
            previous_drift = own_ship_drift(previous_frame, frame) if args.compensate_drift else None
            if view_jump(previous_drift, last_move, gain, args.max_view_jump_px):
                report['view_jump_px'] = [round(v, 1) for v in previous_drift]
                report['view_jump_last_move'] = list(last_move) if last_move else None
                raise ProbeError(
                    'view jumped ' + ' / '.join(f'{v:.0f} px' for v in previous_drift)
                    + ' with no matching command; not an aiming result')
            vertical_errors.append(error[1])
            del vertical_errors[:-8]
            if vertical_reversals(vertical_errors, args.shake_amplitude_px,
                                  args.max_vertical_reversals) >= args.max_vertical_reversals:
                report['vertical_errors_px'] = [round(v, 1) for v in vertical_errors]
                raise ProbeError('vertical sight is oscillating over '
                                 f'{args.shake_amplitude_px:.0f} px; not an aiming result')
            history.append((captured_ns / 1e9, error[0], error[1]))
            del history[:-6]
            pan_log.append(previous_drift)
            del pan_log[:-6]
            # Fit the target's own motion, not the camera pan that also moves
            # the error: an uncompensated fit drove the lead to its ceiling and
            # pushed a stationary target off the reticle (measured live).
            velocity = fit_velocity(pan_compensated(history, pan_log),
                                    minimum=args.lead_min_samples,
                                    min_span_s=args.lead_min_span)
            previous_frame = frame
            previous_captured = captured_ns
            velocity_ema = blend_velocity(velocity_ema, velocity, args.lead_blend)
            if velocity is not None:
                velocity_samples += 1
            flight = hud_flight_seconds(frame)
            scope_evidence = hud_scope_evidence(frame) if (args.scope or args.scope_after) else None
            lead_seconds = args.lead_seconds
            if flight is not None and not args.no_lead:
                lead_seconds = flight['value']
            if args.lead_horizon:
                lead_seconds, horizon_s = lead_with_horizon(
                    flight['value'] if flight is not None else None,
                    args.lead_seconds, period_s, args.lead_horizon_cap)
            else:
                horizon_s = 0.0
            if args.no_lead:
                lead_seconds, horizon_s = 0.0, 0.0
            lead = lead_offset(velocity_ema if velocity_samples >= args.lead_min_samples else None,
                               lead_seconds, ceiling=args.lead_ceiling)
            control = [error[0] + lead[0], error[1] + lead[1]]
            # Range closure used to run only in the binocular view. The gap is
            # a correctness issue in either view (measured: 12.4 km crosshair
            # against a 12.8 km target = 400 m short, and 2.3 km against 11.5 km
            # earlier), and the loop re-reads the crosshair every step, so a
            # wrong per-unit constant only slows it down.
            closure_alive = (args.range_closure and args.last_range_closure is not None
                             and args.last_range_closure.get('aim_km') is not None
                             and args.last_range_closure.get('target_km') is not None)
            if args.range_closure and closure_alive:
                # Elevation is owned by the HUD range closure *while it can
                # read*. Leaving the geometric vertical error here made the two
                # controllers pull the same axis in opposite directions (the
                # hull point wants the reticle low, the closure wants it high),
                # which oscillated between 4 and 14 km and never settled.
                control = [control[0], 0.0]
            distance = math.hypot(*control)
            true_distance = math.hypot(*error)
            row = dict(step=step, reticle=centre,
                       reticle_source=last_source or 'measured',
                       battle_active=battle, weapon_status=weapon,
                       patch=patch['bbox'], patch_expected=None if patch is None else [round(v, 1) for v in expected],
                       match_score=round(patch['score'], 3),
                       aim_offset_px=list(aim_offset), aim_px=[px, py],
                       pre_scope_plate_centre=pre_scope,
                       flight_time=flight, scope_evidence=scope_evidence,
                       # Monotonic capture time and the observed period let a
                       # verifier prove the loop used newer frames, not the same
                       # one twice; the freshness contract is otherwise uncheckable.
                       captured_s=round(captured_ns / 1e9, 3),
                       frame_age_s=(None if not args.live_source else
                                    (time.monotonic_ns() - captured_ns) / 1e9),
                       step_period_s=None if period_s is None else round(period_s, 3),
                       lead_horizon_s=round(horizon_s, 3),
                       lead_total_s=round(lead_seconds, 3) if lead_seconds else lead_seconds,
                       error_px=[round(v, 2) for v in error], distance_px=round(distance, 2),
                       true_distance_px=round(true_distance, 2),
                       velocity_px_s=None if velocity is None else [round(v, 1) for v in velocity],
                       lead_px=[round(v, 1) for v in lead],
                       own_ship_drift_px=(None if previous_drift is None
                                          else [round(v, 1) for v in previous_drift]),
                       gain=[None if g is None else round(g, 3) for g in gain])
            if moved_but_unchanged(previous_error, error, last_move):
                # Keep the pending move until a response is observed. Learning
                # from this frame would turn transport delay into zero gain.
                if stale_reads >= args.max_stale_reads:
                    row['state'] = 'motion_response_unconfirmed'
                    report['steps'].append(row)
                    raise ProbeError('pending mouse move has no observed response after bounded rereads')
                stale_reads += 1
                converged = 0
                row['state'] = 'stale_frame_reread'
                row['stale_reads'] = stale_reads
                report['steps'].append(row)
                if args.settle:
                    time.sleep(args.settle)
                continue
            stale_reads = 0
            if last_move is not None:
                compensated_previous = previous_error
                if previous_drift is not None and previous_error is not None:
                    compensated_previous = [previous_error[0] + previous_drift[0],
                                            previous_error[1] + previous_drift[1]]
                gain, measured = update_gain(gain, compensated_previous, error, last_move,
                                            state=gain_state)
                row['measured_response_px_per_unit'] = [None if v is None else round(v, 3)
                                                        for v in measured]
                row['gain_after'] = [None if g is None else round(g, 3) for g in gain]
                for axis, response in enumerate(measured):
                    # The turrets traverse at a finite rate, so a mid-traverse
                    # reading can look weak. Only a move the pointer actually
                    # delivered and that still shows no response counts, and it
                    # must fail twice before this axis is called unusable.
                    # A weak *per-unit* gain is only useless when the move
                    # itself could not displace the plate beyond quantisation;
                    # several bounded vertical actions move it a few pixels and
                    # the per-unit value is then real.
                    # In the binocular view the vertical response is about one
                    # fifth of the third-person one (measured 0.2 vs 1.3 px per
                    # unit), so a small per-unit gain is real, not noise. Only a
                    # move that displaced nothing at all counts as no response.
                    displaced = None if response is None else abs(response * last_move[axis])
                    if (response is not None and abs(response) < MIN_GAIN
                            and (displaced is None or displaced < 1.0)):
                        weak_moves[axis] += 1
                        row.setdefault('ignored_small_response', {})[axis] = round(response, 3)
                dead = [axis for axis in (0, 1)
                        if gain[axis] is None and weak_moves[axis] >= 2]
                if dead:
                    row['state'] = row.get('state', 'measuring')
                    report['steps'].append(row)
                    raise ProbeError(f'axis {dead} produced no response in 2 delivered moves')
                last_move = None
            # Enter the binocular view only on a settled solution: a plate seen
            # on one scan frame can be mid-rotation, and the zoom then lands on
            # empty water (measured: plate absent in every scope re-read). The
            # fire gate already requires two consecutive settled frames.
            # Only the horizontal axis is required before the scope opens: the
            # third-person camera has almost no pitch response (measured: 3
            # units of vertical input moved the plate 0.1-0.4 px), so waiting
            # for a 2-D convergence there never happens. Elevation is closed
            # inside the binocular view, where the response is ~7 px/unit.
            if pre_scope and abs(error[0]) <= args.scope_after:
                pre_scope_ok += 1
            else:
                pre_scope_ok = 0
            # The operator's rule: open the binocular view once the *coarse*
            # pass is done -- at 12 km nothing is visible in third person, so
            # waiting for a sub-40 px solution there can never happen. Either
            # the horizontal error is small enough, or the coarse budget of
            # steps has been spent; both open the scope.
            # Both axes must be close: the scope magnifies ~5x, so a 150 px
            # vertical error lands ~750 px off-centre and the plate leaves the
            # zoomed view entirely (measured: "scope exited: no enemy healthbar").
            # Operator rule: the vertical is satisfied when the shell-impact
            # range beside the reticle is within 2 km of the enemy's own range --
            # far more robust than the vertical pixel response (0.1-0.4 px/unit
            # in third person). The pixel bound stays as a fallback for frames
            # where the readout is not drawn.
            # Prefer the flight-time derivation; fall back to the text readout.
            # Only what the HUD actually draws may decide this, and the "- -"
            # state it draws beyond maximum range is information: it means the
            # aim is too far and must be lowered, not estimated.
            from fire_hud import impact_readout as _impact
            aim_state, aim_km_now = _impact(frame, centre)
            row['impact_state'] = aim_state
            # "- -" is also what an unreadable readout returns, so it must never
            # move a view that is already on the ship: once the plate is inside
            # the fire tolerance, this branch is how a good aim got thrown away.
            if (aim_state == 'beyond' and args.range_closure
                    and distance > args.fire_tolerance):
                # "--.-" means the reticle is past the gun's reach, but it is
                # also what an unreadable readout returns, so the pitch-down is
                # bounded: after --beyond-max-steps the ordinary control resumes
                # instead of spending the whole budget on one branch (measured
                # 2026-10-21: 13 steps of it and no aiming step at all).
                beyond_moves += 1
                # Inside the zoom one mouse unit is ~7 px, so the third-person
                # 160-unit jolt flies straight past the target; the scoped view
                # gets its own small step and its own budget.
                step = (args.scope_beyond_step_units if scope_for_aim
                        else args.beyond_step_units)
                budget = (args.scope_beyond_max_steps if scope_for_aim
                          else args.beyond_max_steps)
                if beyond_moves <= max(0, budget) and step:
                    move_pointer(0, step, vertical_limit=step)
                    report['inputs'][-1].update(step=step, reason='shorten_aim_beyond_range',
                                                scoped=bool(scope_for_aim))
                    pre_scope_ok = 0
                    converged = 0
                    row['state'] = 'shorten_beyond_range'
                    report['steps'].append(row)
                    continue
                report['beyond_steps_capped'] = beyond_moves
            target_km_now, target_km_source = (
                target_range_km(frame, tracking_candidate['nameplate_bbox'])
                if tracking_candidate else (None, 'no_plate'))
            row['target_km_source'] = target_km_source
            range_ready = (aim_km_now is not None and target_km_now is not None
                           and abs(target_km_now - aim_km_now) <= args.scope_range_gap_km)
            row['scope_range_gap_km'] = (None if (aim_km_now is None or target_km_now is None)
                                         else round(target_km_now - aim_km_now, 2))
            # Prefer the operator's 2 km rule; fall back to the vertical pixel
            # bound when this client does not draw the range at all (measured:
            # the readout is absent on most third-person frames here, and
            # refusing forever would mean never firing).
            row['scope_vertical_source'] = 'range_2km' if range_ready else 'vertical_px'
            vertical_ok = range_ready or abs(error[1]) <= args.scope_vertical_max_px
            coarse_done = (abs(error[0]) <= args.scope_after
                           and vertical_ok and pre_scope_ok >= 2)
            if (not coarse_done and step + 1 >= max(1, args.scope_after_steps)
                    and abs(error[0]) <= args.scope_escape_max_px
                    and vertical_ok):
                coarse_done = True
                report['scope_opened_by_step_budget'] = step + 1
            elif (not coarse_done and step + 1 >= max(1, args.scope_after_steps)):
                report['scope_budget_waiting'] = dict(step=step + 1,
                                                      error_x=round(error[0], 1),
                                                      ceiling=args.scope_escape_max_px)
            if args.scope_after and not scope_for_aim and coarse_done:
                # The zoom shows the gun's own line, so opening it mid-traverse
                # relocates the target by hundreds of pixels and the run spends
                # the rest of its budget chasing that jump. Wait for the mounts,
                # bounded: a turret that never settles must not deadlock the shot.
                waited_traverse = 0.0
                while waited_traverse < getattr(args, 'scope_traverse_wait_s', 3.0):
                    if mounts_traversing(main_battery_fire_evidence(frame)) is not True:
                        break
                    if args.settle:
                        time.sleep(args.settle)
                    waited_traverse += max(0.2, args.settle)
                    frame = next_frame(f'scope-traverse-{int(waited_traverse * 10)}')
                row['scope_traverse_wait_s'] = round(waited_traverse, 2)
                if args.live_source:
                    check_frame('before_scope_input')
                # The user's procedure: put the reticle near the target first,
                # then open the binocular view, then fire. The zoom changes the
                # mouse gain, so everything learned in third person is dropped
                # and re-measured in the new view.
                # Remember the plate's offset before the zoom: if the scope
                # opens onto empty water, this is the only evidence of which way
                # the target went, and it drives the bounded ladder below.
                report['scope_vertical_hint_px'] = round(float(error[1]), 1)
                report['scope_ladder_moves'] = 0
                probe.key(args.scope_key)
                report['inputs'].append(dict(key=args.scope_key, reason='enter_scope_for_aim'))
                # The zoom animates: reading a frame in the first moments found no
                # plate on every attempt and abandoned the scope. Wait for the
                # transition to finish before the first scoped read.
                time.sleep(args.scope_settle)
                report['scope_for_aim'] = dict(step=step, distance_px=round(distance, 2))
                scope_for_aim = True
                # scope_entered drives both the frame-centre reticle fallback
                # (this client paints no ring in the scope) and the exit-scope
                # cleanup; without it the run aborted at the first scoped frame
                # and would have left the player zoomed in.
                scope_entered = True
                # The zoom changes the gain, so third-person gain cannot be
                # reused -- but a scope gain measured earlier **in this same
                # game process** can: it saves the two probe steps that
                # otherwise cost ~4 s and, measured 2026-10-20, pushed the
                # convergence confirmation past the last step.
                # The gain changes with the zoom level, so a value carried over
                # from another zoom walked the camera the wrong way (measured:
                # cached (-1.75, 7.18) turned a (-44,-39) error into (-190,189)).
                # Measure this view now with a small round trip per axis.
                def _scope_plate(frame, _args=args):
                    cands = [c for c in find_enemy(frame, None, _args.aim_fraction,
                                                   scope_observed=True) if c['trackable']]
                    return None if not cands else cands[0]['aim_px']

                # The gain changes with the zoom, and a value cached from an
                # earlier run (or another ship's scope) is worse than no gain at
                # all: measured 2026-10-29 a cached (-5.67,-4.0) turned a -30 px
                # solution into -733 px in two steps. Measure this view, every
                # time; the cache is opt-in for debugging only.
                cached_scope = (load_scope_gain(probe.identity)
                                if (probe.identity and getattr(args, 'scope_gain_cache', False))
                                else None)
                if cached_scope:
                    measured, why = None, 'cached'
                elif args.scope_probe <= 0:
                    # No probe: the zoom probe costs ~2 s of the 15 s budget and
                    # its readings were polluted by the transition frames
                    # (measured 2026-10-23). Keep the third-person seed and let
                    # update_gain() learn from the first real correction.
                    measured, why = None, 'probe_skipped'
                else:
                    measured, why = measure_scope_gain(
                        probe, next_frame, _scope_plate, report,
                        units=(max(2, args.scope_probe), max(1, args.probe_y)), min_shift=8.0)
                if cached_scope:
                    measured = list(cached_scope)
                if measured:
                    # A 3-unit probe moves ~20 px in the scope, and a moving
                    # ship's own jitter is the same size, so a small probe
                    # produced 0.42 px/unit where the truth is ~7 and the loop
                    # then moved 6 units per step (measured 2026-10-29: vertical
                    # error sat at -100 px for 17 steps).
                    sane = []
                    for axis, value in enumerate(measured):
                        if value is None:
                            sane.append(None)
                            continue
                        magnitude = abs(float(value))
                        if GAIN_SANE_MIN <= magnitude <= GAIN_SANE_MAX:
                            sane.append(float(value))
                        else:
                            seed = (getattr(args, 'seed_gain', None) or [None, None])[axis]
                            sane.append(float(seed) if seed else None)
                            report.setdefault('scope_gain_rejected', []).append(
                                dict(axis=axis, measured=round(magnitude, 3)))
                    measured = sane
                    gain = list(measured)
                    if probe.identity:
                        save_scope_gain(probe.identity, gain)
                    report['scope_gain_source'] = 'measured'
                else:
                    gain = [None, None]
                    report['scope_gain_source'] = f'measure_failed:{why}'
                report['scope_gain_seed'] = list(gain)
                report.setdefault('gain_before_scope', list(report.get('gain_before_scope') or gain))
                gain_state = {}
                weak_moves = {0: 0, 1: 0}
                probe_rounds = {}
                history = []
                pan_log = []
                velocity_ema = None
                velocity_samples = 0
                previous_error = None
                previous_captured = None
                last_move = None
                previous_frame = None
                # The patch template was cropped in the old field of view; a
                # zoomed frame cannot match it, so drop it and let the ordinary
                # re-detection path pick the plate up again in the new view.
                template = None
                # The zoom displaces the target by far more than a third-person
                # step: measured 2026-10-20 the first scoped read showed ~500 px
                # offset while the inherited search radius was 160 px, so the
                # tracker lost it immediately. Search wide on the first scoped
                # frame; the loop narrows it again from the planned move.
                next_search = 900
                # The convergence that mattered in third person says nothing
                # about the zoomed view, so it must not count towards the two
                # consecutive converged steps the fire gate needs.
                converged = 0
                row['state'] = 'enter_scope_for_aim'
                row['convergence_reset'] = 'scope_changed_fov'
                report['steps'].append(row)
                if args.settle:
                    time.sleep(args.settle)
                scope_camera = camera_mode_evidence(next_frame('scope-entered-for-aim'),
                                                    baseline=scope_reference)
                report['scope_camera_evidence'] = scope_camera
                row['scope_camera_evidence'] = scope_camera
                continue
            # Divergence guard: a correction that *increases* the elevation
            # error means the stored vertical sign is wrong (measured in the
            # scope: error walked -35 -> -503 px while every step "corrected").
            # Flip once and let the loop continue; the fire gate is unchanged.
            if (last_move is not None and gain[1] is not None and previous_error is not None
                    and abs(last_move[1]) >= 1
                    and abs(error[1]) > abs(previous_error[1]) + 5
                    and abs(error[1]) > abs(previous_error[1]) * 1.02):
                gain[1] = -abs(gain[1])
                row['gain_inverted'] = 'vertical_correction_moved_away'
            # Elevation is closed on the HUD's own ballistic range, not on hull
            # pixels: the hull-window heuristic aimed 0.3 km short at 7.7 km and
            # every shell fell in the water. A missing reading is 'unavailable'
            # and blocks the shot rather than being treated as closed.
            range_state = None
            if args.range_closure and scope_for_aim:
                # hud_aim_km reads the number beside the reticle and needs the
                # reticle's *box*; handing it the centre point silently returns
                # None, which killed range closure and, because closure owns the
                # vertical axis, left elevation uncorrected for whole runs.
                # Range closure reads the drawn distance only (the "- -" state is
                # handled by the shorten branch above), never an estimate.
                from fire_hud import impact_readout as _impact_closure
                _closure_state, aim_km = _impact_closure(frame, centre)
                row['closure_impact_state'] = _closure_state
                # The caption sits under the *nameplate*; the tracked patch is an
                # 80x80 template window and reads nothing (measured: target_km
                # None on every step of a whole run).
                caption_plate = (tracking_candidate or {}).get('nameplate_bbox')
                caption_km = (hud_distance_km(frame, caption_plate)
                              if caption_plate else None)
                target_km, source = target_range(caption_km, args.range_target_km)
                # range_correction already folds the aiming direction in, so it
                # needs the magnitude: the in-place calibration measures a signed
                # km/unit (-0.025 measured) and passing that sign would aim the
                # wrong way.
                # The in-place calibration is only trusted when it is physically
                # plausible (0.01..0.2 km per unit); a wild value (measured -0.5)
                # means the readout flipped between two clusters, and using it
                # would fling the aim.
                km_per_unit = abs(args.range_km_per_unit)
                if not 0.01 <= km_per_unit <= 0.5:
                    km_per_unit = RANGE_KM_PER_UNIT
                    row['range_km_per_unit_rejected'] = args.range_km_per_unit
                # Learn the real km/unit from the last closure move: one step of
                # 160 units moved the aim ~20 km (0.125 km/unit) where the
                # constant claimed 0.025, so a fixed value overshot by 2x-5x.
                if args.range_km_per_unit_learn:
                    previous = args.range_km_per_unit_learn
                    if previous.get('aim_km') is not None and aim_km is not None \
                            and previous.get('units'):
                        observed = abs(float(aim_km) - float(previous['aim_km'])) / abs(previous['units'])
                        if 0.01 <= observed <= 0.5:
                            blended = 0.5 * km_per_unit + 0.5 * observed
                            args.range_km_per_unit = args.range_km_per_unit_learn['value'] = blended
                            row['range_km_per_unit_learned'] = round(blended, 4)
                            km_per_unit = blended
                units, why = range_correction(aim_km, target_km, args.range_tolerance_km,
                                              km_per_unit, args.range_step_units)
                range_state = dict(aim_km=aim_km, target_km=target_km,
                                   target_source=source, units=units, reason=why)
                row['range_closure'] = range_state
                args.last_range_closure = range_state
                # The closure owns the vertical axis, so an unreadable range
                # leaves elevation uncorrected forever and the run can never
                # settle (measured 2026-10-29: x converged to -5.6 px while y sat
                # at 66-84 px for every step). After two unreadable steps, hand
                # the vertical back to the pixel loop.
                if why == 'unavailable':
                    closure_gaps += 1
                    if closure_gaps >= 2:
                        args.range_closure = False
                        report['range_closure_fallback'] = 'pixel_vertical'
                        row['closure_fallback'] = 'two_unreadable_readings'
                else:
                    closure_gaps = 0
                if why == 'move':
                    args.range_km_per_unit_learn = dict(value=km_per_unit, units=units,
                                                        aim_km=aim_km)
                    move_pointer(0, units, vertical_limit=args.range_step_units)
                    report['inputs'].append(dict(relative=[0, units],
                                                 reason='range_closure', step=step))
                    converged = 0
                    report['steps'].append(row)
                    continue
            # Precision comes from the plate in view, not from a constant: the
            # drawn bar is the only thing on screen that says how big the ship
            # is, and it is what the reticle has to sit inside.
            if getattr(args, 'fire_tolerance_from_plate', False):
                resolved, why = resolve_fire_tolerance(
                    (tracking_candidate or {}).get('nameplate_bbox'),
                    args.fire_tolerance_ceiling)
                if resolved is not None and abs(resolved - args.fire_tolerance) > 0.5:
                    row['fire_tolerance_px'] = round(resolved, 1)
                    row['fire_tolerance_source'] = why
                    report['fire_tolerance_source'] = why
                    args.fire_tolerance = resolved
                    args.tolerance = min(args.tolerance, resolved)
            # The target sits |lead| away from the reticle by construction, so
            # the settle rule must admit a real lead: how far off the hull a shot
            # may be is the *fire* tolerance, not twice the fine one.
            lead_bound = max(args.tolerance * 2, args.fire_tolerance)
            # The operator's rule is "close enough to hit", and the fire gate
            # already expresses that as --fire-tolerance. Measured 2026-10-23:
            # the loop sat at 66 px (inside the 140 px fire tolerance) and kept
            # correcting against the 60 px fine tolerance until the step budget
            # ran out, so a perfectly shootable solution never fired.
            accept_px = (args.fire_tolerance if getattr(args, 'settle_with_fire', False)
                         else args.tolerance)
            settled = (distance <= accept_px
                       and (not lead_seconds or true_distance <= lead_bound))
            # The procedure locks *before* the fine loop: the game's own lock
            # holds the horizontal aim on the target, so any horizontal lead is
            # applied on top of a stable track. X toggles, so it may be pressed
            # exactly once per run -- pressing it again just before the shot
            # (the old order) would have unlocked the target.
            if (args.lock and not locked and scope_for_aim
                    and scope_confirmed(scope_evidence, scope_camera)):
                probe.key(args.lock_key)
                report['inputs'].append(dict(key=args.lock_key, step=step,
                                             reason='lock_target_after_scope'))
                locked = True
                row['locked'] = True
                if args.settle:
                    time.sleep(args.settle)
            if settled:
                converged += 1
            else:
                converged = 0
            row['state'] = 'converged' if settled else (
                'within_fire_tolerance' if distance <= args.fire_tolerance else row.get('state'))
            # Record the range gap every step: aim range (crosshair) against the
            # target's own caption. Measured 2026-10-20 at 12.8 km the crosshair
            # read 12.4 km, i.e. 400 m short -- that, not the hull detector, is
            # what makes salvos miss. A readable gap turns a miss into a number.
            try:
                range_row = step_range_evidence(
                    frame,
                    tracking_candidate['nameplate_bbox'] if tracking_candidate else None,
                    reticle=getattr(reticle_centre, 'last_bbox', None))
                row.update(range_row)
            except (ValueError, OSError, KeyError):
                row['range_gap_km'] = None
            ready, fire_reason = fire_ready(distance, converged, step, args.tolerance,
                                            args.fire_min_steps)
            # Smoothness monitor: the error trajectory of this engagement is
            # analysed on every step, and a shaking camera is not allowed to
            # fire (it fires from a moving aim point, which is how straddles
            # happen). --no-smoothness-guard keeps the metric without the veto.
            shot_errors = [r.get('error_px') for r in report['steps']
                           if isinstance(r, dict) and r.get('error_px')]
            if len(shot_errors) >= 2:
                import aim_smoothness as _smooth
                _metrics = _smooth.analyse(shot_errors)
                report['aim_smoothness'] = _metrics
                if (ready and _metrics['smoothness'] == 'oscillating'
                        and not getattr(args, 'no_smoothness_guard', False)):
                    ready, fire_reason = False, 'aim_oscillated'
            if args.range_closure and (range_state is None
                                       or range_state.get('reason') != 'closed'):
                ready, fire_reason = False, 'range_not_closed'
            if args.fire_max_km is not None:
                # Distance is not a legality condition (the nine gates say so),
                # so a long shot is recorded as a warning with the number the
                # miss is expected to be, not as a veto. The reading is the
                # caption under the plate (58 ms) from this step's one range
                # read; the minimap recogniser (4.5 s uncached) is only a
                # fallback inside step_range_evidence.
                near = row.get('target_km')
                if near is not None and near > args.fire_max_km:
                    warning = dict(code='target_beyond_fire_max_km',
                                   target_km=near, fire_max_km=args.fire_max_km,
                                   expected_short_or_long_km=round(near - args.fire_max_km, 2),
                                   range_source=row.get('range_source'))
                    row['fire_max_km_warning'] = warning
                    report['warnings'].append(warning)
            if args.scope_after and not scope_for_aim:
                # The procedure is aim, then scope, then fire: a good third-person
                # solution must not shoot before the binocular view is opened.
                ready, fire_reason = False, None
            if scope_blocks_fire(args.scope or args.scope_after,
                                 scope_confirmed(scope_evidence, scope_camera)):
                # Wanting the scope is not having it: without HUD evidence the
                # shot would leave the barrel from a view nobody confirmed.
                ready, fire_reason = False, 'scope_not_evidenced'
            row['fire_gate'] = fire_reason if not ready else 'ready'
            if args.live_source:
                check_frame('before_servo_action')
            if ready or settled:
                report['steps'].append(row)
                if args.fire and ready:
                    if args.lock and not locked:
                        # Only reachable without the scope path: lock once here.
                        probe.key(args.lock_key)
                        report['inputs'].append(dict(key=args.lock_key, step=step,
                                                     reason='lock_target'))
                        locked = True
                        if args.settle:
                            time.sleep(args.settle)
                    confirm_fire(f'pre-fire-{step}', expected=[px, py], lead=lead,
                                 allow_missing=True)
                    held_button = 1
                    probe.x('mousedown', 1)
                    probe.x('mouseup', 1)
                    held_button = None
                    report['inputs'].append(dict(button=1, pulse=True, step=step))
                    report['fired'] = True
                    report['fire_reason'] = fire_reason
                    report['aim_ms'] = round((time.monotonic() - aim_started) * 1000.0)
                    # Independent evidence that the shot left the barrels: a
                    # muzzle flash lives well under a second, so the frames are
                    # taken back to back with no settle in between.
                    fire_frames = []
                    for index in range(args.post_fire_frames):
                        path = probe.snapshot(f'fire-{index}')
                        fire_frames.append(str(path))
                    report['fire_frames'] = fire_frames
                    break
                if not args.fire:
                    report['steps'][-1]['state'] = 'converged (no fire: pass --fire to shoot)'
                    break
                continue
            converged = 0
            # Fine steps once the target is close: a full-reach correction
            # overshoots and then oscillates around the reticle.
            # Inside the scope one unit moves ~5-7x further, so the horizontal
            # budget must shrink with the zoom or a single step flings the view.
            reach = args.scope_reach if scope_for_aim else args.reach
            if distance < args.tolerance * 3:
                reach = max(20, args.reach // 3)
            # Vertical aiming was needlessly slow: one action is capped at
            # reach-y (3 units, about 4 px in third person and 2 px in the
            # scope) while a horizontal action may carry 150 units. Coarse
            # elevation gets its own, still bounded, budget; the fine phase
            # keeps the small step so it cannot overshoot.
            # The 3-unit vertical step is a third-person safety limit; inside the
            # scope one unit is ~7 px, so the same limit made a 500 px elevation
            # error take ~24 steps and never converge (measured 2026-10-20).
            reach_y = args.scope_reach_y if scope_for_aim else args.reach_y
            # Coarse elevation is for a screen-scale miss, not for every error
            # outside the fine tolerance: a 12 px tolerance would otherwise turn
            # a 20 px error into a 40-unit overshoot that oscillates. The floor
            # is the miss a fine step needs several actions to remove.
            if abs(error[1]) > max(args.tolerance, 40) and gain[1] is not None:
                # The coarse grant must respect the view: the third-person cap is
                # 6 units, but in the scope one unit is ~7 px, so granting the
                # same 6 left the vertical error at -200 px for a whole run
                # (measured 2026-10-29: every correct step sent exactly -6).
                grant = (args.scope_coarse_reach_y if scope_for_aim
                         else args.coarse_reach_y)
                reach_y = max(reach_y, grant)
            # ``control`` is the lead-corrected error: the move must carry the
            # reticle onto the *lead point*. Using the raw error here left the
            # reticle on the hull while ``distance`` stayed at |lead| forever,
            # so the loop never settled and the computed move hit zero.
            # Inside the scope the same mouse unit turns the view ~5-7x further,
            # so a 60-unit gain probe throws the target ~420 px away and loses
            # the track (measured 2026-10-20). Probe small in the binocular view.
            probe_units = args.scope_probe if scope_for_aim else args.probe
            # Already-close solutions get micro corrections only: a full-reach
            # step from 40 px away lands 200 px on the other side, which is how a
            # good aim got thrown away and the run then ran out of steps.
            plan_damping = args.damping
            if distance <= max(2.0 * args.fire_tolerance, args.tolerance * 2):
                reach = min(reach, args.freeze_reach)
                reach_y = min(reach_y, args.reach_y)
                plan_damping = min(plan_damping, args.freeze_damping)
                row['frozen_step'] = True
            row['scope_for_aim'] = bool(scope_for_aim)
            row['reach_y_used'] = int(reach_y)
            row['reach_x_used'] = int(reach)
            move_x, move_y, reason = plan_move(control, gain, probe=probe_units,
                                               reach=reach, damping=plan_damping,
                                               probe_y=args.probe_y, reach_y=reach_y)
            # A probe that produces no measurable response repeats forever: the
            # plate is outside the patch window, so the error never changes and
            # the axis stays unknown while the budget drains (measured 2026-10-29:
            # six probe_x steps in a row, zero convergence). After two tries, fall
            # back to the gain this view had before the zoom and let the
            # plausibility gate refine it from real responses.
            if reason in ('probe_x', 'probe_y'):
                axis = 0 if reason == 'probe_x' else 1
                probe_rounds[axis] = probe_rounds.get(axis, 0) + 1
                if probe_rounds[axis] >= 2:
                    # Fall back, in order, to the gain this view had before the
                    # zoom, then to the caller's seed. Raising here aborted the
                    # run at step 1 when the scope was opened first (measured
                    # 2026-10-29: six runs died with 'axis N gain is
                    # unmeasurable'), which is the dead end this work is about.
                    candidates_fb = [(report.get('gain_before_scope') or [None, None])[axis],
                                     (getattr(args, 'seed_gain', None) or [None, None])[axis]]
                    fallback = next((value for value in candidates_fb if value), None)
                    if fallback:
                        gain[axis] = float(fallback)
                        row['gain_fallback'] = dict(axis=axis, value=round(gain[axis], 3),
                                                    reason='probe_produced_no_response')
                        move_x, move_y, reason = plan_move(control, gain, probe=probe_units,
                                                           reach=reach, damping=plan_damping,
                                                           probe_y=args.probe_y, reach_y=reach_y)
                    else:
                        report['gain_probe_exhausted'] = axis
                        row['gain_probe_exhausted'] = axis
            row['state'] = reason
            report['steps'].append(row)
            if reason == 'gain_too_small':
                raise ProbeError(f'gain {gain} is too small to steer')
            if (move_x, move_y) == (0, 0):
                # The proportional correction quantises to zero once the error
                # is small. That is the converged case, not a failure: count it
                # and let the two-frame confirmation and fire gate decide.
                if distance <= (args.fire_tolerance if getattr(args, 'settle_with_fire', False)
                                else args.tolerance):
                    converged += 1
                    row['state'] = 'converged'
                    row['quantised_move_zero'] = True
                    ready, fire_reason = fire_ready(distance, converged, step,
                                                    args.tolerance, args.fire_min_steps)
                    if args.scope_after and not scope_for_aim:
                        ready, fire_reason = False, None
                    report['steps'].append(row)
                    if ready:
                        break
                    if args.settle:
                        time.sleep(args.settle)
                    continue
                # A zero plan with a large error means the gain is unusable, not
                # that the aim is good: one unit per step cannot remove 700 px and
                # the run then burns the whole budget nudging. Name it instead of
                # pretending the loop is converging.
                if (distance > max(args.fire_tolerance, args.tolerance) * 2
                        and (gain[0] is None or gain[1] is None)):
                    report['zero_plan_error_px'] = [round(v, 1) for v in error]
                    report['zero_plan_gain'] = list(gain)
                    raise ProbeError('plan is zero while the target is '
                                     f'{distance:.0f} px away and the gain is unknown: '
                                     'the view cannot be steered')
                # Out of tolerance but the plan rounded to nothing: a sub-unit
                # error would otherwise end the run. Step one unit along the
                # planned direction and keep going; the fire gate is unchanged.
                axis = 0 if abs(error[0]) >= abs(error[1]) else 1
                direction = -1.0 if error[axis] > 0 else 1.0
                if gain[axis]:
                    direction = -1.0 if (error[axis] / gain[axis]) > 0 else 1.0
                tiny = [0.0, 0.0]
                tiny[axis] = direction
                move_pointer(tiny[0], tiny[1])
                report['inputs'][-1].update(step=step, reason='minimum_step_after_zero_plan')
                row['state'] = 'minimum_step'
                report['steps'].append(row)
                if args.settle:
                    time.sleep(args.settle)
                continue
            if (args.recenter_margin == 0 and
                    getattr(probe, 'relative_transport', 'xtest') == 'xtest'):
                # Any correction needs pointer travel in its direction. Measured:
                # with the seat pointer pinned at the bottom edge (y=1423 of
                # 1440) a downward correction is delivered as 0 and the error
                # stalls at about four pixels per step; the same happens at the
                # right edge for a rightward one. Make room first; the rotation
                # that causes is re-aimed on the next frame.
                pointer = pointer_state(probe.identity['display'])
                try:
                    geometry = probe.x('getdisplaygeometry').split()
                    screen_w, screen_h = (int(geometry[0]), int(geometry[1])) \
                        if len(geometry) == 2 else (width, height)
                except Exception:
                    screen_w, screen_h = width, height
                travel = 60
                stuck_x = ((move_x > 0 and screen_w - pointer['x'] < travel) or
                           (move_x < 0 and pointer['x'] < travel))
                stuck_y = ((move_y > 0 and screen_h - pointer['y'] < travel) or
                           (move_y < 0 and pointer['y'] < travel))
                probe_y_needs_room = (reason == 'probe_y' and
                                      min(pointer['y'], screen_h - pointer['y']) < 40)
                if stuck_x or stuck_y or probe_y_needs_room:
                    retreat_x = (-max(-600, min(600, pointer['x'] - screen_w // 2))
                                 if stuck_x else 0)
                    retreat_y = (-max(-300, min(300, pointer['y'] - screen_h // 2))
                                 if (stuck_y or probe_y_needs_room) else 0)
                    if retreat_x or retreat_y:
                        pointer_probe.move(retreat_x, retreat_y)
                        report['inputs'][-1].update(step=step, reason='pointer_room_recovery')
                        row['pointer_room_recovery'] = [retreat_x, retreat_y]
                        row['pointer_before_recovery'] = [pointer['x'], pointer['y']]
                        report['steps'].append(row)
                        if args.settle:
                            time.sleep(args.settle)
                        continue
            move_pointer(move_x, move_y, vertical_limit=reach_y)
            report['inputs'][-1].update(step=step, reason=reason)
            # A large correction displaces the target further than the patch
            # tracker can search (measured: an 828 px error with a 160 px radius
            # lost the plate at the very next step), so drop the template and let
            # the next step re-detect the nameplate instead of tracking it.
            planned_px = max(abs(move_x * (gain[0] or 1.0)), abs(move_y * (gain[1] or 1.0)))
            if planned_px > args.reacquire_px:
                # Clearing the template alone left the next step tracking with
                # None, which can only fail ("tracking lost" on every run).
                # Clearing the patch as well makes the loop re-detect instead.
                template = None
                patch = None
                tracking_candidate = None
                report['inputs'][-1]['reacquire_after_px'] = round(planned_px, 1)
            # One action carries at most reach-y vertical units and the measured
            # response is about 1.3 screen px per unit, so a 70 px elevation
            # error crawls through the loop. Extra packets are separate bounded
            # actions with a settle between them, never a rapid burst, and each
            # one records its own delivered motion.
            wanted = 1
            if reason == 'probe_y':
                # One 3-unit action moves the plate by well under a pixel in the
                # binocular view (measured 0.1-0.2 px per unit), which the gain
                # read cannot separate from quantisation. Probe with several
                # bounded actions so the displacement is measurable.
                wanted = max(1, min(args.vertical_packets, 4))
            elif move_y and gain[1]:
                # One packet moves about |gain_y| * reach-y screen pixels. Send
                # only as many whole packets as the remaining error needs, so a
                # fine correction stays one small action and a coarse scope
                # offset does not crawl through the loop.
                per_packet = abs(gain[1]) * abs(move_y)
                if per_packet > 0:
                    # The gain estimate lags one step behind, so aim at a
                    # fraction of the error: measured without this the loop
                    # overshot (error -44 px became +11 px) and then walked back,
                    # which reads as the camera creeping after the hull was on
                    # the reticle.
                    wanted = min(args.vertical_packets,
                                 max(1, int(abs(error[1]) * args.damping / per_packet)))
            packets_done = 1
            for packet in range(max(0, wanted - 1)):
                if not move_y:
                    break
                if args.packet_settle:
                    time.sleep(args.packet_settle)
                move_pointer(0, move_y)
                report['inputs'][-1].update(step=step, reason='vertical_packet',
                                            packet=packet + 2)
                delivered_packet = report['inputs'][-1].get('pointer_delta') or [0, 0]
                packets_done += 1
                # The game recentres its own cursor, so a zero pointer delta in
                # the middle of the window is not a clamp. Counting it as one
                # made every step "retreat", which rotated the camera the wrong
                # way and pushed the elevation error from -122 to -266 px.
                pointer_now = pointer_state(probe.identity['display'])
                room = min(pointer_now['y'], height - pointer_now['y'])
                if (not uses_ei_transport(probe) and room < 40
                        and abs(delivered_packet[1]) < abs(move_y)):
                    report['inputs'][-1]['vertical_packet_clamped'] = True
                    # The pointer has run out of vertical travel, which is what
                    # stalls an elevation correction while the error stays put
                    # (measured: 16 packets requested, 2 delivered, error stuck
                    # near -190 px). Restore travel toward the window centre and
                    # re-send; the retreat rotates the camera, so this step is
                    # excluded from gain learning and the next frame re-aims.
                    pointer = pointer_state(probe.identity['display'])
                    room = pointer['y'] - height // 2
                    retreat = -max(-args.reposition_step, min(args.reposition_step, room))
                    if retreat:
                        # The recovery must not go through the aiming bound: the
                        # pointer is pinned at a window edge (measured y=0), and a
                        # 3-unit move cannot restore travel, so every packet was
                        # dropped and the elevation error stopped falling.
                        pointer_probe.move(0, retreat)
                        report['inputs'][-1].update(step=step,
                                                    reason='vertical_reposition_retreat')
                        row['vertical_reposition_retreat'] = retreat
                        last_move = None
                    break
            row['vertical_packets'] = packets_done
            delivered = report['inputs'][-1].get('pointer_delta')
            clamped = (args.recenter_margin > 0 and not uses_ei_transport(probe)
                       and clamped_delivery((move_x, move_y), delivered))
            if clamped:
                # An absolute warp is ignored by the game, so make room with a
                # relative retreat, then deliver the move that never arrived.
                pointer = pointer_state(probe.identity['display'])
                retreat = reposition_move((pointer['x'], pointer['y']), width, height,
                                          (move_x, move_y), args.recenter_margin,
                                          args.reposition_step)
                row['clamped_delivery'] = list(delivered or [])
                if retreat is None:
                    # No room in the requested direction at all: recentre the
                    # pointer relatively. That rotates the camera the wrong way,
                    # but this is a feedback loop and it will correct.
                    pointer = pointer_state(probe.identity['display'])
                    recentre = [int(width // 2 - pointer['x']), int(height // 2 - pointer['y'])]
                    if recentre != [0, 0]:
                        move_pointer(recentre[0], recentre[1])
                        report['inputs'][-1].update(step=step, reason='recentre_pointer')
                    row['state'] = 'recentred_pointer'
                    report['steps'].append(row)
                    previous_error = None
                    last_move = None
                    if args.settle:
                        time.sleep(args.settle)
                    continue
                recovery = move_pointer(retreat[0], retreat[1])
                report['inputs'][-1].update(step=step, reason='reposition_retreat')
                if recovery == tuple(retreat):
                    short = [move_x - (delivered or [0, 0])[0], move_y - (delivered or [0, 0])[1]]
                    move_pointer(short[0], short[1])
                    report['inputs'][-1].update(step=step, reason='resend_after_clamp')
                    next_search = track_search((short[0], short[1]), gain, floor=radius, ceiling=900)
                else:
                    # A bounded retreat has not made all the requested room.
                    # Observe its camera response before another correction.
                    row['state'] = 'bounded_recovery_reread'
                    next_search = track_search(recovery, gain, floor=radius, ceiling=900)
                previous_error = None
                last_move = None
                if args.settle:
                    time.sleep(args.settle)
                continue
            previous_error = list(error)
            previous_captured = captured_ns
            # The response belongs to every packet that was actually delivered,
            # otherwise the gain is learned from one third of the motion and
            # every later correction is that much too small.
            delivered_move = (move_x, move_y * packets_done)
            last_move = delivered_move
            next_search = track_search(delivered_move, gain,
                                       floor=radius, ceiling=900)
            # Wait for the traverse before measuring this move's response.
            if args.settle:
                time.sleep(args.settle)
        report['status'] = 'completed'
        report['converged'] = bool(report['steps'] and report['steps'][-1].get('distance_px',
                                9999) <= args.tolerance)
    except (Exception, KeyboardInterrupt) as error_object:
        report.update(status='failed', error=str(error_object), fired=report.get('fired', False))
        # A one-line error is not enough to find a TypeError in a long chain; the
        # traceback is what made the last two defects findable.
        import traceback as _traceback
        report['error_trace'] = _traceback.format_exc().splitlines()[-6:]
        if isinstance(error_object, FrameTimeError):
            report['frame_rejection'] = error_object.evidence
        if args.fire and not report['fired']:
            report['fire_blocked'] = dict(reason=str(error_object))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if scope_entered and not args.stay_in_scope:
            try:
                probe.key(args.scope_key)
                report['inputs'].append(dict(key=args.scope_key, reason='exit_scope'))
            except Exception as error_object:
                report.setdefault('cleanup_errors', []).append(f'scope exit: {error_object}')
        if held_button:
            try:
                probe.check_identity(); probe.x('mouseup', held_button, cleanup=True); held_button = None
            except Exception as error_object:
                report.setdefault('cleanup_errors', []).append(str(error_object))
        probe.release()
        if old_pointer is not None:
            try:
                probe.focus(); probe.x('mousemove', old_pointer['x'], old_pointer['y'], cleanup=True)
            except Exception as error_object:
                report.setdefault('cleanup_errors', []).append(str(error_object))
        report['unreleased_button'] = held_button is not None
        report['elapsed_s'] = time.monotonic()-probe.started
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        signal.setitimer(signal.ITIMER_REAL, *timer)
        stamp_transport(report, getattr(args, 'relative_transport', 'xtest'))
        (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return int(report['status'] != 'completed')


def build_parser():
    """The command line, exposed so callers can drive run() in-process.

    Spawning a fresh interpreter per shot cost 2.5 s of the 12.9 s wall time,
    all of it before the first frame was read. A caller that keeps this module
    imported can reuse the loaded models and the probe cache between shots.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=float, nargs=2,
                        help='commanded scene point in native capture pixels (x y)')
    parser.add_argument('--scan', action='store_true',
                        help='rotate the camera until a red enemy healthbar is found, then aim at its hull window')
    parser.add_argument('--scan-step', type=int, default=80, help='mouse units per scan step (kept small so the pointer never clips)')
    parser.add_argument('--scan-start-units', type=int, default=0,
                        help='yaw offset in mouse units applied once before the sweep, '
                             'so repeated runs cover the whole circle instead of the '
                             'same arc (a contact behind the ship was never seen)')
    parser.add_argument('--scan-steps', type=int, default=12, help='looks per horizontal sweep')
    parser.add_argument('--scan-rows', type=int, default=1, help='sweeps, tilting between them (1 = no pitch sweep)')
    parser.add_argument('--scan-confirm-steps', type=int, default=4,
                        help='sweep size after a successful minimap bearing; '
                             '0 keeps the full arc')
    parser.add_argument('--scan-min-deg', type=float, default=0.0,
                        help='required yaw coverage of one sweep; the operator procedure '
                             'asks for 360 and a shorter sweep is refused')
    parser.add_argument('--scan-gain-px-per-unit', type=float, default=1.5,
                        help='measured third-person gain used only to report/verify coverage')
    parser.add_argument('--scan-fov-deg', type=float, default=68.0)
    parser.add_argument('--scan-tilt', type=int, default=3,
                        help='signed vertical units between sweeps; bounded by reach-y')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--fire', action='store_true', help='fire once after two converged steps')
    parser.add_argument('--tolerance', type=float, default=12.0)
    parser.add_argument('--compensate-drift', action='store_true', default=True,
                        help='remove the camera shift caused by the ship itself from the response')
    parser.add_argument('--no-compensate-drift', dest='compensate_drift', action='store_false')
    parser.add_argument('--aim-nudge', type=int, default=1,
                        help='tiny mouse move that makes the game draw the aim reticle')
    parser.add_argument('--aim-fraction', type=float, default=None,
                        help='explicit hull-window fraction to aim at (0 top, 1 bottom); '
                             'omit it to use the bar-derived waterline rule')
    parser.add_argument('--scope-fire-tolerance-px', type=int, default=120,
                        help='relaxed tolerance used once the binocular view is '
                             "open: a health bar or ship position that is 'about "
                             "right' is enough to fire")
    parser.add_argument('--refine-min-headroom-s', type=float, default=1.6,
                        help='do not start another correction with less budget left')
    parser.add_argument('--oneshot-refine-rounds', type=int, default=3,
                        help='measured two-axis corrections allowed before firing')
    parser.add_argument('--no-terrain-gate', action='store_true',
                        help='do not read land on the minimap line of fire')
    parser.add_argument('--terrain-ttl-s', type=float, default=8.0,
                        help='reuse the line-of-fire reading for this long')
    parser.add_argument('--fire-tolerance', type=float, default=60.0,
                        help='target association and final reread tolerance; never bypasses convergence')
    parser.add_argument('--fire-tolerance-from-plate', dest='fire_tolerance_from_plate',
                        action='store_true', default=True,
                        help='derive the fire tolerance from the drawn plate width '
                             '(0.30 x width, clamped by --fire-tolerance-ceiling)')
    parser.add_argument('--no-fire-tolerance-from-plate', dest='fire_tolerance_from_plate',
                        action='store_false')
    parser.add_argument('--fire-tolerance-ceiling', type=float, default=60.0,
                        help='upper bound for the plate-derived fire tolerance')
    parser.add_argument('--freeze-reach', type=float, default=40.0,
                        help='horizontal units allowed once the error is within twice '
                             'the fire tolerance (keeps a good aim from being thrown away)')
    parser.add_argument('--freeze-damping', type=float, default=0.35,
                        help='damping applied together with --freeze-reach')
    parser.add_argument('--confirm-reload-wait-s', type=float, default=6.0,
                        help='bounded HUD poll for a reloading gun at the confirm frame '
                             '(a wait, not a refusal)')
    parser.add_argument('--poll-reload-s', type=float, default=0.5)
    parser.add_argument('--fire-min-steps', type=int, default=4,
                        help='earliest zero-based loop step eligible to fire; also needs two tight solutions')
    parser.add_argument('--max-steps', type=int, default=12)
    parser.add_argument('--probe', type=int, default=60, help='horizontal mouse units per axis probe')
    parser.add_argument('--probe-y', type=int, default=3,
                        help='vertical mouse units for servo and gain probes; bounded by reach-y')
    parser.add_argument('--recenter-margin', type=int, default=0,
                        help='pointer room required around a move; a clamp is corrected with a '
                             'relative retreat, since an absolute warp is ignored. Default 0: a '
                             'pre-emptive recentre is a large relative move and the game turns '
                             'the camera with it, which threw an acquired target out of frame')
    parser.add_argument('--reposition-step', type=int, default=400,
                        help='mouse units of relative retreat used to restore pointer room')
    parser.add_argument('--reach', type=int, default=90,
                        help='largest horizontal mouse units per correction step')
    parser.add_argument('--reach-y', type=int, default=3,
                        help='fine vertical units per action, including scanning and pointer '
                             'recovery (1..3)')
    parser.add_argument('--hull-gap-fraction', type=float, default=HULL_AIM_FRACTION,
                        help='fraction of the bar-to-waterline gap to aim below the bar')
    parser.add_argument('--coarse-reach-y', type=int, default=MAX_VERTICAL_STEP_UNITS,
                        help='requested vertical budget for the coarse phase; the '
                             'hard %d-unit cap still applies, so this cannot make a '
                             'single action larger than that' % MAX_VERTICAL_STEP_UNITS)
    parser.add_argument('--vertical-packets', type=int, default=1,
                        help='largest number of bounded vertical actions per step, each a separate '
                             'reach-y move with --packet-settle between (1..40); 1 disables. The '
                             'count is reduced to what the remaining error needs, so fine aim '
                             'stays one action')
    parser.add_argument('--packet-settle', type=float, default=.35,
                        help='seconds between bounded vertical actions (independent of --settle)')
    parser.add_argument('--post-fire-frames', type=int, default=3,
                        help='frames captured immediately after firing, as fire evidence')
    parser.add_argument('--lead', action='store_true',
                        help='measure the target velocity and aim ahead (costs 3 frames)')
    parser.add_argument('--lead-samples', type=int, default=3,
                        help='extra no-move frames used to fit the target velocity')
    parser.add_argument('--lead-sample-s', type=float, default=0.7,
                        help='spacing of those frames (span feeds the fit)')
    parser.add_argument('--lead-min-span', type=float, default=2.0,
                        help='seconds of history a fitted velocity needs before it is used')
    parser.add_argument('--lead-min-samples', type=int, default=3,
                        help='consistent velocity samples required before any lead is applied')
    parser.add_argument('--lead-blend', type=float, default=.4)
    parser.add_argument('--lead-budget-s', type=float, default=14.0,
                        help='aim budget used when --lead sampling is on')
    parser.add_argument('--lead-reach', type=float, default=220.0,
                        help='units the lead move may use (independent of fast-reach)')
    parser.add_argument('--lead-ceiling', type=float, default=320.0,
                        help='largest lead offset in px; velocity noise must not throw the aim')
    parser.add_argument('--lead-seconds', type=float, default=0.0,
                        help='shell flight time used to aim ahead of a moving target; 0 disables')
    parser.add_argument('--no-lead', action='store_true',
                        help='static targets: disable HUD/configured flight-time and latency lead')
    parser.add_argument('--no-reacquire', dest='reacquire', action='store_false',
                        help='give up when the patch track is lost instead of re-detecting')
    parser.add_argument('--damping', type=float, default=.7,
                        help='fraction of the computed correction to apply (0.05..1)')
    parser.add_argument('--live-source', action='store_true',
                        help='read frames through the persistent grab (about 8x faster)')
    parser.add_argument('--max-stale-reads', type=int, default=3,
                        help='re-reads allowed when a move produced no visible change')
    parser.add_argument('--no-lead-horizon', dest='lead_horizon', action='store_false',
                        help='lead by the flight time alone, without the measured latency horizon')
    parser.add_argument('--lead-horizon-cap', type=float, default=.5,
                        help='largest latency horizon added to the lead, seconds')
    parser.add_argument('--measure-gain', action='store_true',
                        help='measure the camera gain in this session before aiming')
    parser.add_argument('--measure-gain-units-y', type=int, default=40,
                        help='vertical probe units for gain measurement')
    parser.add_argument('--measure-gain-units', type=int, default=20,
                        help='horizontal mouse units for measurement; vertical uses probe-y')
    parser.add_argument('--minimap-metres-per-pixel', type=float, default=0.0,
                        help='panel scale for the map in play (0 disables range use)')
    parser.add_argument('--shell-speed', type=float, default=800.0,
                        help='nominal average shell speed for a lead, m/s')
    parser.add_argument('--fov-deg', type=float, default=68.2,
                        help='horizontal field of view of the aim camera')
    parser.add_argument('--bearing-gain', type=float, default=-1.5,
                        help='screen pixels per mouse unit for a yaw turn; measured '
                             '-1.03..-1.82 in third person, so the old -9.77 turned '
                             "a sixth of the commanded angle")
    parser.add_argument('--turn-chunk', type=float, default=400.0,
                        help='largest single yaw piece for a minimap turn; XTest is '
                             'clamped to the pointer-safe 400 units, libei may use up '
                             'to 4000')
    parser.add_argument('--relative-transport', choices=RELATIVE_TRANSPORTS,
                        default='xtest',
                        help='relative mouse transport: xtest moves the seat pointer '
                             'and needs the reposition fallbacks; ei uses libei and '
                             'skips them (default: xtest)')
    parser.add_argument('--allow-concurrent-input', action='store_true',
                        help='run even when the pointer or a control key shows someone else driving')
    parser.add_argument('--autogain', action='store_true',
                        help='measure the mouse gain on the target before the one-shot solve')
    parser.add_argument('--autogain-units', type=int, default=120,
                        help='horizontal target probe units; vertical uses probe-y')
    parser.add_argument('--autogain-verify', action='store_true', default=True,
                        help='confirm the target is still trackable after the reversible probes')
    parser.add_argument('--no-autogain-verify', dest='autogain_verify', action='store_false')
    parser.add_argument('--autogain-min-shift', type=float, default=20.0,
                        help='px the target must move along the probed axis to be a usable gain')
    parser.add_argument('--scan-idle', type=float, default=4.0,
                        help='seconds to let the camera settle on the bow before sweeping')
    parser.add_argument('--autogain-search', type=int, default=260)
    parser.add_argument('--oneshot', action='store_true',
                        help='one open-loop aim, then scope, lock and fire (no feedback)')
    parser.add_argument('--open-loop-gain', type=float, nargs=2, default=[-1.5, -1.4],
                        help='assumed screen px per mouse unit for the one-shot aim')
    parser.add_argument('--scope-after', type=float, default=0.0,
                        help='open the binocular view once the coarse aim is within this many px; 0 disables')
    parser.add_argument('--scope', action='store_true',
                        help='toggle the binocular view (default Shift_L) before aiming')
    parser.add_argument('--scope-key', default='Shift_L')
    # First-axis probe inside the scope; the zoom multiplies every unit.
    parser.add_argument('--scope-probe', type=int, default=20)
    # Seconds to let the binocular zoom finish before the first scoped read.
    parser.add_argument('--scope-settle', type=float, default=0.9)
    # Vertical units per attempt while hunting the plate inside the scope; 0
    # disables the sweep.
    parser.add_argument('--scope-sweep-units', type=int, default=40)
    # Vertical step budget inside the binocular view.
    parser.add_argument('--scope-coarse-reach-y', type=int, default=40,
                        help='coarse vertical grant inside the scope (one scoped unit '
                             'moves about 7 px, so the third-person cap of 6 is useless there)')
    parser.add_argument('--scope-reach-y', type=int, default=60,
                        help='requested vertical units in the scope; clamped to the '
                             'same safety cap as every other vertical action')
    # Horizontal units per step inside the binocular view.
    parser.add_argument('--scope-reach', type=int, default=120)
    # Consecutive scope frames without a plate before the scope is abandoned.
    parser.add_argument('--scope-empty-grace', type=int, default=1)
    parser.add_argument('--scope-retries', type=int, default=2,
                        help='how many times an empty scope may hand back to coarse aim '
                             'before the run is declared unable to see the target')
    # Coarse-pass budget: after this many third-person steps the scope opens
    # even if the horizontal error is still above --scope-after.
    parser.add_argument('--scope-after-steps', type=int, default=3)
    # Re-detect instead of tracking when a single correction is this large.
    parser.add_argument('--reacquire-px', type=float, default=300.0)
    # Opening the binocular view with the target far off-centre zooms into empty
    # water (measured: 359 px error -> "scope transition target missing"). The
    # step-budget escape is therefore capped by this horizontal error.
    parser.add_argument('--scope-escape-max-px', type=float, default=150.0)
    # Vertical error ceiling for opening the scope (see the 5x-magnification note).
    # Only a sanity bound against zooming into empty water. The vertical is
    # closed *inside* the scope (7 px/unit there vs ~0.3 in third person), so a
    # tight value here deadlocked the whole chain: measured errors of -66..-232
    # px kept the scope shut for entire runs.
    parser.add_argument('--scope-gain-cache', action='store_true',
                        help='reuse a scope gain measured by an earlier run (off: the '
                             'gain is measured for the zoom actually in use)')
    parser.add_argument('--scope-traverse-wait-s', type=float, default=3.0,
                        help='wait for the turrets to stop traversing before opening '
                             'the scope (the zoom shows the gun line, not the reticle)')
    parser.add_argument('--scope-ladder-steps', type=int, default=6,
                        help='bounded vertical moves that try to bring an off-frame '
                             'plate into the zoom before closing the scope')
    parser.add_argument('--scope-ladder-units', type=float, default=12.0,
                        help='largest vertical move in one ladder step')
    parser.add_argument('--scope-vertical-max-px', type=float, default=400.0)
    # Operator rule: open the scope once the shell-impact range is this close to
    # the enemy's own range.
    parser.add_argument('--scope-range-gap-km', type=float, default=2.0)
    # Vertical units per step while the HUD shows "- -" (aim beyond max range).
    parser.add_argument('--beyond-max-steps', type=int, default=3)
    parser.add_argument('--scope-beyond-step-units', type=int, default=6,
                        help='vertical units used by the "- -" branch inside the scope '
                             '(one scoped unit is about 7 px)')
    parser.add_argument('--scope-beyond-max-steps', type=int, default=3)
    parser.add_argument('--beyond-step-units', type=int, default=MAX_VERTICAL_STEP_UNITS,
                        help='requested vertical units while the readout is --.- ; '
                             'bounded_move clamps any request to the %d-unit safety cap'
                             % MAX_VERTICAL_STEP_UNITS)
    parser.add_argument('--lock', action='store_true',
                        help='press the target-lock key once the reticle is on the hull, before firing')
    parser.add_argument('--assume-battle', action='store_true',
                        help='operator asserts a battle for an unsupported HUD layout; recorded')
    parser.add_argument('--slow-vision', action='store_true',
                        help='use the full HUD read per step (older behaviour, ~4x slower)')
    parser.add_argument('--lock-key', default='x')
    parser.add_argument('--range-closure', action='store_true',
                        help='close elevation on the HUD ballistic range vs the target '
                             'range before firing; an unreadable range blocks the shot')
    parser.add_argument('--fire-max-km', type=float, default=None,
                        help='record a warning, never a veto, when the caption range '
                             'under the plate is beyond this many km (distance is not '
                             'one of the nine fire conditions)')
    parser.add_argument('--range-source', choices=('caption',), default='caption',
                        help='the shot chain reads range from the main view only')
    parser.add_argument('--range-target-km', type=float, default=None,
                        help='operator-supplied target range used only when the nameplate '
                             'caption is unreadable; recorded as target_source=operator')
    parser.add_argument('--range-tolerance-km', type=float, default=RANGE_TOLERANCE_KM)
    parser.add_argument('--measure-range-units', type=int, default=40,
                        help='vertical units for the in-place range/unit calibration')
    parser.add_argument('--range-km-per-unit', type=float, default=RANGE_KM_PER_UNIT)
    parser.add_argument('--range-step-units', type=int, default=min(RANGE_STEP_UNITS, MAX_VERTICAL_STEP_UNITS))
    parser.add_argument('--stay-in-scope', action='store_true',
                        help='do not toggle the binocular view back when the run ends')
    parser.add_argument('--fast-scope', action='store_true',
                        help='after the coarse solve, open the binocular view and re-solve with '
                             'the scope gain before firing')
    parser.add_argument('--scope-gain', type=float, nargs=2, default=[.91, -.96],
                        help='screen px per mouse unit in the binocular view (measured)')
    parser.add_argument('--descent-step-units', type=int, default=40,
                        help='vertical units per descent action; the measured '
                             'gain is ~0.12 px/unit, so 6 units moved nothing')
    parser.add_argument('--descent-rounds', type=int, default=8)
    parser.add_argument('--fast-max-aim-km', type=float, default=20.0,
                        help='pitch down before solving while the reticle '
                             "reads beyond this range (sky-high camera)")
    parser.add_argument('--fast-deadline-s', type=float, default=6.0,
                        help='stop correcting the fast solve after this many '
                             'seconds; the shot is still capped by the 15 s rule')
    parser.add_argument('--fast-damping', type=float, default=0.6,
                        help='fraction of the planned correction the fast solve applies')
    parser.add_argument('--settle-with-fire', action='store_true',
                        help='treat the fire tolerance as "close enough" in the '
                             'aim loop. Measured 2026-10-23: the loop sat at 66 px '
                             '(inside the 140 px fire tolerance) and kept stepping '
                             'against the 60 px fine tolerance until the budget ran '
                             'out. Default keeps the stricter fine tolerance')
    parser.add_argument('--vertical-cap', type=int, default=None, metavar='UNITS',
                        help='per-action vertical ceiling (default 6, the shared '
                             'safety cap). Raise it only for a view where a '
                             'larger pitch was measured safe; a 475 px error '
                             'cannot close six units at a time inside 15 s')
    parser.add_argument('--move-chunk', type=float, default=0.0,
                        help='split a horizontal action into pieces of at most '
                             'this many units (0 keeps one event); Wine '
                             'accelerates large single deltas so a chunked run '
                             'keeps the measured gain valid across steps')
    parser.add_argument('--recentre-key', default=None, metavar='KEY',
                        help='key that toggles a UI screen which recentres the '
                             'pointer (WoWS: m opens/closes the tactical map). '
                             'Pressed twice when a leg runs out of travel, which '
                             'restores the full range without turning the camera')
    parser.add_argument('--fast-axis-split', action='store_true',
                        help='correct one axis per round (bigger error first) '
                             'because large pitch moves also swing the yaw')
    parser.add_argument('--fast-reach-y', type=float, default=1200.0,
                        help='vertical unit budget per fast correction; the '
                             'measured third-person pitch gain on this client is '
                             '0.12 px/unit, so a 200 px error needs ~1600 units')
    parser.add_argument('--fast-reach', type=float, default=1200.0,
                        help='largest open-loop yaw the fast solve may plan; '
                             'the sender still chunks it into 400-unit pieces')
    parser.add_argument('--fast-search-deg', type=float, default=0.0,
                        help='yaw per search step in degrees, converted with the '
                             'measured gain; 0 keeps --fast-search-step units')
    parser.add_argument('--fast-search-steps', type=int, default=4,
                        help='bounded horizontal looks allowed before a fast shot gives up')
    parser.add_argument('--fast-search-step', type=int, default=800,
                        help='mouse units per fast search look (the third-person view is wide, '
                             'so four looks sweep the whole horizon)')
    parser.add_argument('--fast', action='store_true',
                        help='one-frame open-loop shot: solve, move, confirm, fire; '
                             'reports the end-to-end milliseconds against a 5000 ms budget')
    parser.add_argument('--settle', type=float, default=1.0,
                        help='seconds to wait for turret traverse before measuring a move')
    parser.add_argument('--scan-settle', type=float, default=.25,
                        help='seconds to wait between scan steps')
    parser.add_argument('--hud-retries', type=int, default=2,
                        help='fresh re-reads allowed when a frame does not '
                             'confirm the battle HUD; a flicker is not a loss')
    parser.add_argument('--reticle-retries', type=int, default=4,
                        help='fresh frame reads allowed per step while the reticle is unreadable')
    parser.add_argument('--seconds', type=float, default=30.)
    # Operator rule: <=10 s from the search start to the shot. Measured inside
    # the servo so process start-up is excluded; 0 disables the bound.
    parser.add_argument('--oneshot-refine-px', type=float, default=25.0,
                        help='after the open-loop solve, correct once and re-read '
                             'while the residual exceeds this many pixels')
    parser.add_argument('--no-oneshot-refine', action='store_true')
    parser.add_argument('--no-smoothness-guard', action='store_true',
                        help='record the smoothness metric but never veto a shot on it')
    parser.add_argument('--settle-damping', type=float, default=0.8,
                        help='damping for that single measured correction')
    parser.add_argument('--search-budget-s', type=float, default=0.0,
                        help='budget for the search phase only; the aim budget starts when '
                             'a qualified target is accepted (0 falls back to --aim-budget-s)')
    parser.add_argument('--aim-budget-s', type=float, default=AIM_BUDGET_S,
                        help='seconds from the first search frame to the shot; 0 disables')
    parser.add_argument('--max-vertical-reversals', type=int, default=3,
                        help='vertical corrections that keep flipping direction with a '
                             'large amplitude mean the elevation is shaking, not closing')
    parser.add_argument('--shake-amplitude-px', type=float, default=120.0)
    parser.add_argument('--max-view-jump-px', type=float, default=520.0,
                        help='scene displacement between steps that no commanded move '
                             'explains; a sudden camera jump is not an aiming result')
    # Seed the per-axis gain (px per mouse unit) to skip the probe steps.
    parser.add_argument('--seed-gain', type=float, nargs=2, default=None,
                        metavar=('GX', 'GY'))
    parser.add_argument('--out', type=Path, required=True)
    return parser


def reset_engagement_state():
    """Clear per-engagement caches; gains and models deliberately survive.

    A stale line-of-fire answer must never decide a new engagement, while the
    loaded minimap model and a gain measured in this same game process are
    exactly what makes the next shot faster and more accurate.
    """
    terrain_evidence.result = None
    terrain_evidence.thread = None
    terrain_evidence.contacts = None
    minimap_target_km.cache = None


def prepare_args(args):
    """State that must follow parsing, for callers that drive run() directly.

    In-process callers (the match runner) parse the same argv but never go
    through main(), so the vertical cap stayed at its 6-unit default: a 100 px
    vertical error could not be corrected, and the report showed 6-unit refine
    steps while the command line said 60.
    """
    if getattr(args, 'vertical_cap', None) is not None:
        set_vertical_cap(args.vertical_cap)
        args.reach_y = max(args.reach_y, min(args.vertical_cap, 400))
    args.last_range_closure = None
    args.range_km_per_unit_learn = None
    return args


def main(argv=None):
    reset_engagement_state()
    parser = build_parser()
    args = parser.parse_args(argv)
    prepare_args(args)
    if not 3 <= args.seconds <= 120:
        parser.error('seconds must be 3..120')
    if not 1 <= args.max_steps <= 40:
        # The scoped hull offset alone is ~1.4 bar widths, which magnifies to
        # ~500 px at 4x zoom; at the coarse ceiling of 80 units per step that is
        # six steps before the fine loop even starts. 20 left the loop one step
        # short (measured: 83 px residual), 40 covers scan + coarse + fine.
        parser.error('max-steps must be 1..40')
    if not 1 <= args.vertical_packets <= 40:
        parser.error('vertical-packets must be 1..40')
    if not 0 <= args.packet_settle <= 3:
        parser.error('packet-settle must be 0..3')
    if not 1 <= args.coarse_reach_y <= 80:
        parser.error('coarse-reach-y must be 1..80')
    if not 0 < args.tolerance <= 60:
        parser.error('tolerance must be 0..60 px')
    for value in args.open_loop_gain:
        if not math.isfinite(value) or value == 0 or abs(value) > 10:
            parser.error('open-loop-gain entries must be non-zero and within +-10 px per unit')
    if not 0 <= args.scope_after <= 400:
        parser.error('scope-after must be 0..400 px')
    if not 40 <= args.autogain_units <= 400:
        parser.error('autogain-units must be 40..400 mouse units')
    if not 1 <= args.turn_chunk <= EI_TURN_CHUNK_MAX:
        parser.error(f'turn-chunk must be 1..{EI_TURN_CHUNK_MAX:g} mouse units')
    # Seed the move layer before the probe is built: Probe reads the transport
    # from the environment at construction.
    apply_relative_transport(None, args.relative_transport)
    if not 5 <= args.autogain_min_shift <= 200:
        parser.error('autogain-min-shift must be 5..200 px')
    if not 0 <= args.scan_idle <= 15:
        parser.error('scan-idle must be 0..15 s')
    if not 80 <= args.autogain_search <= 600:
        parser.error('autogain-search must be 80..600 px')
    if not 0 <= args.aim_nudge <= 20:
        parser.error('aim-nudge must be 0..20 mouse units')
    if args.aim_fraction is not None and not 0 <= args.aim_fraction <= 1:
        parser.error('aim-fraction must be 0..1')
    if not args.tolerance <= args.fire_tolerance <= 400:
        parser.error('fire-tolerance must be >= tolerance and <= 400 px')
    if not 1 <= args.fire_min_steps <= 20:
        parser.error('fire-min-steps must be 1..20')
    if not 5 <= args.probe <= 200:
        parser.error('probe must be 5..200 mouse units')
    if not 10 <= args.reach <= 400:
        parser.error('reach must be 10..400 mouse units')
    reach_ceiling = max(3, args.vertical_cap or 3)
    if not 1 <= args.reach_y <= reach_ceiling:
        parser.error(f'reach-y must be 1..{reach_ceiling} vertical mouse units '
                     '(raise --vertical-cap to allow more per action)')
    if not 1 <= args.probe_y <= 3:
        parser.error('probe-y must be 1..3 vertical mouse units')
    if not 1 <= args.measure_gain_units <= 400:
        parser.error('measure-gain-units must be 1..400 horizontal mouse units')
    if not 0 <= args.recenter_margin <= 400:
        parser.error('recenter-margin must be 0..400 px')
    if not 40 <= args.reposition_step <= 900:
        parser.error('reposition-step must be 40..900 mouse units')
    if not 0 <= args.post_fire_frames <= 8:
        parser.error('post-fire-frames must be 0..8')
    if not .5 <= args.lead_min_span <= 15:
        parser.error('lead-min-span must be 0.5..15 s')
    if not 1 <= args.lead_min_samples <= 10:
        parser.error('lead-min-samples must be 1..10')
    if not 0 < args.lead_blend <= 1:
        parser.error('lead-blend must be 0..1')
    if not 20 <= args.lead_ceiling <= 400:
        parser.error('lead-ceiling must be 20..400 px')
    if not 0 <= args.lead_seconds <= 25:
        parser.error('lead-seconds must be 0..25')
    if not .05 <= args.damping <= 1:
        parser.error('damping must be 0.05..1')
    for name, value in (('settle', args.settle), ('scan-settle', args.scan_settle)):
        if not 0 <= value <= 3:
            parser.error(f'{name} must be 0..3 seconds')
    global FAST_VISION, ASSUME_BATTLE
    FAST_VISION = not args.slow_vision
    ASSUME_BATTLE = bool(args.assume_battle)
    if args.lock and not 1 <= len(args.lock_key) <= 4:
        parser.error('lock-key must be 1..4 characters')
    if not 1 <= args.reticle_retries <= 6:
        parser.error('reticle-retries must be 1..6')
    if (args.target is None) == (not args.scan) and not (args.oneshot or args.fast):
        parser.error('give --target or --scan (or --oneshot, which looks at the current frame)')
    if not 20 <= args.scan_step <= 400:
        parser.error('scan-step must be 20..400 mouse units')
    if not 2 <= args.scan_steps <= 48:
        parser.error('scan-steps must be 2..48')
    if not 1 <= args.scan_rows <= 6:
        parser.error('scan-rows must be 1..6')
    if not 1 <= abs(args.scan_tilt) <= args.reach_y:
        parser.error('absolute scan-tilt must be 1..reach-y vertical mouse units')
    if args.target is not None:
        for value in args.target:
            if not math.isfinite(value):
                parser.error('target must be finite')
        if not (0 <= args.target[0] < 2560 and 80 <= args.target[1] < 1520):
            parser.error('target must lie inside the 2560x1440 content area')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
