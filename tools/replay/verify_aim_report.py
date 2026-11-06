#!/usr/bin/env python3
"""Independently check one aim run's report against the frames it saved.

The report is a self-report; this recomputes the parts that can be recomputed
from the run's own artifacts, so a run can be accepted without trusting the
tool: the measured gain from the saved gain frames, the lead accounting, and
the consistency of the fire decision with the recorded steps.

Usage:
  verify_aim_report.py <run-dir> [--out <json>]
Exit code is non-zero when any check fails.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / 'tools' / 'capture'))

import aim_servo as A  # noqa: E402


def read_frame(path):
    import cv2
    image = cv2.imread(str(path))
    if image is None:
        return None
    return np.ascontiguousarray(image[:, :, ::-1])


def check_gain(run, report):
    """Recompute the gain from the frames the measurement itself saved."""
    measured = report.get('measured_gain')
    if not measured:
        return dict(name='gain_recomputed', status='skipped', reason='no measurement recorded')
    base = read_frame(run / 'gain-base.ppm')
    if base is None:
        return dict(name='gain_recomputed', status='skipped', reason='gain-base.ppm absent')
    rows = []
    worst = 0.0
    axis_units = measured.get('axis_units', [measured.get('units', 20)] * 2)
    for axis in (0, 1):
        forward = read_frame(run / f'gain-{axis}-plus.ppm')
        back = read_frame(run / f'gain-{axis}-back.ppm')
        if forward is None or back is None:
            return dict(name='gain_recomputed', status='skipped', reason=f'frames for axis {axis} absent')
        shift = A.estimate_camera_shift(base, forward)
        back_shift = A.estimate_camera_shift(base, back)
        expected = measured['gain'][axis]
        if expected is None:
            # The run did not record a gain for this axis, so there is nothing
            # to contradict; only a recorded value can be wrong.
            rows.append(dict(axis=axis, recomputed=None, recorded=None, delta=None,
                             note='axis not recorded'))
            continue
        if shift is None:
            rows.append(dict(axis=axis, recomputed=None, recorded=expected, delta=None,
                             note='scene shift not measurable'))
            worst = max(worst, 1.0)
            continue
        recomputed = float(shift[axis]) / float(axis_units[axis])
        delta = abs(recomputed - float(expected))
        worst = max(worst, delta)
        rows.append(dict(axis=axis,
                         recomputed=round(recomputed, 4), recorded=round(float(expected), 4),
                         delta=round(delta, 4),
                         return_shift=None if back_shift is None else [round(float(v), 2) for v in back_shift[:2]]))
    return dict(name='gain_recomputed', status='pass' if worst <= 0.25 else 'fail',
                max_delta=round(worst, 4), tolerance=0.25, rows=rows)


def check_lead(report):
    """The lead must be the client flight time plus the measured horizon."""
    steps = [s for s in report.get('steps') or [] if s.get('lead_total_s') is not None]
    if not steps:
        return dict(name='lead_accounting', status='skipped', reason='no step recorded a lead')
    worst = 0.0
    for step in steps:
        flight = (step.get('flight_time') or {}).get('value')
        horizon = step.get('lead_horizon_s') or 0.0
        expected = (flight if flight is not None else 0.0) + horizon
        recorded = step['lead_total_s']
        if expected <= 0:
            continue
        worst = max(worst, abs(recorded - expected))
    return dict(name='lead_accounting', status='pass' if worst <= 0.05 else 'fail',
                max_delta=round(worst, 3), tolerance=0.05, steps=len(steps))


def check_fire(report):
    """Firing needs two converged steps, and a requested scope needs evidence."""
    steps = report.get('steps') or []
    fired = bool(report.get('fired'))
    reasons = []
    converged_run = 0
    best = 0
    for step in steps:
        if step.get('state') == 'converged' or step.get('converged'):
            converged_run += 1
            best = max(best, converged_run)
        else:
            converged_run = 0
    needs = 2
    minimum_step = int(report.get('fire_min_steps') or 0)
    last_step = steps[-1].get('step', -1) if steps else -1
    scope_wanted = bool(report.get('scope')) or bool(report.get('scope_after_px'))
    confirmation = report.get('fire_confirmation') or {}
    scope_seen = (confirmation.get('scope_evidence') or {}).get('observed') is True
    ok = True
    if fired and converged_run < needs:
        ok, reasons = False, [f'fired with only {converged_run} consecutive converged steps (needs {needs})']
    if fired and (not isinstance(last_step, int) or last_step < minimum_step):
        ok, reasons = False, reasons + ['fired before the minimum observation step']
    if fired and scope_wanted and not scope_seen:
        ok, reasons = False, reasons + ['fired after a scope request without HUD scope evidence']
    if fired and (confirmation.get('confirmed') is not True
                  or confirmation.get('battle_active') is not True
                  or confirmation.get('reticle') is None
                  or len(confirmation.get('candidates') or []) != 1):
        ok, reasons = False, reasons + ['fired without current unique target confirmation']
    if fired and report.get('live_source'):
        age = confirmation.get('frame_age_s')
        if not isinstance(age, (int, float)) or not np.isfinite(age) or not 0 <= age <= .75:
            ok, reasons = False, reasons + ['fire confirmation missing or expired frame age']
    blocked = report.get('fire_blocked')
    if blocked and fired:
        ok, reasons = False, reasons + ['fired although the gate recorded a block']
    return dict(name='fire_decision', status='pass' if ok else 'fail',
                fired=fired, best_converged_run=best, final_converged_run=converged_run,
                needed=needs, minimum_step=minimum_step,
                scope_requested=scope_wanted, scope_evidenced=scope_seen,
                reasons=reasons, blocked=blocked)


def check_freshness(report):
    """Every step must have used a strictly newer frame than the step before it."""
    steps = [s for s in report.get('steps') or [] if s.get('captured_s') is not None]
    if len(steps) < 2:
        return dict(name='frame_freshness', status='skipped', reason='fewer than two timed steps')
    repeats = []
    backwards = []
    previous = None
    for step in steps:
        now = step['captured_s']
        if previous is not None:
            if now == previous:
                repeats.append(step.get('step'))
            elif now < previous:
                backwards.append(dict(step=step.get('step'), previous=previous, captured=now))
        previous = now
    ok = not repeats and not backwards
    invalid_ages = []
    if report.get('live_source'):
        for step in steps:
            age = step.get('frame_age_s')
            if not isinstance(age, (int, float)) or not np.isfinite(age) or not 0 <= age <= .75:
                invalid_ages.append(step.get('step'))
        ok = ok and not invalid_ages
    return dict(name='frame_freshness', status='pass' if ok else 'fail',
                timed_steps=len(steps), same_frame_as_previous=repeats[:5],
                going_backwards=backwards[:5], invalid_age_steps=invalid_ages[:5])


def check_errors(report):
    """A recorded step's move must push towards the reticle, not away from it."""
    gain = report.get('gain') or []
    steps = [s for s in report.get('steps') or []
             if s.get('error_px') and s.get('gain') and s.get('moved')]
    if not steps:
        return dict(name='move_direction', status='skipped', reason='no step recorded a move and gain')
    wrong = []
    for step in steps:
        error = step['error_px']
        step_gain = step['gain']
        move = step['moved']
        for axis in (0, 1):
            if not step_gain[axis] or not move[axis]:
                continue
            if error[axis] * move[axis] > 0:
                wrong.append(dict(step=step.get('step'), axis=axis, error=error[axis], move=move[axis]))
    return dict(name='move_direction', status='pass' if not wrong else 'fail',
                checked=len(steps), wrong=wrong[:5])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run', help='directory holding report.json and the saved frames')
    ap.add_argument('--out')
    args = ap.parse_args(argv)
    run = Path(args.run)
    report_path = run / 'report.json'
    if not report_path.exists():
        print(json.dumps({'status': 'failed', 'reason': f'no report at {report_path}'}))
        return 1
    report = json.loads(report_path.read_text())
    checks = [check_gain(run, report), check_lead(report), check_fire(report),
              check_freshness(report), check_errors(report)]
    failed = [c['name'] for c in checks if c['status'] == 'fail']
    result = dict(run=str(run), verdict='fail' if failed else 'pass',
                  failed=failed, checks=checks,
                  report_sha=None)
    import hashlib
    result['report_sha'] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    text = json.dumps(result, indent=1)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
