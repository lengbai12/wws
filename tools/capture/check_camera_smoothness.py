#!/usr/bin/env python3
"""Check one aim_servo report for camera smoothness.

Fails when the run shows the three symptoms measured on 2026-10-21:

* a vertical action larger than MAX_VERTICAL_STEP_UNITS (the view jumps to the
  sky and then slams onto the sea);
* near-perfect up/down alternation (shaking in place) -- either the run's own
  oscillation flag or six reversals inside the last eight vertical moves;
* a horizontal sweep that walks further than SCAN_BUDGET_UNITS, which parks the
  pointer on the screen edge where moves are silently clipped.

Usage: check_camera_smoothness.py REPORT [--expect-smooth]
Exit code 0 = smooth, 1 = a symptom was found, 2 = the report cannot be read.
"""
import argparse
import json
from pathlib import Path

MAX_STEP_Y = 6
SCAN_BUDGET = 700


def vertical_series(report):
    return [int(row['relative'][1]) for row in report.get('inputs') or []
            if row.get('relative') and len(row['relative']) == 2]


def check(report):
    problems = []
    series = vertical_series(report)
    biggest = max((abs(v) for v in series), default=0)
    if biggest > MAX_STEP_Y:
        problems.append('vertical_step_%d_units_exceeds_%d' % (biggest, MAX_STEP_Y))
    signs = [1 if v > 0 else -1 for v in series if v]
    worst_flips = 0
    for end in range(len(signs), 0, -1):
        window = signs[max(0, end - 8):end]
        flips = sum(1 for a, b in zip(window, window[1:]) if a != b)
        worst_flips = max(worst_flips, flips)
    if worst_flips >= 6:
        problems.append('vertical_alternated_%d_times_in_8' % worst_flips)
    clipped = sum(1 for row in report.get('inputs') or [] if row.get('pointer_clipped'))
    if clipped:
        problems.append('pointer_clipped_on_%d_moves' % clipped)
    if report.get('vertical_oscillation_stopped'):
        problems.append('vertical_oscillation_stopped')
    position = 0
    furthest = 0
    for row in report.get('inputs') or []:
        rel = row.get('relative') or [0, 0]
        position += int(rel[0])
        furthest = max(furthest, abs(position))
    if furthest > SCAN_BUDGET + 300:
        problems.append('pointer_walked_%d_units' % furthest)
    errors = []
    for step in report.get('steps') or []:
        err = step.get('error_px')
        if err and len(err) == 2 and err[1] is not None:
            errors.append(float(err[1]))
    grew = sum(1 for a, b in zip(errors, errors[1:]) if abs(b) > abs(a) + 1.0)
    return dict(problems=problems, vertical_max=biggest, worst_flips=worst_flips,
                clipped_moves=clipped,
                pointer_furthest=furthest, steps=len(report.get('steps') or []),
                vertical_growth_steps=grew)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('report', type=Path)
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()
    try:
        report = json.loads(args.report.read_text())
    except (OSError, json.JSONDecodeError) as error:
        print(json.dumps({'error': str(error)}))
        return 2
    result = check(report)
    result['verdict'] = 'smooth' if not result['problems'] else 'not smooth'
    result['report'] = str(args.report)
    if args.out:
        args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result))
    return 0 if not result['problems'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
