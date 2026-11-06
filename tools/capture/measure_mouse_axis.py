#!/usr/bin/env python3
"""Measure one mouse axis against a no-input control, with the same interval.

The camera also moves while the ship moves, so a bare before/after pair cannot
separate input response from drift. This runs control -> input -> revert with an
identical gap and tracks several scene landmarks with the servo's template
tracker. It reports the per-landmark displacement of each pair, so a response is
only claimed when the input pair differs from the control pair.

Default is a dry-run. Execute requires one verified game process and window,
re-verifies identity and focus before every action, and releases every key it
holds (including a scope toggle). It never fires.
"""
import argparse
import json
import math
from pathlib import Path
import signal
import sys
import time

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'tools/capture'))
sys.path.insert(0, str(ROOT / 'tools/replay'))

from probe_aim_input import Probe, ProbeError, game_identity
import aim_servo as A

LANDMARKS = [(700, 560), (1200, 520), (1800, 480), (900, 700), (1500, 640), (400, 820)]


def pair_shift(first, second, landmarks=LANDMARKS, search=180):
    """Track every landmark from one frame in the next and report its shift."""
    one, two = cv2.imread(str(first)), cv2.imread(str(second))
    if one is None or two is None:
        raise ValueError('unreadable frame pair')
    rows = []
    for x, y in landmarks:
        bbox = [x-40, y-40, 80, 80]
        template = A.crop_template(one, bbox)
        if template is None:
            rows.append(dict(point=[x, y], state='flat'))
            continue
        found = A.track_patch(template, bbox, two, search=search)
        if found is None:
            rows.append(dict(point=[x, y], state='lost'))
            continue
        rows.append(dict(point=[x, y], state='tracked', dx=found['bbox'][0]-bbox[0],
                         dy=found['bbox'][1]-bbox[1], score=round(found['score'], 3)))
    return rows


def summarize(control, driven, reverted, axis):
    """Compare the driven pair against the control pair on one axis."""
    index = 0 if axis == 'x' else 1
    def values(rows):
        return [row['dx' if axis == 'x' else 'dy'] for row in rows
                if row['state'] == 'tracked' and row.get('score', 0) >= .6]
    control_values, driven_values, revert_values = values(control), values(driven), values(reverted)
    if not driven_values:
        return dict(axis=axis, state='no_trackable_landmark', control_values=control_values)
    control_span = max(abs(v) for v in control_values) if control_values else None
    driven_median = float(np.median(driven_values))
    revert_median = float(np.median(revert_values)) if revert_values else None
    separates = (control_span is None or abs(driven_median) > control_span + 2)
    return dict(axis=axis, state='response' if separates else 'no_response_beyond_drift',
                landmarks_tracked=len(driven_values), control_values=control_values,
                driven_values=driven_values, revert_values=revert_values,
                control_span=max(map(abs, control_values), default=None),
                driven_median=round(driven_median, 2),
                revert_median=None if revert_median is None else round(revert_median, 2),
                reversible=(None if revert_median is None else
                            abs(revert_median + driven_median) <= max(3.0, abs(driven_median) * .5)))


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(action='measure_mouse_axis', axis=args.axis, units=args.units,
                  scope=args.scope, execute=args.execute, out=str(args.out.resolve()),
                  snapshots=[], inputs=[], focus_checks=[], status='dry-run',
                  gap_s=args.gap, landmarks=LANDMARKS,
                  semantics='input response versus a same-interval control; no aim or fire claim')
    (args.out/'plan.json').write_text(json.dumps(report, indent=2)+'\n')
    if not args.execute:
        print(json.dumps(report, indent=2))
        return 0
    probe = Probe(args.seconds, report)
    handlers, timer, scope_entered = {}, (0., 0.), False
    try:
        def stop(signum, frame):
            raise ProbeError(f'interrupted by signal {signum}')
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
        if args.scope:
            probe.key(args.scope_key)
            report['inputs'].append(dict(key=args.scope_key, reason='enter_scope'))
            scope_entered = True
            time.sleep(max(args.gap, .5))
        start = probe.snapshot('a_control_start')
        time.sleep(args.gap)
        end = probe.snapshot('b_control_end')
        delta = [args.units, 0] if args.axis == 'x' else [0, args.units]
        probe.move(delta[0], delta[1])
        time.sleep(args.gap)
        driven = probe.snapshot('c_driven')
        probe.move(-delta[0], -delta[1])
        time.sleep(args.gap)
        reverted = probe.snapshot('d_reverted')
        report['control'] = pair_shift(start, end, LANDMARKS, args.search)
        report['driven'] = pair_shift(end, driven, LANDMARKS, args.search)
        report['reverted'] = pair_shift(driven, reverted, LANDMARKS, args.search)
        report['summary'] = summarize(report['control'], report['driven'], report['reverted'], args.axis)
        report['status'] = 'completed'
    except (Exception, KeyboardInterrupt) as error:
        report.update(status='failed', error=str(error))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if scope_entered and not args.stay_in_scope:
            try:
                probe.key(args.scope_key)
                report['inputs'].append(dict(key=args.scope_key, reason='exit_scope'))
            except Exception as error:
                report.setdefault('cleanup_errors', []).append(f'scope exit: {error}')
        probe.release()
        report['unreleased_own_keys'] = report.get('unreleased_own_keys', [])
        report['elapsed_s'] = time.monotonic()-probe.started
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        signal.setitimer(signal.ITIMER_REAL, *timer)
        (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return int(report['status'] != 'completed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--axis', choices=('x', 'y'), default='y')
    parser.add_argument('--units', type=int, default=200)
    parser.add_argument('--scope', action='store_true', help='toggle the binocular view first')
    parser.add_argument('--scope-key', default='Shift_L')
    parser.add_argument('--stay-in-scope', action='store_true')
    parser.add_argument('--gap', type=float, default=.35, help='identical wait inside every pair')
    parser.add_argument('--search', type=int, default=180,
                        help='template search radius in px; the binocular view shifts further')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--seconds', type=float, default=25.)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not 20 <= abs(args.units) <= 400:
        parser.error('units must be 20..400')
    if not .1 <= args.gap <= 2:
        parser.error('gap must be 0.1..2 s')
    if not 60 <= args.search <= 600:
        parser.error('search must be 60..600 px')
    if not 5 <= args.seconds <= 60:
        parser.error('seconds must be 5..60')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
