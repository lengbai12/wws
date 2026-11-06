#!/usr/bin/env python3
"""Click a LOGICAL desktop point through the kernel uinput device.

Desktop screenshots are physical pixels (2560x1600) while the compositor seat
and KWin cursor use logical pixels (1707x1067 on this host, scale 1.5). Mixing
the two silently clicks the wrong place, so both are computed here and printed.

Motion is relative through `uinput_input`, and the position is confirmed with
the compositor's own cursor (`kwin_cursor`), never with the X pointer: this
client freezes that one while it owns the mouse.

Default is a dry run. This tool performs a single bounded click and reports any
button it may have left held; it never holds a key and never fires a weapon.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from click_window_point import kwin_cursor, reached, chunks  # noqa: E402

UINPUT = ROOT / 'state/cn360-tools/uinput_input'
TOLERANCE_PX = 4


def logical_point(physical, scale):
    if not all(math.isfinite(v) for v in (*physical, scale)) or scale <= 0:
        raise ValueError('physical point and scale must be finite and positive')
    x, y = physical
    if x < 0 or y < 0:
        raise ValueError('physical point must be non-negative')
    return round(x / scale), round(y / scale)


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    target = logical_point(args.point, args.scale)
    report = dict(action='click_logical_point', execute=args.execute,
                  physical_point=list(args.point), logical_scale=args.scale,
                  logical_point=list(target), inputs=[], pointer_samples=[],
                  status='dry-run',
                  semantics='relative uinput walk + one click; no menu recognition')
    (args.out / 'plan.json').write_text(json.dumps(report, indent=2) + '\n')
    if not args.execute:
        print(json.dumps(report, indent=2))
        return 0
    try:
        for _ in range(args.steps):
            pointer = kwin_cursor()
            report['pointer_samples'].append(pointer)
            if reached(pointer, target, TOLERANCE_PX):
                break
            dx, dy = target[0] - pointer['x'], target[1] - pointer['y']
            for step_x, step_y in chunks(dx, dy):
                result = subprocess.run([str(UINPUT), '--dx', str(step_x), '--dy', str(step_y)],
                                        capture_output=True, text=True, timeout=5)
                if result.returncode:
                    raise RuntimeError(f'uinput move failed: {result.stderr.strip()}')
                report['inputs'].append(dict(relative=[step_x, step_y],
                                             transport='uinput_virtual_mouse'))
                time.sleep(.05)
        else:
            raise RuntimeError(f'cursor did not reach {target} within {args.steps} steps')
        final = kwin_cursor()
        report['pointer_samples'].append(final)
        if not reached(final, target, TOLERANCE_PX):
            raise RuntimeError(f'cursor stopped at {final} short of {target}')
        result = subprocess.run([str(UINPUT), '--click', '--hold-ms', str(args.button_hold_ms)],
                                capture_output=True, text=True, timeout=5)
        if result.returncode:
            raise RuntimeError(f'uinput click failed: {result.stderr.strip()}')
        report['inputs'].append(dict(button=1, pulse=True, hold_ms=args.button_hold_ms,
                                     transport='uinput_virtual_mouse',
                                     logical_point=list(target)))
        report['status'] = 'completed'
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status=report['status'], error=report.get('error'),
                          logical_point=report['logical_point'],
                          inputs=len(report['inputs']))))
    return 0 if report['status'] == 'completed' else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--point', type=float, nargs=2, required=True,
                   help='point in PHYSICAL screenshot pixels (2560x1600)')
    p.add_argument('--scale', type=float, default=1.5,
                   help='physical pixels per logical pixel (this host: 1.5)')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--steps', type=int, default=14)
    p.add_argument('--button-hold-ms', type=int, default=120)
    a = p.parse_args(argv)
    if not 1 <= a.steps <= 30:
        p.error('steps must be 1..30')
    if not 0 <= a.button_hold_ms <= 250:
        p.error('button-hold-ms must be 0..250')
    return run(a)


if __name__ == '__main__':
    raise SystemExit(main())
