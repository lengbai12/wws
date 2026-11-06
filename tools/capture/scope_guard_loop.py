#!/usr/bin/env python3
"""Close an empty binocular scope: the rule is "no enemy in view -> Shift".

The scope is a toggle, so this only ever presses it when the scoped frame has no
trackable enemy plate and this process has not already closed the scope. It does
not aim, lock or fire, and it never touches the scope while an enemy is visible.
Every check is logged so a run can be audited afterwards.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'replay'))

import cv2  # noqa: E402

import aim_servo as A  # noqa: E402

HERE = Path(__file__).resolve().parent


def grab(path):
    subprocess.run(['/usr/bin/python3', str(HERE / 'capture_host.py'), 'grab', str(path)],
                   check=True, capture_output=True)


def shift():
    subprocess.run([sys.executable, '-B', str(HERE / 'replay_view_action.py'),
                    '--action', 'key', '--key-name', 'Shift_L',
                    '--key-hold-ms', '120', '--execute',
                    '--out', str(HERE.parents[2] / 'state' / 'scope-guard-press')],
                   check=False, capture_output=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=600)
    parser.add_argument('--interval', type=float, default=5.0)
    parser.add_argument('--wait', type=float, default=5.0,
                        help='seconds without a visible enemy before closing the scope')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    log = []
    empty_since = None
    closed = False
    deadline = time.monotonic() + args.seconds
    while time.monotonic() < deadline:
        path = args.out / f'frame-{len(log):03d}.ppm'
        try:
            grab(path)
            frame = A.read_frame(path)
        except Exception as error:                      # noqa: BLE001
            log.append(dict(t=round(time.monotonic(), 3), error=str(error)))
            time.sleep(args.interval)
            continue
        centre, battle, weapon, source = A.reticle_centre(frame)
        enemies = [c for c in A.find_enemy(frame, centre, None, scope_observed=True)
                   if c['trackable']]
        scoped = source in ('scope_ruler_center', 'green_ring') or A.hud_aim_km(frame, centre) is not None
        row = dict(t=round(time.monotonic(), 3), scoped=bool(scoped), battle=battle,
                   reticle_source=source, enemies=len(enemies),
                   plates=[c['nameplate_bbox'] for c in enemies][:3])
        now = time.monotonic()
        if scoped and not enemies:
            empty_since = empty_since or now
            row['empty_for_s'] = round(now - empty_since, 2)
            if not closed and now - empty_since >= args.wait:
                row['action'] = 'exit_scope'
                if args.execute:
                    shift()
                closed = True
        else:
            empty_since = None
            if enemies:
                closed = False
        log.append(row)
        (args.out / 'log.json').write_text(json.dumps(log, indent=1))
        time.sleep(args.interval)
    (args.out / 'log.json').write_text(json.dumps(log, indent=1))
    print(json.dumps({'checks': len(log),
                      'closed_scope': any(r.get('action') for r in log),
                      'enemy_checks': sum(1 for r in log if r.get('enemies'))}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
