#!/usr/bin/env python3
"""Keep shooting at a moving target until the battle ends.

The pieces already exist (acquisition via the minimap bearing and a scan, a lead
computed from the target's range rate under the game's lock, the fire gate that
judges the predicted point, and HP diff accounting). What was missing was a loop
that runs them back to back and records every attempt, so a moving-target hit
rate can be measured instead of described.

Every attempt writes one JSONL row: what was seen, which lead geometry was
chosen, whether the shot fired, and the damage read from the nameplate.

Stop early by creating ``state/stop-dynamic.txt`` (or Ctrl-C).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / 'tools/capture', ROOT / 'tools/replay'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import fire_hud  # noqa: E402
import main_view_perception as M  # noqa: E402
import match_runner as MR  # noqa: E402
import shot_accounting as SA  # noqa: E402

STOP_FILE = ROOT / 'state/stop-dynamic.txt'


def snapshot(path):
    import subprocess
    subprocess.run(['/usr/bin/python3', str(ROOT / 'tools/capture/capture_host.py'),
                    'snapshot', str(path)], stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, timeout=30)
    return np.asarray(Image.open(path).convert('RGB'))


def km_guard(frame, plates):
    return plates[0]['nameplate_bbox'] if plates else None


def battle_state(frame):
    try:
        view = fire_hud.FireHudReader().detect(frame)
    except (ValueError, OSError, KeyError):
        return None, None
    return (view.get('battle') or {}).get('active'), view


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seconds', type=float, default=1800.0)
    ap.add_argument('--max-attempts', type=int, default=40)
    ap.add_argument('--lead', dest='lead', action='store_true', default=True)
    ap.add_argument('--no-lead', dest='lead', action='store_false')
    ap.add_argument('--interval', type=float, default=2.0)
    ap.add_argument('--scan-steps', type=int, default=8)
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    ledger_path = out / 'attempts.jsonl'
    started = time.monotonic()
    attempts = 0
    fired = hits = 0
    damage_total = 0

    shot_argv = ['--out', str(out), '--execute', '--max-shots', '1',
                 '--max-steps', '10', '--seconds', '60',
                 '--scan-steps', str(args.scan_steps)]
    if args.lead:
        shot_argv.append('--lead')
    shot_args = MR.build_parser().parse_args(shot_argv)

    with ledger_path.open('a') as ledger:
        while attempts < args.max_attempts and time.monotonic() - started < args.seconds:
            if STOP_FILE.exists():
                print('stop file present; leaving the loop', flush=True)
                break
            attempts += 1
            frame = snapshot(out / 'frame.png')
            active, view = battle_state(frame)
            if active is not True:
                print(f'attempt {attempts}: not in battle, stopping', flush=True)
                break
            plates = M.detect_nameplates(frame)
            bbox = plates[0]['nameplate_bbox'] if plates else None
            km = MR.plate_distance_km(frame, bbox) if bbox else None
            row = dict(attempt=attempts, t=round(time.monotonic() - started, 1),
                       plates=len(plates), target_km=km, fired=False, hit=None, damage=0)
            shot_dir = out / f'attempt-{attempts:02d}'
            report = MR.run_shot_in_process(shot_dir, shot_args, plate_visible=bbox is not None)
            steps = [s for s in (report.get('steps') or []) if isinstance(s, dict)]
            oneshot = [s for s in steps if s.get('mode') == 'oneshot']
            last = oneshot[-1] if oneshot else (steps[-1] if steps else {})
            row.update(fired=bool(report.get('fired')),
                       mode=report.get('mode'),
                       error=report.get('error'),
                       fire_reason=report.get('fire_reason'),
                       aim_ms=report.get('aim_ms'),
                       lead_plan=last.get('lead_plan'),
                       range_lead=last.get('range_lead'),
                       residual=last.get('residual_px'),
                       fire_residual=last.get('fire_residual_px'))
            if row['fired']:
                fired += 1
                flight = float((row.get('range_lead') or {}).get('flight_s') or 12.0)
                time.sleep(flight + 2.0)
                after = snapshot(out / 'after.png')
                accounting = SA.classify_shot(frame, after, bbox) if bbox else {}
                row['hit'] = accounting.get('hit')
                row['damage'] = accounting.get('damage') or 0
                if row['hit']:
                    hits += 1
                damage_total += row['damage'] or 0
            ledger.write(json.dumps(row, default=str) + '\n')
            ledger.flush()
            print(f"attempt {attempts}: km={km} plates={len(plates)} fired={row['fired']} "
                  f"hit={row['hit']} dmg={row['damage']} lead={bool((row.get('lead_plan') or {}).get('mode'))} "
                  f"err={row['error']}", flush=True)
            time.sleep(max(0.5, args.interval))
    summary = dict(attempts=attempts, fired=fired, hits=hits, damage=damage_total,
                   hit_rate=(round(hits / fired, 3) if fired else None),
                   seconds=round(time.monotonic() - started, 1), ledger=str(ledger_path))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
