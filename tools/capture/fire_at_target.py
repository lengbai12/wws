#!/home/lengbai/Downloads/wws/.venv/bin/python
"""Fire at an enemy nameplate that the detectors place under the reticle.

This consumer is deliberately narrow: it never searches, never aims, and never
guesses.  A shot is authorised only when

  * the battle HUD is confirmed by FireHudReader (battle.active),
  * the green reticle is readable, and
  * a trackable red enemy nameplate exists whose aim point is within
    --tolerance pixels of the reticle (default 60).

Everything is decided from the frame, not from a script's own belief: the
report records the nameplate bbox, its red-pixel count before and after the
shot, and the saved frames.  A shot that misses shows up as an unchanged bar.

Default is dry-run.  --execute additionally needs a verified game process and
window, re-checks identity before every action, and releases the button it
holds.  --scope toggles the binocular view once and requires the HUD to show
the scope ruler before firing; without that evidence it revokes instead.

Usage:
  fire_at_target.py --out DIR [--execute] [--scope] [--tolerance 60]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'tools/replay'))

import numpy as np

import aim_servo as A
from fire_hud import FireHudReader
from probe_aim_input import Probe, ProbeError, game_identity


def grab(path: Path) -> np.ndarray:
    result = subprocess.run(['/usr/bin/python3', str(HERE / 'capture_host.py'), 'grab', str(path)],
                            capture_output=True, text=True)
    if result.returncode:
        raise ProbeError(f'grab failed: {result.stderr.strip()}')
    return A.read_frame(path)


def bar_red(frame: np.ndarray, bbox) -> int:
    x, y, w, h = [int(v) for v in bbox]
    sub = frame[max(0, y):y + h, max(0, x):x + w]
    if not sub.size:
        return 0
    red = ((sub[:, :, 0] > 150) &
           (sub[:, :, 0].astype(int) - sub[:, :, 1] > 40) &
           (sub[:, :, 0].astype(int) - sub[:, :, 2] > 40))
    return int(red.sum())


def choose(frame, centre, tolerance):
    hud = FireHudReader().detect(frame)
    if not (hud.get('battle') or {}).get('active'):
        return None, hud, 'battle_hud_unconfirmed'
    if centre is None:
        return None, hud, 'reticle_unreadable'
    reload_state = hud.get('reload') or {}
    if reload_state.get('ready') is False:
        return None, hud, 'gun_not_ready'
    cands = [c for c in A.find_enemy(frame, centre, .75) if c.get('trackable')]
    best = None
    for c in cands:
        aim = c.get('aim_px') or c.get('aim') or c.get('body_px')
        if not aim:
            continue
        dist = ((aim[0] - centre[0]) ** 2 + (aim[1] - centre[1]) ** 2) ** 0.5
        if dist <= tolerance and (best is None or dist < best[0]):
            best = (dist, c, aim)
    if best is None:
        return None, hud, f'no_target_within_{tolerance:g}px'
    return best, hud, 'ok'


def setup_probe(report):
    probe = Probe(30, report)
    probe.identity = game_identity()
    windows = []
    for window in probe.x('search', '--onlyvisible', '--name', '.').splitlines():
        try:
            probe.verify_window(window)
            windows.append(window)
        except ProbeError:
            continue
    if len(windows) != 1:
        raise ProbeError('expected exactly one verified game window')
    probe.window = windows[0]
    probe.focus()
    return probe


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--scope', action='store_true',
                    help='toggle the binocular view before the shot (requires HUD evidence)')
    ap.add_argument('--tolerance', type=float, default=60.0)
    ap.add_argument('--post-fire-frames', type=int, default=4)
    ap.add_argument('--settle', type=float, default=1.2)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)

    report = {
        'kind': 'fire_at_detected_target',
        'execute': args.execute,
        'scope': args.scope,
        'tolerance_px': args.tolerance,
        'status': 'dry-run',
        'reason': None,
        'fired': False,
        'frames': [],
        'inputs': [],
        'identity': None,
        'nameplate_before': None,
        'nameplate_after': None,
        'red_pixels_before': None,
        'red_pixels_after': None,
        'scope_evidence': None,
        'semantics': 'fires only when a tracked enemy is under the reticle; '
                     'a miss is an unchanged health bar, never a success',
    }
    started = time.monotonic()
    probe = None
    held = False
    try:
        frame = grab(args.out / 'before.ppm')
        centre, _battle, _weapon, _src = A.reticle_centre(frame)
        best, hud, reason = choose(frame, centre, args.tolerance)
        report['reason'] = reason
        report['reticle'] = centre
        report['battle'] = (hud.get('battle') or {}).get('status')
        if best is None:
            report['status'] = 'refused'
            return 1
        dist, cand, aim = best
        report['distance_px'] = round(dist, 2)
        report['aim_px'] = [round(float(aim[0]), 2), round(float(aim[1]), 2)]
        report['nameplate_before'] = cand.get('nameplate_bbox')
        report['red_pixels_before'] = bar_red(frame, cand['nameplate_bbox'])
        if not args.execute:
            report['status'] = 'dry-run'
            return 0

        probe = setup_probe(report)
        report['identity'] = probe.identity

        # The capture chain lags the scene by about a second.  A single frame is
        # not enough to authorise a shot: require a second frame, taken right
        # before acting, that still shows a battle-ready gun and the same target
        # under the reticle.
        time.sleep(0.5)
        frame2 = grab(args.out / 'confirm.ppm')
        centre2, _b2, _w2, _s2 = A.reticle_centre(frame2)
        best2, hud2, reason2 = choose(frame2, centre2, args.tolerance)
        report['confirm_reason'] = reason2
        if best2 is None:
            report['status'] = 'refused'
            report['reason'] = f'confirm_frame: {reason2}'
            return 1
        if centre is not None and centre2 is not None:
            drift = ((centre2[0] - centre[0]) ** 2 + (centre2[1] - centre[1]) ** 2) ** 0.5
            report['reticle_drift_px'] = round(drift, 2)
        aim2 = best2[2]
        move = ((aim2[0] - centre2[0]) ** 2 + (aim2[1] - centre2[1]) ** 2) ** 0.5
        if move > args.tolerance:
            report['status'] = 'refused'
            report['reason'] = 'target_moved_out_of_tolerance'
            return 1
        frame = frame2
        cand = best2[1]
        report['nameplate_before'] = cand.get('nameplate_bbox')
        report['red_pixels_before'] = bar_red(frame, cand['nameplate_bbox'])

        if args.scope:
            probe.key('Shift_L')
            report['inputs'].append({'key': 'Shift_L', 'reason': 'open_scope'})
            time.sleep(args.settle)
            probe.focus()
            frame = grab(args.out / 'scoped.ppm')
            centre, _battle, _weapon, _src = A.reticle_centre(frame)
            scope = A.hud_scope_evidence(frame) if hasattr(A, 'hud_scope_evidence') else None
            report['scope_evidence'] = scope
            best, hud, reason = choose(frame, centre, args.tolerance)
            report['reason'] = reason
            if not (scope or {}).get('observed'):
                report['status'] = 'refused'
                report['reason'] = 'scope_not_evidenced'
                return 1
            if best is None:
                report['status'] = 'refused'
                return 1
            dist, cand, aim = best
            report['distance_px'] = round(dist, 2)
            report['aim_px'] = [round(float(aim[0]), 2), round(float(aim[1]), 2)]

        probe.focus()
        held = True
        probe.x('mousedown', 1)
        time.sleep(0.12)
        probe.focus()
        probe.x('mouseup', 1)
        held = False
        report['fired'] = True
        report['inputs'].append({'button': 1, 'hold_ms': 120, 'reason': 'fire'})
        for index in range(max(1, args.post_fire_frames)):
            time.sleep(0.45)
            frame = grab(args.out / f'after-{index}.ppm')
            report['frames'].append(str(args.out / f'after-{index}.ppm'))
        centre, _b, _w, _s = A.reticle_centre(frame)
        best2, _hud, _r = choose(frame, centre, 10_000)
        if best2 is not None:
            report['nameplate_after'] = best2[1].get('nameplate_bbox')
            report['red_pixels_after'] = bar_red(frame, best2[1]['nameplate_bbox'])
        elif report['nameplate_before']:
            report['red_pixels_after'] = bar_red(frame, report['nameplate_before'])
        if (report['red_pixels_before'] is not None and report['red_pixels_after'] is not None
                and report['red_pixels_after'] < report['red_pixels_before']):
            report['hit_evidence'] = 'health_bar_fill_decreased'
        elif (report['red_pixels_before'] == report['red_pixels_after']
              and report['red_pixels_before'] is not None):
            report['hit_evidence'] = 'health_bar_unchanged'
        else:
            report['hit_evidence'] = 'inconclusive'
        report['status'] = 'completed'
        report['elapsed_s'] = time.monotonic() - started
        return 0
    except Exception as error:  # noqa: BLE001 - the report is the evidence
        report['status'] = 'failed'
        report['error'] = f'{type(error).__name__}: {error}'
        return 1
    finally:
        if probe is not None:
            if held:
                try:
                    probe.x('mouseup', 1, cleanup=True)
                except Exception:
                    pass
            try:
                probe.release()
            except Exception:
                pass
            (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps({k: report[k] for k in
                              ('status', 'reason', 'fired', 'distance_px', 'hit_evidence')}))
        else:
            (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps({k: report[k] for k in ('status', 'reason', 'fired')}))


if __name__ == '__main__':
    raise SystemExit(main())
