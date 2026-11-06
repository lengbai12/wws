#!/usr/bin/env python3
"""Play a whole match: acquire, fire, verify, wait for the reload, repeat.

This is the only entry that runs a match end to end. Each shot is a delegated
``aim_pipeline`` run (which in turn drives ``aim_servo``), and every shot is
*verified* from the enemy nameplate's own HP before and after the salvo -- the
script's ``fired`` flag is never treated as a hit.

Per shot the ledger keeps: time to fire, mode, residual, smoothness of the error
trajectory, the line-of-fire terrain test, and the HP delta. The run stops when
the battle HUD disappears, after ``--max-shots`` shots, or when the clock runs
out.

Dry run by default: it reads frames and prints the plan without shooting.
"""
import argparse
import json
import re
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/capture'))
sys.path.insert(0, str(ROOT / 'tools/replay'))

import aim_pipeline as P          # noqa: E402
import aim_smoothness as SM       # noqa: E402
import fire_hud                   # noqa: E402
import shot_accounting as SA      # noqa: E402


def _snapshot(path):
    subprocess.run(['/usr/bin/python3', str(ROOT / 'tools/capture/capture_host.py'),
                    'snapshot', str(path)], capture_output=True, timeout=90)
    from PIL import Image
    return np.asarray(Image.open(path).convert('RGB'))


def _hud(frame):
    hud = fire_hud.FireHudReader().detect(frame)
    return dict(battle=(hud.get('battle') or {}).get('active'),
                reload=(hud.get('reload') or {}).get('ready'),
                flight=(hud.get('flight_time') or {}).get('seconds'))


def _target_bbox(frame):
    from main_view_perception import detect_nameplates
    plates = detect_nameplates(frame)
    if not plates:
        return None
    return plates[0]['nameplate_bbox']


SEED_GAIN_RANGE = (0.2, 4.0)


def next_seed_gain(servo_report, previous=None):
    """Carry this shot's measured gain into the next one.

    The open-loop solve is only as good as its coefficient; measuring it once
    and reusing it inside the same game process removes the under/overshoot that
    cost a correction round (about 1.5 s) on every shot.
    """
    if not isinstance(servo_report, dict):
        return previous
    steps = servo_report.get('steps') or []
    row = steps[-1] if steps and isinstance(steps[-1], dict) else {}
    measured = row.get('gain_effective') or row.get('gain_used')
    if not measured or len(measured) != 2:
        return previous
    seed = list(previous or [None, None])
    low, high = SEED_GAIN_RANGE
    for axis, value in enumerate(measured):
        if value is None:
            continue
        magnitude = abs(float(value))
        if not (low <= magnitude <= high):
            continue
        seed[axis] = round(float(value), 3)
    return None if all(v is None for v in seed) else seed


_SERVO = None


def servo():
    """Import the controller once; a fresh interpreter per shot cost 2.5 s."""
    global _SERVO
    if _SERVO is None:
        import aim_servo as module
        _SERVO = module
    return _SERVO


def servo_argv(out, args, plate_visible, seed_gain=None, budget=None, run_dir=None):
    """The same command line the pipeline uses, as a list."""
    argv = ['--out', str(run_dir or (out / 'servo')), '--seconds', str(int(args.servo_seconds)),
            '--execute', '--aim-nudge', '2', '--move-chunk', '40',
            # One post-fire frame is enough evidence for the ledger; three cost
            # ~1.8 s of capture latency per shot (measured: servo elapsed minus
            # aim_ms was 4.1 s on a shot that aimed in 1.4 s).
            '--post-fire-frames', '1',
            # Four confirmation frames cost ~1.2 s each and the residual is
            # already tight after the refine rounds; two keeps the
            # two-frame-convergence rule at half the price.
            '--fire-min-steps', '2',
            '--vertical-cap', '60', '--settle-with-fire', '--tolerance', '60',
            '--fire-tolerance', str(args.fire_tolerance), '--settle', '0.25',
            '--max-steps', str(args.max_steps)]
    if plate_visible:
        if budget is None:
            # Lead sampling costs seconds on top of the aim, so a lead shot
            # declares its own budget instead of quietly breaking the 10 s rule.
            budget = (args.lead_budget_s if getattr(args, 'lead', False)
                      else args.oneshot_budget_s)
        argv += ['--oneshot', '--aim-budget-s', str(budget)]
    else:
        # A lead shot also has to acquire first, so its budget covers the scan.
        scan_budget = (args.lead_budget_s if getattr(args, 'lead', False)
                       else args.scan_budget_s)
        argv += ['--scan', '--scan-steps', str(args.scan_steps), '--scan-step', '400',
                 '--scan-confirm-steps', str(args.scan_steps), '--scan-settle', '0.12',
                 '--scan-rows', '1', '--scan-tilt', '3',
                 '--aim-budget-s', str(budget if budget is not None else scan_budget)]
    if getattr(args, 'fire', None) if hasattr(args, 'fire') else getattr(args, 'execute', False):
        argv += ['--fire']
    if getattr(args, 'lead', False):
        argv += ['--lead']
    # No minimap in the shot path. Aim and range come from the main view (the
    # nameplate and its caption); the minimap models are not consulted here.
    # The controller takes a seed for both axes or none. A half-measured gain is
    # therefore dropped rather than paired with a guess.
    if seed_gain and all(v is not None for v in seed_gain):
        argv += ['--seed-gain', f'{seed_gain[0]:.3f}', f'{seed_gain[1]:.3f}']
    return argv


def run_shot_in_process(out, args, plate_visible=True, seed_gain=None, attempt=1):
    """One engagement without spawning a process: the fast path.

    Measured 2026-10-23: a spawned shot cost 12.9 s wall for 5.2 s of aim, and
    2.5 s of that was interpreter start plus model import. Keeping the module
    loaded removes it; the loaded minimap model and the probe's preflight cache
    then also carry over, which the subprocess path threw away every shot.
    """
    module = servo()
    module.reset_engagement_state()
    # The controller never overwrites an output directory (evidence rule), so a
    # retry of the same shot needs its own directory -- otherwise the retry died
    # on FileExistsError and read the previous attempt's report.
    # The controller never reuses an output directory, and a retry after an
    # earlier crash would otherwise die on FileExistsError (observed in the
    # night loop). Walk to the first free name instead of trusting the counter.
    base = 'servo' if attempt <= 1 else f'servo-{attempt}'
    run_dir = out / base
    suffix = 2
    while run_dir.exists():
        run_dir = out / f'{base}-{suffix}'
        suffix += 1
    parsed = module.build_parser().parse_args(
        servo_argv(out, args, plate_visible, seed_gain, run_dir=run_dir))
    # main() applies the CLI's side effects; driving run() directly must too, or
    # flags like --vertical-cap silently keep their defaults.
    module.prepare_args(parsed)
    t0 = time.monotonic()
    try:
        module.run(parsed)
    except SystemExit:
        pass
    elapsed = round(time.monotonic() - t0, 2)
    path = run_dir / 'report.json'
    data = json.loads(path.read_text()) if path.is_file() else {}
    data['wall_s'] = elapsed
    return data


def run_shot(out, args, attempt=1, plate_visible=True):
    """One delegated engagement; returns the pipeline report."""
    if not getattr(args, 'spawn', False):
        return run_shot_in_process(out, args, plate_visible=plate_visible,
                                   seed_gain=getattr(args, 'seed_gain', None),
                                   attempt=attempt)
    cmd = [str(ROOT / '.venv/bin/python'), '-B', str(ROOT / 'tools/capture/aim_pipeline.py'),
           '--out', str(out), '--execute', '--fire',
           '--servo-seconds', str(args.servo_seconds),
           '--wait-reload-s', str(args.reload_wait_s)]
    env = dict(os.environ)
    env['PYTHONPATH'] = 'tools/capture:tools/replay'
    subprocess.run(cmd, capture_output=True, text=True,
                   timeout=args.servo_seconds + 60, env=env)
    path = out / 'report.json'
    return json.loads(path.read_text()) if path.is_file() else {}



def plate_distance_km(frame, plate_bbox=None):
    """Target range for the engagement policy: the plate caption, else None.

    Measured 2026-10-23: hits were 2/3 at 8.8 km and 1/6 at 13-13.5 km on the
    same ship and gun, so range is the dominant accuracy factor and the chain
    should close in rather than keep firing at long range.
    """
    import aim_servo as servo_module
    try:
        if plate_bbox is None:
            return None
        return servo_module.hud_distance_km(frame, plate_bbox)
    except (ValueError, OSError, KeyError, AttributeError, TypeError, IndexError):
        return None


def _press_key(key, display=':3'):
    subprocess.run(['xdotool', 'key', key],
                   env={'DISPLAY': display, 'PATH': '/usr/bin:/bin'})


def _click(x, y, display=':3'):
    env = {'DISPLAY': display, 'PATH': '/usr/bin:/bin'}
    subprocess.run(['xdotool', 'mousemove', str(int(x)), str(int(y))], env=env)
    time.sleep(0.4)
    subprocess.run(['xdotool', 'click', '1'], env=env)


def smoothness_of(report):
    """Smoothness of this engagement's error trajectory, from the servo steps."""
    servo = report.get('servo') or (report if 'steps' in report else {})
    steps = (report.get('stages') or {}).get('aim', {})
    errors = []
    for row in (servo.get('steps') or []):
        if isinstance(row, dict) and row.get('error_px'):
            errors.append(row['error_px'])
    if not errors:
        # One-shot engagements do not fill steps[]; they record the residual of
        # every measured correction instead, which is the same trajectory.
        errors = [list(e) for e in (servo.get('oneshot_errors') or []) if e]
    if not errors and steps.get('error_px'):
        errors.append(steps['error_px'])
    if not errors:
        return dict(smoothness='unknown', reason='no_error_series', samples=0)
    result = dict(SM.analyse(errors))
    # How many measurements the verdict rests on. One or two points cannot show
    # oscillation; saying so beats calling it "smooth".
    result['samples'] = len(errors)
    return result


def build_parser():
    """The runner's command line, exposed so tests can check the servo argv against it.

    The in-process path reads attributes off this namespace; a flag that exists
    only in the pipeline made every shot raise AttributeError at the first real
    run, because no test parsed the runner's arguments and fed them to
    servo_argv(). Keeping the parser importable is what makes that checkable.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', required=True)
    parser.add_argument('--execute', action='store_true', help='actually shoot')
    parser.add_argument('--max-shots', type=int, default=8)
    parser.add_argument('--seconds', type=float, default=900.0, help='match time budget')
    parser.add_argument('--servo-seconds', type=float, default=70.0)
    parser.add_argument('--reload-wait-s', type=float, default=35.0)
    parser.add_argument('--flight-margin-s', type=float, default=3.0)
    parser.add_argument('--poll-s', type=float, default=1.0)
    # The shot chain does not move the ship: no autopilot, no bearing steering.
    parser.add_argument('--lead-budget-s', type=float, default=16.0)
    parser.add_argument('--lead', action='store_true',
                        help='aim ahead of a moving target (costs 3 extra frames per shot)')
    parser.add_argument('--display', default=':3')
    parser.add_argument('--fire-tolerance', type=float, default=50.0,
                        help='fire only when the measured aim error is this small')
    parser.add_argument('--max-steps', type=int, default=6)
    parser.add_argument('--spawn', action='store_true',
                        help='run each shot in its own process (slow, for isolating crashes)')
    parser.add_argument('--seed-gain', type=float, nargs=2, default=None,
                        help='first-shot gain seed; later shots reuse the measured one')
    parser.add_argument('--max-reload-waits', type=int, default=3)
    parser.add_argument('--max-no-contact', type=int, default=2,
                        help='attempts with no target information before stopping')
    parser.add_argument('--oneshot-budget-s', type=float, default=10.0,
                        help='scan-to-shot budget for a shot with the target in view')
    parser.add_argument('--scan-budget-s', type=float, default=12.0,
                        help='budget when the controller has to acquire first')
    parser.add_argument('--scan-steps', type=int, default=12)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    ledger, started = [], time.monotonic()
    no_contact_streak = 0
    reload_waits = 0
    index = 0
    retry = 0
    nav = []
    while len(ledger) < args.max_shots and time.monotonic() - started < args.seconds:
        frame = _snapshot(out / f'pre-{index:02d}.png')
        hud = _hud(frame)
        if hud['battle'] is not True:
            ledger.append(dict(shot=len(ledger), event='battle_ended'))
            break
        if hud['reload'] is False and args.execute:
            # Not a refusal: the guns are reloading, so the attempt waits.
            waited = 0.0
            while waited < args.reload_wait_s:
                time.sleep(args.poll_s)
                waited += args.poll_s
                frame = _snapshot(out / f'wait-{index:02d}.png')
                hud = _hud(frame)
                if hud['reload'] is not False:
                    break
            if hud['reload'] is False:
                ledger.append(dict(shot=len(ledger), event='reload_timeout',
                                   waited_s=round(waited, 1)))
                break
        bbox = _target_bbox(frame)
        if bbox is None and args.execute:
            # Nothing in view: one delegated run may still sweep it up.
            pass
        if not args.execute:
            ledger.append(dict(shot=len(ledger), event='dry_run',
                               battle=hud['battle'], reload=hud['reload'],
                               target=bbox))
            break
        shot_dir = out / f'shot-{index:02d}'
        retry += 1
        t0 = time.monotonic()
        # With nothing in view the controller must acquire, not refuse: the
        # in-process path used to claim a plate unconditionally and then bail
        # with "needs current battle HUD, reticle and enemy" on every frame where
        # the target had sailed out of the camera.
        report = run_shot(shot_dir, args, attempt=retry, plate_visible=bbox is not None)
        elapsed = round(time.monotonic() - t0, 1)
        servo = report.get('servo') or {}
        # A fresh report shape: run_shot_in_process returns the servo report
        # itself, while the spawned pipeline wraps it under 'servo'.
        if not servo and 'steps' in report:
            servo = report
        carried = next_seed_gain(servo, getattr(args, 'seed_gain', None))
        if carried != getattr(args, 'seed_gain', None):
            args.seed_gain = carried
            report['seed_gain_next'] = carried
        verdict = report.get('verdict')
        if verdict is None and 'fired' in report:
            # The in-process controller reports `fired`, the spawned pipeline
            # reports `verdict`; the ledger speaks one language.
            verdict = 'fired' if report.get('fired') else 'refused'
        row = dict(shot=len(ledger), seconds=elapsed, verdict=verdict,
                   mode=servo.get('mode'),
                   fire_reason=servo.get('fire_reason'), error=servo.get('error'),
                   residual=(servo.get('residuals') or [None])[-1],
                   # The pipeline path reports terrain under sense; the servo
                   # path reports it where the fire decision was taken. Either is
                   # evidence, and both are recorded, never inferred here.
                   terrain=((report.get('stages') or {}).get('sense', {}).get('terrain')
                            or servo.get('fire_terrain')),
                   smoothness=smoothness_of(report),
                   target_bbox=bbox)
        reason_text = str(row.get('error') or '')
        reload_only = (verdict != 'fired'
                       and reason_text.strip() and
                       all(part.strip() in ('reload_not_ready',)
                           for part in reason_text.replace('fire revoked:', '').split(',')))
        if reload_only:
            # The gun simply is not loaded yet; that is a wait, not a failed
            # attempt, so neither the shot budget nor the no-contact streak move.
            row['event'] = 'waited_for_reload'
            reload_waits += 1
            if reload_waits <= args.max_reload_waits:
                continue
            row['event'] = 'reload_never_confirmed'
        if verdict != 'fired' and args.execute and not row.get('event'):
            # A refusal is not one thing. Only "I found nothing at all" is
            # information starvation worth stopping for; a target that was
            # seen but could not be confirmed (reticle, plate, reload) is a
            # normal refusal and must not end the match.
            found_nothing = bool(re.search(r'no enemy (healthbar|nameplate)', reason_text))
            if found_nothing:
                sense = (report.get('stages') or {}).get('sense') or {}
                no_contact_streak += 1
                row['event'] = ('no_information' if not sense.get('minimap')
                                else 'no_plate_found')
                row['no_contact_streak'] = no_contact_streak
                if no_contact_streak >= args.max_no_contact:
                    ledger.append(row)
                    index += 1
                    ledger.append(dict(kind='stopped', reason='no_information'))
                    break
            else:
                row['event'] = 'refused'
                no_contact_streak = 0
        if verdict != 'fired' and args.execute:
            # No nameplate found: either nothing is spotted, or land is in the
            # way. The terrain test answers which, and the two cases are handled
            # differently -- a hidden target is not shootable, so the ship moves
            # to get a line of fire instead of rescanning the same blocked view.
            sense = (report.get('stages') or {}).get('sense') or {}
            terrain = sense.get('terrain') or {}
            if terrain.get('clear') is False:
                row['event'] = 'target_hidden_by_terrain'
                row['terrain'] = terrain
        if verdict == 'fired':
            wait = float(hud['flight'] or 12.0) + args.flight_margin_s
            time.sleep(wait)
            after = _snapshot(out / f'post-{index:02d}.png')
            row['accounting'] = SA.classify_shot(frame, after, bbox)
            row['hit'] = row['accounting'].get('hit')
            row['damage'] = row['accounting'].get('damage')
        ledger.append(row)
        index += 1
        reload_waits = 0
        retry = 0
        (out / 'ledger.json').write_text(json.dumps(ledger, indent=2, default=str))

    # Range-closing moves are part of the record but not part of the shot budget.
    ledger.extend(nav)
    # The ledger is written on every exit path, including "the battle ended".
    (out / 'ledger.json').write_text(json.dumps(ledger, indent=2, default=str))

    fired = [r for r in ledger if r.get('verdict') == 'fired']
    hits = [r for r in fired if r.get('hit') is True]
    seconds = [r['seconds'] for r in fired if r.get('seconds')]
    summary = dict(
        shots=len(fired), attempts=len([r for r in ledger if r.get('verdict')]),
        hits=len(hits), hit_rate=round(len(hits) / len(fired), 3) if fired else None,
        damage=sum((r.get('damage') or 0) for r in fired),
        seconds=dict(min=min(seconds), median=round(statistics.median(seconds), 1),
                     max=max(seconds)) if seconds else None,
        smoothness=[r['smoothness'].get('smoothness') for r in fired],
        terrain=[(r.get('terrain') or {}).get('clear') for r in fired],
        ledger='ledger.json')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
