#!/usr/bin/env python3
"""Fire repeated shots at a moving target, judging each one from the health bar.

One iteration = one run of the verified chain (scan -> converge -> optional
scope -> lock -> HUD-range closure -> confirm -> fire) followed by an evidence
pair (before/after frames) handed to shot_accounting.classify_shot. The loop
stops after N shots or when the budget runs out, and writes a table with the
per-shot verdict, so the 2-in-10 acceptance can be read off the file.
"""
import argparse
import json
import os
import cv2
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'tools/replay'))


def snapshot(path):
    subprocess.run(['/usr/bin/python3', str(HERE / 'capture_host.py'), 'snapshot', str(path)],
                   check=True, capture_output=True)


def _rgb(path):
    """classify_shot takes arrays, not paths: load the saved frame as RGB."""
    image = cv2.imread(str(path))
    if image is None:
        raise SystemExit(f'unreadable frame {path}')
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class SalvoCounter:
    """Counts salvos from reload transitions, so clicks stop being a numerator.

    A click only fires when the gun is loaded, so "clicks" overstates shots.
    ``ready -> not ready`` is one salvo leaving the barrels; unknown readings are
    counted separately and never break the pending transition.
    """

    def __init__(self):
        self.salvos = 0
        self.unknown = 0
        self._was_ready = None
        self._unknown_streak = 0

    def observe(self, ready):
        # This client does not report "reloading" as False: the turret bars
        # become unreadable the moment a salvo leaves (measured: ready True ->
        # unknown on every live shot). So a salvo is either True -> False, or
        # True -> two consecutive unreadable samples.
        if ready is None:
            self.unknown += 1
            self._unknown_streak += 1
            if self._was_ready is True and self._unknown_streak >= 2:
                self.salvos += 1
                self._was_ready = None
            return
        self._unknown_streak = 0
        if self._was_ready is True and ready is False:
            self.salvos += 1
        self._was_ready = bool(ready)


def servo_argv(args, run_dir):
    """The aim phase is bounded by the operator budget, not by a step count."""
    return [args.python, '-B', str(HERE / 'aim_servo.py'),
            # A 13-look sweep costs more than the whole 10 s budget on its own;
            # a coarser 6-look sweep still catches a 123 px wide nameplate.
            # Bearing prior first: a blind sweep either eats the whole 10 s
            # budget (13 looks > 25 s) or misses the plate (6 looks). The model
            # gives the enemy's bearing, so turn to it and only confirm.
            *([] if os.environ.get('WW_BEARING', '1') == '0' else
              ['--minimap-bearing', '--bearing-gain', '-1.5', '--fov-deg', '68.2']),
            '--scan', '--scan-step', '320', '--scan-steps', '6', '--scan-rows', '2',
            '--scan-settle', '0.05',
            '--reach', '400', '--damping', '0.9', '--reach-y', '3', '--coarse-reach-y', '60',
            '--max-steps', '16', '--tolerance', '60', '--fire-tolerance', '100',
            '--fire-min-steps', '2',
            # --fire-max-km would call minimap_target_km() every step: measured
            # 4546 ms uncached (45 ms cached) against 58 ms for the caption
            # readout. The far-range guard is now the caption's job.

            '--settle', '0.3', '--max-stale-reads', '1',
            # Re-detect every step: template tracking lost the plate after real
            # moves on 6 of 6 recent cycles (search radii 160..900 px), while a
            # fresh nameplate detection costs ~0.1 s.
            '--reacquire-px', '0',
            # The radar ring is only painted after mouse motion; a slightly
            # larger nudge plus an extra read keeps third-person aiming alive
            # without ever inventing a reticle position.
            '--aim-nudge', '3', '--reticle-retries', '3',
            # Measure the gain in-session (the vertical seed alone overshoots 8x).
            # seed claimed: with -1.4 every vertical correction overshot ~6x
            # (measured -83 -> -500 -> -412) and the scope never opened.
            '--seed-gain', '-1.5', '-1.16',
            # Fire from the binocular view: third-person salvos were the whole
            # reason every shell missed. Coarse-aim in third person, open the
            # scope, then fine-aim with the scope's own gain.
            # Aim, then open the binocular view, then fire (operator's order).
            # The scope gain is now seeded from the per-process cache, and the
            # step cap leaves room for the two-frame confirmation.
            # 90 px was too loose: the zoom landed on empty water and the scope
            # re-read found no nameplate (measured 2026-10-20). The binocular
            # view magnifies ~5x, so the coarse solution must already sit within
            # ~40 px before it is worth opening.
            # No --scope-after: this client draws no nameplate inside the
            # binocular view (measured pre/post pair), so the scope removes the
            # evidence the shot is judged on. Third person keeps the plate.
            # '--scope-after', '40',
            # Fixed km/unit (magnitude of the measured -0.025) instead of the
            # 2-frame in-place calibration, and 160 units per closure step:
            # together they keep a cycle inside the 10 s the operator requires.
            '--measure-range-units', '0', '--range-km-per-unit', '0.125',
            # Operator rule only: the scope opens when the shell-impact range is
            # within 2 km of the enemy's range. The pixel fallback is disabled so
            # a "lucky" pixel alignment cannot pass for the range criterion.
            '--scope-vertical-max-px', '60',
            # Close the crosshair range onto the target's own caption before
            # firing; without it the salvo lands short by however much the two
            # readouts disagree (measured 0.4-9 km).
            '--range-closure', '--range-source', 'caption', '--range-tolerance-km', '0.5',
            '--range-step-units', '40',

            # No --lock: with a locked target this client draws the *target's*
            # range beside the reticle instead of the aimed water distance, and
            # then the 2 km range criterion compares a number with itself. The
            # criterion needs the aimed distance, so the lock stays off.
            '--assume-battle',
            '--seconds', '90', '--aim-budget-s', f'{max(4.0, args.aim_seconds):.1f}',
            '--out', str(run_dir),
            *(['--execute', '--fire'] if args.execute else [])]


def _reload_ready(path):
    """Reload readiness from a saved frame, or None when it cannot be read."""
    try:
        from fire_hud import FireHudReader
        image = cv2.imread(str(path))
        if image is None:
            return None
        return FireHudReader().detect(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))['reload'].get('ready')
    except (ValueError, OSError):
        return None


def burst_fire(probe, seconds=20.0, interval=0.5, clock=time.monotonic, sleep=time.sleep,
               sample=None, sample_every=2.0):
    """Hold the solution and keep firing: no aiming inside this window.

    Operator rule: once the reticle is on the target, stop measuring for 20 s
    and just click -- the guns reload faster than a fresh aim takes.
    """
    if not (seconds > 0) or not (interval > 0):
        raise ValueError('burst needs a positive window and interval')
    if sample is not None and not (sample_every > 0):
        raise ValueError('sampling needs a positive period')
    started = clock()
    clicks = 0
    next_sample = sample_every
    while clock() - started < seconds:
        probe.x('mousedown', 1)
        probe.x('mouseup', 1)
        clicks += 1
        # A click does not prove a salvo: the gun has to be loaded. Reading the
        # reload state is not aiming, so it may happen inside the burst and it
        # turns "clicks" into a real shot denominator.
        if sample is not None and clock() - started >= next_sample:
            sample()
            next_sample += sample_every
        sleep(interval)
    return clicks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--shots', type=int, default=10)
    # Operator budget: <=10 s from opening the view to the first shell, then a
    # 20 s burst of clicks at 0.5 s with no re-aiming in between.
    parser.add_argument('--aim-seconds', type=float, default=10.0)
    parser.add_argument('--burst-seconds', type=float, default=20.0)
    parser.add_argument('--burst-interval', type=float, default=0.5)
    # Reading the reload state inside the burst is not aiming; it turns clicks
    # into an observed salvo count.
    parser.add_argument('--burst-sample-s', type=float, default=2.0)
    parser.add_argument('--seconds', type=float, default=900)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--python', default=str(ROOT / '.venv/bin/python'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    from shot_accounting import classify_shot
    log, shots = [], 0
    deadline = time.monotonic() + args.seconds
    while time.monotonic() < deadline and shots < args.shots:
        index = shots + 1
        before = args.out / f'before-{index:02d}.png'
        snapshot(before)
        aim_started = time.monotonic()
        run_dir = args.out / f'run-{index:02d}'
        result = subprocess.run(servo_argv(args, run_dir), capture_output=True, text=True,
                                env={**os.environ, 'WW_FRAME_REUSE_S': '0.75',
                                     # The bearing prior costs 2215 ms uncached and
                                     # 45 ms cached; 8 s of staleness is harmless
                                     # for a coarse turn that the scan confirms.
                                     'WW_MINIMAP_TTL': '8'})
        report_path = run_dir / 'report.json'
        report = json.loads(report_path.read_text()) if report_path.is_file() else {}
        fired = bool(report.get('fired'))
        row = dict(shot=index, fired=fired,
                   reason=report.get('fire_reason') or report.get('error'),
                   steps=len(report.get('steps') or []))
        aim_ms = report.get('aim_ms')
        if aim_ms is None:
            aim_ms = (time.monotonic() - aim_started) * 1000.0
        if fired:
            shots += 1
            row['aim_ms'] = round(aim_ms)
            row['over_aim_budget'] = aim_ms > args.aim_seconds * 1000.0
            # Operator rule: the solution is finished, so hold it and keep
            # clicking for the whole burst window instead of re-aiming.
            probe = None
            clicks = 0
            counter = SalvoCounter()
            if args.execute and args.burst_seconds > 0:
                from probe_aim_input import Probe, ProbeError, game_identity
                probe = Probe(args.burst_seconds + 8.0, {})
                try:
                    # Probe.preflight() reads self.identity: bind the game first,
                    # exactly like aim_servo does before its own preflight.
                    probe.identity = game_identity()
                    probe.preflight()
                    probe.check_identity()
                    def sampler(shot=index):
                        path = args.out / f'burst-{shot:02d}-{counter.salvos}-{counter.unknown}.png'
                        snapshot(path)
                        counter.observe(_reload_ready(path))

                    clicks = burst_fire(probe, args.burst_seconds, args.burst_interval,
                                        sample=sampler, sample_every=args.burst_sample_s)
                except (ProbeError, OSError) as error:
                    row['burst_error'] = str(error)
                finally:
                    probe.release()
            row['burst_clicks'] = clicks
            row['salvos'] = counter.salvos
            row['reload_unknown'] = counter.unknown
            # Shells need their flight time to land: sample the same plate a
            # few times and keep the strongest reading instead of one snapshot
            # taken while the salvo is still in the air.
            flight = report.get('fire') if isinstance(report.get('fire'), dict) else None
            flight = (flight or {}).get('flight_time_s')
            if not flight:
                for step in reversed(report.get('steps') or []):
                    value = (step.get('flight_time') or {}).get('value')
                    if value:
                        flight = value
                        break
            flight = float(flight or 12.0)
            wait = min(24.0, max(9.0, float(flight) + 4.0))
            best = None
            for attempt in range(4):
                time.sleep(wait / 4.0)
                after = args.out / f'after-{index:02d}.png'
                snapshot(after)
                verdict = classify_shot(_rgb(before), _rgb(after))
                if best is None or (verdict.get('hit') is True and best.get('hit') is not True):
                    best = verdict
                if verdict.get('hit') is True:
                    break
            row.update(hit=best.get('hit'), damage=best.get('damage'),
                       verdict=best.get('reason'))
        log.append(row)
        (args.out / 'shots.json').write_text(json.dumps(log, indent=1))
        time.sleep(3)
    hits = sum(1 for r in log if r.get('hit') is True)
    summary = dict(shots=shots, hits=hits, misses=sum(1 for r in log if r.get('hit') is False),
                   unknown=sum(1 for r in log if r.get('fired') and r.get('hit') is None),
                   rate=None if not shots else round(hits / shots, 3),
                   target='2 hits in 10 shots (0.2)')
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
