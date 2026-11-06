#!/usr/bin/env python3
"""One entry point for the whole engagement, with a stage-by-stage verdict.

The stack grew around a live experiment: perception lived in main_view_perception
and fire_hud, the controller in aim_servo, the gates in fire_gate, and last night
two more paths were bolted on (a minimap-only shot and a flight-time elevation
loop). This module is the single place that says, in order, what the pipeline is:

    sense   -> what the frame actually contains (battle, reticle, plate, contact)
    target  -> which contact the shot may use, and on what basis
    aim     -> the measured error and the one action that reduces it
    gate    -> the nine fire conditions, as a single decision
    fire    -> the trigger, only when the gate allows it
    verify  -> what the next frame says about the shot

Each stage records the raw numbers it used, so an acceptance run can be checked
against the frame instead of against the program's own opinion.

Default is a single-frame dry run: it senses, targets and evaluates the gate, and
prints what it would do without touching the input devices.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/replay'))
sys.path.insert(0, str(ROOT / 'tools/capture'))

import aim_servo as A                      # noqa: E402
import fire_gate as G                      # noqa: E402
import minimap_shot as MS                  # noqa: E402
from minimap_recognizer import MinimapRecognizer  # noqa: E402

SCHEMA = 'aim-pipeline/1'
RECENTRE_KEY_DEFAULT = 'm'


def _frame(path=None):
    """One RGB frame from the capture host, or from a saved PNG."""
    if path:
        from PIL import Image
        return np.asarray(Image.open(path).convert('RGB'))
    import subprocess
    target = ROOT / 'state' / '_pipeline_frame.png'
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['/usr/bin/python3', str(ROOT / 'tools/capture/capture_host.py'),
                    'snapshot', str(target)], capture_output=True, timeout=90)
    from PIL import Image
    return np.asarray(Image.open(target).convert('RGB'))


_MINIMAP_CACHE = {'at': 0.0, 'value': None, 'seen': None}
MINIMAP_TTL_S = 4.0


def minimap_seen(frame, ttl=MINIMAP_TTL_S):
    """The raw contact read (own pixel, nearest pixel), cached with the contact."""
    now = time.monotonic()
    if _MINIMAP_CACHE['seen'] is not None and now - _MINIMAP_CACHE['at'] <= ttl:
        return _MINIMAP_CACHE['seen']
    seen = A.minimap_contacts_from_frame(frame)[0]
    _MINIMAP_CACHE.update(at=now, seen=seen)
    return seen


def minimap_pixels(frame):
    """(own_pixel, enemy_pixel) or (None, None)."""
    seen = minimap_seen(frame)
    if not seen or not seen.get('nearest'):
        return None, None
    return seen['own_pixel'], seen['nearest'].get('pixel')


def minimap_cached(frame, ttl=MINIMAP_TTL_S):
    """The nearest minimap contact, cached for a few seconds.

    Measured 2026-10-23: one contact read costs ~417 ms -- the single largest
    item in the sense stage, and the contact itself moves at most a few tenths
    of a pixel in that time at 20 kt. Re-reading it every frame bought nothing
    and pushed the engagement out of its time budget.
    """
    now = time.monotonic()
    if _MINIMAP_CACHE['value'] is not None and now - _MINIMAP_CACHE['at'] <= ttl:
        return _MINIMAP_CACHE['value']
    value = MS.contact_from_frame(frame, A.minimap_contacts_from_frame)
    _MINIMAP_CACHE.update(at=now, value=value)
    return value


def stage_sense(frame):
    """Everything the frame contains, with the reader that produced it."""
    # One HUD read for everything it already contains: the full detector
    # returns the reticle, the weapon, the reload state and the readouts, so the
    # separate reticle/readout calls would only re-scan the same pixels.
    try:
        import fire_hud
        hud = fire_hud.FireHudReader().detect(frame)
    except Exception:
        hud = {}
    reticle_row = hud.get('reticle') or {}
    if reticle_row.get('status') == 'observed':
        reticle = [reticle_row.get('x'), reticle_row.get('y')]
        if reticle[0] is None or reticle[1] is None:
            reticle = None
        source = (reticle_row.get('raw') or {}).get('source', 'fire_hud')
    else:
        reticle = None
        source = None
    _, battle, weapon, _ = A.reticle_centre(frame) if reticle is None else (
        None, (hud.get('battle') or {}).get('active'), (hud.get('weapon') or {}).get('kind'), None)
    if reticle is None:
        # A fixed HUD element: every successful read in this project returned
        # [1280.5, 805]. Recorded as a fallback, never as an observation.
        reticle_source = 'fixed_centre'
        reticle = (1280.5, 805.0)
    else:
        reticle_source = source
    aim_km = A.hud_aim_km(frame, reticle)
    if aim_km is None:
        aim_km = A.aim_distance_km(frame, reticle)
        if aim_km is not None:
            aim_km_source = 'offset_fan'
        else:
            aim_km_source = None
    else:
        aim_km_source = 'anchored'
    flight = None
    if aim_km is None:
        # Measured 70 ms and only needed when the km readout failed, so it is
        # read lazily rather than on every frame.
        try:
            info = A.hud_flight_seconds(frame)
            flight = None if not info else info.get('value')
        except Exception:
            flight = None
    plates = []
    for row in A.find_enemy(frame, reticle):
        plates.append(dict(bbox=row['nameplate_bbox'], trackable=row.get('trackable'),
                           distance_px=row.get('distance_from_reticle_px'),
                           body_visible=row.get('body_visible')))
    contact = minimap_cached(frame)
    own_px, enemy_px = minimap_pixels(frame)
    terrain_row = None
    if own_px is not None and enemy_px is not None:
        try:
            import minimap_terrain as _terrain
            panel = frame[796:1519, 1836:2556]
            terrain_row = _terrain.shot_terrain(panel, own_px, enemy_px)
        except Exception as error:                      # never break a frame on this
            terrain_row = dict(clear=None, reason=f'terrain_error:{type(error).__name__}')
    reload = (hud.get('reload') or {})
    turrets = [t.get('ready') for t in (reload.get('turrets') or [])]
    weapon_kind = (hud.get('weapon') or {}).get('kind')
    return dict(
        battle=battle, weapon=weapon, reticle=list(reticle), reticle_source=reticle_source,
        aim_km=aim_km, aim_km_source=aim_km_source, flight_s=flight,
        plates=plates, plate_count=len(plates),
        minimap=None if contact is None else dict(
            km=round(contact[0], 2), relative_bearing_deg=round(contact[1], 1)),
        terrain=terrain_row,
        hud=dict(weapon_kind=weapon_kind,
                 reload_ready=reload.get('ready'),
                 all_ready=reload.get('all_ready'),
                 turrets_ready=turrets,
                 reload_status=reload.get('status'),
                 aiming_view=(hud.get('aiming_view') or {}).get('mode')),
    )


def stage_target(sense, allow_minimap):
    """Which target the shot may use and on what basis.

    A nameplate is preferred: it is the only evidence that also carries the
    enemy's identity on screen. When no plate is ever drawn (a submarine, or a
    target beyond the cyclone's detection range) the minimap contact is the only
    measurement left, and it is used *explicitly* -- the report keeps the basis.
    """
    if sense['plate_count']:
        best = min(sense['plates'], key=lambda row: row['distance_px'] or 1e9)
        return dict(basis='nameplate', plate=best, contact=sense['minimap'])
    if allow_minimap and sense['minimap']:
        return dict(basis='minimap_contact', plate=None, contact=sense['minimap'])
    return dict(basis='none', plate=None, contact=sense['minimap'], reason='no_target_evidence')


def stage_aim(sense, target, gain=(1.5, 5.0), km_per_unit=0.004):
    """The measured error and the single action that reduces it.

    Only numbers that exist are used: a missing reading is reported as missing,
    never replaced by an assumption about where the target is.
    """
    row = dict(gain=list(gain), action=None, error_px=None, error_km=None,
               lead_px=[0.0, 0.0], notes=[])
    if sense['battle'] is not True:
        row['notes'].append('battle_unconfirmed')
        return row
    plate = target.get('plate')
    if plate is not None:
        reticle = sense['reticle']
        # The plate's aim point is recomputed from the same rule the servo uses,
        # so the pipeline cannot drift from the controller it drives.
        dx = (plate['bbox'][0] + plate['bbox'][2] / 2) - reticle[0]
        dy = (plate['bbox'][1] + plate['bbox'][3] / 2) - reticle[1]
        row['error_px'] = [round(dx, 1), round(dy, 1)]
        row['action'] = dict(kind='plate', units=[round(-dx / gain[0], 1),
                                                  round(-dy / gain[1], 1)])
        return row
    contact = target.get('contact')
    if contact is None:
        row['notes'].append('no_contact')
        return row
    units = MS.yaw_units(contact['relative_bearing_deg'], gain[0])
    row['action'] = dict(kind='minimap', yaw_units=round(units, 1))
    if sense['aim_km'] is not None:
        row['error_km'] = round(sense['aim_km'] - contact['km'], 2)
        row['action']['elevation_units'] = round(
            MS.elevation_units(sense['aim_km'], contact['km'], km_per_unit), 1)
    elif sense['flight_s'] is not None:
        row['notes'].append('elevation_on_flight_time')
        row['action']['target_flight_s'] = round(contact['km'] * 0.6, 2)
    else:
        row['notes'].append('no_elevation_sensor')
    return row


def aim_error_for_gate(target, aim, fire_tolerance_px, km_tolerance=0.4):
    """The error the gate should judge, expressed in its own units.

    With a nameplate this is the pixel distance to the aim point. With a minimap
    contact there is no pixel error to measure -- the shot's quality *is* the
    range and bearing gap -- so it is reported as 0 when both gaps are inside
    tolerance and as "outside" otherwise, and the basis is recorded.
    """
    if aim.get('error_px') is not None:
        return float(np.hypot(*aim['error_px']))
    if target.get('basis') == 'minimap_contact':
        gap = aim.get('error_km')
        bearing = (target.get('contact') or {}).get('relative_bearing_deg')
        if gap is None or bearing is None:
            return None
        return 0.0 if abs(gap) <= km_tolerance and abs(bearing) <= 2.0 else fire_tolerance_px + 1.0
    return None


def stage_gate(sense, target, aim, fire_tolerance_px, identity_ok=None, focus_ok=None):
    """One decision object for the nine conditions."""
    basis = target.get('basis')
    evidence = G.Evidence(
        battle_active=sense['battle'],
        own_alive=sense['battle'],
        weapon_kind=(sense.get('hud') or {}).get('weapon_kind') or sense.get('weapon'),
        reload_ready=(sense.get('hud') or {}).get('reload_ready'),
        enemy_plate=(None if target.get('plate') is None else dict(
            bbox=target['plate']['bbox'], hp=None, red_px=1)),
        frame_fresh=True,
        aim_error_px=aim_error_for_gate(target, aim, fire_tolerance_px),
        fire_tolerance_px=fire_tolerance_px,
        identity_ok=identity_ok,
        focus_ok=focus_ok,
        scope_requested=False,
        scope_evidenced=None,
        scope_target_visible=None,
        body_visible=(None if target.get('plate') is None
                      else target['plate'].get('body_visible')),
        terrain_clear=(sense.get('terrain') or {}).get('clear'),
        terrain_detail=sense.get('terrain'),
        shot_basis=('minimap_contact' if basis == 'minimap_contact' else 'plate'),
        minimap_contact=target.get('contact'),
        # Freshness needs an actual contact: a basis label alone must never
        # satisfy the plate gate (caught by test_minimap_basis_..._only_when_fresh).
        minimap_contact_fresh=(True if (basis == 'minimap_contact'
                                        and target.get('contact')) else None),
    )
    return G.fire_decision(evidence)


def run(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', required=True)
    parser.add_argument('--frame', default=None, help='saved PNG instead of a live grab')
    parser.add_argument('--execute', action='store_true', help='send the aim action')
    parser.add_argument('--fire', action='store_true', help='pull the trigger when the gate allows')
    parser.add_argument('--allow-minimap', action='store_true',
                        help='accept a minimap contact when no nameplate exists')
    parser.add_argument('--fire-tolerance', type=float, default=140.0)
    parser.add_argument('--km-per-unit', type=float, default=0.004)
    parser.add_argument('--gain', type=float, nargs=2, default=[1.5, 5.0])
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--servo-seconds', type=float, default=45.0,
                        help='time budget handed to the delegated controller')
    parser.add_argument('--max-steps', type=int, default=20)
    parser.add_argument('--oneshot-budget-s', type=float, default=10.0)
    parser.add_argument('--scan-budget-s', type=float, default=60.0)
    parser.add_argument('--scan-steps', type=int, default=12,
                        help='sweep steps of 400 units (~16 deg each) before aiming')
    parser.add_argument('--wait-reload-s', type=float, default=35.0,
                        help='how long to poll for a loaded gun before giving up')

    parser.add_argument('--recentre-key', default=RECENTRE_KEY_DEFAULT)
    parser.add_argument('--display', default=':2')
    args = parser.parse_args(argv)

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    report = dict(schema=SCHEMA, execute=args.execute, fire_requested=args.fire,
                  allow_minimap=args.allow_minimap, stage='sense', stages={}, reasons=[])

    reload_deadline = time.monotonic() + args.wait_reload_s
    for index in range(max(1, args.rounds)):
        frame = _frame(args.frame if index == 0 else None)
        if args.execute and not args.frame:
            # Cheap pre-check: 43 ms instead of the full ~1.2 s read (HUD and
            # minimap), because in an executing run the sense stage only picks
            # the controller's mode.
            import main_view_perception as _MVP
            plates = _MVP.detect_nameplates(frame)
            sense = dict(battle=True, weapon=None, reticle=[1280.5, 805.0],
                         reticle_source='not_read_in_execute', aim_km=None,
                         aim_km_source=None, flight_s=None,
                         plates=[dict(bbox=p['nameplate_bbox'],
                                      distance_px=abs(p['nameplate_bbox'][0] + p['nameplate_bbox'][2] / 2
                                                      - 1280.5),
                                      body_visible=p.get('body_visible'),
                                      trackable=True) for p in plates],
                         plate_count=len(plates), minimap=None,
                         hud=dict(weapon_kind=None, reload_ready=None, all_ready=None,
                                  turrets_ready=[], reload_status=None, aiming_view=None),
                         light=True)
        else:
            sense = stage_sense(frame)
        report['stages']['sense'] = sense
        target = stage_target(sense, args.allow_minimap)
        report['stages']['target'] = target
        aim = stage_aim(sense, target, tuple(args.gain), args.km_per_unit)
        report['stages']['aim'] = aim
        decision = stage_gate(sense, target, aim, args.fire_tolerance)
        report['stages']['gate'] = decision
        report['frame'] = index
        if sense['battle'] is not True:
            report['verdict'] = 'refused'
            report['reasons'].append('battle_unconfirmed')
            break
        if target['basis'] == 'none' and not args.execute:
            # A dry run has nothing else to go on. In an executing run our own
            # sense only *selects the mode*: the controller owns acquisition and
            # can sweep for a target this single frame did not show. Measured
            # 2026-10-23: four of ten engagements refused here while the
            # controller would have found the target.
            report['verdict'] = 'refused'
            report['reasons'].append('no_target_evidence')
            break
        action = aim.get('action') or {}
        moved = False
        if args.execute and not args.frame:
            # One HUD read (~120 ms) tells us whether a shot is even possible.
            # Measured 2026-10-23: two of eight engagements spent 38 s polling
            # inside the controller and still refused. Scheduling the attempt
            # for when the guns are loaded is not a gate change -- the same
            # reload evidence decides -- it just stops wasting the attempt.
            ready = _reload_ready(frame)
            waited = 0.0
            while ready is False and waited < args.wait_reload_s:
                time.sleep(0.5)
                waited += 0.5
                frame = _frame()
                ready = _reload_ready(frame)
            if ready is False:
                report['verdict'] = 'refused'
                report['reasons'].append('reload_not_ready_after_wait')
                break
            report['reload_wait_s'] = round(waited, 1)
        if args.execute:
            # Acting is delegated to the sanctioned controller: this module
            # measures and decides, aim_servo performs the motion and owns the
            # gates. (Project rule: the model aims and fires, never an ad-hoc
            # xdotool move from a script.)
            report['delegated_to'] = 'aim_servo.py'
            delegated = _run_servo(out, args,
                                    plate_visible=bool(sense['plate_count']))
            report['servo'] = delegated
            report['verdict'] = 'fired' if delegated.get('fired') else 'refused'
            report['reasons'].extend(delegated.get('reasons') or [])
            break
        allowed = bool(decision.get('allow'))
        blockers = set(decision.get('blockers') or [])
        if (not allowed and blockers and blockers <= {'reload_not_ready'}
                and args.fire and args.execute
                and time.monotonic() < reload_deadline):
            # Guns not loaded is a "wait", not a refusal: the reload bar is
            # readable now (amber = traversing, countdown = reloading), so the
            # pipeline polls instead of giving up.
            report['reasons'].append('waited_for_reload')
            time.sleep(0.4)
            continue
        report['verdict'] = ('ready' if allowed and not args.fire else
                             ('fired' if allowed else 'refused'))
        if not allowed:
            report['reasons'].extend(decision.get('blockers') or [])
        if allowed and args.fire and not args.execute:
            # Dry run: the gate's verdict is the deliverable; the trigger is the
            # controller's job.
            report['reasons'].append('fire_needs_execute')
            report['verdict'] = 'ready'
        break

    (out / 'report.json').write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(dict(verdict=report.get('verdict'), reasons=report.get('reasons'),
                          stage=report.get('stage')), indent=2, default=str))
    return 0 if report.get('verdict') in ('ready', 'fired') else 1


def _reload_ready(frame):
    """True/False/None from the reload bars, cheaply (no minimap, no plates)."""
    try:
        import fire_hud
        hud = fire_hud.FireHudReader().detect(frame)
    except Exception:
        return None
    return (hud.get('reload') or {}).get('ready')


def _run_servo(out, args, plate_visible=False):
    """Run the sanctioned controller once and return its verdict.

    Two modes, chosen from our own sense stage:
      * a plate is already in view -> ``--oneshot``: one look, one open-loop
        solve, one trigger (measured ~2 s);
      * nothing in view -> ``--scan``: acquire first (12 steps x 400 units).
    Both are aim_servo runs; this module never issues input itself.
    """
    import os
    import subprocess
    cmd = [str(ROOT / '.venv/bin/python'), '-B', str(ROOT / 'tools/capture/aim_servo.py'),
           '--out', str(out / 'servo'), '--seconds', str(int(args.servo_seconds)),
           '--execute', '--aim-nudge', '2', '--move-chunk', '40',
           '--vertical-cap', '60', '--settle-with-fire', '--tolerance', '60',
           '--fire-tolerance', str(args.fire_tolerance), '--settle', '0.25',
           '--max-steps', str(args.max_steps)]
    if plate_visible:
        cmd += ['--oneshot', '--aim-budget-s', str(args.oneshot_budget_s)]
    else:
        cmd += ['--scan', '--scan-steps', str(args.scan_steps), '--scan-step', '400',
                '--scan-confirm-steps', str(args.scan_steps), '--scan-settle', '0.12',
                '--scan-rows', '1', '--scan-tilt', '3',
                '--aim-budget-s', str(args.scan_budget_s)]
    if args.fire:
        cmd += ['--fire']
    env = dict(os.environ)
    env['WW_ALLOW_FOREIGN_CAPTURE'] = '1'
    mode = 'oneshot' if plate_visible else 'scan'

    deadline = time.monotonic() + max(0.0, args.wait_reload_s)
    retried_scan = False
    attempt = 0
    while True:
        attempt += 1
        # The controller refuses to reuse an output directory, so a retry needs
        # its own: otherwise the retry died on FileExistsError and this loop read
        # the previous attempt's report as if it were the new verdict.
        run_dir = out / ('servo' if attempt <= 1 else f'servo-{attempt}')
        cmd[cmd.index('--out') + 1] = str(run_dir)
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=args.servo_seconds + 30, env=env)
        path = run_dir / 'report.json'
        if not path.is_file():
            return dict(fired=False, reasons=['servo_wrote_no_report'], mode=mode,
                        attempts=attempt, stderr=result.stderr[-400:])
        data = json.loads(path.read_text())
        fast = data.get('fast') or {}
        verdict = dict(fired=bool(data.get('fired')), fire_reason=data.get('fire_reason'),
                       error=data.get('error'), mode=mode, attempts=attempt,
                       residuals=fast.get('residuals'),
                       aim_ms=fast.get('aim_ms') or data.get('aim_ms'),
                       reasons=[data.get('fire_reason') or data.get('error') or 'refused'],
                       # Carried through so the match ledger records the line of
                       # fire and the aim trajectory this shot was decided on,
                       # instead of a summary that hides both.
                       fire_terrain=data.get('fire_terrain'),
                       oneshot_errors=data.get('oneshot_errors'),
                       oneshot_refine_rounds=(data.get('steps') or [{}])[-1].get(
                           'oneshot_refine_rounds'),
                       fire_decision=data.get('fire_decision'))
        reason_text = f"{data.get('fire_reason') or ''} {data.get('error') or ''}".lower()
        # A reload is a wait, not a failure: poll inside the budget.
        if (not verdict['fired'] and 'reload' in reason_text
                and time.monotonic() < deadline):
            time.sleep(1.5)
            continue
        # The target was in view when *we* looked and gone when the controller
        # looked (measured: 3 of 8 engagements). Re-run in acquisition mode
        # instead of counting it as a miss.
        if (not verdict['fired'] and plate_visible and not retried_scan
                and ('enemy' in reason_text or 'reticle' in reason_text
                     or 'nameplate' in reason_text or 'residual' in reason_text)):
            retried_scan = True
            cmd = [c for c in cmd if c not in ('--oneshot',)]
            index = cmd.index('--aim-budget-s')
            cmd[index + 1] = str(args.scan_budget_s)
            cmd += ['--scan', '--scan-steps', str(args.scan_steps), '--scan-step', '400',
                    '--scan-confirm-steps', str(args.scan_steps), '--scan-settle', '0.12',
                    '--scan-rows', '1', '--scan-tilt', '3']
            mode = 'scan_after_oneshot'
            verdict['mode'] = mode
            continue
        return verdict


if __name__ == '__main__':
    raise SystemExit(run())
