#!/usr/bin/env python3
"""Bounded name-identified, UI-locked main-view aiming and shot requests.

Default is an offline dry-run. Execute requires an explicit training-room
attestation, which is an operator assertion, not visual proof of battle mode.
Capture timestamps bound pipeline age; upstream render latency remains unknown.
A delivered trigger request never proves actual discharge, hits or damage.
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
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / 'tools/replay'))
from probe_aim_input import Probe, ProbeError, pointer_state
from locked_capture import LockedCapture, require_fresh
import aim_servo as A
from fire_hud import FireHudReader
from main_view_perception import detect_nameplates
from target_range_selection import inspect_candidates, DEFAULT_MIN_RANGE_KM, DEFAULT_MAX_RANGE_KM
from cross_zoom_identity import CrossZoomIdentity
from target_confirmation import confirm_target
from locked_observation import LockedObservation
from locked_aim_control import LockedAimControl

# A visible enemy whose range glyph is unreadable on one frame is transient
# evidence, not an out-of-range decision. Waiting is bounded in both time and
# frames, sends no input, and never starts the shot budget; the distance gate
# itself is unchanged, so an ambiguous digit still cannot select a target.
CANDIDATE_WAIT_MAX_NS = 20_000_000_000
CANDIDATE_WAIT_MAX_FRAMES = 60
# In scope the CN360 15.9 HUD prints flight time but no aim-range number, so a
# numeric range comparison is unavailable. World of Warships derives the
# trajectory from the reticle point, so the scoped vertical loop is "put the
# reticle on the hull". This tolerance is a screen-pixel aim band, not a claim
# about a numeric range, and it is only used by the explicit --scoped-hull path.
SCOPED_HULL_TOLERANCE_PX = 25


class ShotBudget:
    """First qualified acquisition starts a bounded thirty-second attempt."""
    def __init__(self):
        self.accepted_ns = self.deadline_ns = None

    def accept(self, captured_ns):
        if self.accepted_ns is None:
            self.accepted_ns = captured_ns
            self.deadline_ns = captured_ns + 30_000_000_000

    def require(self, now_ns, reserve_ns=0):
        if self.deadline_ns is not None and now_ns + reserve_ns >= self.deadline_ns:
            raise ProbeError('thirty-second target-to-shot budget exhausted')

    def record(self, now_ns):
        return dict(accepted_monotonic_ns=self.accepted_ns,
                    deadline_monotonic_ns=self.deadline_ns,
                    elapsed_ns=None if self.accepted_ns is None else now_ns-self.accepted_ns,
                    limit_ns=30_000_000_000, actual_discharge_verified=False)


class AcquisitionProbe(Probe):
    """Check pinned capture, focus, age and budget on both sides of focus."""
    evidence = None
    capture = None
    shot_budget = None
    left_owned = False
    scope_input_attempted = False

    def guard(self, reserve_ns=0):
        if self.capture is None or self.evidence is None:
            raise ProbeError('current pinned capture evidence missing')
        self.capture.check()
        require_fresh(self.evidence)
        if self.evidence.get('identity') != self.identity or self.evidence.get('window') != self.window:
            raise ProbeError('capture and input identity mismatch')
        if self.shot_budget is not None:
            self.shot_budget.require(time.monotonic_ns(), reserve_ns)
        if time.monotonic() + reserve_ns/1e9 >= self.action_deadline:
            raise ProbeError('bounded action deadline reached')

    def focus(self):
        super().focus()
        self.guard()

    def command(self, argv, cleanup=False, identity=None):
        budget = self.shot_budget
        original = self.action_deadline
        read_only = len(argv) > 1 and argv[0] == 'xdotool' and argv[1] in (
            'getwindowname', 'getwindowpid', 'getwindowfocus', 'getwindowgeometry', 'search')
        if not cleanup and not read_only:
            trigger = len(argv) > 1 and argv[0] == 'xdotool' and argv[1] == 'mousedown'
            self.guard(150_000_000 if trigger else 0)
            if budget is not None and budget.deadline_ns is not None:
                self.action_deadline = min(original, budget.deadline_ns/1e9)
            if trigger:
                self.report['trigger_requested_monotonic_ns'] = time.monotonic_ns()
                self.report['shot_outcome'] = 'input_attempted_discharge_unverified'
                self.report['inputs'].append(dict(button='left',hold_ms=120,transport='XTest',status='attempted'))
            if ((len(argv) > 2 and argv[:3] == ['xdotool','keydown','Shift_L']) or
                    ('--key' in argv and argv[argv.index('--key')+1] == '42')):
                self.scope_input_attempted = True
        try:
            return super().command(argv, cleanup=cleanup, identity=identity)
        finally:
            self.action_deadline = original

    def move(self, dx, dy):
        vertical_limit = getattr(self, '_vertical_move_limit', 3)
        if type(dx) is not int or type(dy) is not int or abs(dx) > 150 or abs(dy) > vertical_limit:
            raise ProbeError(f'movement exceeds horizontal 150 / vertical {vertical_limit} bound')
        if self.shot_budget is not None:
            self.shot_budget.require(time.monotonic_ns(), 350_000_000)
        if self.relative_transport != 'xtest':
            self.focus()
        if self.relative_transport == 'xtest':
            result = super().move(dx,dy)
            delivered = self.report['inputs'][-1].get('pointer_delta') if self.report['inputs'] else None
            if delivered is not None and any(abs(d) < abs(r)*.85 for d,r in zip(delivered,(dx,dy)) if r):
                if self.report.get('pointer_recoveries', 0):
                    raise ProbeError('XTest pointer clipping repeated after bounded recovery')
                # Retreat turns the camera too. Defer it to the next fresh
                # confirmed frame, then reobserve without retrying correction.
                self._pointer_retreat = [-dx if dx and abs(delivered[0]) < abs(dx)*.85 else 0,
                                         -dy if dy and abs(delivered[1]) < abs(dy)*.85 else 0]
                self.report['pointer_recoveries'] = 1
                return False
            return True
        binary = ROOT/'state/cn360-tools/uinput_input'
        if not binary.is_file():
            raise ProbeError('bounded uinput transport unavailable')
        row = dict(relative=[dx,dy], transport='uinput_virtual_mouse', transport_status='attempted')
        self.report['inputs'].append(row)
        self.command([str(binary), '--dx', str(dx), '--dy', str(dy)])
        row['transport_status'] = 'command_completed'

    def coarse_move(self, dx, dy):
        """A measured wide-view correction may use at most forty vertical units."""
        self._vertical_move_limit = 40
        try:
            return self.move(dx, dy)
        finally:
            self._vertical_move_limit = 3

    def recover_pointer(self):
        retreat = getattr(self, '_pointer_retreat', None)
        if retreat is None:
            return False
        self.focus()
        self.guard(350_000_000)
        self._pointer_retreat = None
        if abs(retreat[1]) > 3:
            self.coarse_move(*retreat)
        else:
            self.move(*retreat)
        self.report['inputs'][-1]['purpose'] = 'pointer_boundary_retreat_camera_changed'
        return True

    def key(self, key):
        codes = {'Shift_L': 42, 'x': 45}
        if key not in codes:
            raise ProbeError('unsupported engagement key')
        self.focus()
        if self.key_is_down(key) or (key == 'Shift_L' and self.key_is_down('Shift_R')):
            raise ProbeError('requested key already held; refusing ownership')
        if self.relative_transport == 'xtest':
            self.focus()
            self.guard(150_000_000)
            row = dict(key=key,transport='XTest',hold_ms=120,
                       image_confirmation_required=True,status='attempted')
            self.report['inputs'].append(row)
            result = super().key(key)
            # Probe records its own completion row; keep one input event here.
            if self.report['inputs'][-1] is not row:
                self.report['inputs'].pop()
            row['status'] = 'command_completed'
            return result
        binary = ROOT/'state/cn360-tools/uinput_input'
        if not binary.is_file():
            raise ProbeError('bounded uinput key transport unavailable')
        self.focus()
        self.guard(600_000_000)
        row = dict(key=key, transport='uinput_virtual_keyboard', hold_ms=120,
                   image_confirmation_required=True, transport_status='attempted')
        self.report['inputs'].append(row)
        self.command([str(binary), '--key', str(codes[key]), '--hold-ms', '120'])
        row['transport_status'] = 'command_completed'

    def x(self, *args, cleanup=False, identity=None):
        if args[:1] == ('mousedown',):
            self.focus()
            self.guard(150_000_000)
            if pointer_state(self.identity['display'])['mask'] & 256:
                raise ProbeError('left button already held; refusing ownership')
            self.guard(150_000_000)
            self.left_owned = True
        result = super().x(*args,cleanup=cleanup,identity=identity)
        if args[:1] == ('mouseup',):
            self.left_owned = False
        return result

    def release(self):
        if self.left_owned:
            try:
                self.x('mouseup',1,cleanup=True,identity=self.identity)
            except Exception as exc:
                self.report.setdefault('cleanup_errors',[]).append(str(exc))
        super().release()
        self.report['unreleased_own_left_button'] = self.left_owned


def amber_reticle(frame, *, presence_only=False):
    """Native orange ring geometry, independent of the white caret template.

    Hough circles only propose centres. Annular support in every quadrant
    rejects orange text, ship details and isolated arcs near the reticle.
    """
    if frame.shape != (1600,2560,3):
        return None
    left,top = 900,660
    patch = frame[top:940,left:1660]
    r,g,b = (patch[...,i].astype(np.int16) for i in range(3))
    mask = ((r > 140) & (g > 90) & (r-g > 25) & (g-b > 25) & (r-b > 65)).astype(np.uint8)
    circles = cv2.HoughCircles(mask*255,cv2.HOUGH_GRADIENT,1,25,
                              param1=80,param2=9,minRadius=12,maxRadius=25)
    if circles is None:
        return None
    angles = np.arange(128)*2*np.pi/128
    matches = []
    for cx,cy,radius in circles[0]:
        # The turret ring can trail the fixed caret by hundreds of pixels
        # during traverse. Keep full annular evidence throughout this band.
        if abs(cy+top-800) > 100:
            continue
        yy_grid,xx_grid = np.ogrid[:patch.shape[0],:patch.shape[1]]
        interior = (xx_grid-cx)**2+(yy_grid-cy)**2 < (radius*.55)**2
        if mask[interior].mean() > .15:
            continue
        support = np.zeros(128,dtype=bool)
        for delta in (-2,-1,0,1,2):
            xx = np.rint(cx+(radius+delta)*np.cos(angles)).astype(int)
            yy = np.rint(cy+(radius+delta)*np.sin(angles)).astype(int)
            if xx.min() < 0 or yy.min() < 0 or xx.max() >= patch.shape[1] or yy.max() >= patch.shape[0]:
                continue
            support |= mask[yy,xx].astype(bool)
        quarters = support.reshape(4,32).mean(axis=1)
        if support.mean() >= .5 and quarters.min() >= .35:
            matches.append(dict(centre=[float(cx+left),float(cy+top)],radius_px=float(radius),
                                support=float(support.mean()),quarters=quarters.tolist(),
                                source='native_amber_ring_geometry'))
    if presence_only:
        return bool(matches)
    return matches[0] if len(matches) == 1 else None


def view_state(hud, frame):
    view = hud.get('aiming_view') or {}
    if view.get('status') == 'observed' and view.get('mode') == 'scope_ruler':
        return 'scope'
    camera = A.camera_mode_evidence(frame)
    native_ring = (hud.get('reticle') or {}).get('raw', {}).get('search') in (
        'native_green_ring', 'native_amber_ring')
    if camera.get('binocular') is False and (
            native_ring or A.green_reticle(frame) is not None or
            (hud.get('battle',{}).get('active') is True and
             amber_reticle(frame, presence_only=True))):
        return 'wide'
    return 'unknown'


def current_centre(hud, frame, state):
    point = hud.get('aim_point')
    if isinstance(point, dict) and point.get('center') is not None:
        return point['center'], 'current_white_chevron_geometry'
    return None, 'white_aim_point_unconfirmed'


def locked_same_target(lock, candidates, target_id):
    index = lock.get('detection_index')
    return (lock.get('confirmed') is True and type(index) is int and
            0 <= index < len(candidates) and target_id is not None and
            candidates[index].get('target_id') == target_id)


def gun_ready(hud):
    """Current gun-direction evidence; orange/missing means wait, not abort."""
    direction = hud.get('gun_direction') or {}
    return (direction.get('status') == 'observed' and direction.get('ready') is True
            and (direction.get('ring') or {}).get('color') == 'green'
            and (hud.get('aim_point') or {}).get('center') is not None)


def final_shot_check(hud, state, target, candidates, lock, observation, target_id, max_range_km):
    if hud.get('battle', {}).get('active') is not True or state != 'scope':
        raise ProbeError('final scoped battle image unconfirmed')
    weapon = hud.get('weapon') or {}
    if weapon.get('status') != 'observed' or weapon.get('kind') != 'gun':
        raise ProbeError('final selected main battery unconfirmed')
    if (hud.get('reload') or {}).get('ready') is not True:
        raise ProbeError('final main battery readiness unconfirmed')
    if not gun_ready(hud):
        raise ProbeError('final green gun direction at white aim point unconfirmed')
    if target.get('target_id') != target_id or target.get('range_eligible') is not True:
        raise ProbeError('final same-target distance eligibility unconfirmed')
    interval = (target.get('distance_evidence') or {}).get('interval_km')
    if (not isinstance(interval, (list,tuple)) or len(interval) != 2 or
            not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in interval) or
            interval[0] > interval[1] or
            not DEFAULT_MIN_RANGE_KM <= sum(interval)/2 <= min(max_range_km, DEFAULT_MAX_RANGE_KM)):
        raise ProbeError('final distance interval outside policy')
    if not locked_same_target(lock, candidates, target_id):
        raise ProbeError('final unique same-target UI lock unconfirmed')
    control = observation.get('control') or {}
    alignment = control.get('range_alignment') or {}
    error, tolerance = control.get('horizontal_error_px'), control.get('horizontal_tolerance_px')
    if (observation.get('status') != 'aligned' or observation.get('locked') is not True or
            control.get('status') != 'aligned' or alignment.get('contained') is not True or
            alignment.get('uncertain') is not False or
            not isinstance(error,(int,float)) or not math.isfinite(error) or
            not isinstance(tolerance,(int,float)) or not 0 <= tolerance <= LockedAimControl.X_TOLERANCE_PX or abs(error) > tolerance):
        raise ProbeError('final horizontal lead or predicted range alignment unconfirmed')


def final_shot_check_hull(hud, state, target, candidates, lock, target_id, max_range_km,
                          error, tolerance):
    """Final gate for the scoped hull-point alignment mode.

    Everything except the range-interval comparison is identical to the numeric
    path; the alignment evidence is the current-frame hull-point error, which is
    recorded so this mode can never be mistaken for a numeric range alignment.
    """
    if hud.get('battle', {}).get('active') is not True or state != 'scope':
        raise ProbeError('final scoped battle image unconfirmed')
    weapon = hud.get('weapon') or {}
    if weapon.get('status') != 'observed' or weapon.get('kind') != 'gun':
        raise ProbeError('final selected main battery unconfirmed')
    if (hud.get('reload') or {}).get('ready') is not True:
        raise ProbeError('final main battery readiness unconfirmed')
    if not gun_ready(hud):
        raise ProbeError('final green gun direction at white aim point unconfirmed')
    if target.get('target_id') != target_id or target.get('range_eligible') is not True:
        raise ProbeError('final same-target distance eligibility unconfirmed')
    interval = (target.get('distance_evidence') or {}).get('interval_km')
    if (not isinstance(interval, (list,tuple)) or len(interval) != 2 or
            not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in interval) or
            interval[0] > interval[1] or
            not DEFAULT_MIN_RANGE_KM <= sum(interval)/2 <= min(max_range_km, DEFAULT_MAX_RANGE_KM)):
        raise ProbeError('final distance interval outside policy')
    if not locked_same_target(lock, candidates, target_id):
        raise ProbeError('final unique same-target UI lock unconfirmed')
    if (not isinstance(error, (list, tuple)) or len(error) != 2 or
            not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
                    for v in error) or
            max(abs(error[0]), abs(error[1])) > tolerance):
        raise ProbeError('final scoped hull alignment unconfirmed')


def run(args, *, acquisition=None):
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(status='dry-run', inputs=[], focus_checks=[], frames=[], fired=False,
                  actual_discharge_verified=False, fire_authorized=False,
                  fire_requested=args.fire, shot_outcome='not_fired' if args.fire else 'not_requested',
                  target_source='main_view_healthbar_and_unique_name',
                  body_confirmation_claimed=False, minimap_used=False,
                  training_room_attested=bool(getattr(args,'training_room',False)),
                  training_room_evidence='operator_assertion_not_visual_proof',
                  capture_age_semantics='gst_segment_running_time;upstream_render_latency_unknown',
                  render_age_verified=False, key_hold_ms=120)
    if args.max_range_km is not None and math.isfinite(args.max_range_km):
        args.max_range_km = min(args.max_range_km, DEFAULT_MAX_RANGE_KM)
    report['shot_policy'] = dict(min_range_km=DEFAULT_MIN_RANGE_KM,
        max_range_km=args.max_range_km, range_tolerance_km=LockedAimControl.RANGE_TOLERANCE_KM,
        horizontal_tolerance_px=LockedAimControl.X_TOLERANCE_PX,
        target_budget_s=30, result_acceptance='independently_verified_main_battery_hit')
    budget = ShotBudget()
    report['shot_budget'] = budget.record(time.monotonic_ns())
    saved = []
    capture = probe = None
    handlers = {}
    scope_owned = False
    identity = CrossZoomIdentity()
    motion = LockedObservation()
    last_ns = None
    source_key = None
    previous = None
    aligned = None
    scope_pending = None
    lock_requested = False
    lock_pending = False
    coarse_gain = [None,None]
    scope_only = bool(getattr(args, 'scope_only', False))
    empty_scope_close_pending = False
    coarse_pending = None
    recovery_pixels = None
    coarse_baseline = []
    coarse_centre_source = None
    coarse_ready = None
    search_dx = 150
    candidate_waits = 0
    candidate_wait_started_ns = None
    hud_waits = 0
    hud_wait_started_ns = None

    def write_report():
        (args.out/'report.json').write_text(json.dumps(report, indent=2, default=str))

    def stop(signum, _):
        raise ProbeError(f'bounded action interrupted by signal {signum}')

    try:
        if args.fire and not args.execute:
            raise ProbeError('--fire requires --execute')
        if not args.execute and not args.inspect:
            return 0
        if not scope_only and (args.max_range_km is None or not math.isfinite(args.max_range_km) or args.max_range_km <= DEFAULT_MIN_RANGE_KM):
            raise ProbeError('explicit finite main battery maximum range above 4 km required')
        if args.execute and not getattr(args,'training_room',False):
            raise ProbeError('--execute requires explicit --training-room operator attestation')
        if not 1 <= args.steps <= 100 or not 5 <= args.seconds <= 120:
            raise ProbeError('invalid bounded steps or seconds')
        capture = LockedCapture()
        probe = AcquisitionProbe(args.seconds, report)
        probe.capture, probe.shot_budget = capture, budget
        probe.identity, probe.window = capture.identity, capture.window
        if acquisition is not None:
            seed_meta = acquisition['capture']
            require_fresh(seed_meta)
            if seed_meta.get('identity') != capture.identity or seed_meta.get('window') != capture.window:
                raise ProbeError('scan acquisition game identity changed')
            source = {k: seed_meta.get(k) for k in ('source','identity','window')}
            source_key = json.dumps([source[k] for k in ('source','identity','window')],sort_keys=True)
            identity.observe_source(source,battle_active=True)
            seed_selection = inspect_candidates(acquisition['frame'],acquisition['candidates'],
                                                max_range_km=args.max_range_km)
            previous = seed_selection.get('range_selected')
            if previous is None:
                raise ProbeError('scan enemy outside range or ambiguous; scan remains stopped')
            budget.accept(seed_meta['captured_monotonic_ns'])
            _, seed_identity = identity.identify(acquisition['frame'],acquisition['candidates'],
                selected=previous,frame_source='scan_acquisition')
            report['scan_acquisition'] = dict(capture=seed_meta,identity=seed_identity)
            if seed_identity.get('status') != 'confirmed':
                raise ProbeError('scan enemy name identity unconfirmed; scan remains stopped')
        probe.relative_transport = getattr(args,'transport','xtest')
        report['relative_and_key_transport'] = probe.relative_transport
        reader = FireHudReader()
        if args.execute:
            for sig in (signal.SIGALRM, signal.SIGINT, signal.SIGTERM):
                handlers[sig] = signal.signal(sig,stop)
            signal.setitimer(signal.ITIMER_REAL,max(.001,probe.action_deadline-time.monotonic()))
        report['status'] = 'running'
        for step in range(args.steps):
            if time.monotonic() >= probe.action_deadline:
                raise ProbeError('bounded action deadline reached')
            if not scope_only:
                budget.require(time.monotonic_ns())
            pixels, meta = capture.acquire(wait_update=True)
            last_ns = require_fresh(meta, previous_ns=last_ns)
            key = json.dumps([meta.get(k) for k in ('source','identity','window')],sort_keys=True)
            if source_key is not None and key != source_key:
                raise ProbeError('pinned capture source changed')
            source_key = key
            path = args.out/f'frame-{step:03d}.png'
            saved.append((path,pixels,meta))
            frame = np.asarray(pixels)
            if frame.shape != (1600,2560,3) or frame.dtype != np.uint8:
                raise ProbeError('requires exact native 2560x1600 RGB; no image rescaling')
            probe.evidence = meta
            hud = reader.detect(frame)
            state = view_state(hud,frame)
            centre, centre_source = current_centre(hud,frame,state)
            candidates = []
            if hud.get('battle', {}).get('active') is True:
                candidates = [dict(c,trackable=True) for c in detect_nameplates(frame,scope_observed=state=='scope')]
                candidates = [{k:v for k,v in c.items() if k not in
                               ('target_id','track_id','association_id','identity','identity_key')}
                              for c in candidates]
            if scope_only:
                guard_row = dict(frame=str(path), capture=meta, hud=hud, view=state,
                                 candidates=candidates, reticle=centre, reticle_source=centre_source)
                if args.inspect:
                    report['frames'].append(guard_row)
                    guard_row['action'] = ('scope_state_unknown_no_input' if state == 'unknown' else
                                           'not_scoped_no_input' if state == 'wide' else
                                           'empty_scope_would_exit' if not candidates else
                                           'enemy_present_would_aim')
                    report['status'] = 'inspected'
                    break
                if empty_scope_close_pending:
                    report['frames'].append(guard_row)
                    if state != 'wide':
                        raise ProbeError('empty-scope exit not confirmed by subsequent image')
                    report['status'] = 'scope_closed'
                    guard_row['action'] = 'empty_scope_exit_image_confirmed'
                    break
                if state != 'scope':
                    report['frames'].append(guard_row)
                    if state != 'wide':
                        guard_row['action'] = 'scope_state_unknown_no_input'
                        raise ProbeError('current scope state unconfirmed; no toggle requested')
                    report['status'] = 'scope_inactive'
                    guard_row['action'] = 'not_scoped_no_input'
                    break
                if hud.get('battle', {}).get('active') is not True:
                    raise ProbeError('scoped battle HUD unconfirmed')
                if not candidates:
                    report['frames'].append(guard_row)
                    # Closing an empty view is not a shot and may outlive its
                    # target budget; the original total action deadline remains.
                    probe.shot_budget = None
                    empty_scope_close_pending = True
                    probe.key('Shift_L')
                    guard_row['action'] = 'empty_scope_exit_requested'
                    continue
                report['scope_enemy_present'] = True
                if args.max_range_km is None or not math.isfinite(args.max_range_km) or args.max_range_km <= DEFAULT_MIN_RANGE_KM:
                    raise ProbeError('explicit finite main battery maximum range above 4 km required for aiming')
            initial = inspect_candidates(frame,candidates,max_range_km=args.max_range_km)
            # Start before name extraction or scope transition, even when that
            # subsequent evidence fails. Reacquisition cannot restart this clock.
            if initial.get('range_selected') is not None:
                budget.accept(last_ns)
            # Numeric evidence is from this frame, never a previous capture.
            candidates = initial['candidates']
            report['shot_budget'] = budget.record(time.monotonic_ns())
            row = dict(frame=str(path),capture=meta,hud=hud,view=state,reticle=centre,
                       reticle_source=centre_source,selection=initial,candidates=candidates)
            report['frames'].append(row)
            budget.require(time.monotonic_ns())
            if identity.observe_source(dict(source=meta.get('source'),identity=meta.get('identity'),
                                             window=meta.get('window')),
                                       battle_active=hud.get('battle',{}).get('active') is True):
                raise ProbeError('accepted target identity source lost')
            candidates, evidence = identity.identify(frame,candidates,selected=initial.get('range_selected'),
                                                       frame_source=str(path))
            row.update(identity=evidence,candidates=candidates)
            selection = inspect_candidates(frame,candidates,max_range_km=args.max_range_km,previous=previous)
            row['selection'] = selection
            target = selection.get('range_selected')
            if identity.target_id is not None:
                matches = [c for c in selection['candidates'] if c.get('target_id') == identity.target_id
                           and c.get('range_eligible') is True]
                target = matches[0] if len(matches) == 1 else None
            row['target'] = target
            lock = confirm_target(frame,candidates)
            row['lock'] = lock
            if args.inspect:
                if target is not None and centre is not None and state == 'scope':
                    row['observation'] = motion.observe(frame,hud,candidates,target,centre,meta)
                report['status'] = 'inspected'
                break
            require_fresh(meta)
            if hud.get('battle',{}).get('active') is not True or centre is None:
                # A single frame without battle HUD or caret evidence is a
                # measurement gap (reload sprites hidden, caret occluded by
                # smoke or sea), not proof the fight ended. Wait on new frames
                # with no input and the original budget; persistence still fails.
                if hud_wait_started_ns is None:
                    hud_wait_started_ns = last_ns
                if (hud_waits < CANDIDATE_WAIT_MAX_FRAMES and
                        last_ns - (hud_wait_started_ns or last_ns) <= CANDIDATE_WAIT_MAX_NS):
                    hud_waits += 1
                    row['action'] = 'wait_battle_or_reticle_evidence'
                    row['hud_waits'] = hud_waits
                    row['hud_wait_s'] = (last_ns-hud_wait_started_ns)/1e9
                    continue
                raise ProbeError('current battle HUD or reticle unconfirmed')
            hud_waits = 0
            hud_wait_started_ns = None
            if scope_pending is not None:
                scope_identity_ready = (evidence.get('status') == 'confirmed' and
                    sum(c.get('target_id') == identity.target_id for c in candidates) == 1)
                # Zoom animation can temporarily hide the name behind its lock
                # circle. Missing evidence permits waiting only, never identity
                # reuse or input; observed conflicting/competing names still fail.
                name_occluded = (len(candidates) == 1 and
                    candidates[0].get('identity_evidence', {}).get('supported') is False)
                if state != 'scope' or not scope_identity_ready:
                    if (time.monotonic_ns()-scope_pending <= 750_000_000 and
                            (scope_identity_ready or name_occluded)):
                        row['action'] = 'wait_scope_visual_response'
                        continue
                    raise ProbeError('scope request not confirmed by subsequent image')
                scope_pending = None
                motion.reset()
                aligned = coarse_pending = None
                coarse_gain = [None,None]
                coarse_baseline.clear()
                coarse_centre_source = None
                coarse_ready = None
                row['scope_transition'] = 'image_confirmed_history_reset'
            if state == 'unknown':
                raise ProbeError('current camera view unconfirmed')
            # An unreadable range must not turn a pending retreat into an
            # identity failure. Keep it pending, with no input or budget reset.
            same_name = [c for c in selection['candidates']
                         if c.get('target_id') == identity.target_id]
            if (target is None and budget.accepted_ns is not None and identity.target_id is not None
                    and evidence.get('status') == 'confirmed' and len(same_name) == 1
                    and set(same_name[0].get('reasons', [])) <= {'distance_unknown', 'hull_unconfirmed'}
                    and 'distance_unknown' in same_name[0].get('reasons', [])):
                motion.reset()
                aligned = None
                coarse_ready = None
                if coarse_pending is not None and last_ns-coarse_pending[7] > 750_000_000:
                    raise ProbeError('coarse input response unresolved after distance gap')
                coarse_baseline.clear()
                row['action'] = 'wait_same_target_distance'
                row['range_wait'] = 'current_name_confirmed; no_input; original_budget_retained'
                continue
            if getattr(probe, '_pointer_retreat', None) is not None:
                searching = budget.accepted_ns is None and identity.target_id is None
                if searching:
                    if state != 'wide' or candidates:
                        raise ProbeError('search recovery requires empty current wide view')
                    search_dx = probe._pointer_retreat[0]
                    if not search_dx or probe._pointer_retreat[1]:
                        raise ProbeError('search recovery requires horizontal retreat')
                elif target is None or evidence.get('status') != 'confirmed':
                    raise ProbeError('pointer recovery requires current same-target evidence')
                probe.recover_pointer()
                coarse_ready = None
                motion.reset()
                coarse_gain = [None,None]
                coarse_pending = aligned = None
                coarse_baseline.clear()
                recovery_pixels = meta['raw_pixel_sha256']
                row['action'] = ('search_boundary_reverse_reobserve' if searching else
                                 'pointer_boundary_retreat_reobserve')
                continue
            if recovery_pixels is not None:
                coarse_ready = None
                motion.reset()
                coarse_gain = [None,None]
                coarse_pending = aligned = None
                coarse_baseline.clear()
                if meta['raw_pixel_sha256'] == recovery_pixels:
                    row['action'] = 'wait_pointer_recovery_visual_response'
                    continue
                recovery_pixels = None
            if target is None:
                # A single frame where the name text cannot be measured at all
                # is missing evidence, not proof of a different enemy. Wait on
                # genuinely new frames, bounded, with no input and no budget
                # reset; a readable name that disagrees still fails below.
                name_unreadable = bool(candidates) and all(
                    c.get('identity_evidence', {}).get('supported') is False
                    for c in candidates)
                if name_unreadable and candidate_wait_started_ns is None:
                    candidate_wait_started_ns = last_ns
                if (name_unreadable and candidate_waits < CANDIDATE_WAIT_MAX_FRAMES and
                        last_ns - (candidate_wait_started_ns or last_ns) <= CANDIDATE_WAIT_MAX_NS):
                    candidate_waits += 1
                    row['action'] = 'wait_candidate_name_readable'
                    row['candidate_waits'] = candidate_waits
                    row['candidate_wait_s'] = (last_ns-candidate_wait_started_ns)/1e9
                    continue
                if budget.accepted_ns is not None:
                    raise ProbeError('accepted target name changed, ambiguous, lost or outside range')
                if state != 'wide':
                    raise ProbeError('scoped target identity unavailable')
                if candidates:
                    # A candidate whose distance is only unreadable on this frame
                    # is transient evidence, not proof of an out-of-range enemy.
                    # Wait for genuinely new pixels without input; every real
                    # distance/range gate still applies, and the wait is bounded.
                    unreadable = [c for c in candidates
                                  if 'distance_unknown' in (c.get('reasons') or [])]
                    if unreadable and candidate_wait_started_ns is None:
                        candidate_wait_started_ns = last_ns
                    if (unreadable and candidate_waits < CANDIDATE_WAIT_MAX_FRAMES and
                            last_ns - (candidate_wait_started_ns or last_ns) <= CANDIDATE_WAIT_MAX_NS):
                        candidate_waits += 1
                        row['action'] = 'wait_candidate_distance_readable'
                        row['candidate_waits'] = candidate_waits
                        row['candidate_wait_s'] = (last_ns-candidate_wait_started_ns)/1e9
                        continue
                    raise ProbeError('enemy candidate found but range or identity unconfirmed; scan remains stopped')
                delivered = probe.move(search_dx,0)
                if delivered is False:
                    recovery_pixels = meta['raw_pixel_sha256']
                row['action'] = ('pointer_boundary_recovery_reobserve' if delivered is False
                                 else 'search_wide_view')
                continue
            if evidence.get('status') != 'confirmed' or target.get('target_id') != identity.target_id:
                raise ProbeError('unique same-target name identity unconfirmed')
            previous = target
            candidate_waits = 0
            candidate_wait_started_ns = None
            x,y,w,h = target['nameplate_bbox']
            flight_ev = hud.get('flight_time') or {}
            aim_distance_ev = flight_ev.get('distance') or {}
            # The scoped HUD of this client prints flight time but no aim-range
            # number. Only the explicit --scoped-hull path substitutes the
            # image-derived hull point; the default path keeps waiting instead.
            scoped_hull = (state == 'scope' and bool(getattr(args, 'scoped_hull', False)) and
                           aim_distance_ev.get('status') != 'observed')
            hull_ready = False
            if state == 'wide' or scoped_hull:
                if scoped_hull:
                    hull, hull_evidence = A.hull_aim_point(frame, target['nameplate_bbox'])
                    coarse_point = [float(hull[0]), float(hull[1])]
                    anchor_kind = 'scoped_hull_point'
                else:
                    coarse_point = [x+w/2,y+h/2]
                    anchor_kind = 'healthbar_center_approach_only'
                    hull_evidence = None
            else:
                coarse_point = [x+w/2,y+h/2]
                anchor_kind = 'healthbar_center_approach_only'
                hull_evidence = None
            error = [coarse_point[i]-centre[i] for i in (0,1)]
            row['coarse'] = dict(anchor=anchor_kind,point=coarse_point,error_px=error,
                                 body_confirmation=False)
            if hull_evidence is not None:
                row['coarse']['hull_evidence'] = hull_evidence
            if lock_pending:
                if not locked_same_target(lock,candidates,identity.target_id):
                    raise ProbeError('single lock request not confirmed by subsequent image')
                lock_pending = False
                motion.reset()
                aligned = None
            if state == 'wide' or scoped_hull:
                motion.reset()
                aligned = None
                if coarse_centre_source is not None and centre_source != coarse_centre_source:
                    coarse_gain = [None,None]
                    coarse_baseline.clear()
                    if coarse_pending is not None:
                        raise ProbeError('coarse reticle reference changed during pending input')
                coarse_centre_source = centre_source
                # A wide-view plate must be close on BOTH axes before zooming.
                # This is a coarse HUD anchor, never a hull or firing solution.
                coarse_tolerance = 60 if state == 'wide' else SCOPED_HULL_TOLERANCE_PX
                near = max(abs(error[0]),abs(error[1])) <= coarse_tolerance
                row['coarse'].update(tolerance_px=coarse_tolerance, near=near)
                if not near or coarse_pending is not None:
                    coarse_ready = None
                    if coarse_pending is not None:
                        axis,units,old_error,old_hash,old_ns,drift,uncertainty,delivered_ns = coarse_pending
                        if last_ns <= delivered_ns:
                            row['action'] = 'wait_capture_after_coarse_input'
                            continue
                        if last_ns-delivered_ns > 750_000_000:
                            raise ProbeError('coarse input response unresolved after drift subtraction')
                        if meta['raw_pixel_sha256'] == old_hash:
                            row['action'] = 'wait_coarse_visual_response'
                            continue
                        elapsed = (last_ns-old_ns)/1e9
                        isolated = error[axis]-old_error-drift*elapsed
                        row['coarse']['response'] = dict(axis=axis, measured=error[axis]-old_error,
                            natural_drift_projection=drift*elapsed, isolated=isolated,
                            uncertainty_px=uncertainty*elapsed+1)
                        if (abs(isolated) <= uncertainty*elapsed+1 or
                                abs(isolated/units) < .15):
                            # Delivery may precede the visible response. Preserve the
                            # original pending move and budget; never repeat input.
                            row['action'] = 'wait_coarse_measurable_response'
                            continue
                        response = isolated/units
                        if abs(response) > 20:
                            raise ProbeError('coarse isolated gain outside measurable bounds')
                        coarse_gain[axis] = response
                        coarse_pending = None
                        coarse_baseline.clear()
                        if near:
                            row['action'] = 'wait_coarse_settled_confirmation'
                            continue
                    if coarse_baseline and meta['raw_pixel_sha256'] == coarse_baseline[-1][2]:
                        row['action'] = 'wait_coarse_baseline_new_pixels'
                        continue
                    if coarse_baseline and last_ns-coarse_baseline[-1][0] > 750_000_000:
                        coarse_baseline.clear()
                    coarse_baseline.append((last_ns,error,meta['raw_pixel_sha256']))
                    coarse_baseline[:] = coarse_baseline[-3:]
                    if len(coarse_baseline) < 3:
                        row['action'] = 'wait_coarse_no_input_baseline'
                        continue
                    axis = 0 if abs(error[0]) > coarse_tolerance else 1
                    bound = 150 if axis == 0 else (
                        40 if state == 'wide' and coarse_gain[1] is not None and
                        abs(error[1]) > coarse_tolerance else 3)
                    rates = [(b[1][axis]-a[1][axis])/((b[0]-a[0])/1e9)
                             for a,b in zip(coarse_baseline,coarse_baseline[1:])]
                    drift = sum(rates)/len(rates)
                    uncertainty = max(abs(rate-drift) for rate in rates)
                    row['coarse']['baseline'] = dict(rate_px_s=drift,
                        variation_px_s=uncertainty, semantics='local_error_drift_not_world_motion')
                    units = (20 if axis == 0 else 3) if coarse_gain[axis] is None else round(
                        np.clip(-error[axis]/coarse_gain[axis]*.5,-bound,bound))
                    if units == 0:
                        raise ProbeError('coarse correction quantized to zero')
                    row['coarse']['vertical_limit_units'] = bound if axis == 1 else 3
                    mover = probe.coarse_move if axis == 1 and bound > 3 else probe.move
                    delivered = mover(units if axis == 0 else 0,units if axis == 1 else 0)
                    if delivered is False:
                        recovery_pixels = meta['raw_pixel_sha256']
                        coarse_baseline.clear()
                        coarse_gain = [None,None]
                        coarse_pending = aligned = None
                        motion.reset()
                        row['action'] = 'pointer_boundary_recovery_reobserve'
                        continue
                    delivered_ns = time.monotonic_ns()
                    coarse_pending = (axis,units,error[axis],meta['raw_pixel_sha256'],last_ns,
                                      drift,uncertainty,delivered_ns)
                    row['coarse']['input_completed_monotonic_ns'] = delivered_ns
                    row['action'] = 'healthbar_coarse_approach'
                    continue
            if state == 'wide' or scoped_hull:
                current_coarse = dict(stamp=last_ns,pixels=meta['raw_pixel_sha256'],
                                      source=centre_source,target_id=identity.target_id)
                if (coarse_ready is None or coarse_ready['source'] != centre_source or
                        coarse_ready['target_id'] != identity.target_id or
                        not 0 < last_ns-coarse_ready['stamp'] <= 750_000_000 or
                        coarse_ready['pixels'] == current_coarse['pixels']):
                    coarse_ready = current_coarse
                    row['action'] = ('wait_second_coarse_alignment' if state == 'wide'
                                     else 'wait_second_scoped_hull_alignment')
                    continue
                row['coarse']['two_frame_confirmed'] = True
                if scoped_hull:
                    # Alignment evidence is the current-frame hull-point error;
                    # no numeric range comparison is claimed for this mode.
                    coarse_ready = None
                    coarse_pending = None
                    coarse_baseline.clear()
                    hull_ready = True
                    row['alignment_mode'] = 'scoped_hull_point'
                else:
                    probe.key('Shift_L')
                    coarse_ready = None
                    scope_owned = True
                    scope_pending = time.monotonic_ns()
                    coarse_pending = aligned = None
                    coarse_gain = [None,None]
                    motion.reset()
                    row['action'] = 'request_scope_image_confirmation_required'
                    continue
            if not locked_same_target(lock,candidates,identity.target_id):
                if lock.get('confirmed') or lock.get('state') == 'ambiguous' or lock.get('candidates'):
                    raise ProbeError('different or ambiguous UI lock; refusing target toggle')
                if lock_requested:
                    raise ProbeError('same-target lock lost after single request')
                # X can select a nearby plate; never guess between competitors.
                if len(candidates) != 1 or abs(error[0]) > 150 or abs(error[1]) > 400:
                    raise ProbeError('lock request requires unique same target near reticle')
                lock_requested = True
                probe.key('x')
                lock_pending = True
                row['action'] = 'request_lock_once'
                continue
            if hull_ready:
                row['action'] = 'two_frame_scoped_hull_alignment'
                if not args.fire:
                    report['status'] = 'prepared'
                    break
                if not gun_ready(hud) or (hud.get('reload') or {}).get('ready') is not True:
                    row['action'] = 'wait_gun_ready'
                    row['wait_reason'] = ('gun_direction_not_ready' if not gun_ready(hud)
                                          else 'reload_not_ready')
                    continue
                final_shot_check_hull(hud,state,target,candidates,lock,identity.target_id,
                                      args.max_range_km,error,coarse_tolerance)
                probe.guard(150_000_000)
                probe.focus()
                final_shot_check_hull(hud,state,target,candidates,lock,identity.target_id,
                                      args.max_range_km,error,coarse_tolerance)
                probe.guard(150_000_000)
                report['alignment_mode'] = 'scoped_hull_point'
                report['fire_authorized'] = True
                try:
                    probe.x('mousedown',1)
                    time.sleep(.12)
                finally:
                    if probe.left_owned:
                        probe.x('mouseup',1,cleanup=True)
                report['trigger_released_monotonic_ns'] = time.monotonic_ns()
                report['inputs'][-1]['status'] = 'command_completed'
                report['status'] = 'shot_requested'
                row['action'] = 'single_trigger_request_discharge_unverified'
                break
            flight = hud.get('flight_time') or {}
            distance = flight.get('distance') or {}
            if (flight.get('status') != 'observed' or distance.get('status') != 'observed'):
                # Missing numbers never license input or reuse old range. Do not
                # destroy an in-flight response: the observation/control layer
                # retains its original 750 ms gap/response deadline on recovery.
                aligned = None
                row['action'] = 'wait_current_aim_numbers'
                continue
            observation = motion.observe(frame,hud,candidates,target,centre,meta)
            row['observation'] = observation
            row['solution'] = observation.get('solution')
            if observation.get('status') == 'rejected':
                raise ProbeError(observation.get('reason') or 'locked observation rejected')
            if observation.get('status') != 'aligned':
                aligned = None
                dx,dy = observation.get('move',[0,0])
                if dx or dy:
                    try:
                        delivered = probe.move(dx,dy)
                    except Exception:
                        motion.reset()
                        raise
                    if delivered is False:
                        recovery_pixels = meta['raw_pixel_sha256']
                        motion.reset()
                        aligned = None
                        row['action'] = 'pointer_boundary_recovery_reobserve'
                        continue
                    completed_ns = time.monotonic_ns()
                    motion.input_completed(completed_ns)
                    row['input_completed_monotonic_ns'] = completed_ns
                    row['action'] = 'locked_horizontal_lead_and_range_correction'
                else:
                    row['action'] = 'wait_locked_evidence'
                continue
            current = dict(stamp=last_ns,pixels=meta['raw_pixel_sha256'],target_id=identity.target_id,
                           view_id=observation.get('view_id'))
            if (aligned is None or current['view_id'] is None or aligned['view_id'] != current['view_id'] or
                    aligned['pixels'] == current['pixels'] or aligned['stamp'] >= current['stamp'] or
                    aligned['target_id'] != current['target_id']):
                aligned = current
                row['action'] = 'wait_second_independent_alignment'
                continue
            if not args.fire:
                report['status'] = 'prepared'
                row['action'] = 'two_frame_locked_alignment'
                break
            if not gun_ready(hud) or (hud.get('reload') or {}).get('ready') is not True:
                row['action'] = 'wait_gun_ready'
                row['wait_reason'] = ('gun_direction_not_ready' if not gun_ready(hud)
                                      else 'reload_not_ready')
                # Preserve the target/budget; next frame must independently
                # revalidate identity, lock and the complete alignment solution.
                continue
            final_shot_check(hud,state,target,candidates,lock,observation,identity.target_id,args.max_range_km)
            probe.guard(150_000_000)
            probe.focus()
            # Focus can consume the age/budget or reveal a replaced game/source.
            final_shot_check(hud,state,target,candidates,lock,observation,identity.target_id,args.max_range_km)
            probe.guard(150_000_000)
            report['fire_authorized'] = True
            try:
                probe.x('mousedown',1)
                time.sleep(.12)
            finally:
                if probe.left_owned:
                    probe.x('mouseup',1,cleanup=True)
            report['trigger_released_monotonic_ns'] = time.monotonic_ns()
            report['inputs'][-1]['status'] = 'command_completed'
            report['status'] = 'shot_requested'
            row['action'] = 'single_trigger_request_discharge_unverified'
            break
        else:
            raise ProbeError('bounded aiming step budget exhausted')
    except Exception as exc:
        report.update(status='refused',reason=str(exc))
        if args.fire:
            report['shot_outcome'] = ('input_attempted_discharge_unknown' if
                report.get('trigger_requested_monotonic_ns') is not None else
                'timeout_not_fired' if 'budget exhausted' in str(exc) or 'deadline' in str(exc) else
                'rejected_not_fired')
    finally:
        if handlers:
            signal.setitimer(signal.ITIMER_REAL,0)
        if probe is not None:
            probe.release()
            # A timed-out transport may have delivered Shift. A failed guard
            # before the recorded transport attempt never grants scope ownership.
            scope_owned = scope_owned or probe.scope_input_attempted
            if scope_owned and not empty_scope_close_pending:
                try:
                    # Cleanup has the original total bound, never a new deadline.
                    probe.shot_budget = None
                    probe.action_deadline = probe.deadline
                    pixels,meta = capture.acquire(wait_update=True)
                    last_ns = require_fresh(meta,previous_ns=last_ns)
                    saved.append((args.out/'cleanup.png',pixels,meta))
                    probe.evidence = meta
                    cleanup_frame = np.asarray(pixels)
                    cleanup_hud = reader.detect(cleanup_frame)
                    if view_state(cleanup_hud,cleanup_frame) == 'scope':
                        probe.key('Shift_L')
                        report['scope_cleanup'] = 'fresh_scoped_image_exit_requested_not_image_verified'
                    else:
                        report['scope_cleanup'] = 'fresh_image_not_scope_no_toggle'
                except Exception as exc:
                    report['scope_cleanup_error'] = str(exc)
            probe.release()
            if report.get('unreleased_own_keys') or report.get('unreleased_own_left_button'):
                report.update(status='refused',reason='owned input release incomplete')
        report['shot_budget'] = budget.record(time.monotonic_ns())
        for sig,handler in handlers.items():
            signal.signal(sig,handler)
        report['frame_artifacts_complete'] = False
        write_report()
        try:
            for path,pixels,meta in saved:
                capture.save(path,pixels,meta)
            report['frame_artifacts_complete'] = True
        except Exception as exc:
            report['artifact_error'] = str(exc)
        finally:
            if capture is not None:
                try:
                    capture.close()
                except Exception as exc:
                    report['capture_close_error'] = str(exc)
            write_report()
    print(json.dumps(dict(status=report['status'],reason=report.get('reason'),
                          fired=False,shot_outcome=report['shot_outcome'])))
    return 0 if report['status'] in ('dry-run','inspected','prepared','shot_requested',
                                     'scope_closed','scope_inactive') else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--execute',action='store_true')
    modes.add_argument('--inspect',action='store_true',help='inspect one frame without focus or input')
    modes.add_argument('--dry-run',action='store_true',help='record the offline plan without capture or input')
    parser.add_argument('--training-room',action='store_true',help='attest current battle is a training room')
    parser.add_argument('--max-range-km',type=float,help='actual selected main battery maximum range')
    parser.add_argument('--steps',type=int,default=32)
    parser.add_argument('--seconds',type=float,default=30)
    parser.add_argument('--transport',choices=('xtest','uinput'),default='xtest',
                        help='relative/key transport; trigger requests always use bounded XTest')
    parser.add_argument('--scope-only', action='store_true',
                        help='act only in confirmed scope; close an empty scope, otherwise aim')
    parser.add_argument('--scoped-hull', action='store_true',
                        help='when the scoped HUD shows no aim-range number, align the reticle '
                             'to the image-derived hull point instead of a numeric range')
    parser.add_argument('--fire',action='store_true',help='request one 120ms trigger after final checks')
    args = parser.parse_args(argv)
    if not 1 <= args.steps <= 100 or not 5 <= args.seconds <= 120:
        parser.error('steps must be 1..100 and seconds 5..120')
    if args.max_range_km is not None and (not math.isfinite(args.max_range_km) or args.max_range_km <= DEFAULT_MIN_RANGE_KM):
        parser.error('max-range-km must be finite and greater than 4')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
