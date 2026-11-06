#!/home/lengbai/Downloads/wws/.venv/bin/python
"""Bloodbar coarse aim, scoped learned bow correction and bounded shot requests.

Window update age does not prove source render age. Bow candidates require
reviewed model weights and independent operational acceptance before firing.
This controller does not manufacture model acceptance: current BowRecognizer
outputs remain non-authoritative, so live fire is blocked until accepted model
and render-age producers exist. Preparation and refusal paths are runnable.
"""
import argparse
import json
import math
from pathlib import Path
import signal
import time

import cv2
import numpy as np
from PIL import Image
from wows_window_capture import WindowCapture
from probe_aim_input import Probe, ProbeError
import aim_servo as A
from fire_hud import FireHudReader
from bow_recognizer import BowRecognizer
from target_range_selection import inspect_candidates
from cross_zoom_identity import CrossZoomIdentity
from locked_target_motion import LockedTargetMotion
from locked_aim_control import LockedAimControl
from main_view_perception import estimate_camera_shift
from target_confirmation import confirm_target


class DynamicLockedAim:
    """Local translation approximation; never supplies live fire authority."""
    def __init__(self):
        self.motion = LockedTargetMotion()
        self.control = LockedAimControl()
        self.previous_frame = None
        self.previous_boxes = []
        self.previous_view = None

    def reset(self):
        self.motion.reset()
        self.control.reset()
        self.previous_frame = None
        self.previous_boxes = []
        self.previous_view = None

    def observe(self, frame, hud, candidates, target, centre, meta):
        lock = confirm_target(frame, candidates)
        flight = hud.get('flight_time') or {}
        distance = flight.get('distance') or {}
        view = hud.get('aiming_view') or {}
        target_id = target.get('target_id')
        index = lock.get('detection_index')
        locked = (lock.get('confirmed') is True and type(index) is int and
                  0 <= index < len(candidates) and target_id is not None and
                  candidates[index].get('target_id') == target_id)
        result = dict(lock=lock, locked=locked,
                      target_range_interval_km=target.get('distance_interval_km'),
                      aim_range_km=distance.get('km'), flight_s=flight.get('seconds'),
                      fire_authorized=False,
                      acquisition_semantics='drawable_update_bound; source_render_age_unverified')
        reason = None
        if not locked:
            reason = 'same_target_ui_lock_unconfirmed'
        elif (view.get('status') != 'observed' or view.get('mode') != 'scope_ruler' or
              view.get('zoom_confirmed') is not True or view.get('zoom') is None):
            reason = 'scope_zoom_identity_unconfirmed'
        elif distance.get('status') != 'observed' or flight.get('status') != 'observed':
            reason = 'current_aim_range_or_flight_time_unconfirmed'
        elif not meta.get('source') or not meta.get('identity') or not meta.get('window'):
            reason = 'capture_view_identity_missing'
        if reason:
            self.reset()
            return dict(result, status='unknown', reason=reason, move=[0, 0])
        view_id = json.dumps([meta['source'], meta['identity'], meta['window'],
                              view['mode'], view['zoom']], sort_keys=True)
        shift = None
        boxes = [c['nameplate_bbox'] for c in candidates]
        if self.previous_frame is not None and self.previous_view == view_id:
            excluded = np.zeros(frame.shape[:2], dtype=bool)
            for x, y, w, h in self.previous_boxes + boxes:
                left, right = max(0, int(x-w)), min(frame.shape[1], int(x+2*w))
                top, bottom = max(0, int(y-175)), min(frame.shape[0], int(y+6*w))
                excluded[top:bottom, left:right] = True
            shift = estimate_camera_shift(self.previous_frame, frame, exclude=excluded)
        stamp = meta.get('drawable_update_lower_bound_monotonic_ns')
        solution = self.motion.observe(
            target_id=target_id, view_id=view_id, captured_ns=stamp,
            now_ns=time.monotonic_ns(), range_interval_km=target.get('distance_interval_km'),
            aim_range_km=distance['km'], flight_s=flight['seconds'], locked=locked,
            image_shift=shift, target_x=float(target['aim_px'][0]))
        control = self.control.update(
            solution=solution, bow_x=float(target['aim_px'][0]), reticle_x=float(centre[0]),
            aim_range_km=distance['km'], captured_ns=stamp, view_id=view_id,
            pixel_hash=meta.get('raw_pixel_sha256'), image_shift=shift)
        self.previous_frame = frame.copy()
        self.previous_boxes, self.previous_view = boxes, view_id
        return dict(result, status=control['status'], reason=control.get('reason'),
                    solution=solution, control=control, image_shift=shift, move=control['move'])


class AcquisitionProbe(Probe):
    evidence = None

    def focus(self):
        super().focus()
        if self.evidence is not None:
            require_recent_acquisition(self.evidence)
        if getattr(self, 'shot_budget', None) is not None:
            self.shot_budget.require(time.monotonic_ns())

    def command(self, argv, cleanup=False, identity=None):
        budget = getattr(self, 'shot_budget', None)
        if cleanup or budget is None or budget.deadline_ns is None:
            return super().command(argv, cleanup=cleanup, identity=identity)
        budget.require(time.monotonic_ns())
        original = self.action_deadline
        self.action_deadline = min(original, budget.deadline_ns / 1e9)
        try:
            return super().command(argv, cleanup=cleanup, identity=identity)
        finally:
            self.action_deadline = original

    def key(self, key):
        if key != 'Shift_L':
            return super().key(key)
        self.focus()
        if self.key_is_down('Shift_L') or self.key_is_down('Shift_R'):
            raise ProbeError('scope key already held; refusing ownership')
        binary = A.ROOT / 'state/cn360-tools/uinput_input'
        if not binary.is_file():
            raise ProbeError('bounded uinput scope transport unavailable')
        if time.monotonic() + .6 >= self.action_deadline:
            raise ProbeError('scope pulse exceeds action deadline')
        self.focus()
        self.command([str(binary), '--key', '42', '--hold-ms', '120'])
        self.report['inputs'].append(dict(key=key, transport='uinput_virtual_keyboard',
                                         hold_ms=120, image_confirmation_required=True))


def require_recent_acquisition(meta):
    from wows_window_capture import DRAWABLE_UPDATE_SEMANTICS
    stamp = meta.get('drawable_update_lower_bound_monotonic_ns')
    if (type(stamp) is not int or
            meta.get('drawable_update_semantics') != DRAWABLE_UPDATE_SEMANTICS):
        raise ProbeError('window update age unverified; acquisition alone is insufficient')
    age = time.monotonic_ns() - stamp
    if not 0 <= age <= 750_000_000:
        raise ProbeError('window acquisition expired; render age is separately unverified')


def view_state(hud, frame):
    view = hud.get('aiming_view') or {}
    if view.get('status') == 'observed' and view.get('mode') == 'scope_ruler':
        return 'scope'
    camera = A.camera_mode_evidence(frame)
    native_ring = (hud.get('reticle') or {}).get('raw', {}).get('search') in (
        'native_green_ring', 'native_amber_ring')
    if (native_ring or A.green_reticle(frame) is not None) and camera.get('binocular') is False:
        return 'wide'
    return 'unknown'


def same_target(before, after, maximum_shift=180):
    if before is None or after is None:
        return False
    if before.get('target_id') is not None or after.get('target_id') is not None:
        return (before.get('target_id') is not None and
                before.get('target_id') == after.get('target_id'))
    a, b = before['nameplate_bbox'], after['nameplate_bbox']
    return (max(a[2], b[2]) <= 1.5 * min(a[2], b[2]) and
            math.dist((a[0]+a[2]/2, a[1]+a[3]/2),
                      (b[0]+b[2]/2, b[1]+b[3]/2)) <= maximum_shift)


class ShotBudget:
    """One accepted target owns one deadline; reacquisition cannot restart it."""
    def __init__(self):
        self.accepted_ns = None
        self.deadline_ns = None

    def accept(self, now_ns):
        if self.accepted_ns is None:
            self.accepted_ns = now_ns
            self.deadline_ns = now_ns + 5_000_000_000

    def require(self, now_ns, reserve_ns=0):
        if self.deadline_ns is not None and now_ns + reserve_ns >= self.deadline_ns:
            raise ProbeError('five-second target-to-shot budget exhausted')

    def record(self, now_ns):
        return dict(accepted_monotonic_ns=self.accepted_ns, deadline_monotonic_ns=self.deadline_ns,
                    elapsed_ns=None if self.accepted_ns is None else now_ns-self.accepted_ns,
                    limit_ns=5_000_000_000, actual_discharge_verified=False)


def final_shot_check(hud, state, bow, error, continuity):
    if hud.get('battle', {}).get('active') is not True or state != 'scope':
        raise ProbeError('final scoped battle evidence missing')
    weapon = hud.get('weapon') or {}
    if weapon.get('status') != 'observed' or weapon.get('kind') != 'gun':
        raise ProbeError('final selected main battery unconfirmed')
    if (hud.get('reload') or {}).get('ready') is not True:
        raise ProbeError('final main battery readiness unconfirmed')
    if not continuity or not np.isfinite(error).all() or np.linalg.norm(error) > 35:
        raise ProbeError('final same-target bow alignment unconfirmed')
    if not bow or not bow.get('model_sha256') or not bow.get('bow_point'):
        raise ProbeError('current learned bow identity missing')
    if bow.get('fire_authorized') is not True:
        raise ProbeError('BowNet operational acceptance unavailable; firing refused')


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(status='dry-run', inputs=[], focus_checks=[], frames=[], fired=False,
                  target_source='bloodbar_coarse_then_scoped_bow', human_target_coordinates=False,
                  render_age_verified=False, fire_authorized=False, key_hold_ms=120,
                  lead_disabled_by_user=True, lead_px=[0, 0], minimap_used=False)
    budget = ShotBudget()
    fire_requested = getattr(args, 'fire', False)
    dynamic_locked = getattr(args, 'dynamic_locked', False)
    report['dynamic_locked'] = dynamic_locked
    if dynamic_locked:
        report.update(lead_disabled_by_user=False,
                      dynamic_fire_acceptance=False,
                      motion_semantics='local_screen_translation_approximation; not_world_velocity')
    if dynamic_locked and fire_requested:
        report.update(status='refused', shot_outcome='rejected_not_fired',
                      reason='dynamic fire acceptance unavailable; preparation only')
        (args.out/'report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report))
        return 1
    report.update(fire_requested=fire_requested, shot_outcome='not_requested' if not fire_requested else 'not_fired',
                  shot_budget=budget.record(time.monotonic_ns()))
    if not args.execute and not args.inspect:
        (args.out/'report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report))
        return 0
    if args.execute and args.bow_model is None:
        report.update(status='refused', shot_outcome='rejected_not_fired',
                      reason='reviewed bow model missing; refusing before capture or input')
        (args.out/'report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report))
        return 1
    if getattr(args, 'max_range_km', None) is None:
        report.update(status='refused', shot_outcome='rejected_not_fired',
                      reason='explicit main battery maximum range required')
        (args.out/'report.json').write_text(json.dumps(report, indent=2))
        return 1
    cv2.setNumThreads(8)
    capture = WindowCapture()
    probe = AcquisitionProbe(args.seconds, report)
    probe.shot_budget = budget
    probe.identity, probe.window = capture.identity, capture.window
    probe.relative_transport = 'uinput'
    reader = FireHudReader()
    previous, pending = None, None
    gain = [None, None]
    scope_owned = False
    transition = None
    handlers = {}
    identity = CrossZoomIdentity()
    saved_frames = []
    dynamic = DynamicLockedAim() if dynamic_locked else None

    def stop(signum, _):
        raise ProbeError(f'bounded action interrupted by signal {signum}')

    bow_model = None
    try:
        if args.bow_model is not None:
            bow_model = BowRecognizer(args.bow_model)
        if args.execute:
            for sig in (signal.SIGALRM, signal.SIGINT, signal.SIGTERM):
                handlers[sig] = signal.signal(sig, stop)
            signal.setitimer(signal.ITIMER_REAL, max(.001, probe.action_deadline-time.monotonic()))
        report['status'] = 'running'
        for step in range(args.steps):
            if time.monotonic() >= probe.action_deadline:
                raise ProbeError('action deadline reached')
            budget.require(time.monotonic_ns())
            path = args.out/f'frame-{step:03d}.png'
            pixels, meta = capture.acquire(wait_update=True)
            saved_frames.append((path, pixels, meta))
            image = np.asarray(pixels)
            if image.shape != (1440, 2560, 3):
                raise ProbeError('requires exact 2560x1440 content; no image rescaling')
            frame = np.pad(image, ((80,80),(0,0),(0,0)))
            hud = reader.detect(frame)
            centre = (hud.get('reticle') or {}).get('center')
            if centre is None:
                green = A.green_reticle(frame)
                centre = None if green is None else green['centre']
            state = view_state(hud, frame)
            if state == 'scope':
                from main_view_perception import detect_nameplates
                candidates = [dict(c, trackable=True) for c in
                              detect_nameplates(frame, scope_observed=True)]
            else:
                candidates = A.find_enemy(frame, centre)
            # Range selection owns acquisition; reticle proximity never chooses
            # the immutable appearance anchor. Ignore caller/replay identities.
            candidates = [{k: v for k, v in c.items() if k not in
                           ('target_id', 'track_id', 'association_id', 'identity', 'identity_key')}
                          for c in candidates]
            selection = inspect_candidates(frame, candidates, max_range_km=args.max_range_km)
            row = dict(frame=str(path), capture=meta, hud=hud, reticle=centre,
                       candidates=candidates, view=state, gain=gain.copy())
            report['frames'].append(row)
            source_lost = identity.observe_source(
                ({'capture': meta['source'], 'identity': meta.get('identity', capture.identity),
                  'window': meta.get('window', capture.window)} if meta.get('source') else None),
                battle_active=hud.get('battle', {}).get('active') is True)
            if source_lost:
                previous, pending = None, None
                gain = [None, None]
                row['identity_reset'] = 'battle_or_capture_source_lost'
                raise ProbeError('accepted target identity source lost')
            if state != 'unknown':
                candidates, identity_evidence = identity.identify(
                    frame, candidates, selected=selection.get('range_selected') if state == 'wide' else None,
                    frame_source=str(path))
                row['identity'] = identity_evidence
                row['candidates'] = candidates
                selection = inspect_candidates(frame, candidates, max_range_km=args.max_range_km,
                                               previous=previous)
            row['selection'] = selection
            if dynamic is not None and state != 'scope':
                dynamic.reset()
            if dynamic is not None and args.inspect:
                current = selection.get('range_selected')
                if current is None or centre is None:
                    row['dynamic'] = dict(status='unknown', reason='current_target_or_reticle_missing',
                                          move=[0,0], fire_authorized=False)
                else:
                    row['dynamic'] = dynamic.observe(frame,hud,candidates,current,centre,meta)
            if args.inspect:
                report['status']='inspected'
                break
            require_recent_acquisition(meta)
            probe.evidence=meta
            budget.require(time.monotonic_ns())
            if state == 'unknown':
                row['action']='wait_current_view_evidence'
                time.sleep(.1)
                continue
            if state == 'scope':
                if not candidates:
                    raise ProbeError('scoped enemy healthbar missing or ambiguous')
                matched = [c for c in candidates if c.get('target_id') == identity.target_id
                           and identity.target_id is not None]
                if len(matched) != 1:
                    raise ProbeError('cross-zoom target identity unconfirmed; appearance missing or ambiguous')
                target = matched[0]
                if bow_model is None:
                    raise ProbeError('bow model unavailable; no scoped geometry fallback')
                bow = bow_model.detect(frame, target['nameplate_bbox'])
                row['bow'] = bow
                if bow is None:
                    raise ProbeError('current scoped bow unconfirmed; refusing correction')
                target = dict(target, aim_px=bow['bow_point'])
                if pending is not None and previous is not None and previous.get('aim_stage') != 'bow':
                    pending = None
                target['aim_stage'] = 'bow'
            # BowNet only inspects the matched original plate. All other plates
            # remain in the range report and in the appearance ambiguity check.
            if state == 'scope':
                current = dict(target, bow_confirmation={'confirmed': True, 'current': True})
                candidates = [current if c.get('target_id') == identity.target_id else c for c in candidates]
                selection = inspect_candidates(frame, candidates, max_range_km=args.max_range_km,
                                               previous=previous)
            row['selection'] = selection
            chosen = selection['selected'] if state == 'scope' else selection.get('range_selected')
            if identity.target_id is not None:
                matches = [c for c in selection['candidates'] if c.get('target_id') == identity.target_id
                           and c['eligible' if state == 'scope' else 'range_eligible']]
                target = matches[0] if len(matches) == 1 else None
                if target is None:
                    raise ProbeError('accepted target identity changed, lost, outside range or ambiguous')
            else:
                target = chosen
                if target is not None:
                    raise ProbeError('cross-zoom target identity unconfirmed; anchor name text unavailable')
            if target is not None:
                budget.accept(meta['drawable_update_lower_bound_monotonic_ns'])
                report['shot_budget'] = budget.record(time.monotonic_ns())
            elif fire_requested and budget.accepted_ns is not None:
                raise ProbeError('accepted target lost, outside range or ambiguous')
            if transition is not None:
                if state != transition:
                    raise ProbeError('scope toggle not confirmed by subsequent image')
                transition=None
            if hud['battle']['active'] is not True or centre is None:
                raise ProbeError('current battle HUD or reticle unconfirmed')
            if target is None:
                pending, previous = None, None
                gain=[None,None]
                probe.move(150,0)
                row['action']='search_wide_view'
                time.sleep(.25)
                continue
            error = np.array(target['aim_px'])-np.array(centre)
            row['error_px']=error.tolist()
            if dynamic is not None and state == 'scope':
                pending = None
                row['dynamic'] = dynamic.observe(frame, hud, candidates, target, centre, meta)
                previous = target
                if row['dynamic']['status'] == 'aligned':
                    row['action'] = 'dynamic_locked_aligned_preparation_only'
                    report['status'] = 'prepared'
                    break
                dx, dy = row['dynamic']['move']
                if dx or dy:
                    require_recent_acquisition(meta)
                    budget.require(time.monotonic_ns())
                    probe.move(int(dx), int(dy))
                    row['action'] = 'dynamic_locked_correction'
                else:
                    row['action'] = 'wait_dynamic_locked_evidence'
                time.sleep(.1)
                continue
            if pending is not None:
                axis, units, old_error, old_target, old_pixels = pending
                if meta.get('raw_pixel_sha256') == old_pixels:
                    row['action']='wait_pending_visual_response'
                    row['pending_move']=[axis, units]
                    time.sleep(.1)
                    continue
                if same_target(old_target, target):
                    response=(error[axis]-old_error)/units
                    row['observed_response']=float(response)
                    if .15 <= abs(response) <= 20:
                        gain[axis]=float(response)
                else:
                    gain=[None,None]
                    row['calibration_rejected']='target_continuity_unconfirmed'
                pending=None
            prior_target = previous
            previous=target
            target['aligned'] = bool(np.isfinite(error).all() and np.linalg.norm(error) <= 35)
            if np.linalg.norm(error) <= 35:
                if state == 'wide':
                    probe.key('Shift_L')
                    scope_owned=True
                    transition='scope'
                    gain=[None,None]
                    if dynamic is not None:
                        dynamic.reset()
                    pending = None
                    previous = dict(target, aim_stage='zoom_transition')
                    row['action']='request_scope'
                    time.sleep(.3)
                    continue
                if not fire_requested:
                    row['action']='scoped_bow_aligned'
                    report['status']='prepared'
                    break
                if (prior_target is None or prior_target.get('aim_stage') != 'bow' or
                        prior_target.get('aligned') is not True or
                        meta.get('raw_pixel_sha256') == report.get('aligned_pixels')):
                    report['aligned_pixels'] = meta.get('raw_pixel_sha256')
                    row['action'] = 'wait_final_bow_confirmation'
                    continue
                final_shot_check(hud, state, row.get('bow'), error, same_target(prior_target, target))
                from target_confirmation import confirm_target
                row['lock'] = confirm_target(frame, [target])
                if row['lock'].get('confirmed') is not True:
                    raise ProbeError('final unique target lock unconfirmed')
                if meta.get('render_age_verified') is not True:
                    raise ProbeError('source render age unverified; firing refused')
                require_recent_acquisition(meta)
                probe.focus()
                budget.require(time.monotonic_ns(), reserve_ns=150_000_000)
                report['fire_authorized'] = True
                report['shot_outcome'] = 'input_attempted_discharge_unverified'
                report['trigger_requested_monotonic_ns'] = time.monotonic_ns()
                try:
                    probe.x('mousedown', 1)
                    time.sleep(.12)
                finally:
                    probe.x('mouseup', 1, cleanup=True)
                report['trigger_released_monotonic_ns'] = time.monotonic_ns()
                report['shot_budget'] = budget.record(time.monotonic_ns())
                report['status'] = 'shot_requested'
                row['action'] = 'shot_input_delivered'
                break
            axis = 0 if abs(error[0]) > 25 else 1
            units = (20 if axis==0 else 3) if gain[axis] is None else round(
                np.clip(-error[axis]/gain[axis]*.5, -150 if axis==0 else -3, 150 if axis==0 else 3))
            if not units:
                raise ProbeError('correction quantized to zero before alignment')
            probe.move(int(units) if axis==0 else 0, int(units) if axis==1 else 0)
            pending=(axis,units,float(error[axis]),target,meta.get('raw_pixel_sha256'))
            row['action']='local_visual_correction'
            time.sleep(.3)
        else:
            raise ProbeError('local coarse-aim step budget exhausted')
    except Exception as exc:
        identity.reset()
        report.update(status='refused', reason=str(exc))
        if fire_requested:
            if report.get('trigger_requested_monotonic_ns') is not None:
                report['shot_outcome'] = 'input_attempted_discharge_unknown'
            else:
                report['shot_outcome'] = ('timeout_not_fired' if 'budget exhausted' in str(exc)
                                          else 'rejected_not_fired')
    finally:
        if handlers:
            signal.setitimer(signal.ITIMER_REAL,0)
        # Reserve bounds scope restoration too. Never use stale scene evidence
        # for cleanup, and never restore a scope state that we did not change.
        if scope_owned and report['status']!='prepared':
            try:
                probe.evidence=None
                probe.shot_budget = None
                probe.action_deadline=probe.deadline
                probe.key('Shift_L')
                report['scope_cleanup']='exit_toggle_delivered; image_confirmation_pending'
            except Exception as exc:
                report['scope_cleanup_error']=str(exc)
        probe.release()
        report['shot_budget'] = budget.record(time.monotonic_ns())
        for sig, handler in handlers.items():
            signal.signal(sig,handler)
        # Input outcome must survive termination during potentially slow PNG saves.
        report['frame_artifacts_complete'] = False
        (args.out/'report.json').write_text(json.dumps(report,indent=2,default=str))
        for path, pixels, meta in saved_frames:
            capture.save(path, pixels, meta)
        report['frame_artifacts_complete'] = True
        (args.out/'report.json').write_text(json.dumps(report,indent=2,default=str))
    print(json.dumps(dict(status=report['status'],reason=report.get('reason'),frames=len(report['frames']),
                          fired=report['fired'], shot_outcome=report['shot_outcome'])))
    return 0 if report['status'] in ('prepared','inspected','shot_requested') else 1


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--execute',action='store_true')
    mode.add_argument('--inspect',action='store_true',help='read one current frame without focus or input')
    p.add_argument('--seconds',type=float,default=90)
    p.add_argument('--steps',type=int,default=32)
    p.add_argument('--bow-model',type=Path,help='reviewed BowNet checkpoint; required for input')
    p.add_argument('--max-range-km',type=float,help='actual selected main battery maximum range; required for capture/input')
    p.add_argument('--fire',action='store_true',help='request one shot after final same-target confirmation within five seconds')
    p.add_argument('--dynamic-locked', action='store_true',
                   help='prepare locked horizontal lead and predicted range; no live fire authority')
    args=p.parse_args(argv)
    if not 1<=args.steps<=100 or not 5<=args.seconds<=120:
        p.error('invalid bounds')
    if args.max_range_km is not None and (not math.isfinite(args.max_range_km) or args.max_range_km <= 8):
        p.error('max-range-km must be finite and greater than 8')
    return run(args)


if __name__=='__main__':
    raise SystemExit(main())
