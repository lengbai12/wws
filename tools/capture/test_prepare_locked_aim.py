from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import prepare_locked_aim as P


class LockedControllerBehavior(unittest.TestCase):
    @staticmethod
    def scene(view='scope', locked=True, **changes):
        result = dict(view=view, locked=locked, label='XKnight_Alpha', age_ns=20_000_000,
                      advance_ns=100_000_000, observation='aligned', focus=True,
                      candidates=[dict(nameplate_bbox=[1230,750,100,15],trackable=True,
                                       distance_interval_km=[10.25,10.35])])
        result.update(changes)
        return result

    def run_scenes(self, scenes, *, fire=True, inspect=False, execute=True,
                   training_room=True, max_range_km=15, focus_delay_ns=0, save_failure=False,
                   scope_only=False, acquisition=False, scoped_hull=False):
        clock = {'ns':2_000_000_000}
        current = {'scene':None,'index':0}
        events = []
        original_probe = P.AcquisitionProbe

        class Capture:
            identity = dict(pid=123,display=':test',start_ticks='test')
            window = 'test-window'

            def check(self):
                scene = current['scene'] or {}
                if scene.get('check_failure'):
                    raise P.ProbeError('pinned capture source changed')

            def acquire(self, **kwargs):
                if current['index'] >= len(scenes):
                    raise P.ProbeError('no fresh cleanup image available')
                index = current['index']
                scene = scenes[index]
                current.update(scene=scene,index=index+1)
                clock['ns'] += scene['advance_ns']
                image = np.zeros((1600,2560,3),np.uint8)
                image[100,100] = [index%255,20,30]
                for candidate in scene['candidates']:
                    x,y,w,h = candidate['nameplate_bbox']
                    label = candidate.get('label',scene['label'])
                    if label:
                        cv2.putText(image,label,(x+w//2-60,y-8),cv2.FONT_HERSHEY_SIMPLEX,
                                    .55,(245,245,245),1,cv2.LINE_AA)
                return image, dict(source=scene.get('source','test_pipeline'),
                    identity=scene.get('identity',Capture.identity),window='test-window',
                    captured_monotonic_ns=scene.get('stamp',clock['ns']-scene['age_ns']),
                    raw_pixel_sha256=scene.get('pixels',hashlib.sha256(image.tobytes()).hexdigest()),
                    capture_age_semantics='gst_segment_running_time;upstream_render_latency_unknown')

            def save(self,path,pixels,meta):
                events.append(('save',str(path)))
                self_case.assertIn(('release',),events)
                self_case.assertTrue((path.parent/'report.json').is_file())
                if save_failure:
                    raise OSError('PNG encoding failed')

            def close(self):
                events.append(('close',))

        class Reader:
            def detect(self,frame):
                scene = current['scene']
                return dict(battle=dict(active=scene.get('battle',True)),
                            weapon=dict(status='observed',kind=scene.get('weapon','gun')),
                            reload=dict(ready=scene.get('ready',True)),
                            reticle=dict(center=[1280,800]),
                            aim_point=dict(center=[1280,800],source='current_white_chevron_geometry'),
                            gun_direction=scene.get('gun_direction',dict(status='observed',ready=True,
                                               ring=dict(color='green',center=[1280,800]),alignment_tolerance_px=30)),
                            aiming_view=dict(status='observed',mode='scope_ruler' if scene['view']=='scope' else 'wide'),
                            flight_time=dict(status=scene.get('flight_status','observed'),seconds=6,
                                             distance=dict(status=scene.get('aim_range_status','observed'),km=10.3)))

        class Motion:
            def reset(self):
                events.append(('motion_reset',))

            def input_completed(self, stamp):
                events.append(('input_completed', stamp))

            def observe(self,*args):
                scene = current['scene']
                status = scene['observation']
                return dict(status=status,reason='motion_evidence_rejected',
                            locked=scene['locked'],move=scene.get('move',[0,0]),
                            fire_authorized=False,view_id=scene.get('view_id','scope-epoch-one'),
                            solution=dict(predicted_range_interval_km=[10.25,10.35]),
                            control=dict(status=status,horizontal_error_px=scene.get('horizontal_error',0),
                                         horizontal_tolerance_px=P.LockedAimControl.X_TOLERANCE_PX,
                                         range_alignment=dict(contained=scene.get('contained',True),
                                                              uncertain=scene.get('uncertain',False))))

        class Input(original_probe):
            def move(self, dx, dy):
                if current['scene'].get('recovered'):
                    events.append(('boundary_recovery',dx,dy))
                    if current['scene'].get('deferred'):
                        self._pointer_retreat = [-dx,-dy]
                    return False
                return super().move(dx,dy)

            def release(self):
                events.append(('release',))
                super().release()

        def focus(probe):
            events.append(('focus',))
            clock['ns'] += focus_delay_ns
            if not current['scene']['focus']:
                raise P.ProbeError('game window does not have input focus')

        def command(probe,argv,cleanup=False,identity=None):
            events.append(('command',tuple(argv),cleanup))
            clock['ns'] += 10_000_000
            return ''

        def lock(frame,candidates):
            scene = current['scene']
            if scene.get('ambiguous_lock'):
                return dict(confirmed=False,state='ambiguous',candidates=[{},{}])
            return dict(confirmed=scene['locked'],state='confirmed' if scene['locked'] else 'unknown',
                        detection_index=scene.get('lock_index',0) if scene['locked'] else None,
                        candidates=[])

        self_case = self
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            args = SimpleNamespace(out=Path(directory)/'run',fire=fire,execute=execute,inspect=inspect,
                                   training_room=training_room,max_range_km=max_range_km,
                                   steps=max(1,len(scenes)),seconds=30,transport='uinput',
                                   scope_only=scope_only, scoped_hull=scoped_hull)
            for target,name,value in ((P,'LockedCapture',Capture),(P,'AcquisitionProbe',Input),
                    (P,'FireHudReader',Reader),(P,'LockedObservation',Motion),
                    (P,'view_state',lambda *a:current['scene']['view']),
                    (P,'current_centre',lambda *a:(current['scene'].get('centre',[1280,800]),
                        current['scene'].get('centre_source','test_fixed_reticle'))),
                    (P,'detect_nameplates',lambda *a,**kw:current['scene']['candidates']),
                    (P,'confirm_target',lock),(P,'pointer_state',lambda *a:dict(mask=0)),
                    (P.Probe,'focus',focus),(P.Probe,'command',command),
                    (P.AcquisitionProbe,'key_is_down',lambda *a:False),
                    (P.time,'monotonic_ns',lambda:clock['ns']),
                    (P.time,'monotonic',lambda:clock['ns']/1e9),
                    (P.time,'sleep',lambda seconds:clock.update(ns=clock['ns']+int(seconds*1e9))),
                    (P.Path,'is_file',lambda *a:True),
                    (P.signal,'signal',lambda *a:None),(P.signal,'setitimer',lambda *a:None),
                    (P.A,'hull_aim_point',lambda frame,box:(current['scene'].get('hull',[1280,800]),
                                                            dict(source='test_scoped_hull')))):
                stack.enter_context(patch.object(target,name,value))
            stack.enter_context(patch('builtins.print'))
            seed = None
            if acquisition:
                seed_frame, seed_meta = Capture().acquire()
                seed = dict(frame=seed_frame,capture=seed_meta,
                            candidates=current['scene']['candidates'])
            code = P.run(args, acquisition=seed)
            report = json.loads((args.out/'report.json').read_text())
        return code,report,events

    @staticmethod
    def commands(events,kind):
        if kind == 'x':
            return [e for e in events if e[0]=='command' and '--key' in e[1] and '45' in e[1]]
        if kind == 'shift':
            return [e for e in events if e[0]=='command' and '--key' in e[1] and '42' in e[1]]
        return [e for e in events if e[0]=='command' and kind in e[1]]

    def test_orange_missing_or_ambiguous_gun_direction_never_triggers(self):
        for direction in (None, dict(status='observed',ready=False,ring=dict(color='orange')),
                          dict(status='unknown',ready=True,ring=dict(color='green')),
                          dict(status='observed',ready=False,alignment_tolerance_px=30,
                               ring=dict(color='green',center=[1380,800]))):
            code, report, events = self.run_scenes([
                self.scene(),self.scene(gun_direction=direction)])
            self.assertEqual(code,1)
            self.assertEqual(report['frames'][-1]['action'],'wait_gun_ready')
            self.assertFalse(self.commands(events,'mousedown'))

    def test_transient_orange_or_reload_wait_recovers_without_restart_or_motion(self):
        for change in (dict(gun_direction=dict(status='observed',ready=False,ring=dict(color='orange'))),
                       dict(ready=False)):
            code, report, events = self.run_scenes([
                self.scene(),self.scene(**change),self.scene()])
            self.assertEqual(code,0)
            self.assertEqual(report['frames'][1]['action'],'wait_gun_ready')
            self.assertEqual(len(self.commands(events,'mousedown')),1)
            self.assertFalse(self.commands(events,'--dx'))
            self.assertEqual(report['shot_budget']['accepted_monotonic_ns'],
                             report['frames'][0]['capture']['captured_monotonic_ns'])

    def test_gun_wait_does_not_bridge_changed_target_or_budget(self):
        waiting=dict(gun_direction=dict(status='observed',ready=False,ring=dict(color='orange')))
        for last in (self.scene(label='Other_Player'),self.scene(advance_ns=30_000_000_000)):
            code, report, events = self.run_scenes([self.scene(),self.scene(**waiting),last])
            self.assertEqual(code,1)
            self.assertFalse(self.commands(events,'mousedown'))

    def test_visible_enemy_with_out_of_range_distance_stops_scan_immediately(self):
        # An observed distance outside the engagement band is a decision, not
        # missing evidence: the scan must stop without moving or waiting.
        for interval in ([3.85, 3.95], [15.95, 16.05]):
            candidate = dict(nameplate_bbox=[1230,750,100,15], trackable=True,
                             distance_interval_km=interval)
            code, report, events = self.run_scenes([
                self.scene(view='wide', candidates=[candidate]),
                self.scene(view='wide', candidates=[])], fire=False)
            self.assertEqual(code, 1)
            self.assertIn('scan remains stopped', report['reason'])
            self.assertEqual(len(report['frames']), 1)
            self.assertFalse([e for e in events if e[0] in ('command', 'focus')])

    def test_same_name_unknown_distance_waits_without_input_then_needs_two_alignments(self):
        missing = dict(nameplate_bbox=[1230,750,100,15],trackable=True,
                       distance_interval_km=None)
        code, report, events = self.run_scenes([
            self.scene(), self.scene(candidates=[missing]), self.scene(), self.scene()])
        self.assertEqual(code, 0)
        self.assertEqual(report['frames'][1]['action'], 'wait_same_target_distance')
        self.assertIsNone(report['frames'][1]['target'])
        self.assertEqual(report['frames'][2]['action'], 'wait_second_independent_alignment')
        self.assertEqual(len(self.commands(events, 'mousedown')), 1)
        self.assertEqual(report['shot_budget']['accepted_monotonic_ns'],
                         report['frames'][0]['capture']['captured_monotonic_ns'])
        self.assertFalse(self.commands(events, '--dx'))

    def test_unknown_distance_never_fires_or_restarts_target_budget(self):
        missing = dict(nameplate_bbox=[1230,750,100,15],trackable=True,
                       distance_interval_km=None)
        code, report, events = self.run_scenes([
            self.scene(), self.scene(candidates=[missing]),
            self.scene(candidates=[missing],advance_ns=30_000_000_000)])
        self.assertEqual(code, 1)
        self.assertIn('budget', report['reason'])
        self.assertFalse(self.commands(events, 'mousedown'))
        self.assertFalse(self.commands(events, '--dx'))

    def test_same_name_observed_outside_range_still_stops_immediately(self):
        for interval in ([3.85,3.95], [15.95,16.05]):
            row = dict(nameplate_bbox=[1230,750,100,15],trackable=True,
                       distance_interval_km=interval)
            code, report, events = self.run_scenes([self.scene(), self.scene(candidates=[row])])
            self.assertEqual(code, 1)
            self.assertIn('outside range', report['reason'])
            self.assertNotEqual(report['frames'][-1].get('action'), 'wait_same_target_distance')
            self.assertFalse(self.commands(events, 'mousedown'))

    def test_unknown_distance_different_name_or_occluded_target_cannot_wait(self):
        for change in ({'label':'Other_Enemy'}, {'body_status':'occluded_or_unresolved'},
                       {'association_ambiguous':True}):
            row = dict(nameplate_bbox=[1230,750,100,15],trackable=True,
                       distance_interval_km=None, **change)
            code, report, events = self.run_scenes([self.scene(),self.scene(candidates=[row])])
            self.assertEqual(code, 1)
            self.assertNotEqual(report['frames'][-1].get('action'), 'wait_same_target_distance')
            self.assertFalse(self.commands(events, 'mousedown'))

    def test_scope_guard_confirmed_wide_sends_no_input(self):
        code, report, events = self.run_scenes([self.scene(view='wide')],
            fire=False, scope_only=True, max_range_km=None)
        self.assertEqual(code, 0)
        self.assertEqual(report['status'], 'scope_inactive')
        self.assertFalse([e for e in events if e[0] in ('command', 'focus')])

    def test_scope_guard_unknown_refuses_without_claiming_exit_or_toggling(self):
        code, report, events = self.run_scenes([self.scene(view='unknown')],
            fire=False, scope_only=True, max_range_km=None)
        self.assertEqual(code, 1)
        self.assertEqual(report['status'], 'refused')
        self.assertIn('scope state unconfirmed', report['reason'])
        self.assertFalse([e for e in events if e[0] in ('command', 'focus')])

    def test_scope_guard_closes_empty_scope_once_and_confirms_wide(self):
        code, report, events = self.run_scenes([
            self.scene(candidates=[]), self.scene(view='wide', candidates=[])],
            fire=False, scope_only=True, max_range_km=None)
        self.assertEqual(code, 0)
        self.assertEqual(report['status'], 'scope_closed')
        self.assertEqual(len(self.commands(events, 'shift')), 1)
        self.assertFalse(self.commands(events, 'mousedown'))

    def test_scope_guard_never_repeats_unconfirmed_close(self):
        code, report, events = self.run_scenes([
            self.scene(candidates=[]), self.scene(candidates=[])],
            fire=False, scope_only=True)
        self.assertEqual(code, 1)
        self.assertIn('exit not confirmed', report['reason'])
        self.assertEqual(len(self.commands(events, 'shift')), 1)

    def test_scope_guard_enemy_continues_existing_aim_without_shooting(self):
        code, report, events = self.run_scenes([self.scene(), self.scene()],
                                             fire=False, scope_only=True)
        self.assertEqual(code, 0)
        self.assertEqual(report['status'], 'prepared')
        self.assertTrue(report['scope_enemy_present'])
        self.assertFalse(self.commands(events, 'mousedown'))
        self.assertFalse(self.commands(events, 'shift'))

    def test_scope_guard_inspect_reports_empty_without_toggle(self):
        code, report, events = self.run_scenes([self.scene(candidates=[])],
            fire=False, scope_only=True, inspect=True, execute=False, max_range_km=None)
        self.assertEqual(code, 0)
        self.assertEqual(report['frames'][0]['action'], 'empty_scope_would_exit')
        self.assertFalse([e for e in events if e[0] in ('command', 'focus')])

    def test_scope_guard_ignores_menu_healthbar_lookalikes(self):
        code, report, events = self.run_scenes([self.scene(battle=False, view='unknown')],
            fire=False, scope_only=True, inspect=True, execute=False, max_range_km=None)
        self.assertEqual(code, 0)
        self.assertEqual(report['frames'][0]['candidates'], [])
        self.assertEqual(report['frames'][0]['action'], 'scope_state_unknown_no_input')
        self.assertFalse([e for e in events if e[0] in ('command', 'focus')])

    def test_scan_acquisition_loss_refuses_without_resuming_search(self):
        code, report, events = self.run_scenes([
            self.scene(view='wide'), self.scene(view='wide', candidates=[])],
            fire=False, acquisition=True)
        self.assertEqual(code, 1)
        self.assertIn('accepted target', report['reason'])
        self.assertFalse([e for e in events if e[0] == 'command'])
        self.assertIsNotNone(report['shot_budget']['accepted_monotonic_ns'])

    def test_scan_acquisition_keeps_original_clock_and_prepares_same_target(self):
        code, report, events = self.run_scenes([self.scene(),self.scene(),self.scene()],
                                             fire=False, acquisition=True)
        self.assertEqual(code, 0)
        self.assertEqual(report['status'], 'prepared')
        self.assertEqual(report['shot_budget']['accepted_monotonic_ns'],
                         report['scan_acquisition']['capture']['captured_monotonic_ns'])
        self.assertEqual(report['frames'][0]['target']['target_id'],
                         report['scan_acquisition']['identity']['target_id'])

    def test_empty_search_boundary_reverses_then_accepts_target(self):
        code, report, events = self.run_scenes([
            self.scene(view='wide', candidates=[], recovered=True, deferred=True),
            self.scene(view='wide', candidates=[], pixels='retreat'),
            self.scene(view='wide', candidates=[], pixels='retreat'),
            self.scene(view='wide', candidates=[]),
            self.scene(), self.scene()], fire=False)
        self.assertEqual(code, 0)
        self.assertEqual([r['action'] for r in report['frames']], [
            'pointer_boundary_recovery_reobserve', 'search_boundary_reverse_reobserve',
            'wait_pointer_recovery_visual_response', 'search_wide_view',
            'wait_second_independent_alignment', 'two_frame_locked_alignment'])
        moves = [e[1] for e in self.commands(events, '--dx')]
        self.assertEqual([m[m.index('--dx')+1] for m in moves], ['-150', '-150'])
        self.assertFalse(self.commands(events, 'mousedown'))

    def test_search_recovery_stops_when_unresolved_enemy_appears(self):
        code, report, events = self.run_scenes([
            self.scene(view='wide', candidates=[], recovered=True, deferred=True),
            self.scene(view='wide', label='', candidates=[dict(
                nameplate_bbox=[1230,750,100,15], trackable=True)])], fire=False)
        self.assertEqual(code, 1)
        self.assertIn('empty current wide view', report['reason'])
        self.assertFalse(self.commands(events, '--dx'))
        self.assertFalse(self.commands(events, 'mousedown'))

    def test_scoped_hull_mode_aligns_without_numeric_aim_range_and_fires(self):
        # The scoped HUD may print flight time only. With --scoped-hull the
        # vertical loop uses the image-derived hull point instead of a numeric
        # range; the numeric path stays the default and unchanged.
        near = self.scene(view='scope', aim_range_status='unknown', hull=[1300, 812])
        code, report, events = self.run_scenes([near, self.scene(
            view='scope', aim_range_status='unknown', hull=[1300, 812])],
            fire=True, scoped_hull=True)
        self.assertEqual(code, 0)
        self.assertEqual(report['status'], 'shot_requested')
        self.assertEqual(report['alignment_mode'], 'scoped_hull_point')
        self.assertEqual(len(self.commands(events, 'mousedown')), 1)
        self.assertEqual(report['frames'][0]['coarse']['anchor'], 'scoped_hull_point')
        self.assertEqual(report['frames'][0]['action'], 'wait_second_scoped_hull_alignment')

    def test_scoped_hull_mode_never_fires_from_one_frame_or_far_hull_point(self):
        code, report, events = self.run_scenes([
            self.scene(view='scope', aim_range_status='unknown', hull=[1300, 812]),
            self.scene(view='scope', aim_range_status='unknown', hull=[1500, 900]),
            self.scene(view='scope', aim_range_status='unknown', hull=[1500, 900])],
            fire=True, scoped_hull=True)
        self.assertEqual(code, 1)
        self.assertFalse(self.commands(events, 'mousedown'))
        self.assertNotEqual(report.get('status'), 'shot_requested')

    def test_default_path_still_waits_without_numeric_aim_range(self):
        code, report, events = self.run_scenes([
            self.scene(view='scope', aim_range_status='unknown', hull=[1280, 804])],
            fire=True, scoped_hull=False)
        self.assertEqual(code, 1)
        self.assertEqual(report['frames'][0]['action'], 'wait_current_aim_numbers')
        self.assertFalse(self.commands(events, 'mousedown'))

    def test_transient_missing_battle_or_reticle_waits_then_continues(self):
        # A caret that is momentarily unreadable (smoke, sea backdrop) is a
        # measurement gap, not a lost battle: wait without input, keep budget.
        code, report, events = self.run_scenes([
            self.scene(view='wide'), self.scene(view='wide', centre=None),
            self.scene(view='wide'), self.scene(view='wide')], fire=False)
        self.assertEqual([r.get('action') for r in report['frames'][:3]],
                         ['wait_second_coarse_alignment',
                          'wait_battle_or_reticle_evidence',
                          'request_scope_image_confirmation_required'])
        self.assertEqual(report['frames'][1]['hud_waits'], 1)
        self.assertEqual(report['shot_budget']['accepted_monotonic_ns'],
                         report['frames'][0]['capture']['captured_monotonic_ns'])
        self.assertFalse(self.commands(events, '--dx'))
        self.assertFalse(self.commands(events, 'mousedown'))

    def test_missing_battle_evidence_still_fails_when_it_persists(self):
        code, report, events = self.run_scenes([
            self.scene(view='wide', centre=None, advance_ns=2_000_000_000)
            for _ in range(13)], fire=False)
        self.assertEqual(code, 1)
        self.assertIn('battle HUD or reticle unconfirmed', report['reason'])
        waits = [r for r in report['frames'] if r.get('action') == 'wait_battle_or_reticle_evidence']
        self.assertEqual(len(waits), 11)
        self.assertFalse(self.commands(events, '--dx'))
        self.assertFalse(self.commands(events, 'mousedown'))

    def test_transient_missing_name_text_waits_instead_of_declaring_target_lost(self):
        # The accepted enemy's name becomes unmeasurable for one frame (no
        # native text strokes at all) and is readable again afterwards: keep the
        # same target, do not reset the budget, and send no input. Identity is
        # the real consumer here; an undrawn name is genuine missing evidence.
        code, report, events = self.run_scenes([
            self.scene(view='wide'), self.scene(view='wide', label=''),
            self.scene(view='wide'), self.scene(view='wide')], fire=False)
        self.assertEqual([r.get('action') for r in report['frames'][:3]],
                         ['wait_second_coarse_alignment',
                          'wait_candidate_name_readable',
                          'request_scope_image_confirmation_required'])
        self.assertEqual(report['frames'][1]['identity']['status'], 'unconfirmed')
        self.assertEqual(report['frames'][1]['candidate_waits'], 1)
        self.assertEqual(report['frames'][2]['identity']['status'], 'confirmed')
        self.assertEqual(report['shot_budget']['accepted_monotonic_ns'],
                         report['frames'][0]['capture']['captured_monotonic_ns'])
        self.assertFalse(self.commands(events, '--dx'))
        self.assertFalse(self.commands(events, 'mousedown'))

    def test_transient_unreadable_distance_waits_without_input_then_accepts(self):
        unreadable = [dict(nameplate_bbox=[1230,750,100,15], trackable=False,
                           distance_interval_km=None, reasons=['distance_unknown'])]
        code, report, events = self.run_scenes([
            self.scene(view='wide', candidates=unreadable),
            self.scene(view='wide', candidates=unreadable),
            self.scene(view='wide'), self.scene(view='wide')], fire=False)
        self.assertEqual([r['action'] for r in report['frames'][:2]],
                         ['wait_candidate_distance_readable'] * 2)
        self.assertEqual([r.get('candidate_waits') for r in report['frames'][:2]], [1, 2])
        self.assertEqual(report['shot_budget']['accepted_monotonic_ns'],
                         report['frames'][2]['capture']['captured_monotonic_ns'])
        self.assertFalse(self.commands(events, '--dx'))
        self.assertFalse(self.commands(events, 'mousedown'))

    def test_unreadable_distance_wait_is_time_bounded_and_never_inputs(self):
        unreadable = [dict(nameplate_bbox=[1230,750,100,15], trackable=False,
                           distance_interval_km=None, reasons=['distance_unknown'])]
        code, report, events = self.run_scenes([
            self.scene(view='wide', candidates=unreadable, advance_ns=2_000_000_000)
            for _ in range(13)], fire=False)
        self.assertEqual(code, 1)
        self.assertIn('range or identity unconfirmed', report['reason'])
        waits = [r for r in report['frames'] if r.get('action') == 'wait_candidate_distance_readable']
        self.assertEqual(len(waits), 11)
        self.assertLessEqual(waits[-1]['candidate_wait_s'], 20)
        self.assertFalse(self.commands(events, '--dx'))
        self.assertFalse(self.commands(events, 'mousedown'))

    def test_out_of_range_candidate_still_refuses_immediately(self):
        far = [dict(nameplate_bbox=[1230,750,100,15], trackable=False,
                    distance_interval_km=[20.0, 20.1], reasons=['range_out_of_bounds'])]
        code, report, events = self.run_scenes([
            self.scene(view='wide', candidates=far), self.scene(view='wide')], fire=False)
        self.assertEqual(code, 1)
        self.assertIn('range or identity unconfirmed', report['reason'])
        self.assertEqual(len(report['frames']), 1)
        self.assertFalse(self.commands(events, '--dx'))

    def test_pending_retreat_waits_for_current_same_target_distance(self):
        missing = dict(nameplate_bbox=[1230,750,100,15], trackable=True,
                       distance_interval_km=None)
        code, report, events = self.run_scenes([
            self.scene(observation='correcting', move=[150,0], recovered=True, deferred=True),
            self.scene(candidates=[missing]), self.scene(pixels='retreat'),
            self.scene(), self.scene()], fire=False)
        self.assertEqual(code, 0)
        self.assertEqual([r['action'] for r in report['frames']], [
            'pointer_boundary_recovery_reobserve', 'wait_same_target_distance',
            'pointer_boundary_retreat_reobserve', 'wait_second_independent_alignment',
            'two_frame_locked_alignment'])
        self.assertEqual(len(self.commands(events, '--dx')), 1)
        self.assertEqual(report['shot_budget']['accepted_monotonic_ns'],
                         report['frames'][0]['capture']['captured_monotonic_ns'])
        self.assertFalse(self.commands(events, 'mousedown'))

    def test_vertical_coarse_offset_requires_correction_before_scope(self):
        candidate = dict(nameplate_bbox=[1230,540,100,15],trackable=True,
                         distance_interval_km=[10.25,10.35])
        code, report, events = self.run_scenes([
            self.scene(view='wide', candidates=[candidate]) for _ in range(3)], fire=False)
        self.assertEqual(code, 1)
        self.assertFalse(self.commands(events, 'shift'))
        moves = [e[1] for e in self.commands(events, '--dx')]
        self.assertEqual(len(moves), 1)
        self.assertEqual(moves[0][moves[0].index('--dx')+1], '0')
        self.assertEqual(moves[0][moves[0].index('--dy')+1], '3')
        self.assertFalse(self.commands(events, 'mousedown'))

    def test_vertical_coarse_large_step_requires_gain_and_shrinks_near_target(self):
        def wide(error):
            return self.scene(view='wide', candidates=[dict(
                nameplate_bbox=[1230,793+error,100,15], trackable=True,
                distance_interval_km=[10.25,10.35])])
        # The first three-unit probe measures -2 px/unit. Only then is the
        # far correction enlarged; inside the 60 px band it is small again.
        code, report, events = self.run_scenes([
            wide(-400), wide(-400), wide(-400),
            wide(-406), wide(-406), wide(-406),
            wide(-40), wide(-40), wide(-40)], fire=False)
        moves = [e[1] for e in self.commands(events, '--dx')]
        self.assertEqual([int(m[m.index('--dy')+1]) for m in moves], [3,-40])
        self.assertEqual([r['coarse']['vertical_limit_units'] for r in report['frames']
                          if r.get('action') == 'healthbar_coarse_approach'], [3,40])
        # Inside the 60 px band the loop stops moving entirely; it never crawls
        # with three-unit steps, so the near-target rule is satisfied strictly.
        self.assertIn('wait_second_coarse_alignment',
                      [r.get('action') for r in report['frames']])
        self.assertFalse(self.commands(events, 'mousedown'))

    def test_two_current_coarse_alignments_allow_scope_then_require_scoped_target(self):
        code, report, events = self.run_scenes([
            self.scene(view='wide'), self.scene(view='wide'),
            self.scene(), self.scene(), self.scene(view='wide')], fire=False)
        self.assertEqual(code, 0)
        self.assertEqual(report['status'], 'prepared')
        self.assertEqual(report['frames'][0]['action'], 'wait_second_coarse_alignment')
        self.assertTrue(report['frames'][1]['coarse']['two_frame_confirmed'])
        self.assertEqual(report['frames'][2]['scope_transition'], 'image_confirmed_history_reset')
        self.assertFalse(self.commands(events, 'mousedown'))
        code, report, events = self.run_scenes([
            self.scene(view='wide'), self.scene(view='wide'),
            self.scene(candidates=[]), self.scene(view='wide')], fire=False)
        self.assertEqual(code, 1)
        self.assertFalse(self.commands(events, 'mousedown'))

    def test_scope_roundtrip_rebuilds_coarse_drift_before_new_input(self):
        far = dict(nameplate_bbox=[1500,750,100,15], trackable=True,
                   distance_interval_km=[10.25,10.35])
        code, report, events = self.run_scenes([
            self.scene(view='wide', candidates=[far]),
            self.scene(view='wide', candidates=[far]),
            self.scene(view='wide'), self.scene(view='wide'),
            self.scene(observation='waiting'),
            self.scene(view='wide', candidates=[far]),
            self.scene(view='wide')], fire=False)
        self.assertEqual(code, 1)
        self.assertEqual(report['frames'][4]['scope_transition'],
                         'image_confirmed_history_reset')
        self.assertEqual(report['frames'][5]['action'], 'wait_coarse_no_input_baseline')
        self.assertFalse(self.commands(events, '--dx'))
        self.assertFalse(self.commands(events, 'mousedown'))

    def test_duplicate_coarse_pixels_do_not_authorize_zoom(self):
        code, report, events = self.run_scenes([
            self.scene(view='wide', pixels='same') for _ in range(3)], fire=False)
        self.assertEqual(code, 1)
        self.assertFalse(self.commands(events, 'shift'))
        self.assertFalse(self.commands(events, '--dx'))

    def test_old_coarse_alignment_or_changed_reference_needs_another_frame(self):
        for change in (dict(advance_ns=800_000_000), dict(centre_source='other_reticle')):
            with self.subTest(change=change):
                code, report, events = self.run_scenes([
                    self.scene(view='wide'), self.scene(view='wide', **change)], fire=False)
                self.assertEqual(code, 1)
                self.assertFalse(self.commands(events, 'shift'))
                self.assertFalse(self.commands(events, '--dx'))

    def test_departure_or_distance_gap_breaks_coarse_confirmation(self):
        for candidate in (
                dict(nameplate_bbox=[1230,540,100,15],trackable=True,
                     distance_interval_km=[10.25,10.35]),
                dict(nameplate_bbox=[1230,750,100,15],trackable=True,
                     distance_interval_km=None)):
            with self.subTest(candidate=candidate):
                code, report, events = self.run_scenes([
                    self.scene(view='wide'), self.scene(view='wide', candidates=[candidate]),
                    self.scene(view='wide')], fire=False)
                self.assertEqual(code, 1)
                self.assertFalse(self.commands(events, 'shift'))

    def test_different_enemy_cannot_supply_second_coarse_confirmation(self):
        code, report, events = self.run_scenes([
            self.scene(view='wide'), self.scene(view='wide', label='Other_Enemy')], fire=False)
        self.assertEqual(code, 1)
        self.assertFalse(self.commands(events, 'shift'))

    def test_coarse_gain_subtracts_no_input_target_drift(self):
        def wide(x):
            return self.scene(view='wide', candidates=[dict(
                nameplate_bbox=[x,750,100,15], trackable=True,
                distance_interval_km=[10.25,10.35])])
        code, report, events = self.run_scenes([
            wide(1600), wide(1605), wide(1610), wide(1595),
            wide(1600), wide(1605), self.scene(), self.scene(), self.scene()], fire=False)
        self.assertEqual(code, 0)
        response = report['frames'][3]['coarse']['response']
        self.assertAlmostEqual(response['natural_drift_projection'], 5.5)
        self.assertAlmostEqual(response['isolated'], -20.5)
        moves = [e[1] for e in self.commands(events, '--dx')]
        self.assertEqual([m[m.index('--dx')+1] for m in moves], ['20', '150'])
        self.assertFalse(self.commands(events, 'mousedown'))

    def test_coarse_natural_motion_alone_cannot_calibrate_actuator(self):
        scenes = [self.scene(view='wide', candidates=[dict(
            nameplate_bbox=[x,750,100,15], trackable=True,
            distance_interval_km=[10.25,10.35])]) for x in range(1600,1660,5)]
        code, report, events = self.run_scenes(scenes, fire=False)
        self.assertEqual(code, 1)
        self.assertIn('drift subtraction', report['reason'])
        self.assertIn('wait_coarse_measurable_response', [r.get('action') for r in report['frames']])
        responses = [r['coarse']['response'] for r in report['frames'] if 'response' in r.get('coarse',{})]
        self.assertTrue(all(abs(r['isolated']) <= 1 for r in responses))
        self.assertEqual(len(self.commands(events, '--dx')), 1)
        self.assertFalse(self.commands(events, 'mousedown'))

    def test_delayed_coarse_response_waits_without_repeating_move(self):
        def wide(x):
            return self.scene(view='wide', candidates=[dict(
                nameplate_bbox=[x,750,100,15], trackable=True,
                distance_interval_km=[10.25,10.35])])
        code, report, events = self.run_scenes([
            wide(1600), wide(1605), wide(1610), wide(1615), wide(1600),
            wide(1605), wide(1610), self.scene(), self.scene(), self.scene()], fire=False)
        self.assertEqual(code, 0)
        self.assertEqual(report['frames'][3]['action'], 'wait_coarse_measurable_response')
        self.assertLess(report['frames'][4]['coarse']['response']['isolated'], -19)
        moves = [e[1] for e in self.commands(events, '--dx')]
        self.assertEqual([m[m.index('--dx')+1] for m in moves], ['20', '150'])
        self.assertEqual(report['shot_budget']['accepted_monotonic_ns'],
                         report['frames'][0]['capture']['captured_monotonic_ns'])
        self.assertFalse(self.commands(events, 'mousedown'))

    def test_subminimum_coarse_gain_waits_for_full_response_without_new_input(self):
        def wide(x, advance_ns=100_000_000):
            return self.scene(view='wide', advance_ns=advance_ns, candidates=[dict(
                nameplate_bbox=[x,750,100,15], trackable=True,
                distance_interval_km=[10.25,10.35])])
        code, report, events = self.run_scenes([
            wide(1600), wide(1600), wide(1600), wide(1602), wide(1580)], fire=False)
        self.assertEqual(report['frames'][3]['action'], 'wait_coarse_measurable_response')
        self.assertEqual(report['frames'][3]['coarse']['response']['isolated'], 2)
        self.assertEqual(report['frames'][4]['coarse']['response']['isolated'], -20)
        self.assertEqual(report['frames'][4]['action'], 'wait_coarse_no_input_baseline')
        self.assertEqual(len(self.commands(events, '--dx')), 1)
        self.assertFalse(self.commands(events, 'shift'))
        self.assertFalse(self.commands(events, 'mousedown'))

    def test_subminimum_coarse_response_still_expires_without_new_input(self):
        def wide(x, advance_ns=100_000_000):
            return self.scene(view='wide', advance_ns=advance_ns, candidates=[dict(
                nameplate_bbox=[x,750,100,15], trackable=True,
                distance_interval_km=[10.25,10.35])])
        code, report, events = self.run_scenes([
            wide(1600), wide(1600), wide(1600), wide(1602),
            wide(1602, advance_ns=800_000_000)], fire=False)
        self.assertEqual(code, 1)
        self.assertIn('unresolved after drift subtraction', report['reason'])
        self.assertEqual(len(self.commands(events, '--dx')), 1)
        self.assertFalse(self.commands(events, 'shift'))
        self.assertFalse(self.commands(events, 'mousedown'))

    def test_distance_gap_preserves_pending_coarse_input_without_repeating_it(self):
        def wide(x, missing=False, advance_ns=100_000_000):
            return self.scene(view='wide', advance_ns=advance_ns, candidates=[dict(
                nameplate_bbox=[x,750,100,15], trackable=True,
                distance_interval_km=None if missing else [10.25,10.35])])
        code, report, events = self.run_scenes([
            wide(1600), wide(1605), wide(1610), wide(1615, missing=True),
            wide(1600), wide(1605), wide(1610),
            self.scene(), self.scene(), self.scene()], fire=False)
        self.assertEqual(code, 0)
        self.assertEqual(report['frames'][3]['action'], 'wait_same_target_distance')
        self.assertLess(report['frames'][4]['coarse']['response']['isolated'], -19)
        moves = [e[1] for e in self.commands(events, '--dx')]
        self.assertEqual([m[m.index('--dx')+1] for m in moves], ['20', '150'])
        self.assertFalse(self.commands(events, 'mousedown'))
        code, report, events = self.run_scenes([
            wide(1600), wide(1605), wide(1610),
            wide(1615, missing=True, advance_ns=800_000_000)], fire=False)
        self.assertEqual(code, 1)
        self.assertIn('unresolved after distance gap', report['reason'])
        self.assertEqual(len(self.commands(events, '--dx')), 1)
        self.assertFalse(self.commands(events, 'mousedown'))

    def test_pending_move_rejects_changed_reticle_reference(self):
        def wide(source):
            return self.scene(view='wide', centre_source=source, candidates=[dict(
                nameplate_bbox=[1600,750,100,15], trackable=True,
                distance_interval_km=[10.25,10.35])])
        code, report, events = self.run_scenes([
            wide('green_ring'), wide('green_ring'), wide('green_ring'),
            wide('amber_ring')], fire=False)
        self.assertEqual(code, 1)
        self.assertIn('reticle reference changed', report['reason'])
        self.assertEqual(len(self.commands(events, '--dx')), 1)
        self.assertFalse(self.commands(events, 'mousedown'))

    def test_recovery_drops_alignment_and_waits_for_changed_pixels(self):
        code, report, events = self.run_scenes([
            self.scene(observation='correcting',move=[150,0],recovered=True,pixels='old'),
            self.scene(pixels='old'), self.scene(), self.scene()],fire=False)
        self.assertEqual(code,0)
        self.assertEqual([r['action'] for r in report['frames']], [
            'pointer_boundary_recovery_reobserve', 'wait_pointer_recovery_visual_response',
            'wait_second_independent_alignment', 'two_frame_locked_alignment'])
        self.assertEqual(len([e for e in events if e[0]=='boundary_recovery']),1)
        self.assertFalse(self.commands(events,'mousedown'))

    def test_recovery_cannot_restart_original_shot_budget(self):
        code, report, events = self.run_scenes([
            self.scene(observation='correcting',move=[150,0],recovered=True),
            self.scene(advance_ns=30_000_000_000)])
        self.assertEqual(code,1)
        self.assertIn('budget',report['reason'])
        self.assertFalse(self.commands(events,'mousedown'))

    def test_deferred_retreat_needs_new_identity_frame_and_rebuilds_pair(self):
        code, report, events = self.run_scenes([
            self.scene(observation='correcting',move=[150,0],recovered=True,deferred=True),
            self.scene(pixels='retreat'), self.scene(pixels='retreat'),
            self.scene(), self.scene()],fire=False)
        self.assertEqual(code,0)
        self.assertEqual([r['action'] for r in report['frames']], [
            'pointer_boundary_recovery_reobserve','pointer_boundary_retreat_reobserve',
            'wait_pointer_recovery_visual_response','wait_second_independent_alignment',
            'two_frame_locked_alignment'])
        self.assertEqual(len(self.commands(events,'--dx')),1)
        self.assertFalse(self.commands(events,'mousedown'))

    def test_unlocked_requests_x_once_and_waits_for_image_lock(self):
        code,report,events = self.run_scenes([
            self.scene(view='wide',locked=False),self.scene(view='wide',locked=False),self.scene(locked=False),
            self.scene(),self.scene(),self.scene(view='wide')])
        self.assertEqual(code,0)
        self.assertEqual(len(self.commands(events,'x')),1)
        self.assertEqual(len(self.commands(events,'shift')),1)
        self.assertEqual(len(self.commands(events,'mousedown')),1)
        self.assertEqual(report['frames'][0]['action'],'wait_second_coarse_alignment')
        self.assertEqual(report['frames'][1]['action'],'request_scope_image_confirmation_required')
        self.assertEqual(report['frames'][2]['action'],'request_lock_once')
        self.assertEqual(report['frames'][2]['scope_transition'],'image_confirmed_history_reset')
        self.assertFalse(report['fired'])
        self.assertFalse(report['actual_discharge_verified'])

    def test_existing_lock_never_toggles_x_and_requests_one_left(self):
        code,report,events = self.run_scenes([self.scene(),self.scene()])
        self.assertEqual(code,0)
        self.assertFalse(self.commands(events,'x'))
        self.assertEqual(len(self.commands(events,'mousedown')),1)
        self.assertEqual(len(self.commands(events,'mouseup')),1)
        self.assertEqual(report['status'],'shot_requested')
        self.assertEqual(report['inputs'][-1]['hold_ms'],120)
        self.assertEqual(report['shot_outcome'],'input_attempted_discharge_unverified')
        self.assertIn(('release',),events)

    def test_lock_request_cannot_repeat_when_confirmation_missing(self):
        code,report,events = self.run_scenes([self.scene(locked=False),self.scene(locked=False)])
        self.assertEqual(code,1)
        self.assertIn('single lock request',report['reason'])
        self.assertEqual(len(self.commands(events,'x')),1)
        self.assertFalse(self.commands(events,'mousedown'))

    def test_name_change_or_competing_same_name_refuses(self):
        duplicate = dict(nameplate_bbox=[1450,750,100,15],trackable=True,
                         distance_interval_km=[11.25,11.35])
        for scene in (self.scene(label='Enemy_Different'),
                      self.scene(candidates=self.scene()['candidates']+[duplicate])):
            with self.subTest(scene=scene):
                code,report,events = self.run_scenes([self.scene(),scene])
                self.assertEqual(code,1)
                self.assertFalse(self.commands(events,'mousedown'))
                self.assertIn(('release',),events)

    def test_scope_transition_waits_without_repeating_shift_or_movement(self):
        code, report, events = self.run_scenes([
            self.scene(view='wide'), self.scene(view='wide'),
            self.scene(view='wide'), self.scene(), self.scene(), self.scene(view='wide')], fire=False)
        self.assertEqual(code, 0)
        self.assertEqual(report['frames'][2]['action'], 'wait_scope_visual_response')
        self.assertEqual(report['frames'][3]['scope_transition'], 'image_confirmed_history_reset')
        self.assertEqual(len(self.commands(events,'shift')),1)
        self.assertFalse(self.commands(events,'--dx'))
        self.assertFalse(self.commands(events,'mousedown'))

    def test_scope_animation_unknown_waits_without_authorizing_input(self):
        code, report, events = self.run_scenes([
            self.scene(view='wide'), self.scene(view='wide'),
            self.scene(view='unknown'), self.scene(), self.scene(), self.scene(view='wide')], fire=False)
        self.assertEqual(code, 0)
        self.assertEqual(report['frames'][2]['action'], 'wait_scope_visual_response')
        self.assertEqual(report['frames'][3]['scope_transition'], 'image_confirmed_history_reset')
        self.assertEqual(len(self.commands(events, 'shift')), 1)
        self.assertFalse(self.commands(events, '--dx'))
        self.assertFalse(self.commands(events, 'mousedown'))

    def test_scope_animation_unknown_still_expires(self):
        code, report, events = self.run_scenes([
            self.scene(view='wide'), self.scene(view='wide'),
            self.scene(view='unknown', advance_ns=800_000_000)], fire=False)
        self.assertEqual(code, 1)
        self.assertIn('scope request not confirmed', report['reason'])
        self.assertEqual(len(self.commands(events, 'shift')), 1)
        self.assertFalse(self.commands(events, 'mousedown'))

    def test_scope_animation_missing_name_waits_but_never_reuses_identity(self):
        code, report, events = self.run_scenes([
            self.scene(view='wide'), self.scene(view='wide'),
            self.scene(view='unknown', label=''), self.scene(), self.scene(), self.scene(view='wide')], fire=False)
        self.assertEqual(code, 0)
        self.assertEqual(report['frames'][2]['identity']['status'], 'unconfirmed')
        self.assertIsNone(report['frames'][2]['target'])
        self.assertEqual(report['frames'][2]['action'], 'wait_scope_visual_response')
        self.assertEqual(report['frames'][3]['scope_transition'], 'image_confirmed_history_reset')
        self.assertEqual(len(self.commands(events, 'shift')), 1)
        self.assertFalse(self.commands(events, 'mousedown'))

    def test_scope_animation_missing_name_expires_without_firing(self):
        code, report, events = self.run_scenes([
            self.scene(view='wide'), self.scene(view='wide'),
            self.scene(view='unknown', label='', advance_ns=800_000_000)], fire=False)
        self.assertEqual(code, 1)
        self.assertIn('scope request not confirmed', report['reason'])
        self.assertFalse(self.commands(events, 'mousedown'))

    def test_scope_animation_different_observed_name_never_waits_as_occlusion(self):
        code, report, events = self.run_scenes([
            self.scene(view='wide'), self.scene(view='wide'),
            self.scene(view='unknown', label='Another_Target')], fire=False)
        self.assertEqual(code, 1)
        self.assertIn('scope request not confirmed', report['reason'])
        self.assertNotEqual(report['frames'][-1].get('action'), 'wait_scope_visual_response')
        self.assertFalse(self.commands(events, 'mousedown'))

    def test_scope_identity_confirmation_does_not_reuse_missing_range(self):
        missing = self.scene(candidates=[dict(nameplate_bbox=[1230,750,100,15],
                                             trackable=True, distance_interval_km=None)])
        code, report, events = self.run_scenes([
            self.scene(view='wide'), self.scene(view='wide'), missing,
            self.scene(), self.scene(), self.scene(view='wide')], fire=False)
        self.assertEqual(code, 0)
        self.assertEqual(report['frames'][2]['scope_transition'], 'image_confirmed_history_reset')
        self.assertEqual(report['frames'][2]['action'], 'wait_same_target_distance')
        self.assertIsNone(report['frames'][2]['target'])
        self.assertFalse(self.commands(events, 'mousedown'))

    def test_missing_aim_numbers_wait_without_reusing_alignment_or_firing(self):
        for field in ('flight_status', 'aim_range_status'):
            with self.subTest(field=field):
                code, report, events = self.run_scenes([
                    self.scene(), self.scene(**{field:'unknown'}), self.scene(),
                    self.scene(), self.scene(view='wide')], fire=False)
                self.assertEqual(code, 0)
                self.assertEqual(report['frames'][1]['action'], 'wait_current_aim_numbers')
                self.assertNotIn('observation', report['frames'][1])
                self.assertEqual(report['frames'][2]['action'], 'wait_second_independent_alignment')
                self.assertFalse(self.commands(events, 'mousedown'))

    def test_scope_requires_next_image_and_cleanup_never_blind_toggles(self):
        code,report,events = self.run_scenes([
            self.scene(view='wide'),self.scene(view='wide'),self.scene(view='wide',advance_ns=800_000_000)])
        self.assertEqual(code,1)
        self.assertIn('scope request not confirmed',report['reason'])
        self.assertEqual(len(self.commands(events,'shift')),1)
        self.assertFalse(self.commands(events,'mousedown'))
        self.assertIn('no fresh cleanup',report['scope_cleanup_error'])

    def test_scope_cleanup_exits_only_fresh_confirmed_owned_scope(self):
        code,report,events = self.run_scenes([
            self.scene(view='wide'),self.scene(view='wide'),self.scene(),self.scene(),self.scene()])
        self.assertEqual(code,0)
        self.assertEqual(len(self.commands(events,'shift')),2)
        self.assertIn('fresh_scoped_image',report['scope_cleanup'])
        self.assertLess(events.index(('release',)),next(i for i,e in enumerate(events) if e[0]=='save'))

    def test_range_interval_misalignment_cannot_fire(self):
        code,report,events = self.run_scenes([self.scene(contained=False),self.scene(contained=False)])
        self.assertEqual(code,1)
        self.assertIn('range alignment',report['reason'])
        self.assertFalse(self.commands(events,'mousedown'))

    def test_two_frames_require_new_pixels_and_times(self):
        code,report,events = self.run_scenes([self.scene(pixels='same'),self.scene(pixels='same')])
        self.assertEqual(code,1)
        self.assertFalse(self.commands(events,'mousedown'))
        code,report,events = self.run_scenes([self.scene(stamp=2_080_000_000),
                                            self.scene(stamp=2_080_000_000)])
        self.assertEqual(code,1)
        self.assertFalse(self.commands(events,'mousedown'))

    def test_view_epoch_change_requires_new_alignment_pair(self):
        code,report,events = self.run_scenes([self.scene(view_id='one'),self.scene(view_id='two')])
        self.assertEqual(code,1)
        self.assertFalse(self.commands(events,'mousedown'))

    def test_focus_expiry_before_scope_attempt_does_not_grant_cleanup_ownership(self):
        code,report,events = self.run_scenes([self.scene(view='wide'),self.scene()],
                                            focus_delay_ns=800_000_000)
        self.assertEqual(code,1)
        self.assertFalse(self.commands(events,'shift'))
        self.assertNotIn('scope_cleanup',report)
        self.assertNotIn('scope_cleanup_error',report)

    def test_waiting_or_correcting_breaks_consecutive_alignment(self):
        code,report,events = self.run_scenes([self.scene(),self.scene(observation='waiting'),self.scene()])
        self.assertEqual(code,1)
        self.assertFalse(self.commands(events,'mousedown'))

    def test_expired_capture_budget_source_and_focus_fail_closed_and_release(self):
        cases = [([self.scene(age_ns=800_000_000)],0),
                 ([self.scene(),self.scene(advance_ns=30_000_000_000)],0),
                 ([self.scene(),self.scene(source='other')],0),
                 ([self.scene(),self.scene(focus=False)],0),
                 ([self.scene(),self.scene(check_failure=True)],0),
                 ([self.scene(),self.scene()],800_000_000)]
        for scenes,delay in cases:
            with self.subTest(scenes=scenes,delay=delay):
                code,report,events = self.run_scenes(scenes,focus_delay_ns=delay)
                self.assertEqual(code,1)
                self.assertFalse(self.commands(events,'mousedown'))
                self.assertIn(('release',),events)
                self.assertFalse(report['fired'])

    def test_first_qualified_frame_starts_budget_before_name_extraction(self):
        code,report,events = self.run_scenes([self.scene(label='')])
        self.assertEqual(code,1)
        self.assertEqual(report['shot_budget']['accepted_monotonic_ns'],2_080_000_000)
        self.assertFalse(self.commands(events,'mousedown'))

    def test_final_reload_weapon_distance_and_ambiguous_lock_reject(self):
        for change in (dict(ready=False),dict(weapon='torpedo'),dict(ambiguous_lock=True),
                       dict(horizontal_error=31),dict(uncertain=True),
                       dict(candidates=[dict(nameplate_bbox=[1230,750,100,15],trackable=True,
                                             distance_interval_km=[3.85,3.95])])):
            with self.subTest(change=change):
                code,report,events = self.run_scenes([self.scene(),self.scene(**change)])
                self.assertEqual(code,1)
                self.assertFalse(self.commands(events,'mousedown'))

    def test_inspect_never_focuses_or_inputs(self):
        code,report,events = self.run_scenes([self.scene()],fire=False,execute=False,inspect=True)
        self.assertEqual(code,0)
        self.assertEqual(report['status'],'inspected')
        self.assertFalse([e for e in events if e[0] in ('focus','command')])

    def test_preparation_requires_two_frames_and_does_not_fire(self):
        code,report,events = self.run_scenes([self.scene(),self.scene()],fire=False)
        self.assertEqual(code,0)
        self.assertEqual(report['status'],'prepared')
        self.assertFalse(self.commands(events,'mousedown'))
        self.assertFalse(report['body_confirmation_claimed'])

    def test_relaxed_range_and_horizontal_alignment_can_request_shot(self):
        for interval in ([3.95,4.05], [6.95,7.05], [14.95,15.05]):
            candidate = dict(nameplate_bbox=[1230,750,100,15],trackable=True,
                             distance_interval_km=interval)
            code, report, events = self.run_scenes([
                self.scene(candidates=[candidate], horizontal_error=30),
                self.scene(candidates=[candidate], horizontal_error=30)])
            self.assertEqual(code,0,report.get('reason'))
            self.assertEqual(report['status'],'shot_requested')
            self.assertEqual(len(self.commands(events,'mousedown')),1)
            self.assertEqual(report['shot_policy']['range_tolerance_km'],.5)

    def test_actual_weapon_range_still_limits_target(self):
        code, report, events = self.run_scenes([self.scene()],max_range_km=10)
        self.assertEqual(code,1)
        self.assertFalse(self.commands(events,'mousedown'))

    def test_policy_caps_long_range_weapon_at_fifteen(self):
        candidate = dict(nameplate_bbox=[1230,750,100,15],trackable=True,
                         distance_interval_km=[15.95,16.05])
        code, report, events = self.run_scenes([self.scene(candidates=[candidate])],max_range_km=20)
        self.assertEqual(code,1)
        self.assertFalse(self.commands(events,'mousedown'))
        self.assertEqual(report['shot_policy']['max_range_km'],15)

    def test_dry_run_and_execute_preflight_have_no_model_requirement(self):
        code,report,events = self.run_scenes([],fire=False,execute=False)
        self.assertEqual(code,0)
        self.assertEqual(report['status'],'dry-run')
        self.assertEqual(events,[])
        code,report,events = self.run_scenes([self.scene(),self.scene()],fire=False)
        self.assertEqual(code,0)
        self.assertNotIn('model',report.get('reason',''))
        for changes,reason in ((dict(training_room=False),'training-room'),
                               (dict(max_range_km=None),'maximum range')):
            code,report,events = self.run_scenes([],fire=False,**changes)
            self.assertEqual(code,1)
            self.assertIn(reason,report['reason'])
            self.assertFalse(events)

    def test_report_survives_encoding_failure_after_release(self):
        code,report,events = self.run_scenes([self.scene(),self.scene()],save_failure=True)
        self.assertEqual(code,0)
        self.assertEqual(report['status'],'shot_requested')
        self.assertEqual(report['artifact_error'],'PNG encoding failed')
        self.assertFalse(report['frame_artifacts_complete'])
        self.assertIn(('release',),events)


class InputGuardContracts(unittest.TestCase):
    def test_budget_reservation_is_strict_and_cannot_restart(self):
        budget = P.ShotBudget()
        budget.accept(1_000_000_000)
        budget.accept(5_000_000_000)
        self.assertEqual(budget.deadline_ns,31_000_000_000)
        budget.require(6_000_000_000)
        with self.assertRaisesRegex(P.ProbeError,'thirty-second'):
            budget.require(30_850_000_000,150_000_000)

    def test_real_native_amber_ring_proves_wide_with_independent_camera_evidence(self):
        from PIL import Image
        frame = np.asarray(Image.open(P.HERE/'aim_fixtures/native-amber-wide-20261019.png').convert('RGB'))
        hud = P.FireHudReader().detect(frame)
        ring = P.amber_reticle(frame)
        self.assertIsNotNone(ring)
        self.assertGreaterEqual(ring['support'],.5)
        self.assertEqual(P.view_state(hud,frame),'wide')
        scoped = dict(hud,aiming_view=dict(status='observed',mode='scope_ruler'))
        self.assertEqual(P.view_state(scoped,frame),'scope')
        without_ring = frame.copy()
        without_ring[690:785,1235:1325] = 0
        self.assertIsNone(P.amber_reticle(without_ring))
        # Preserve the observed caret/battle evidence: a caret alone is not wide.
        no_ring_hud = dict(hud,reticle=dict(status='observed',center=[1280,800],raw=dict(search='native_caret')))
        with patch.object(P.A,'green_reticle',return_value=None):
            self.assertEqual(P.view_state(no_ring_hud,without_ring),'unknown')
        with patch.object(P.A,'camera_mode_evidence',return_value=dict(binocular=True)):
            self.assertEqual(P.view_state(hud,frame),'unknown')

    def test_white_aim_point_never_substitutes_actual_gun_ring(self):
        frame = np.zeros((1600,2560,3),np.uint8)
        cv2.circle(frame,(1291,811),8,(30,230,160),2)
        hud = dict(battle=dict(active=True),aim_point=dict(center=[1280.5,805]),
                   gun_direction=dict(status='observed',ready=False,ring=dict(color='orange')))
        for state in ('wide','scope'):
            self.assertEqual(P.current_centre(hud,frame,state),
                             ([1280.5,805],'current_white_chevron_geometry'))
        hud['aim_point'] = None
        self.assertIsNone(P.current_centre(hud,frame,'wide')[0])

    def test_real_left_traversing_ring_outside_old_roi_confirms_wide(self):
        from PIL import Image
        frame = np.asarray(Image.open(P.HERE/'aim_fixtures/native-amber-left-traverse-20261019.png').convert('RGB'))
        ring = P.amber_reticle(frame)
        self.assertIsNotNone(ring)
        self.assertLess(ring['centre'][0], 1140)
        hud = P.FireHudReader().detect(frame)
        self.assertEqual(P.view_state(hud, frame), 'wide')
        without_ring = frame.copy()
        without_ring[770:830, 1090:1160] = 0
        self.assertIsNone(P.amber_reticle(without_ring))
        with patch.object(P.A, 'green_reticle', return_value=None):
            self.assertEqual(P.view_state(hud, without_ring), 'unknown')

    def test_real_multiple_turret_rings_prove_view_but_not_unique_centre(self):
        from PIL import Image
        frame = np.asarray(Image.open(P.HERE/'aim_fixtures/native-amber-traverse-20261019.png').convert('RGB'))
        self.assertIsNone(P.amber_reticle(frame))
        self.assertTrue(P.amber_reticle(frame, presence_only=True))
        hud = P.FireHudReader().detect(frame)
        self.assertEqual(P.view_state(hud, frame), 'wide')
        hud['reticle'] = dict(status='unknown', center=None, raw={})
        hud['aim_point'] = None
        with patch.object(P.A, 'green_reticle', return_value=None):
            self.assertIsNone(P.current_centre(hud, frame, 'wide')[0])
        no_ring = frame.copy()
        no_ring[660:940, 900:1660] = 0
        self.assertFalse(P.amber_reticle(no_ring, presence_only=True))
        with patch.object(P.A, 'green_reticle', return_value=None):
            self.assertEqual(P.view_state(hud, no_ring), 'unknown')

    def test_real_amber_ring_supplies_missing_native_caret_centre(self):
        from PIL import Image
        frame = np.asarray(Image.open(P.HERE/'aim_fixtures/native-amber-no-caret-20261019.png').convert('RGB'))
        hud = P.FireHudReader().detect(frame)
        # Exercise ring-only evidence even if native templates later improve.
        hud['reticle'] = dict(status='unknown',center=None,raw={})
        hud['aim_point'] = None
        with patch.object(P.A,'green_reticle',return_value=None):
            self.assertEqual(P.view_state(hud,frame),'wide')
            self.assertIsNone(P.current_centre(hud,frame,'wide')[0])
            no_ring = frame.copy()
            no_ring[660:940,1140:1420] = 0
            self.assertIsNone(P.amber_reticle(no_ring))
            self.assertEqual(P.view_state(hud,no_ring),'unknown')
            self.assertIsNone(P.current_centre(hud,no_ring,'unknown')[0])
            self.assertIsNone(P.current_centre(hud,frame,'scope')[0])

    def test_amber_arc_or_solid_orange_blob_is_not_ring(self):
        frame = np.zeros((1600,2560,3),np.uint8)
        cv2.circle(frame,(1280,740),18,(230,160,30),-1)
        self.assertIsNone(P.amber_reticle(frame))
        frame[:] = 0
        cv2.ellipse(frame,(1280,740),(18,18),0,0,90,(230,160,30),2)
        self.assertIsNone(P.amber_reticle(frame))

    def test_scope_requires_ruler_image_not_requested_intent(self):
        frame = np.zeros((1,1,3),np.uint8)
        with patch.object(P.A,'camera_mode_evidence',return_value=dict(binocular=False)), \
             patch.object(P.A,'green_reticle',return_value=None):
            self.assertEqual(P.view_state(dict(scope_requested=True),frame),'unknown')
            self.assertEqual(P.view_state(dict(aiming_view=dict(status='observed',mode='scope_ruler')),frame),'scope')

    def test_oversized_input_rejected_before_focus(self):
        probe = P.AcquisitionProbe(10,dict(inputs=[]))
        with patch.object(probe,'focus') as focus:
            for move in ((151,0),(0,4)):
                with self.assertRaises(P.ProbeError):
                    probe.move(*move)
            focus.assert_not_called()

    def test_coarse_move_allows_forty_then_restores_three(self):
        probe = P.AcquisitionProbe(10,dict(inputs=[]))
        probe.relative_transport = 'xtest'
        with patch.object(probe,'focus'), patch.object(probe,'guard'), \
             patch.object(P.Probe,'move') as move:
            probe.coarse_move(0,-40)
            self.assertEqual(move.call_args.args, (0,-40))
            with self.assertRaises(P.ProbeError):
                probe.move(0,-40)
            with self.assertRaises(P.ProbeError):
                probe.coarse_move(0,-41)
            self.assertEqual(move.call_count, 1)
            probe.move(0,-3)
            self.assertEqual(move.call_count, 2)

    def test_default_xtest_key_and_move_reuse_guarded_probe(self):
        probe = P.AcquisitionProbe(10,dict(inputs=[]))
        probe.relative_transport = 'xtest'
        with patch.object(probe,'focus'),patch.object(probe,'guard'), \
             patch.object(probe,'key_is_down',return_value=False), \
             patch.object(P.Probe,'move') as move,patch.object(P.Probe,'key') as key:
            probe.move(150,3)
            probe.key('Shift_L')
        move.assert_called_once_with(150,3)
        key.assert_called_once_with('Shift_L')
        self.assertTrue(probe.report['inputs'][0]['image_confirmation_required'])

    def test_clipping_retreats_once_without_retrying_original_correction(self):
        probe = P.AcquisitionProbe(10,dict(inputs=[]))
        probe.relative_transport = 'xtest'
        def clipped(*args):
            probe.report['inputs'].append(dict(pointer_delta=[5,0] if args[0]>0 else list(args)))
        with patch.object(probe,'focus'),patch.object(probe,'guard'), \
             patch.object(P.Probe,'move',side_effect=clipped) as move:
            self.assertIs(probe.move(150,0),False)
            self.assertEqual([c.args for c in move.call_args_list],[(150,0)])
            self.assertTrue(probe.recover_pointer())
            self.assertEqual([c.args for c in move.call_args_list],[(150,0),(-150,0)])
            with self.assertRaisesRegex(P.ProbeError,'repeated'):
                probe.move(150,0)
            self.assertEqual(move.call_count,3)

    def test_recovery_expiry_prevents_retreat(self):
        probe = P.AcquisitionProbe(10,dict(inputs=[]))
        probe.relative_transport = 'xtest'
        def clipped(*args):
            probe.report['inputs'].append(dict(pointer_delta=[0,0]))
        with patch.object(probe,'focus'), patch.object(probe,'guard') as guard, \
             patch.object(P.Probe,'move',side_effect=clipped) as move:
            self.assertIs(probe.move(150,0),False)
            guard.side_effect = P.ProbeError('expired')
            with self.assertRaisesRegex(P.ProbeError,'expired'):
                probe.recover_pointer()
            self.assertEqual(move.call_count,1)

    def test_failed_mouseup_is_retried_by_release(self):
        probe = P.AcquisitionProbe(10,dict(inputs=[],focus_checks=[]))
        probe.left_owned = True
        with patch.object(P.Probe,'x',side_effect=[P.ProbeError('lost release'),None]) as transport:
            with self.assertRaises(P.ProbeError):
                probe.x('mouseup',1,cleanup=True)
            probe.release()
        self.assertEqual(transport.call_count,2)
        self.assertFalse(probe.left_owned)

    def test_left_button_owned_by_user_is_never_released(self):
        probe = P.AcquisitionProbe(10,dict(inputs=[],focus_checks=[]))
        probe.identity = dict(display=':test')
        with patch.object(probe,'focus'),patch.object(probe,'guard'), \
             patch.object(P,'pointer_state',return_value=dict(mask=256)), \
             patch.object(P.Probe,'x') as transport:
            with self.assertRaisesRegex(P.ProbeError,'already held'):
                probe.x('mousedown',1)
            probe.release()
            transport.assert_not_called()

    def test_missing_future_and_expired_pipeline_timestamps_reject(self):
        meta = dict(captured_monotonic_ns=1_000_000_000,
                    capture_age_semantics='gst_segment_running_time;upstream_render_latency_unknown')
        with patch.object(P.time,'monotonic_ns',return_value=2_000_000_000):
            for evidence in ({},meta,dict(meta,captured_monotonic_ns=3_000_000_000)):
                with self.assertRaises(Exception):
                    P.require_fresh(evidence)


if __name__ == '__main__':
    unittest.main()
