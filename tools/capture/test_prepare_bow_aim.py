from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import cv2
import numpy as np
import prepare_bow_aim as P


class PreparationContracts(unittest.TestCase):
    def test_read_delay_is_rejected_without_claiming_render_age(self):
        from wows_window_capture import DRAWABLE_UPDATE_SEMANTICS
        meta = dict(drawable_update_lower_bound_monotonic_ns=1,
                    drawable_update_semantics=DRAWABLE_UPDATE_SEMANTICS,
                    render_age_verified=False)
        with patch.object(P.time,'monotonic_ns',return_value=800_000_001):
            with self.assertRaisesRegex(P.ProbeError,'expired'):
                P.require_recent_acquisition(meta)
        with patch.object(P.time,'monotonic_ns',return_value=700_000_001):
            P.require_recent_acquisition(meta)
        with self.assertRaisesRegex(P.ProbeError,'unverified'):
            P.require_recent_acquisition({'acquisition_started_monotonic_ns':1})

    def test_scope_requires_current_ruler_not_toggle_intent(self):
        frame=np.zeros((1,1,3),np.uint8)
        with patch.object(P.A,'camera_mode_evidence',return_value={'binocular':False}), \
             patch.object(P.A,'green_reticle',return_value=None):
            self.assertEqual(P.view_state({'scope_requested':True},frame),'unknown')
            self.assertEqual(P.view_state({'aiming_view':{'status':'observed','mode':'scope_ruler'}},frame),'scope')

    def test_scope_uses_bounded_kernel_key_and_does_not_claim_image_success(self):
        report = {'inputs': []}
        probe = P.AcquisitionProbe(10, report)
        with patch.object(probe, 'focus'), patch.object(probe, 'key_is_down', return_value=False), \
                patch.object(probe, 'command') as command, patch.object(P.Path, 'is_file', return_value=True):
            probe.key('Shift_L')
        self.assertEqual(command.call_args.args[0][-4:], ['--key','42','--hold-ms','120'])
        self.assertTrue(report['inputs'][0]['image_confirmation_required'])
        with patch.object(probe, 'focus'), patch.object(probe, 'key_is_down', return_value=True), \
                patch.object(probe, 'command') as command, self.assertRaises(P.ProbeError):
            probe.key('Shift_L')
        command.assert_not_called()

    def test_target_loss_or_switch_does_not_calibrate(self):
        a={'nameplate_bbox':[100,100,100,15]}
        self.assertFalse(P.same_target(a,None))
        self.assertFalse(P.same_target(a,{'nameplate_bbox':[500,100,100,15]}))
        self.assertFalse(P.same_target(a,{'nameplate_bbox':[110,100,200,15]}))
        self.assertTrue(P.same_target(a,{'nameplate_bbox':[130,100,100,15]}))

    def test_verified_appearance_identity_outranks_screen_proximity(self):
        before = {'nameplate_bbox':[100,300,100,15], 'target_id':'appearance-one'}
        far = {'nameplate_bbox':[1800,700,100,15], 'target_id':'appearance-one'}
        replacement = {'nameplate_bbox':[100,300,100,15], 'target_id':'appearance-two'}
        self.assertTrue(P.same_target(before, far))
        self.assertFalse(P.same_target(before, replacement))

    def test_dynamic_fire_refuses_before_capture_even_with_weights(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(P, 'WindowCapture') as capture, patch('builtins.print'):
            out = Path(directory) / 'refused'
            args = SimpleNamespace(out=out, execute=True, inspect=False,
                                   bow_model=Path('weights.pt'), fire=True, dynamic_locked=True)
            self.assertEqual(P.run(args), 1)
            capture.assert_not_called()
            report = json.loads((out/'report.json').read_text())
            self.assertEqual(report['inputs'], [])
            self.assertFalse(report['fire_authorized'])
            self.assertEqual(report['shot_outcome'], 'rejected_not_fired')

    def test_dynamic_requires_current_zoom_and_lock_on_accepted_target(self):
        controller = P.DynamicLockedAim()
        target = dict(target_id='one', nameplate_bbox=[100,300,100,15],
                      aim_px=[1280,800], distance_interval_km=[10.25,10.35])
        other = dict(target, target_id='two')
        hud = dict(aiming_view=dict(status='observed', mode='scope_ruler', zoom_confirmed=None),
                   flight_time=dict(status='observed',seconds=5,
                                    distance=dict(status='observed', km=8)))
        frame = np.zeros((1600,2560,3), np.uint8)
        with patch.object(P, 'confirm_target', return_value=dict(confirmed=True,detection_index=0)):
            result = controller.observe(frame,hud,[target],target,[1280,800],{})
            self.assertEqual(result['reason'], 'scope_zoom_identity_unconfirmed')
            self.assertEqual(result['move'], [0,0])
        with patch.object(P, 'confirm_target', return_value=dict(confirmed=True,detection_index=1)):
            result = controller.observe(frame,hud,[target,other],target,[1280,800],{})
            self.assertEqual(result['reason'], 'same_target_ui_lock_unconfirmed')
            self.assertFalse(result['fire_authorized'])

    def test_dynamic_uses_target_range_separately_from_reticle_range(self):
        controller = P.DynamicLockedAim()
        target = dict(target_id='one', nameplate_bbox=[100,300,100,15],
                      aim_px=[1280,800], distance_interval_km=[10.25,10.35])
        hud = dict(aiming_view=dict(status='observed', mode='scope_ruler', zoom_confirmed=True, zoom=2),
                   flight_time=dict(status='observed',seconds=5,
                                    distance=dict(status='observed', km=8)))
        meta = dict(source='test',identity={'pid':1},window='one',raw_pixel_sha256='new',
                    drawable_update_lower_bound_monotonic_ns=1_000_000_000)
        with patch.object(P, 'confirm_target', return_value=dict(confirmed=True,detection_index=0)), \
                patch.object(P.time,'monotonic_ns',return_value=1_100_000_000):
            result = controller.observe(np.zeros((1600,2560,3),np.uint8),hud,
                                        [target],target,[1280,800],meta)
        self.assertEqual(result['target_range_interval_km'], [10.25,10.35])
        self.assertEqual(result['aim_range_km'], 8)
        self.assertEqual(result['solution']['aim_range_km'], 8)
        self.assertEqual(result['move'], [0,0])
        self.assertFalse(result['fire_authorized'])


class PreparationConsumerContracts(unittest.TestCase):
    @staticmethod
    def scene(view='wide', error=0, target=True, age_ns=100_000_000):
        candidates = [] if not target else [{
            'trackable': True,
            'nameplate_bbox': [100, 300, 100, 15],
            'aim_px': [1280 + error, 800],
            'distance_from_reticle_px': abs(error),
            'distance_interval_km': [10.25, 10.35],
            'body_association': 'confirmed',
        }]
        return {
            'view': view,
            'age_ns': age_ns,
            'label': 'XKnight_Alpha',
            'candidates': candidates,
            'hud': {
                'battle': {'active': True},
                'weapon': {'status':'observed', 'kind':'gun'},
                'reload': {'ready':True},
                'reticle': {'center': [1280, 800]},
                'aiming_view': {
                    'status': 'observed',
                    'mode': 'scope_ruler' if view == 'scope' else 'wide',
                },
            },
        }

    def run_scenes(self, scenes, *, execute=True, focus_delay_ns=0, save_failure=False,
                   missing_model=False, bow_missing=False, bow_offset=0, fire=False, accepted=False,
                   dynamic_locked=False):
        clock = {'ns': 2_000_000_000}
        current = {'scene': None, 'index': 0}
        events = []
        released = []

        class Capture:
            identity = {'pid': 123, 'display': ':test'}
            window = 'test-window'

            def acquire(self, **kwargs):
                index = current['index']
                if index >= len(scenes):
                    raise AssertionError('consumer captured beyond supplied scenes')
                scene = scenes[index]
                current.update(scene=scene, index=index + 1)
                clock['ns'] += scene.get('advance_ns', 0)
                started = clock['ns'] - scene['age_ns']
                pixels = image.copy()
                for candidate in scene['candidates']:
                    x, y, w, _ = candidate['nameplate_bbox']
                    label = candidate.get('label', scene.get('label', 'XKnight_Alpha'))
                    if label:
                        cv2.putText(pixels, label, (x+w//2-55, y-80-8),
                                    cv2.FONT_HERSHEY_SIMPLEX, .55, (245,245,245), 1, cv2.LINE_AA)
                return pixels, {
                    'acquisition_started_monotonic_ns': started,
                    'source': 'xcomposite_window_pixmap',
                    'identity': scene.get('source_identity', Capture.identity),
                    'window': 'test-window',
                    'drawable_update_lower_bound_monotonic_ns': started,
                    'drawable_update_semantics': 'xdamage_after_cleared_baseline_before_xgetimage;upstream_render_latency_unknown',
                    'render_age_verified': accepted,
                    'source_render_monotonic_ns': None,
                    'raw_pixel_sha256': scene.get('pixel_identity', str(index)),
                }

            def save(self, path, pixels, meta):
                if save_failure:
                    raise OSError('simulated PNG write failure')

        class RecordingProbe(P.AcquisitionProbe):
            def __init__(self, seconds, report):
                self.action_deadline = self.deadline = 100.0
                self.evidence = None

            def move(self, dx, dy):
                self.focus()
                events.append(('move', dx, dy))

            def key(self, key):
                self.focus()
                events.append(('key', key))

            def x(self, *args, **kwargs):
                events.append(('button', *args))

            def release(self):
                released.append(True)

        def focus(_probe):
            clock['ns'] += focus_delay_ns

        image = np.zeros((1440, 2560, 3), np.uint8)
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            out = Path(directory) / 'result'
            args = SimpleNamespace(out=out, execute=execute, inspect=False,
                                   bow_model=None if missing_model else Path('reviewed-test-model.pt'),
                                   seconds=30, steps=max(1, len(scenes)), max_range_km=15, fire=fire,
                                   dynamic_locked=dynamic_locked)
            capture_factory = stack.enter_context(patch.object(P, 'WindowCapture', return_value=Capture()))
            probe_factory = stack.enter_context(patch.object(P, 'AcquisitionProbe', side_effect=RecordingProbe))
            stack.enter_context(patch.object(P.Probe, 'focus', focus))
            stack.enter_context(patch.object(P.time, 'monotonic_ns', side_effect=lambda: clock['ns']))
            stack.enter_context(patch.object(P.time, 'monotonic', side_effect=lambda: clock['ns'] / 1e9))
            stack.enter_context(patch.object(P.time, 'sleep'))
            stack.enter_context(patch.object(P.signal, 'signal'))
            stack.enter_context(patch.object(P.signal, 'setitimer'))
            stack.enter_context(patch.object(P, 'FireHudReader', return_value=SimpleNamespace(
                detect=lambda frame: current['scene']['hud'])))
            stack.enter_context(patch.object(P.A, 'camera_mode_evidence', side_effect=lambda frame: {
                'binocular': current['scene']['view'] != 'wide'}))
            stack.enter_context(patch.object(P.A, 'green_reticle', side_effect=lambda frame:
                None if current['scene']['hud']['reticle']['center'] is None else {'centre': [1280,800]}))
            stack.enter_context(patch.object(P.A,'find_enemy',side_effect=lambda *args, **kwargs: current['scene']['candidates']))
            stack.enter_context(patch('main_view_perception.detect_nameplates',
                side_effect=lambda *a, **k: current['scene']['candidates']))
            stack.enter_context(patch.object(P, 'BowRecognizer', return_value=SimpleNamespace(
                detect=lambda *a: None if bow_missing else dict(
                    bow_point=[1280+bow_offset, 800], bow_confidence=.9,
                    model_sha256='test-model', fire_authorized=accepted,
                    orientation_confidence=.9, segmentation_confidence=.9))))
            stack.enter_context(patch('builtins.print'))
            try:
                result = P.run(args)
            except OSError:
                if not save_failure:
                    raise
                result = None
            report = json.loads((out / 'report.json').read_text())
            initialized = (capture_factory.call_count, probe_factory.call_count)
        return result, report, events, released, initialized

    def test_png_failure_preserves_outcome_after_input_release(self):
        result, report, events, released, _ = self.run_scenes([
            self.scene('wide'), self.scene('scope')], save_failure=True)
        self.assertIsNone(result)
        self.assertEqual(report['status'], 'prepared')
        self.assertFalse(report['frame_artifacts_complete'])
        self.assertFalse(report['fired'])
        self.assertEqual(released, [True])

    def test_aligned_wide_view_requires_shift_and_observed_scope(self):
        result, report, events, released, _ = self.run_scenes([
            self.scene('wide'), self.scene('scope'),
        ])
        self.assertEqual(result, 0)
        self.assertEqual(report['status'], 'prepared')
        self.assertEqual(events, [('key', 'Shift_L')])
        self.assertEqual([row['action'] for row in report['frames']],
                         ['request_scope', 'scoped_bow_aligned'])
        self.assertFalse(report['fired'])
        self.assertFalse(report['fire_authorized'])
        self.assertEqual(released, [True])

    def test_dynamic_scope_without_lock_cannot_reuse_static_alignment(self):
        result, report, events, released, _ = self.run_scenes([
            self.scene('wide'), self.scene('scope'),
        ], dynamic_locked=True)
        self.assertEqual(result, 1)
        self.assertEqual(events, [('key', 'Shift_L'), ('key', 'Shift_L')])
        self.assertEqual(report['frames'][-1]['action'], 'wait_dynamic_locked_evidence')
        self.assertEqual(report['frames'][-1]['dynamic']['reason'], 'same_target_ui_lock_unconfirmed')
        self.assertFalse(report['fire_authorized'])
        self.assertEqual(released, [True])

    def test_missing_weights_refuse_before_any_coarse_input(self):
        result, report, events, released, initialized = self.run_scenes([], missing_model=True)
        self.assertEqual(result, 1)
        self.assertIn('model missing', report['reason'])
        self.assertEqual(events, [])
        self.assertEqual(released, [])
        self.assertEqual(initialized, (0, 0))

    def test_scope_uses_bow_not_aligned_healthbar(self):
        result, report, events, _, _ = self.run_scenes([
            self.scene('wide'), self.scene('scope')], bow_offset=100)
        self.assertEqual(result, 1)
        self.assertEqual(report['frames'][1]['error_px'], [100,0])
        self.assertIn(('move',20,0), events)
        self.assertFalse(report['fire_authorized'])
        self.assertTrue(report['lead_disabled_by_user'])
        self.assertFalse(report['minimap_used'])

    def test_scope_missing_bow_does_not_use_bar_geometry(self):
        result, report, events, _, _ = self.run_scenes([
            self.scene('wide'), self.scene('scope')], bow_missing=True)
        self.assertEqual(result, 1)
        self.assertIn('bow unconfirmed', report['reason'])
        self.assertEqual(events, [('key','Shift_L'), ('key','Shift_L')])
        self.assertFalse(report['fired'])

    def test_no_reticle_with_multiple_candidates_does_not_choose_or_move(self):
        scene = self.scene('unknown')
        scene['hud']['reticle']['center'] = None
        scene['candidates'] *= 2
        for row in scene['candidates']:
            row['distance_from_reticle_px'] = None
        with patch.object(P.A, 'green_reticle', return_value=None):
            result, report, events, _, _ = self.run_scenes([scene])
        self.assertEqual(result, 1)
        self.assertEqual(events, [])
        self.assertEqual(report['frames'][0]['action'], 'wait_current_view_evidence')

    def test_unknown_view_never_moves_and_requires_new_evidence(self):
        unknown = self.scene('unknown', error=100)
        unknown['hud']['aiming_view'] = {'status': 'unknown'}
        result, report, events, _, _ = self.run_scenes([
            unknown, self.scene('wide'), self.scene('scope')])
        self.assertEqual(result, 0)
        self.assertEqual(report['frames'][0]['action'], 'wait_current_view_evidence')
        self.assertEqual(events, [('key', 'Shift_L')])

    def test_unknown_view_exhaustion_never_sends_input(self):
        unknown = self.scene('unknown')
        unknown['hud']['aiming_view'] = {'status': 'unknown'}
        result, report, events, _, _ = self.run_scenes([unknown, unknown])
        self.assertEqual(result, 1)
        self.assertIn('budget exhausted', report['reason'])
        self.assertEqual(events, [])

    def test_toggle_without_scope_image_cannot_report_prepared(self):
        result, report, events, _, _ = self.run_scenes([
            self.scene('wide'), self.scene('wide'),
        ])
        self.assertEqual(result, 1)
        self.assertEqual(report['status'], 'refused')
        self.assertIn('not confirmed', report['reason'])
        self.assertEqual(events, [('key', 'Shift_L'), ('key', 'Shift_L')])
        self.assertNotIn('scoped_bow_aligned',
                         [row.get('action') for row in report['frames']])

    def test_empty_scope_exits_once_and_never_searches_or_moves(self):
        result, report, events, released, _ = self.run_scenes([
            self.scene('wide'), self.scene('scope', target=False), self.scene('wide', error=100),
        ])
        self.assertEqual(result, 1)
        self.assertEqual(report['status'], 'refused')
        self.assertEqual(len(report['frames']), 2)
        self.assertIn('healthbar missing', report['reason'])
        self.assertEqual(events, [('key', 'Shift_L'), ('key', 'Shift_L')])
        self.assertEqual(released, [True])

    def test_lost_target_refuses_without_relabeling_or_reusing_calibration(self):
        result, report, events, _, _ = self.run_scenes([
            self.scene(error=100), self.scene(target=False), self.scene(error=60),
        ])
        self.assertEqual(result, 1)
        self.assertIn('identity changed, lost', report['reason'])
        self.assertEqual(events, [('move', 20, 0)])
        self.assertEqual(len(report['frames']), 2)

    def test_unchanged_pixels_preserve_pending_move_until_response(self):
        before = self.scene(error=100)
        before['pixel_identity'] = 'before'
        repeated = self.scene(error=100)
        repeated['pixel_identity'] = 'before'
        result, report, events, released, _ = self.run_scenes([
            before, repeated, repeated, self.scene(error=60),
            self.scene(), self.scene('scope')])
        self.assertEqual(result, 0)
        self.assertEqual(events, [('move', 20, 0), ('move', 15, 0),
                                  ('key', 'Shift_L')])
        self.assertEqual([r['action'] for r in report['frames'][1:3]],
                         ['wait_pending_visual_response'] * 2)
        self.assertNotIn('observed_response', report['frames'][2])
        self.assertEqual(report['frames'][3]['observed_response'], -2.0)
        self.assertEqual(released, [True])

    def test_unchanged_pixels_exhaust_budget_without_duplicate_input(self):
        scene = self.scene(error=100)
        scene['pixel_identity'] = 'frozen'
        result, report, events, released, _ = self.run_scenes([scene] * 4)
        self.assertEqual(result, 1)
        self.assertEqual(events, [('move', 20, 0)])
        self.assertFalse(report['fired'])
        self.assertEqual(released, [True])

    def test_offline_dry_run_initializes_neither_capture_nor_input(self):
        result, report, events, released, initialized = self.run_scenes([], execute=False)
        self.assertEqual(result, 0)
        self.assertEqual(report['status'], 'dry-run')
        self.assertEqual(report['frames'], [])
        self.assertEqual(report['inputs'], [])
        self.assertEqual(events, [])
        self.assertEqual(released, [])
        self.assertEqual(initialized, (0, 0))

    def test_expired_acquisition_blocks_every_scene_driven_input(self):
        for name, scene in [
            ('scan', self.scene(target=False)),
            ('enter_scope', self.scene()),
            ('correction', self.scene(error=100)),
            ('exit_empty_scope', self.scene('scope', target=False)),
        ]:
            with self.subTest(action=name):
                scene['age_ns'] = 750_000_001
                result, report, events, released, _ = self.run_scenes([scene])
                self.assertEqual(result, 1)
                self.assertEqual(report['status'], 'refused')
                self.assertIn('expired', report['reason'])
                self.assertEqual(events, [])
                self.assertEqual(released, [True])

    def test_focus_delay_cannot_turn_recent_frame_into_stale_input(self):
        for name, scene in [
            ('scan', self.scene(target=False)),
            ('enter_scope', self.scene()),
            ('correction', self.scene(error=100)),
        ]:
            with self.subTest(action=name):
                result, report, events, released, _ = self.run_scenes(
                    [scene], focus_delay_ns=750_000_001)
                self.assertEqual(result, 1)
                self.assertEqual(report['status'], 'refused')
                self.assertIn('expired', report['reason'])
                self.assertEqual(events, [])
                self.assertEqual(released, [True])

    def test_five_second_expiry_records_no_shot_and_releases(self):
        expired = self.scene('scope')
        expired['advance_ns'] = 5_000_000_000
        _, report, events, released, _ = self.run_scenes([self.scene(), expired], fire=True)
        self.assertEqual(report['shot_outcome'], 'timeout_not_fired')
        self.assertFalse(report['fired'])
        self.assertEqual(released, [True])
        self.assertEqual(events, [('key','Shift_L'), ('key','Shift_L')])

    def test_final_current_bow_without_acceptance_cannot_trigger(self):
        _, report, events, _, _ = self.run_scenes(
            [self.scene(), self.scene('scope'), self.scene('scope')], fire=True)
        self.assertEqual(report['shot_outcome'], 'rejected_not_fired')
        self.assertIn('operational acceptance', report['reason'])
        self.assertFalse(any(event[0] == 'button' for event in events))

    def test_current_bow_resolves_scoped_hull_without_inheriting_coarse_confirmation(self):
        wide, scoped = self.scene(), self.scene('scope')
        for scene in (wide, scoped):
            scene['candidates'][0]['body_association'] = 'unknown'
            scene['candidates'][0]['body_status'] = 'visible_silhouette_candidate'
        result, report, events, _, _ = self.run_scenes([wide, scoped])
        self.assertEqual(result, 0)
        self.assertEqual(report['status'], 'prepared')
        final = report['frames'][-1]['selection']['selected']
        self.assertEqual(final['bow_confirmation'], {'confirmed': True, 'current': True})
        self.assertFalse(any(event[0] == 'button' for event in events))

    def test_trigger_request_does_not_claim_actual_discharge(self):
        with patch('target_confirmation.confirm_target', return_value={'confirmed': True}):
            _, report, events, released, _ = self.run_scenes(
                [self.scene(), self.scene('scope'), self.scene('scope')], fire=True, accepted=True)
        self.assertEqual(report['status'], 'shot_requested')
        self.assertEqual(report['shot_outcome'], 'input_attempted_discharge_unverified')
        self.assertFalse(report['fired'])
        self.assertFalse(report['shot_budget']['actual_discharge_verified'])
        self.assertIn(('button','mousedown',1),events)
        self.assertIn(('button','mouseup',1),events)
        self.assertEqual(released,[True])

    def test_cross_zoom_without_identity_refuses_instead_of_retargeting(self):
        wide, scoped = self.scene(), self.scene('scope')
        wide['label'] = ''
        scoped['label'] = ''
        _, report, events, _, _ = self.run_scenes([wide, scoped], fire=True)
        self.assertIn('cross-zoom target identity', report['reason'])
        self.assertFalse(any(e[0] == 'button' for e in events))

    def test_replacement_target_does_not_inherit_first_targets_budget_or_fire(self):
        wide, scoped = self.scene(), self.scene('scope')
        scoped['label'] = 'Unrelated_Target'
        _, report, events, _, _ = self.run_scenes([wide, scoped], fire=True)
        self.assertIn('cross-zoom target identity', report['reason'])
        self.assertFalse(any(e[0] == 'button' for e in events))

    def test_distance_selected_original_survives_extra_scoped_candidate(self):
        wide, scoped = self.scene(), self.scene('scope')
        original = wide['candidates'][0]
        original.update(distance_interval_km=[9.2,9.3], distance_from_reticle_px=200)
        other = dict(original, nameplate_bbox=[600,400,120,15],
                     label='Nearest_Reticle', distance_interval_km=[12.2,12.3],
                     distance_from_reticle_px=0)
        wide['candidates'] = [other, original]
        scoped_original = dict(scoped['candidates'][0], nameplate_bbox=[1600,700,100,15])
        scoped['candidates'] = [dict(other), scoped_original]
        result, report, events, _, _ = self.run_scenes([wide, scoped])
        self.assertEqual(result, 0)
        first, last = report['frames']
        self.assertEqual(first['identity']['matched_index'], 1)
        self.assertEqual(last['identity']['matched_index'], 1)
        self.assertEqual(first['selection']['range_selected']['target_id'],
                         last['selection']['selected']['target_id'])
        self.assertEqual(last['gain'], [None,None])
        self.assertEqual(events, [('key','Shift_L')])

    def test_identical_scoped_names_refuse_before_bow_or_correction(self):
        wide, scoped = self.scene(), self.scene('scope')
        scoped['candidates'].append(dict(scoped['candidates'][0], nameplate_bbox=[600,400,100,15]))
        _, report, events, _, _ = self.run_scenes([wide, scoped])
        self.assertIn('cross-zoom target identity', report['reason'])
        self.assertEqual(report['frames'][-1]['identity']['reason'], 'appearance_ambiguous')
        self.assertNotIn('bow', report['frames'][-1])
        self.assertFalse(any(event[0] in ('move','button') for event in events))

    def test_capture_source_change_discards_identity_without_reanchoring(self):
        wide, scoped = self.scene(), self.scene('scope')
        scoped['source_identity'] = {'pid':124,'display':':test'}
        _, report, events, _, _ = self.run_scenes([wide, scoped])
        self.assertIn('identity source lost', report['reason'])
        self.assertEqual(report['frames'][-1]['identity_reset'], 'battle_or_capture_source_lost')
        self.assertFalse(any(event[0] in ('move','button') for event in events))


    def test_battle_loss_clears_identity_before_any_new_target_action(self):
        wide, scoped = self.scene(), self.scene('scope')
        scoped['hud']['battle']['active'] = False
        _, report, events, _, _ = self.run_scenes([wide, scoped])
        self.assertIn('identity source lost', report['reason'])
        self.assertEqual(report['frames'][-1]['identity_reset'], 'battle_or_capture_source_lost')
        self.assertFalse(any(event[0] in ('move','button') for event in events))

class ShotBudgetContracts(unittest.TestCase):
    def test_deadline_is_not_reset_by_reacquisition_and_boundary_refuses(self):
        budget = P.ShotBudget()
        budget.accept(1_000_000_000)
        budget.accept(4_000_000_000)
        budget.require(5_999_999_999)
        with self.assertRaisesRegex(P.ProbeError, 'five-second'):
            budget.require(6_000_000_000)
        with self.assertRaises(P.ProbeError):
            budget.require(5_900_000_000, reserve_ns=150_000_000)
        self.assertEqual(budget.accepted_ns, 1_000_000_000)

    def test_input_subprocess_uses_target_deadline_and_cleanup_can_release_after_it(self):
        probe = P.AcquisitionProbe(30, {'inputs':[]})
        probe.shot_budget = P.ShotBudget()
        probe.shot_budget.accept(1_000_000_000)
        original = probe.action_deadline
        deadlines = []
        def command(_self, argv, **kwargs):
            deadlines.append(_self.action_deadline)
        with patch.object(P.Probe, 'command', command), patch.object(P.time, 'monotonic_ns',return_value=2_000_000_000):
            probe.command(['xdotool','mousedown','1'])
        self.assertEqual(deadlines,[6.0])
        self.assertEqual(probe.action_deadline,original)
        with patch.object(P.Probe, 'command', command), patch.object(P.time, 'monotonic_ns',return_value=7_000_000_000):
            with self.assertRaises(P.ProbeError):
                probe.command(['xdotool','mousedown','1'])
            probe.command(['xdotool','mouseup','1'],cleanup=True)
        self.assertEqual(deadlines[-1],original)

    def test_final_weapon_readiness_identity_and_scope_are_current(self):
        hud = dict(battle={'active': True}, weapon={'status':'observed','kind':'gun'},
                   reload={'ready':True})
        bow = dict(model_sha256='identity', bow_point=[1280,800], fire_authorized=True)
        P.final_shot_check(hud, 'scope', bow, np.array([2,3]), True)
        for bad_hud, state, model, error, continuous in [
                (hud,'wide',bow,[0,0],True),
                ({**hud,'weapon':{'status':'observed','kind':'torpedo'}},'scope',bow,[0,0],True),
                ({**hud,'reload':{'ready':None}},'scope',bow,[0,0],True),
                (hud,'scope',None,[0,0],True), (hud,'scope',bow,[100,0],True),
                (hud,'scope',bow,[0,0],False)]:
            with self.subTest(state=state, error=error, continuous=continuous), self.assertRaises(P.ProbeError):
                P.final_shot_check(bad_hud,state,model,np.array(error),continuous)


if __name__=='__main__':
    unittest.main()
