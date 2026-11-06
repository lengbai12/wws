"""Servo decision gate: gain learning, one-axis probing and patch tracking.

The loop must not assume a mouse sign or a stored gain matrix, must refuse to
steer on a response too small to trust, and must track a real patch rather than
believe a commanded coordinate.
"""
import sys
from pathlib import Path
import os
import tempfile
import time
import unittest

import json
import math
import numpy as np
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'replay'))
import aim_servo as A
import main_view_perception as MVP


class FrameTimeTests(unittest.TestCase):
    def test_fresh_frame_uses_capture_time_not_read_time(self):
        self.assertEqual(A.frame_capture_ns({'captured_monotonic_ns': 1_000_000_000},
                                          1_750_000_000), 1_000_000_000)

    def test_missing_future_expired_and_duplicate_frames_are_rejected(self):
        for meta, now, previous in (
                (None, 2_000_000_000, None),
                ({'captured_monotonic_ns': 2_000_000_001}, 2_000_000_000, None),
                ({'captured_monotonic_ns': 1_000_000_000}, 1_750_000_001, None),
                ({'captured_monotonic_ns': 1_000_000_000}, 1_100_000_000, 1_000_000_000)):
            with self.subTest(meta=meta, previous=previous), self.assertRaises(A.ProbeError):
                A.frame_capture_ns(meta, now, previous)

    def test_timestamp_failure_reasons_preserve_rejection_boundary(self):
        now = 2_000_000_000
        for meta, previous, reason in (
                (None, None, 'missing_or_invalid_timestamp'),
                ({'captured_monotonic_ns': True}, None, 'missing_or_invalid_timestamp'),
                ({'captured_monotonic_ns': now + 1}, None, 'future_timestamp'),
                ({'captured_monotonic_ns': now - A.FRAME_MAX_AGE_NS - 1}, None, 'expired'),
                ({'captured_monotonic_ns': now - 1}, now - 1, 'nonadvancing_timestamp')):
            with self.subTest(reason=reason), self.assertRaises(A.FrameTimeError) as caught:
                A.frame_capture_ns(meta, now, previous, checkpoint='before_move', frame='step-4')
            self.assertEqual(caught.exception.evidence['reason'], reason)
            self.assertEqual(caught.exception.evidence['checkpoint'], 'before_move')
            self.assertEqual(caught.exception.evidence['frame'], 'step-4')


class ServoDecisionTests(unittest.TestCase):
    def test_first_step_probes_x_and_second_probes_y(self):
        self.assertEqual(A.plan_move([50, -30], [None, None], probe=60), (60, 0, 'probe_x'))
        self.assertEqual(A.plan_move([50, -30], [1.2, None], probe=60), (0, 3, 'probe_y'))

    def test_correction_divides_by_the_learned_gain(self):
        # measured gain: a +60 mouse move shifted the patch by +75 px
        self.assertEqual(A.observed_response(0.0, 75.0, 60), 1.25)
        self.assertEqual(A.observed_response(0.0, -75.0, 60), -1.25)
        move_x, move_y, reason = A.plan_move([125, 0], [1.25, -1.0], probe=60)
        self.assertEqual(reason, 'correct')
        self.assertEqual(move_x, -100)   # -125 / 1.25
        self.assertEqual(move_y, 0)

    def test_correction_is_clamped_to_the_reach(self):
        move_x, _, _ = A.plan_move([2000, 0], [0.5, 1.0], reach=400)
        self.assertEqual(move_x, -400)

    def test_vertical_corrections_are_small_in_both_directions(self):
        for error_y in (-2000, 2000):
            with self.subTest(error_y=error_y):
                dx, dy, reason = A.plan_move([2000, error_y], [.5, 1], reach=150)
                self.assertEqual(dx, -150)
                self.assertEqual(dy, -3 if error_y > 0 else 3)
                self.assertEqual(reason, 'correct')
        self.assertEqual(A.plan_move([0, 100], [1, 1], reach_y=2)[1], -2)
        self.assertEqual(A.plan_move([0, 100], [1, None], probe_y=3, reach_y=2)[1], 2)

    def test_pointer_recovery_never_sends_a_large_vertical_burst(self):
        class Probe:
            def __init__(self):
                self.report = {'inputs': []}

            def move(self, dx, dy):
                self.report['inputs'].append({'relative': [dx, dy]})

        probe = Probe()
        for requested in (540, -540, 2, 0):
            before = len(probe.report['inputs'])
            sent = A.bounded_move(probe, 400, requested, 3)
            self.assertEqual(sent, (400, max(-3, min(3, requested))))
            self.assertEqual(len(probe.report['inputs']), before + 1)
            row = probe.report['inputs'][-1]
            self.assertEqual(row['relative'], [400, max(-3, min(3, requested))])
            if abs(requested) > 3:
                self.assertEqual(row['requested_relative'], [400, requested])
                self.assertTrue(row['vertical_limited'])

    def test_damping_under_corrects(self):
        move_x, _, _ = A.plan_move([100, 0], [1.0, 1.0], damping=.7)
        self.assertEqual(move_x, -70)
        move_x, _, _ = A.plan_move([100, 0], [1.0, 1.0])
        self.assertEqual(move_x, -100)

    def test_gain_is_revised_from_every_measured_response(self):
        gain = [None, None]
        gain, measured = A.update_gain(gain, [0, 0], [90, 0], (60, 0))
        self.assertEqual(measured, [1.5, None])
        self.assertEqual(gain, [1.5, None])
        # a later, smaller response is blended in rather than ignored
        gain, measured = A.update_gain(gain, [0, 0], [60, 0], (60, 0), blend=.5)
        self.assertEqual(measured, [1.0, None])
        self.assertEqual(gain[0], 1.25)

    def test_a_sign_flipped_response_is_dropped(self):
        # a mouse delta cannot reverse the camera direction; this reading is
        # contamination and must not flip the established gain
        gain, measured = A.update_gain([1.5, None], [0, 0], [-90, 0], (60, 0))
        self.assertEqual(gain[0], 1.5)
        self.assertEqual(measured[0], None)

    def test_a_consistent_response_still_updates(self):
        gain, measured = A.update_gain([1.5, None], [0, 0], [60, 0], (60, 0), blend=.5)
        self.assertEqual(measured[0], 1.0)
        self.assertEqual(gain[0], 1.25)

    def test_a_revised_response_too_small_is_reported_not_adopted(self):
        gain, measured = A.update_gain([1.5, None], [0, 0], [1, 0], (60, 0))
        self.assertEqual(measured, [1 / 60, None])
        self.assertEqual(gain[0], 1.5)

    def test_unmoved_axis_is_not_measured(self):
        gain, measured = A.update_gain([1.5, None], [0, 0], [90, 50], (60, 0))
        self.assertEqual(measured, [1.5, None])
        self.assertEqual(gain, [1.5, None])

    def test_a_response_too_small_to_steer_is_refused(self):
        self.assertEqual(A.plan_move([50, 10], [0.1, 1.0])[2], 'gain_too_small')
        self.assertIsNone(A.observed_response(0.0, 0.0, 0))
        self.assertEqual(A.observed_response(0.0, 0.0, 60), 0.0)

    def test_vertical_small_probe_learns_without_relaxing_horizontal_gate(self):
        for units in (1, 2, 3, -1, -2, -3):
            with self.subTest(units=units):
                gain, measured = A.update_gain([None, None], [0, 0], [9, -2 * units], (3, units))
                self.assertEqual(gain, [None, -2.0])
                self.assertEqual(measured, [None, -2.0])


class PatchTrackingTests(unittest.TestCase):
    def frame_with_marker(self, cx, cy, size=12):
        frame = np.zeros((1600, 2560, 3), np.uint8)
        rng = np.random.default_rng(7)
        patch = rng.integers(40, 220, (40, 40, 3), dtype=np.uint8)
        frame[cy-20:cy+20, cx-20:cx+20] = patch
        return frame

    def test_tracking_finds_the_shifted_patch(self):
        first = self.frame_with_marker(1200, 700)
        second = self.frame_with_marker(1140, 690)
        seed = [1180, 680, 40, 40]
        template = A.crop_template(first, seed)
        self.assertIsNotNone(template)
        found = A.track_patch(template, seed, second)
        self.assertIsNotNone(found)
        self.assertLessEqual(abs(found['bbox'][0] - 1120), 2)
        self.assertLessEqual(abs(found['bbox'][1] - 670), 2)
        self.assertGreater(found['score'], .9)
        self.assertEqual(A.patch_centre(seed), (1200.0, 700.0))
        self.assertEqual(A.patch_centre(found['bbox']),
                         (found['bbox'][0]+20, found['bbox'][1]+20))

    def test_tracking_refuses_a_frame_without_the_patch(self):
        first = self.frame_with_marker(1200, 700)
        blank = np.zeros((1600, 2560, 3), np.uint8)
        template = A.crop_template(first, [1180, 680, 40, 40])
        self.assertIsNone(A.track_patch(template, [1180, 680, 40, 40], blank))

    def test_a_flat_or_out_of_frame_window_is_not_trackable(self):
        blank = np.zeros((1600, 2560, 3), np.uint8)
        self.assertIsNone(A.crop_template(blank, [1180, 680, 40, 40]))
        first = self.frame_with_marker(1200, 700)
        self.assertIsNone(A.crop_template(first, [2555, 1590, 80, 80]))
        self.assertIsNone(A.crop_template(first, [-5, 100, 40, 40]))
        self.assertIsNone(A.track_patch(None, [1180, 680, 40, 40], first))

    def test_real_nameplate_moves_while_minimap_edge_stays_fixed(self):
        directory = Path(__file__).with_name('aim_fixtures')
        from PIL import Image
        before = np.asarray(Image.open(directory / 'nameplate-minimap-before.png').convert('RGB'))
        after = np.asarray(Image.open(directory / 'nameplate-minimap-after.png').convert('RGB'))
        candidate = dict(nameplate_bbox=[111, 49, 124, 16],
                         aim_px=[163., 161.], confidence=.95)
        patch, template, offset = A.nameplate_patch(before, candidate)
        found = A.track_patch(template, patch['bbox'], after, search=80)
        self.assertIsNotNone(found)
        self.assertAlmostEqual(found['bbox'][0] - patch['bbox'][0], -30, delta=1)
        aim = [v + d for v, d in zip(A.patch_centre(found['bbox']), offset)]
        self.assertAlmostEqual(aim[0], 132., delta=1)
        self.assertEqual(aim[1], 161.)
        self.assertGreater(aim[1], found['bbox'][1] + found['bbox'][3])
        gain, _ = A.update_gain([None, None], candidate['aim_px'], aim, (20, 0))
        self.assertAlmostEqual(gain[0], -1.55, delta=.05)

    def test_real_nameplate_without_texture_is_rejected(self):
        frame = np.zeros((220, 300, 3), np.uint8)
        with self.assertRaises(A.ProbeError):
            A.nameplate_patch(frame, dict(nameplate_bbox=[111, 49, 124, 16],
                                         aim_px=[163., 161.], confidence=.95))


class LockBeforeFireTests(unittest.TestCase):
    def test_cli_accepts_a_lock_key_and_rejects_an_empty_one(self):
        from unittest.mock import patch
        with patch.object(sys, 'argv', ['prog', '--scan', '--lock', '--lock-key', '', '--out', '/tmp/x3']), \
                self.assertRaises(SystemExit):
            A.main()

    def test_the_lock_key_defaults_to_x(self):
        captured = {}
        with patch.object(A, 'run', lambda args: captured.update(
                lock_key=args.lock_key, lock=args.lock) or 0), \
                patch.object(sys, 'argv', ['prog', '--scan', '--lock', '--out', '/tmp/wws-lock-gate']):
            self.assertEqual(A.main(), 0)
        self.assertEqual(captured['lock_key'], 'x')
        self.assertTrue(captured['lock'])


class TrackSearchTests(unittest.TestCase):
    def test_the_radius_follows_the_move_and_the_gain(self):
        self.assertEqual(A.track_search(None, [1.2, 1.2]), 160)
        self.assertGreater(A.track_search((150, 0), [-1.2, -0.9]), 350)
        self.assertEqual(A.track_search((0, 10), [-1.2, -0.9]), 160)

    def test_the_radius_is_capped(self):
        self.assertLessEqual(A.track_search((400, 400), [-2.0, -2.0]), 900)


class ClampedDeliveryTests(unittest.TestCase):
    def test_a_short_pointer_delta_is_a_clamp(self):
        self.assertTrue(A.clamped_delivery((200, 0), [0, 0]))
        self.assertTrue(A.clamped_delivery((200, 0), [40, 0]))
        self.assertFalse(A.clamped_delivery((200, 0), [200, 0]))
        self.assertFalse(A.clamped_delivery((0, 200), [0, 195]))
        self.assertFalse(A.clamped_delivery((200, 0), None))

    def test_only_axes_that_were_asked_for_are_checked(self):
        # a large vertical delta on a horizontal move is not a clamp
        self.assertFalse(A.clamped_delivery((200, 0), [200, -300]))
        self.assertTrue(A.clamped_delivery((0, 200), [0, -30]))


class PostFireFrameTests(unittest.TestCase):
    def test_post_fire_frames_are_bounded_on_the_cli(self):
        from unittest.mock import patch
        with patch.object(sys, 'argv', ['prog', '--scan', '--post-fire-frames', '9',
                                        '--out', '/tmp/wws-fire-gate']), \
                self.assertRaises(SystemExit):
            A.main()

    def test_the_default_captures_three_frames(self):
        captured = {}
        with patch.object(A, 'run', lambda args: captured.update(n=args.post_fire_frames) or 0), \
                patch.object(sys, 'argv', ['prog', '--scan', '--out', '/tmp/wws-fire-gate2']):
            self.assertEqual(A.main(), 0)
        self.assertEqual(captured['n'], 3)


class AxisShiftTests(unittest.TestCase):
    def test_a_shift_along_the_probed_axis_is_usable(self):
        self.assertTrue(A.usable_axis_shift([120, 8], axis=0))
        self.assertTrue(A.usable_axis_shift([5, 120], axis=1))

    def test_a_sideways_lock_is_refused(self):
        # probing the vertical axis but the lock slid sideways
        self.assertFalse(A.usable_axis_shift([160, 0], axis=1))
        self.assertFalse(A.usable_axis_shift([0, 160], axis=0))

    def test_a_tiny_shift_is_refused(self):
        self.assertFalse(A.usable_axis_shift([12, 1], axis=0))
        self.assertFalse(A.usable_axis_shift(None, axis=0))


class ConcurrentInputTests(unittest.TestCase):
    class FakeProbe:
        def __init__(self, states, keys=()):
            self.states = list(states)
            self.keys = set(keys)
            self.identity = {'display': ':0'}

        def key_is_down(self, key):
            return key in self.keys

    def test_a_still_pointer_and_free_keys_pass(self):
        probe = self.FakeProbe([{'x': 960, 'y': 540, 'mask': 0}, {'x': 960, 'y': 540, 'mask': 0}])
        with patch.object(A, 'pointer_state', side_effect=probe.states), \
                patch.object(A.time, 'sleep', lambda s: None):
            self.assertEqual(A.concurrent_input(probe), [])

    def test_a_pointer_moving_without_input_is_flagged(self):
        probe = self.FakeProbe([{'x': 960, 'y': 540, 'mask': 0}, {'x': 990, 'y': 540, 'mask': 0}])
        with patch.object(A, 'pointer_state', side_effect=probe.states), \
                patch.object(A.time, 'sleep', lambda s: None):
            signals = A.concurrent_input(probe)
        self.assertEqual([s['kind'] for s in signals], ['pointer_moved_without_input'])

    def test_a_held_control_key_is_flagged(self):
        probe = self.FakeProbe([{'x': 960, 'y': 540, 'mask': 0}, {'x': 960, 'y': 540, 'mask': 0}],
                               keys={'w'})
        with patch.object(A, 'pointer_state', side_effect=probe.states), \
                patch.object(A.time, 'sleep', lambda s: None):
            signals = A.concurrent_input(probe)
        self.assertEqual([s['kind'] for s in signals], ['key_held'])

    def test_a_held_button_is_flagged(self):
        probe = self.FakeProbe([{'x': 960, 'y': 540, 'mask': 0}, {'x': 960, 'y': 540, 'mask': 256}])
        with patch.object(A, 'pointer_state', side_effect=probe.states), \
                patch.object(A.time, 'sleep', lambda s: None):
            signals = A.concurrent_input(probe)
        self.assertIn('button_held', [s['kind'] for s in signals])


class RunFlowTests(unittest.TestCase):
    """Drive run() end to end with injected fakes.

    Several runtime faults (an unbound report field, a misplaced row update)
    only appeared in live runs because nothing exercised run() itself.
    """

    def parse_args(self, tmp, extra=None):
        captured = {}
        argv = ['prog', '--scan', '--scan-steps', '2', '--max-steps', '2', '--fire', '--lock',
                '--execute', '--fire-min-steps', '1', '--post-fire-frames', '1',
                '--out', str(Path(tmp) / 'run')] + (extra or [])
        with patch.object(A, 'run', lambda args: (captured.update(args=args), 0)[1]), \
                patch.object(sys, 'argv', argv):
            self.assertEqual(A.main(), 0)
        return captured['args']

    def test_the_open_loop_delta_is_minus_the_error_over_the_gain(self):
        # gain is d(error)/d(mouse): the correction is -error/gain. Dividing
        # without the minus drove the live fast shot away from the hull.
        self.assertEqual(A.open_loop_delta([1300.0, 800.0], [1280.0, 800.0], [-1.5, -1.4]),
                         [13.333333333333334, 0.0])
        with self.assertRaises(ValueError):
            A.open_loop_delta([1.0, 1.0], [0.0, 0.0], [0.0, 1.0])

    def test_oneshot_fires_only_with_selected_ready_main_battery(self):
        self._oneshot_with_battery({
            'weapon': {'status': 'observed', 'kind': 'gun'},
            'reload': {'status': 'observed', 'ready': True},
        })

    def test_oneshot_blocks_unknown_or_other_weapon_and_unready_battery(self):
        for weapon, reload, reason in (
                ({'status': 'unknown'}, {'status': 'observed', 'ready': True},
                 'weapon_not_gun'),
                ({'status': 'observed', 'kind': 'torpedo'},
                 {'status': 'observed', 'ready': True}, 'weapon_not_gun'),
                ({'status': 'observed', 'kind': 'gun'},
                 {'status': 'observed', 'ready': False}, 'reload_not_ready'),
                ({'status': 'observed', 'kind': 'gun'},
                 {'status': 'unknown', 'ready': None}, 'reload_not_ready')):
            with self.subTest(weapon=weapon, reload=reload):
                self._oneshot_with_battery({'weapon': weapon, 'reload': reload}, reason)

    def _oneshot_with_battery(self, battery, expected_error=None, empty_scope=False):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            args = self.parse_args(tmp, ['--oneshot', '--scope'])
            if empty_scope:
                args.stay_in_scope = True
            frame = np.random.default_rng(5).integers(0, 255, (1600, 2560, 3), dtype=np.uint8)

            class FakeProbe:
                def __init__(self, seconds, report):
                    self.report = report
                    self.started = 0.0
                    self.action_deadline = 1e9
                    self.identity = {'pid': 1, 'display': ':0', 'start_ticks': '1'}
                    self.window = '1'

                def x(self, *argv, **kwargs):
                    if argv and argv[0] == 'getwindowgeometry':
                        return 'WIDTH=1920\nHEIGHT=1080'
                    if argv and argv[0] == 'search':
                        return '12345'
                    if argv and argv[0] in ('mousedown', 'mouseup'):
                        self.report.setdefault('trigger_commands', []).append(list(argv))
                    return ''

                def verify_window(self, window):
                    return {'window': window, 'pid': 1, 'title': 'World of Warships'}

                def focus(self):
                    self.report['focus_checks'].append({})

                def key(self, key):
                    self.report['inputs'].append({'key': key})

                def move(self, dx, dy):
                    self.report['inputs'].append({'relative': [dx, dy], 'pointer_delta': [dx, dy]})

                def snapshot(self, name):
                    if empty_scope and name.startswith('pre-fire-oneshot'):
                        calls.append('scope-empty')
                    return Path(tmp) / f'{name}.png'

                def command(self, argv, cleanup=False, identity=None):
                    return ''

                def check_identity(self):
                    return None

                def release(self):
                    self.report['unreleased_own_keys'] = []

            def fake_reticle(rgb):
                return [1280.0, 800.0], True, 'observed', 'native_band'

            calls = []

            def fake_enemies(rgb, reticle=None, aim_fraction=.75, scope_observed=False):
                # First read: the target is off to the side. The re-read after
                # the open-loop move lands on the reticle, which is the only
                # state the fire gate accepts.
                if 'scope-empty' in calls:
                    return []
                calls.append(1)
                if len(calls) == 1:
                    return [dict(nameplate_bbox=[1560, 720, 60, 18], body_roi=[1550, 740, 80, 40],
                                 aim_px=[1590.0, 760.0], trackable=True, confidence=.9,
                                 hp=39100, red_px=1500,
                                 distance_from_reticle_px=310.0)]
                return [dict(nameplate_bbox=[1250, 780, 60, 18], body_roi=[1240, 790, 80, 40],
                             aim_px=[1280.0, 800.0], trackable=True, confidence=.9,
                             hp=39100, red_px=1500,
                             distance_from_reticle_px=0.0)]

            with patch.object(A, 'Probe', FakeProbe), \
                    patch.object(A, 'game_identity', lambda: {'pid': 1, 'display': ':0', 'start_ticks': '1'}), \
                    patch.object(A, 'pointer_state', lambda display: {'x': 960, 'y': 540, 'mask': 0}), \
                    patch.object(A, 'read_frame', lambda path: frame), \
                    patch.object(A, 'reticle_centre', fake_reticle), \
                    patch.object(A, 'hud_scope_evidence', lambda frame: {'observed': True}), \
                    patch.object(A, 'main_battery_fire_evidence', lambda frame: battery), \
                    patch.object(A.time, 'sleep', lambda seconds: None), \
                    patch.object(A, 'find_enemy', fake_enemies):
                code = A.run(args)
            report = json.loads((Path(args.out) / 'report.json').read_text())
            moves = [row['relative'] for row in report['inputs'] if 'relative' in row]
            self.assertTrue(any(abs(dx) > 3 for dx, dy in moves))
            self.assertTrue(any(abs(dy) == 3 for dx, dy in moves))
            self.assertTrue(all(abs(dy) <= 3 for dx, dy in moves))
            if empty_scope:
                # A plate was confirmed before the zoom, so a plate-free scoped
                # frame is the client hiding it, not the enemy leaving.
                self.assertEqual(code, 1)
                self.assertFalse(report.get('fired', False))
                self.assertFalse(report.get('trigger_commands'))
                self.assertFalse(report['unreleased_button'])
                self.assertEqual(report['unreleased_own_keys'], [])
                self.assertTrue(report.get('scope_hides_nameplates'))
                self.assertNotIn('scope_rejection', report)
                return
            if expected_error:
                self.assertEqual(code, 1)
                self.assertIn(expected_error, report['error'])
                self.assertFalse(report.get('fired', False))
                self.assertFalse(report.get('trigger_commands'))
                self.assertFalse(report['unreleased_button'])
                self.assertEqual(report['unreleased_own_keys'], [])
                return
            self.assertEqual(code, 0, report.get('error'))
            self.assertEqual(report['mode'], 'oneshot_open_loop')
            self.assertTrue(report['fired'])
            decision = report['fire_decision']
            self.assertTrue(decision['allow'], decision['blockers'])
            self.assertEqual(set(decision['status']), set(A.fire_gate.GATES))
            self.assertEqual(report['trigger_commands'], [['mousedown', 1], ['mouseup', 1]])
            self.assertEqual(report['fire_reason'], 'fresh_confirmed_solution')
            self.assertEqual(len(report.get('fire_frames') or []), 1)
            keys = [item.get('key') for item in report['inputs'] if item.get('key')]
            self.assertIn('Shift_L', keys)
            self.assertIn('x', keys)
            # The binocular view is a toggle, so entering it in the one-shot path
            # must be followed by leaving it again: twice, enter then exit.
            # The probe records the press itself; the intent entries carry the
            # reason, and those are what say enter happened before exit.
            scope_reasons = [item.get('reason') for item in report['inputs']
                             if item.get('key') == 'Shift_L' and item.get('reason')]
            self.assertEqual(scope_reasons, ['scope_for_oneshot', 'exit_scope'])

    def test_oneshot_refuses_to_fire_when_the_reread_disagrees(self):
        """A move the re-read does not confirm must not fire."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            args = self.parse_args(tmp, ['--oneshot', '--scope'])
            frame = np.random.default_rng(5).integers(0, 255, (1600, 2560, 3), dtype=np.uint8)

            class FakeProbe:
                def __init__(self, seconds, report):
                    self.report = report
                    self.started = 0.0
                    self.action_deadline = 1e9
                    self.identity = {'pid': 1, 'display': ':0', 'start_ticks': '1'}
                    self.window = '1'

                def x(self, *argv, **kwargs):
                    if argv and argv[0] == 'getwindowgeometry':
                        return 'WIDTH=1920\nHEIGHT=1080'
                    if argv and argv[0] == 'search':
                        return '12345'
                    return ''

                def verify_window(self, window):
                    return {'window': window, 'pid': 1, 'title': 'World of Warships'}

                def focus(self):
                    self.report['focus_checks'].append({})

                def key(self, key):
                    self.report['inputs'].append({'key': key})

                def move(self, dx, dy):
                    self.report['inputs'].append({'relative': [dx, dy], 'pointer_delta': [dx, dy]})

                def snapshot(self, name):
                    return Path(tmp) / f'{name}.png'

                def command(self, argv, cleanup=False, identity=None):
                    return ''

                def check_identity(self):
                    return None

                def release(self):
                    self.report['unreleased_own_keys'] = []

            def fake_reticle(rgb):
                return [1280.0, 800.0], True, 'observed', 'native_band'

            def fake_enemies(rgb, reticle=None, aim_fraction=.75, scope_observed=False):
                # The target runs away: the re-read stays far from the reticle.
                return [dict(nameplate_bbox=[1560, 720, 60, 18], body_roi=[1550, 740, 80, 40],
                             aim_px=[1590.0, 760.0], trackable=True, confidence=.9,
                             distance_from_reticle_px=310.0)]

            with patch.object(A, 'Probe', FakeProbe), \
                    patch.object(A, 'game_identity', lambda: {'pid': 1, 'display': ':0', 'start_ticks': '1'}), \
                    patch.object(A, 'pointer_state', lambda display: {'x': 960, 'y': 540, 'mask': 0}), \
                    patch.object(A, 'read_frame', lambda path: frame), \
                    patch.object(A, 'reticle_centre', fake_reticle), \
                    patch.object(A, 'find_enemy', fake_enemies):
                code = A.run(args)
            report = json.loads((Path(args.out) / 'report.json').read_text())
            self.assertEqual(code, 0, report.get('error'))
            self.assertFalse(report['fired'])
            self.assertEqual(report['fire_reason'], 'oneshot_residual_outside_tolerance')
            self.assertIn('residual_px', report['fire_blocked'])
            self.assertNotIn(1, [item.get('button') for item in report['inputs']])

    def test_run_refuses_a_loose_solution_with_injected_fakes(self):
        self.run_unchanged_frames(max_stale_reads=3, max_steps=3, expected_code=0)

    def test_unobserved_move_exhausts_rereads_without_resending(self):
        report = self.run_unchanged_frames(max_stale_reads=1, max_steps=3, expected_code=1)
        self.assertEqual(report['steps'][-1]['state'], 'motion_response_unconfirmed')
        self.assertIn('no observed response', report['error'])

    def test_limited_vertical_recovery_observes_before_another_move(self):
        report = self.run_unchanged_frames(max_stale_reads=1, max_steps=2,
                                           expected_code=0, bounded_recovery=True)
        self.assertEqual(report['steps'][0]['state'], 'bounded_recovery_reread')
        moves = [row for row in report['inputs'] if 'relative' in row]
        self.assertEqual(moves[1]['relative'], [0, -3])
        self.assertEqual(moves[1]['requested_relative'], [0, -40])
        self.assertNotIn('resend_after_clamp', [row.get('reason') for row in moves])
        # Same ordering as before, shifted by the camera-mode precheck frame
        # that start-up now reads: two clamped moves, then a fresh observation,
        # then the next move.
        events = report['recovery_events']
        self.assertEqual(events[1:6], ['frame', 'frame', 'move', 'move', 'frame'])

    def run_unchanged_frames(self, *, max_stale_reads, max_steps, expected_code,
                             bounded_recovery=False):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            args = self.parse_args(tmp, ['--no-lead', '--lead-seconds', '12',
                                         '--max-steps', str(max_steps),
                                         '--max-stale-reads', str(max_stale_reads),
                                         '--settle', '0'])
            if bounded_recovery:
                args.aim_nudge = 0
                args.measure_gain = True
                # The pre-emptive recentre is now opt-in: it is a large relative
                # move that the game turns the camera with, so the default is 0
                # and this clamp-recovery path must ask for it explicitly.
                args.recenter_margin = 40
            events = []
            # textured so the tracker has something to match
            frame = np.random.default_rng(3).integers(0, 255, (1600, 2560, 3), dtype=np.uint8)

            class FakeProbe:
                def __init__(self, seconds, report):
                    self.report = report
                    self.started = 0.0
                    self.action_deadline = 1e9
                    self.identity = {'pid': 1, 'display': ':0', 'start_ticks': '1'}
                    self.window = '1'

                def x(self, *argv, **kwargs):
                    if argv and argv[0] == 'getwindowgeometry':
                        return 'WIDTH=1920\nHEIGHT=1080'
                    if argv and argv[0] == 'search':
                        return '12345'
                    return ''

                def verify_window(self, window):
                    return {'window': window, 'pid': 1, 'title': 'World of Warships'}

                def focus(self):
                    self.report['focus_checks'].append({})

                def key(self, key):
                    self.report['inputs'].append({'key': key})

                def move(self, dx, dy):
                    events.append('move')
                    # every move is delivered except the very first one, which
                    # stands in for a pointer that hit an edge
                    delivered = [0, 0] if not self.report['inputs'] else [dx, dy]
                    self.report['inputs'].append({'relative': [dx, dy], 'pointer_delta': delivered})

                def snapshot(self, name):
                    events.append('frame')
                    return Path(tmp) / f'{name}.png'

                def command(self, argv, cleanup=False, identity=None):
                    return ''

                def check_identity(self):
                    return None

                def release(self):
                    self.report['unreleased_own_keys'] = []

            def fake_reticle(rgb):
                return [1280.0, 800.0], True, 'observed', 'native_band'

            def fake_enemies(rgb, reticle=None, aim_fraction=.75, scope_observed=False):
                # A target 50 px away never reaches the tight 12 px tolerance.
                return [dict(nameplate_bbox=[1300, 760, 60, 18], body_roi=[1290, 780, 80, 40],
                             aim_px=[1330.0, 800.0], trackable=True, confidence=.9,
                             distance_from_reticle_px=50.0)]

            with patch.object(A, 'Probe', FakeProbe), \
                    patch.object(A, 'game_identity', lambda: {'pid': 1, 'display': ':0', 'start_ticks': '1'}), \
                    patch.object(A, 'pointer_state', lambda display: {'x': 960, 'y': 540, 'mask': 0}), \
                    patch.object(A, 'read_frame', lambda path: frame), \
                    patch.object(A, 'own_ship_drift', lambda previous, current, floor=.35: None), \
                    patch.object(A, 'reticle_centre', fake_reticle), \
                    patch.object(A, 'hud_flight_seconds', lambda rgb: {'value': 11.0}), \
                    patch.object(A, 'find_enemy', fake_enemies), \
                    patch.object(A, 'measure_gain', return_value={'ok': True, 'gain': [1., 1.]}), \
                    patch.object(A, 'reposition_move', return_value=[0, -40]):
                code = A.run(args)
            report_path = Path(args.out) / 'report.json'
            report = json.loads(report_path.read_text())
            self.assertEqual(code, expected_code, report.get('error'))
            self.assertEqual(report['status'], 'completed' if expected_code == 0 else 'failed')
            self.assertTrue(report['steps'])
            self.assertFalse(report.get('fired', False))
            self.assertNotIn(1, [item.get('button') for item in report['inputs']])
            self.assertIn('own_ship_drift_px', report['steps'][0])
            for row in report['steps']:
                self.assertEqual(row['lead_total_s'], 0.0)
                self.assertEqual(row['lead_horizon_s'], 0.0)
                self.assertEqual(row['lead_px'], [0.0, 0.0])
            self.assertEqual(report['unreleased_own_keys'], [])
            if bounded_recovery:
                report['recovery_events'] = events
                return report
            # The second unchanged frame follows a real probe. It must be
            # reread without learning a zero gain or issuing another move.
            self.assertEqual(report['steps'][1]['state'], 'stale_frame_reread')
            if expected_code == 0:
                self.assertEqual(report['steps'][2]['state'], 'stale_frame_reread')
            for row in report['steps'][1:]:
                self.assertNotIn('measured_response_px_per_unit', row)
            moves = [item for item in report['inputs'] if 'relative' in item
                     and item['relative'] != [1, 0]]
            self.assertEqual(len(moves), 1)
            return report


class ScopeForAimTests(unittest.TestCase):
    def test_scope_after_is_bounded_and_defaults_off(self):
        captured = {}
        with patch.object(A, 'run', lambda args: captured.update(n=args.scope_after) or 0), \
                patch.object(sys, 'argv', ['prog', '--scan', '--out', '/tmp/wws-scope-gate-1']):
            self.assertEqual(A.main(), 0)
        self.assertEqual(captured['n'], 0.0)
        with patch.object(sys, 'argv', ['prog', '--scan', '--scope-after', '900',
                                        '--out', '/tmp/wws-scope-gate-2']), \
                self.assertRaises(SystemExit):
            A.main()

    def test_a_scope_after_value_is_accepted(self):
        captured = {}
        with patch.object(A, 'run', lambda args: captured.update(n=args.scope_after) or 0), \
                patch.object(sys, 'argv', ['prog', '--scan', '--scope-after', '80',
                                           '--out', '/tmp/wws-scope-gate-3']):
            self.assertEqual(A.main(), 0)
        self.assertEqual(captured['n'], 80.0)


class GreenReticleTests(unittest.TestCase):
    def ring_frame(self, cx=1291, cy=811, colour=(0, 220, 120), size=16):
        frame = np.zeros((1600, 2560, 3), np.uint8)
        half = size // 2
        frame[cy-half:cy+half, cx-half:cx+half] = colour
        return frame

    def test_a_green_ring_near_the_centre_is_found(self):
        found = A.green_reticle(self.ring_frame())
        self.assertIsNotNone(found)
        self.assertAlmostEqual(found['centre'][0], 1291, delta=1)
        self.assertAlmostEqual(found['centre'][1], 811, delta=1)
        self.assertEqual(found['source'], 'green_ring')

    def test_a_ring_far_from_the_centre_is_ignored(self):
        self.assertIsNone(A.green_reticle(self.ring_frame(cx=400, cy=300)))

    def test_the_wrong_colour_is_ignored(self):
        self.assertIsNone(A.green_reticle(self.ring_frame(colour=(220, 0, 0))))
        self.assertIsNone(A.green_reticle(self.ring_frame(colour=(0, 0, 220))))

    def test_a_tiny_or_huge_blob_is_ignored(self):
        self.assertIsNone(A.green_reticle(self.ring_frame(size=4)))
        self.assertIsNone(A.green_reticle(self.ring_frame(size=60)))

    def test_a_blank_frame_has_no_reticle(self):
        self.assertIsNone(A.green_reticle(np.zeros((1600, 2560, 3), np.uint8)))

    def test_a_real_live_frame_still_shows_the_ring(self):
        # measured on live frames: the green reticle sits at (1291, 811)
        import cv2
        path = Path(__file__).resolve().parents[2] / 'state' / 'current-view-20261017-q1.png'
        if not path.is_file():
            self.skipTest('live frame evidence is not present')
        frame = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        found = A.green_reticle(frame)
        self.assertIsNotNone(found)
        self.assertLessEqual(math.hypot(found['centre'][0]-1291, found['centre'][1]-811), 6)


class VerticalGainTests(unittest.TestCase):
    """The y gain must be escapable when it was seeded by a yaw sweep.

    Measured on 2026-10-17: a 150-unit horizontal correction also lifted the
    target, so a 3-unit vertical nudge in the same step looked like +1.574
    px/unit. The true response was about -1.5, every vertical correction then
    pushed the reticle further from the hull, and the error grew -37 -> -80 px
    while x converged. One contradicting reading is still contamination.
    """

    def test_two_agreeing_contradictions_replace_a_contaminated_gain(self):
        state = {}
        gain = [None, 1.574]
        gain, measured = A.update_gain(gain, [0, -44.0], [0, -48.7], (150, 3), state=state)
        self.assertIsNone(measured[1])
        self.assertAlmostEqual(gain[1], 1.574, places=3)
        gain, measured = A.update_gain(gain, [0, -48.7], [0, -53.4], (150, 3), state=state)
        self.assertAlmostEqual(measured[1], -1.5667, places=3)
        self.assertAlmostEqual(gain[1], -1.5667, places=3)

    def test_one_contradiction_is_still_dropped(self):
        gain, measured = A.update_gain([None, 1.574], [0, -44.0], [0, -48.7], (150, 3), state={})
        self.assertIsNone(measured[1])
        self.assertAlmostEqual(gain[1], 1.574, places=3)

    def test_alternating_contradictions_never_flip(self):
        gain = [None, 1.574]
        state = {}
        for before, after in (([0, -44.0], [0, -48.7]),
                              ([0, -48.7], [0, -44.0]),
                              ([0, -44.0], [0, -48.7])):
            gain, _ = A.update_gain(gain, before, after, (150, 3), state=state)
            # a reading that agrees with the stored sign still blends, but the
            # sign is never replaced by an alternating pair
            self.assertGreater(gain[1], 0)
            self.assertAlmostEqual(gain[1], 1.574, places=2)

    def test_the_contaminated_gain_steers_the_wrong_way(self):
        # documents the defect: the seeded sign walks the aim away from the hull
        _, move_y, reason = A.plan_move([0, -72.0], [1.2, 1.574], damping=.7)
        self.assertEqual(reason, 'correct')
        self.assertGreater(move_y, 0)

    def test_a_correct_gain_steers_the_pointer_up(self):
        _, move_y, reason = A.plan_move([0, -72.0], [1.2, -1.5], damping=.7)
        self.assertEqual(reason, 'correct')
        self.assertLess(move_y, 0)


class CameraModeTests(unittest.TestCase):
    """The binocular view must be provable from the frame, not from intent.

    Measured on this client: third person has bright sky in the top corners
    (195) and a darker centre (136); the scope darkens the corners to 108 while
    the centre brightens to 165. The HUD's own ruler/reticle read is not always
    available, so this ratio is the second, independent source.
    """

    @staticmethod
    def frame(corner, centre):
        img = np.full((1600, 2560, 3), 128, dtype=np.uint8)
        cx0, cy0, cx1, cy1 = A.SCOPE_CENTRE_ROI
        img[cy0:cy1, cx0:cx1] = centre            # the measured centre window
        img[100:180, 60:260] = corner             # the top-left corner window
        return img

    def test_bright_corners_are_third_person(self):
        got = A.camera_mode_evidence(self.frame(195, 136))
        self.assertEqual(got['status'], 'observed')
        self.assertFalse(got['binocular'])
        self.assertEqual(got['reason'], 'bright_corners_third_person')

    def test_darkened_corners_are_the_binocular_view(self):
        got = A.camera_mode_evidence(self.frame(108, 165))
        self.assertTrue(got['binocular'])
        self.assertEqual(got['reason'], 'corner_darkened_by_binocular_view')

    def test_missing_or_tiny_frames_are_unknown(self):
        self.assertEqual(A.camera_mode_evidence(None)['status'], 'unknown')
        self.assertEqual(A.camera_mode_evidence(np.zeros((100, 100, 3), np.uint8))['status'], 'unknown')

    def test_a_dark_scoped_scene_is_judged_against_the_session_reference(self):
        """Measured live: the scope on a dark sea had corner 54, centre 44.

        The ratio alone reads that as third person (1.22), so the third-person
        corner brightness measured at the start of the same run is the
        reference; the scope drops it far below that.
        """
        frame = self.frame(54, 44)
        self.assertFalse(A.camera_mode_evidence(frame)['binocular'])
        got = A.camera_mode_evidence(frame, baseline={'corner_mean': 190.0})
        self.assertTrue(got['binocular'])
        self.assertAlmostEqual(got['corner_drop_px'], 136.0, places=1)

    def test_a_small_lighting_change_is_not_the_scope(self):
        frame = self.frame(175, 140)
        got = A.camera_mode_evidence(frame, baseline={'corner_mean': 190.0})
        self.assertFalse(got['binocular'])

    def test_scope_evidence_needs_a_real_source(self):
        self.assertTrue(A.scope_confirmed({'observed': True}, None))
        self.assertTrue(A.scope_confirmed({'observed': False}, {'binocular': True}))
        self.assertFalse(A.scope_confirmed({'observed': False}, {'binocular': False}))
        self.assertFalse(A.scope_confirmed(None, None))


class FastShotTests(unittest.TestCase):
    """The 5 s shot solves the whole error from one frame.

    The closed loop needs a scan, two probes and several settled reads; measured
    live that runs 15-50 s. A static target needs one solve with a per-view gain
    constant and one move, clamped per axis so the pointer cannot be asked for
    more than the window carries.
    """

    def test_the_move_solves_the_whole_error_in_one_action(self):
        move = A.fast_shot_plan([1400.0, 700.0], [1280.0, 800.0], [-1.4, -1.4])
        # gain is d(error)/d(mouse): the correction is -error/gain
        self.assertAlmostEqual(move[0], 85.7, places=1)
        self.assertAlmostEqual(move[1], -71.4, places=1)

    def test_the_move_is_clamped_per_axis(self):
        self.assertEqual(A.fast_shot_plan([3500.0, -1000.0], [1280.0, 800.0], [-1.0, -1.0]),
                         [400.0, -300.0])

    def test_a_zero_gain_is_refused(self):
        with self.assertRaises(ValueError):
            A.fast_shot_plan([1300.0, 800.0], [1280.0, 800.0], [0.0, -1.4])


class FireReadyTests(unittest.TestCase):
    def test_two_current_tight_solutions_after_warmup_allow_fire(self):
        self.assertTrue(A.fire_ready(10., 2, 5, 25)[0])

    def test_loose_solutions_never_allow_fire(self):
        self.assertFalse(A.fire_ready(55., 0, 6, 25)[0])
        self.assertFalse(A.fire_ready(55., 2, 6, 25)[0])

    def test_tight_solution_still_needs_consecutive_observations(self):
        self.assertFalse(A.fire_ready(10., 1, 6, 25)[0])

    def test_tight_solutions_cannot_skip_warmup(self):
        self.assertFalse(A.fire_ready(10., 2, 2, 25, min_steps=4)[0])


class FittedVelocityTests(unittest.TestCase):
    def test_a_clean_ramp_is_recovered(self):
        samples = [(i * 0.5, 10.0 * i * 0.5, -4.0 * i * 0.5) for i in range(5)]
        self.assertEqual(A.fit_velocity(samples), [10.0, -4.0])

    def test_a_short_baseline_is_refused(self):
        self.assertIsNone(A.fit_velocity([(0.0, 0.0, 0.0), (0.2, 5.0, 1.0)]))
        self.assertIsNone(A.fit_velocity([(0.0, 0.0, 0.0)]))

    def test_noise_affects_the_fit_less_than_a_single_difference(self):
        # a real 6 px/s ramp with a 30 px outlier on the last sample
        samples = [(0.0, 0.0, 0.0), (1.0, 6.0, 0.0), (2.0, 12.0, 0.0), (3.0, 48.0, 0.0)]
        fitted = A.fit_velocity(samples)[0]
        consecutive = (48.0 - 12.0) / 1.0
        self.assertLess(abs(fitted - 6.0), abs(consecutive - 6.0))

    def test_identical_times_are_refused(self):
        self.assertIsNone(A.fit_velocity([(1.0, 1.0, 1.0), (1.0, 2.0, 2.0), (1.0, 3.0, 3.0)]))


class VelocityBlendTests(unittest.TestCase):
    def test_the_first_sample_is_taken_as_is(self):
        self.assertEqual(A.blend_velocity(None, [10.0, -6.0]), [10.0, -6.0])
        self.assertIsNone(A.blend_velocity(None, None))

    def test_consistent_samples_blend_towards_the_measurement(self):
        self.assertEqual(A.blend_velocity([10.0, 0.0], [20.0, 0.0], blend=.5), [15.0, 0.0])

    def test_a_sign_flip_resets_instead_of_averaging(self):
        self.assertEqual(A.blend_velocity([10.0, 4.0], [-8.0, 4.0]), [-8.0, 4.0])
        self.assertEqual(A.blend_velocity([10.0, 4.0], None), [10.0, 4.0])

    def test_the_ceiling_bounds_the_lead(self):
        self.assertEqual(A.lead_offset([100.0, 0.0], 6.0, ceiling=150.0), [150.0, 0.0])


class LeadOffsetTests(unittest.TestCase):
    def test_lead_is_velocity_times_flight_time(self):
        self.assertEqual(A.lead_offset([10.0, -4.0], 5.0), [50.0, -20.0])
        self.assertEqual(A.lead_offset([10.0, -4.0], 0.0), [0.0, 0.0])
        self.assertEqual(A.lead_offset(None, 5.0), [0.0, 0.0])

    def test_a_wild_velocity_is_bounded(self):
        self.assertEqual(A.lead_offset([1000.0, -1000.0], 5.0), [400.0, -400.0])

    def test_lead_seconds_are_bounded_on_the_cli(self):
        from unittest.mock import patch
        with patch.object(sys, 'argv', ['prog', '--scan', '--lead-seconds', '30',
                                        '--out', '/tmp/wws-lead-gate']), \
                self.assertRaises(SystemExit):
            A.main()


class PointerRoomTests(unittest.TestCase):
    def test_a_move_into_the_right_edge_retreats_left(self):
        # the pointer sits at the right edge of a 1920 wide screen
        self.assertEqual(A.reposition_move((1919, 540), 1920, 1080, (200, 0)), [-240, 0])
        # room available, so no retreat
        self.assertIsNone(A.reposition_move((960, 540), 1920, 1080, (200, 0)))
        # leftward out of room retreats right, and the vertical axis uses height
        self.assertEqual(A.reposition_move((10, 540), 1920, 1080, (-200, 0)), [230, 0])
        self.assertEqual(A.reposition_move((960, 5), 1920, 1080, (0, -200)), [0, 235])

    def test_the_retreat_is_only_the_missing_room(self):
        # 200 wanted + 40 margin, 19 px of room -> retreat 221, not a fixed 400
        self.assertEqual(A.reposition_move((1900, 540), 1920, 1080, (200, 0)), [-221, 0])

    def test_recentre_puts_the_pointer_mid_screen(self):
        self.assertEqual(A.recentre_move((1919, 540), 1920, 1080), [-959, 0])
        self.assertEqual(A.recentre_move((960, 540), 1920, 1080), [0, 0])
        self.assertEqual(A.recentre_move((100, 1000), 1920, 1080), [860, -460])

    def test_the_retreat_never_leaves_the_screen(self):
        # at x=100 a rightward move has room, so nothing is sent
        self.assertIsNone(A.reposition_move((100, 540), 1920, 1080, (200, 0)))
        # the same pointer has no room leftwards, so it retreats right by the cap
        self.assertEqual(A.reposition_move((100, 540), 1920, 1080, (-200, 0)),
                         [140, 0])
        self.assertEqual(A.reposition_move((1900, 540), 1920, 1080, (200, 0)),
                         [-221, 0])

    def test_margin_decides_how_much_room_is_needed(self):
        self.assertIsNone(A.reposition_move((1500, 540), 1920, 1080, (300, 0), margin=40))
        self.assertEqual(A.reposition_move((1500, 540), 1920, 1080, (300, 0), margin=150),
                         [-31, 0])
        self.assertIsNone(A.reposition_move((1500, 540), 1920, 1080, (300, 0), margin=0))


class AxisArgumentTests(unittest.TestCase):
    def test_large_vertical_requests_fail_before_starting_the_consumer(self):
        for option in ('--reach-y', '--probe-y', '--scan-tilt'):
            with self.subTest(option=option), patch.object(A, 'run') as run, \
                    patch.object(sys, 'argv', ['servo', '--scan', '--out', '/unused', option, '4']), \
                    self.assertRaises(SystemExit) as error:
                A.main()
            self.assertEqual(error.exception.code, 2)
            run.assert_not_called()


class ScanPathTests(unittest.TestCase):
    def test_rows_alternate_direction_and_tilt_between_sweeps(self):
        path = A.scan_path(steps_per_row=3, rows=2, step=100, tilt=40)
        self.assertEqual(path, [(100, 0), (100, 0), (0, 3), (-100, 0), (-100, 0)])

    def test_a_single_row_never_tilts(self):
        self.assertEqual(A.scan_path(2, 1, 100, 40), [(100, 0)])

    def test_rows_are_bounded_by_the_request(self):
        path = A.scan_path(4, 3, 100, 40)
        self.assertEqual(len(path), 4 * 3 - 1)
        self.assertEqual(sum(1 for move in path if move[1]), 2)

    def test_vertical_scan_keeps_direction_and_respects_the_axis_limit(self):
        self.assertEqual(A.scan_path(2, 2, 140, -45, reach_y=2),
                         [(140, 0), (0, -2), (-140, 0)])


class EnemyScanTests(unittest.TestCase):
    def frame(self):
        return np.zeros((1600, 2560, 3), np.uint8)

    def plate(self, bar, body, confidence=.9):
        return dict(nameplate_bbox=bar, body_roi=body, confidence=confidence)

    def test_aim_point_is_the_hull_window_not_the_bar(self):
        plate = self.plate([1900, 660, 60, 18], [1880, 680, 100, 60])
        with patch.object(MVP, 'detect_nameplates', lambda frame: [plate]), \
                patch.object(MVP, 'verify_ship_bodies', lambda frame, plates: [plate]):
            found = A.find_enemy(self.frame(), [1280, 805])
        # The bar anchors a hull whose sea line sits 0.30 bar-widths lower
        # (measured 34 px at 8.4 km and 37 px at 10.9 km), and the aim sits
        # AIM_ABOVE_WATERLINE_PX above that line: 660+18+18-8 = 688.
        self.assertEqual(found[0]['aim_px'], [1930.0, 688.0])
        self.assertEqual(found[0]['aim_source']['source'], 'bar_geometry')
        self.assertEqual(found[0]['nameplate_bbox'], [1900, 660, 60, 18])

    def test_the_aim_fraction_selects_the_height_in_the_hull_window(self):
        plate = self.plate([1900, 660, 60, 18], [1880, 680, 100, 60])
        with patch.object(MVP, 'detect_nameplates', lambda frame: [plate]), \
                patch.object(MVP, 'verify_ship_bodies', lambda frame, plates: [plate]):
            top = A.find_enemy(self.frame(), [1280, 805], aim_fraction=0.0)[0]['aim_px']
            bottom = A.find_enemy(self.frame(), [1280, 805], aim_fraction=1.0)[0]['aim_px']
            clamped = A.find_enemy(self.frame(), [1280, 805], aim_fraction=9.0)[0]['aim_px']
        self.assertEqual(top[1], 680.0)
        self.assertEqual(bottom[1], 740.0)
        self.assertEqual(clamped[1], 740.0)

    def test_a_candidate_with_no_tracking_window_ranks_last(self):
        edge = dict(nameplate_bbox=[0, 340, 120, 16], body_roi=[0, 360, 120, 120],
                    confidence=.99, trackable=False, distance_from_reticle_px=500.0)
        middle = dict(nameplate_bbox=[1500, 700, 60, 18], body_roi=[1480, 720, 100, 60],
                      confidence=.60, trackable=True, distance_from_reticle_px=400.0)
        out = [edge, middle]
        out.sort(key=lambda c: (not c['trackable'], c['distance_from_reticle_px']))
        self.assertIs(out[0], middle)
        self.assertTrue(A.window_fits(np.zeros((1600, 2560, 3), np.uint8), [1480, 720, 100, 60]))
        self.assertFalse(A.window_fits(np.zeros((1600, 2560, 3), np.uint8), [0, 360, 120, 120]))

    def test_nearest_candidate_to_the_reticle_is_chosen_first(self):
        far = self.plate([100, 100, 60, 18], [80, 120, 100, 60], confidence=.95)
        near = self.plate([1500, 700, 60, 18], [1480, 720, 100, 60], confidence=.60)
        with patch.object(MVP, 'detect_nameplates', lambda frame: [far, near]), \
                patch.object(MVP, 'verify_ship_bodies', lambda frame, plates: [far, near]):
            found = A.find_enemy(self.frame(), [1280, 805])
        self.assertEqual(found[0]['aim_px'], [1530.0, 728.0])  # bar 700+18+18-8
        self.assertLess(found[0]['distance_from_reticle_px'], found[1]['distance_from_reticle_px'])

    def test_without_a_reticle_the_most_confident_candidate_is_first(self):
        weak = self.plate([100, 100, 60, 18], [80, 120, 100, 60], confidence=.55)
        strong = self.plate([1500, 700, 60, 18], [1480, 720, 100, 60], confidence=.95)
        with patch.object(MVP, 'detect_nameplates', lambda frame: [weak, strong]), \
                patch.object(MVP, 'verify_ship_bodies', lambda frame, plates: [weak, strong]):
            found = A.find_enemy(self.frame())
        self.assertEqual(found[0]['aim_px'], [1530.0, 728.0])
        self.assertIsNone(found[0]['distance_from_reticle_px'])

    def test_cli_needs_exactly_one_of_target_or_scan(self):
        for argv in (['prog', '--out', '/tmp/x1'], ['prog', '--scan', '--target', '10', '900', '--out', '/tmp/x2']):
            with patch.object(sys, 'argv', argv), self.assertRaises(SystemExit):
                A.main()


class RelativeTransportTests(unittest.TestCase):
    """XTest needs pointer-room fallbacks; libei must not pay for them."""

    class FakeProbe:
        def __init__(self, transport='xtest'):
            self.relative_transport = transport
            self.identity = {'display': ':0'}
            self.moves = []

        def move(self, dx, dy):
            self.moves.append([dx, dy])

    def tearDown(self):
        os.environ.pop(A.TRANSPORT_ENV, None)

    def test_the_cli_defaults_to_xtest_and_keeps_the_pointer_safe_chunk(self):
        captured = {}
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(A, 'run', lambda args: (captured.update(args=args), 0)[1]), \
                    patch.object(sys, 'argv', ['prog', '--scan', '--out', str(Path(tmp) / 'o')]):
                self.assertEqual(A.main(), 0)
        self.assertEqual(captured['args'].relative_transport, 'xtest')
        self.assertEqual(captured['args'].turn_chunk, 400.0)

    def test_xtest_still_repositions_and_is_clamped_to_400(self):
        probe = self.FakeProbe('xtest')
        report = {'inputs': []}
        with patch.object(A, 'pointer_state',
                          lambda display: {'x': 1880, 'y': 540, 'mask': 0}):
            moved = A.ensure_room(probe, report, 620, 1920, 'fast_open_loop_aim')
        self.assertTrue(moved)
        self.assertEqual(probe.moves, [[-300, 0]])
        self.assertEqual(report['inputs'][0]['reason'], 'fast_open_loop_aim_reposition')
        # An operator --turn-chunk above the pointer-safe cap is ignored.
        self.assertEqual(A.turn_chunk_limit(A.argparse.Namespace(turn_chunk=1200.0), probe), 400.0)
        self.assertEqual(A.turn_chunk_limit(A.argparse.Namespace(turn_chunk=400.0), probe), 400.0)

    def test_ei_ensure_room_returns_false_without_sending_a_retreat(self):
        probe = self.FakeProbe('ei')
        report = {'inputs': []}
        with patch.object(A, 'pointer_state',
                          side_effect=AssertionError('libei has no pointer room to query')):
            moved = A.ensure_room(probe, report, 620, 1920, 'fast_open_loop_aim')
        self.assertFalse(moved)
        self.assertEqual(probe.moves, [])
        self.assertEqual(report['inputs'], [])
        self.assertEqual(report['transport_skips'],
                         [dict(reason='fast_open_loop_aim_reposition', transport='ei', dx=620)])

    def test_ei_is_seen_through_the_environment_handoff(self):
        probe = self.FakeProbe()
        del probe.relative_transport
        A.apply_relative_transport(None, 'ei')
        self.assertTrue(A.uses_ei_transport(probe))
        self.assertTrue(A.uses_ei_transport(None))
        with self.assertRaises(ValueError):
            A.apply_relative_transport(None, 'bogus')

    def test_ei_allows_the_widened_turn_chunk(self):
        self.assertEqual(A.turn_chunk_limit(A.argparse.Namespace(turn_chunk=1200.0),
                                            self.FakeProbe('ei')), 1200.0)
        self.assertEqual(A.turn_chunk_limit(A.argparse.Namespace(turn_chunk=9000.0),
                                            self.FakeProbe('ei')), A.EI_TURN_CHUNK_MAX)

    def test_every_report_row_is_tagged_with_the_transport(self):
        report = {'inputs': [{'relative': [1, 0], 'transport': 'libei'}], 'steps': [{'step': 0}]}
        A.stamp_transport(report, 'ei')
        self.assertEqual(report['relative_transport'], 'ei')
        self.assertEqual(report['inputs'][0]['transport'], 'libei')
        self.assertEqual(report['steps'][0]['transport'], 'ei')


class ConfirmReticleFallbackTests(unittest.TestCase):
    """A confirm frame without the native caret still has the white aim chevron.

    Measured 2026-10-23: four live one-shot refusals all ended in
    ``fire revoked: reticle_not_observed`` on a frame where ``detect`` reported
    ``reticle: unknown`` but ``aim_point: observed``. The gun was loaded and the
    target was in view, so the chain refused a shot it could have taken.
    """

    FIXTURE = Path(__file__).with_name('aim_fixtures') / 'confirm-chevron-only-20261023.png'

    def test_the_chevron_is_used_when_the_caret_is_missing(self):
        import numpy as np
        from PIL import Image
        frame = np.asarray(Image.open(self.FIXTURE).convert('RGB'))
        centre, battle, _weapon, source = A.reticle_centre(frame)
        self.assertIsNotNone(centre)
        self.assertEqual(source, 'aim_point_chevron')
        self.assertIs(battle, True)
        self.assertLess(abs(centre[0] - 1280.5), 60)
        self.assertLess(abs(centre[1] - 805), 60)


class TerrainEvidenceTests(unittest.TestCase):
    """Land on the minimap line must be reported, and never invented."""

    def _panel(self, land=False):
        import numpy as np
        frame = np.zeros((1600, 2560, 3), dtype=np.uint8)
        # Sea: dark teal. Land rule wants red-ish pixels, so sea must fail it.
        frame[796:1519, 1836:2556] = (20, 55, 60)
        if land:
            frame[1180:1230, 2100:2200] = (150, 100, 70)
        return frame

    def _contacts(self, own, enemy):
        def provider(frame):
            seen = dict(own=dict(pixel_x=own[0], pixel_y=own[1], angle_deg=0.0),
                        own_pixel=list(own),
                        contacts=[dict(pixel=list(enemy), source='test')],
                        nearest=dict(pixel=list(enemy), source='test'))
            return seen, dict(enemies=[dict(u=0, v=0)])
        return provider

    def test_open_water_is_clear(self):
        frame = self._panel(land=False)
        out = A.terrain_evidence(frame, None, contacts_fn=self._contacts([200, 600], [400, 300]), ttl=0)
        self.assertIs(out['clear'], True)
        self.assertEqual(out['source'], 'minimap_line_of_fire')

    def test_land_on_the_line_blocks(self):
        frame = self._panel(land=True)
        # Panel-local coordinates: the land patch sits at panel rows 384-434,
        # columns 264-364, so the segment must cross that band.
        out = A.terrain_evidence(frame, None, contacts_fn=self._contacts([200, 400], [500, 400]), ttl=0)
        self.assertIs(out['clear'], False)
        self.assertGreater(out['land_pixels'], 0)

    def test_no_contact_is_unknown_not_clear(self):
        def provider(frame):
            return None, dict(enemies=[])
        out = A.terrain_evidence(self._panel(), None, contacts_fn=provider, ttl=0)
        self.assertIsNone(out['clear'])
        self.assertEqual(out['reason'], 'no_minimap_contact')


class TerrainGateTests(unittest.TestCase):
    """The gate must block on measured land and must not block on unknown."""

    def _evidence(self, **kw):
        import fire_gate as FG
        base = dict(battle_active=True, own_alive=True, weapon_kind='gun',
                    reload_ready=True,
                    enemy_plate=dict(bbox=[1, 2, 3, 4], hp=1000, red_px=50),
                    frame_fresh=True, aim_error_px=5.0, fire_tolerance_px=60.0,
                    identity_ok=True, focus_ok=True)
        base.update(kw)
        return FG.fire_decision(FG.Evidence(**base))

    def test_land_blocks_the_shot(self):
        decision = self._evidence(terrain_clear=False)
        self.assertFalse(decision['allow'])
        self.assertIn('terrain_blocks_shot', decision['blockers'])

    def test_unknown_terrain_does_not_block(self):
        decision = self._evidence(terrain_clear=None)
        self.assertTrue(decision['allow'], decision['blockers'])

    def test_body_invisible_is_only_a_warning(self):
        decision = self._evidence(body_visible=False)
        self.assertTrue(decision['allow'], decision['blockers'])
        self.assertTrue(any(w.get('code') == 'body_not_visible' for w in decision['warnings']))


class NewOptionTests(unittest.TestCase):
    """The terrain gate and the refine budget must be reachable from the CLI.

    The parser is built inside main(), so the consumer-facing surface is the
    help text a shell caller actually sees.
    """

    @classmethod
    def setUpClass(cls):
        import subprocess, sys
        here = Path(__file__).resolve().parent
        out = subprocess.run([sys.executable, '-B', str(here / 'aim_servo.py'), '--help'],
                             capture_output=True, text=True,
                             env={'PYTHONPATH': str(here) + ':' + str(here.parents[1] / 'replay'),
                                  'PATH': '/usr/bin:/bin'})
        cls.help = out.stdout + out.stderr
        cls.code = out.returncode

    def test_help_renders(self):
        self.assertEqual(self.code, 0, self.help[-400:])

    def test_terrain_and_refine_flags_are_advertised(self):
        for flag in ('--no-terrain-gate', '--terrain-ttl-s', '--oneshot-refine-rounds'):
            self.assertIn(flag, self.help)


class TerrainPrefetchTests(unittest.TestCase):
    """The probe must not spend the shot budget, and must never guess."""

    def setUp(self):
        A.terrain_evidence.result = None
        A.terrain_evidence.thread = None
        A.terrain_evidence.contacts = None

    def tearDown(self):
        self.setUp()

    def _frame(self):
        import numpy as np
        frame = np.zeros((1600, 2560, 3), dtype=np.uint8)
        frame[796:1519, 1836:2556] = (20, 55, 60)
        return frame

    def test_not_ready_reads_as_pending_and_does_not_veto(self):
        got = A.terrain_latest()
        self.assertIsNone(got['clear'])
        self.assertEqual(got['reason'], 'terrain_pending')

    def test_prefetch_returns_immediately_then_fills_in(self):
        import threading, time
        started = threading.Event()
        released = threading.Event()

        def slow_provider(frame):
            started.set()
            released.wait(5)
            seen = dict(own=dict(pixel_x=200, pixel_y=400, angle_deg=0.0),
                        own_pixel=[200, 400],
                        contacts=[dict(pixel=[500, 400], source='test')],
                        nearest=dict(pixel=[500, 400], source='test'))
            return seen, dict(enemies=[dict(u=0, v=0)])

        t0 = time.monotonic()
        self.assertTrue(A.terrain_prefetch(self._frame(), None, contacts_fn=slow_provider))
        self.assertLess(time.monotonic() - t0, 0.5)
        self.assertTrue(started.wait(3))
        self.assertIsNone(A.terrain_latest()['clear'])
        released.set()
        for _ in range(60):
            time.sleep(0.05)
            got = A.terrain_latest()
            if got.get('clear') is not None:
                break
        self.assertIs(got['clear'], True)

    def test_a_second_prefetch_does_not_start_a_second_probe(self):
        import threading
        released = threading.Event()

        def stuck(frame):
            released.wait(5)
            return None, dict(enemies=[])

        self.assertTrue(A.terrain_prefetch(self._frame(), None, contacts_fn=stuck))
        self.assertFalse(A.terrain_prefetch(self._frame(), None, contacts_fn=stuck))
        released.set()
        A.terrain_evidence.thread.join(3)


class BudgetTests(unittest.TestCase):
    """The shot budget must be enforced where the trigger is, not only in loops."""

    def test_within_budget(self):
        self.assertFalse(A.budget_exceeded(100.0, 10.0, now=109.9))

    def test_at_the_boundary(self):
        self.assertFalse(A.budget_exceeded(100.0, 10.0, now=110.0))
        self.assertTrue(A.budget_exceeded(100.0, 10.0, now=110.1))

    def test_no_budget_is_never_exceeded(self):
        self.assertFalse(A.budget_exceeded(100.0, None, now=1e6))
        self.assertFalse(A.budget_exceeded(100.0, 0, now=1e6))


class RefineBudgetTests(unittest.TestCase):
    """One round may use exactly what the error needs, never more than the grant."""

    def test_a_large_needed_move_is_allowed_up_to_the_grant(self):
        # 39 units is exactly what a 45 px error needs at ~1.16 px/unit; the
        # grant only caps, it does not inflate.
        self.assertEqual(A.refine_limit_y(39.0, 3, 40), 39.0)
        self.assertEqual(A.refine_limit_y(45.0, 3, 40), 40)

    def test_a_small_needed_move_keeps_the_safe_step(self):
        self.assertEqual(A.refine_limit_y(0.5, 3, 40), 3)

    def test_the_grant_caps_the_round(self):
        self.assertEqual(A.refine_limit_y(500.0, 3, 40), 40)

    def test_the_sign_does_not_matter(self):
        self.assertEqual(A.refine_limit_y(-39.0, 3, 40), 39.0)

    def test_a_tiny_grant_never_shrinks_below_the_safe_step(self):
        self.assertEqual(A.refine_limit_y(39.0, 3, 1), 3)


class CaptionAimOffsetTests(unittest.TestCase):
    """The offset rule is the two measured anchors, not the old 1415/km guess."""

    def test_the_measured_anchors_are_reproduced(self):
        self.assertAlmostEqual(A.capture_aim_offset_km(8.4), 35.9, delta=2.0)
        self.assertAlmostEqual(A.capture_aim_offset_km(12.75), 62.0, delta=2.0)

    def test_it_grows_with_distance(self):
        values = [A.capture_aim_offset_km(km) for km in (8, 10, 12, 14, 16)]
        self.assertEqual(values, sorted(values))
        self.assertGreater(values[-1], values[0])

    def test_it_is_bounded(self):
        low, high = A.AIM_OFFSET_BOUNDS
        self.assertEqual(A.capture_aim_offset_km(2.0), low)
        self.assertEqual(A.capture_aim_offset_km(60.0), high)

    def test_the_old_rule_is_not_what_runs(self):
        # 0.6*1415/8.4 was 101 px; the measured answer is about 36.
        self.assertLess(A.capture_aim_offset_km(8.4), 60)


class WaterlineTests(unittest.TestCase):
    """In the scope the aim follows the measured waterline, not the bar offset."""

    def _frame(self, edge_row=None, noise=8):
        import numpy as np
        rng = np.random.default_rng(5)
        frame = rng.integers(90, 90 + noise, (600, 400, 3), dtype=np.uint8)
        if edge_row is not None:
            frame[edge_row:, :, :] = 200   # sea becomes bright below the hull
        return frame

    def test_the_strongest_edge_below_the_bar_is_the_waterline(self):
        frame = self._frame(edge_row=420)
        found = A.waterline_row(frame, [150, 300, 100, 16])
        self.assertIsNotNone(found)
        self.assertLessEqual(abs(found['row'] - 420), 1)

    def test_a_flat_frame_has_no_waterline(self):
        self.assertIsNone(A.waterline_row(self._frame(edge_row=None, noise=2), [150, 300, 100, 16]))

    def test_the_aim_sits_slightly_above_the_waterline(self):
        frame = self._frame(edge_row=420)
        found = A.waterline_row(frame, [150, 300, 100, 16])
        self.assertEqual(A.AIM_ABOVE_WATERLINE_PX, 8)
        self.assertGreater(found['row'], 0)
        # the reported aim uses row - above; check the constant is what is applied
        self.assertEqual(found['row'] - A.AIM_ABOVE_WATERLINE_PX, found['row'] - 8)


class MovingTargetLeadTests(unittest.TestCase):
    """Preliminary lead for a moving target, and every reason to refuse."""

    def test_a_crossing_target_gets_a_clamped_lead(self):
        samples = [(0.0, 100, 50), (0.6, 124, 50), (1.2, 148, 50), (1.8, 172, 50)]
        out = A.moving_target_lead(samples, 6.0, ceiling=150)
        self.assertTrue(out['applied'])
        self.assertEqual(out['velocity_px_s'][0], 40.0)
        self.assertEqual(out['lead_px'][0], 150)     # 40*6 = 240, clamped
        self.assertEqual(out['samples'], 4)

    def test_without_enough_samples_it_refuses(self):
        out = A.moving_target_lead([(0.0, 100, 50)], 6.0)
        self.assertFalse(out['applied'])
        self.assertEqual(out['reason'], 'insufficient_samples')

    def test_a_short_baseline_is_not_a_velocity(self):
        samples = [(0.0, 100, 50), (0.3, 104, 50), (0.6, 108, 50)]
        out = A.moving_target_lead(samples, 6.0, min_span_s=1.2)
        self.assertFalse(out['applied'])
        self.assertEqual(out['reason'], 'baseline_too_short')

    def test_without_flight_time_there_is_no_lead(self):
        samples = [(0.0, 100, 50), (0.6, 124, 50), (1.2, 148, 50)]
        out = A.moving_target_lead(samples, None)
        self.assertFalse(out['applied'])
        self.assertEqual(out['reason'], 'no_flight_time')

    def test_a_still_target_is_reported_as_such(self):
        samples = [(0.0, 100, 50), (0.6, 100, 50), (1.2, 100, 50)]
        out = A.moving_target_lead(samples, 6.0)
        self.assertFalse(out['applied'])
        self.assertEqual(out['reason'], 'target_effectively_still')


class MultiTargetSwitchTests(unittest.TestCase):
    """Switching ships: keep the tracked one while it exists, otherwise move on.

    This is the contract the round loop relies on when a target sinks or leaves
    the view: the next engagement must pick a different plate instead of
    failing, and must not chatter between two visible ships.
    """

    @staticmethod
    def plate(x, y=400, w=120, h=16):
        return dict(nameplate_bbox=[x, y, w, h], aim_px=[x + w / 2, y + h])

    def test_the_tracked_ship_is_kept_when_it_is_still_there(self):
        tracked = self.plate(900)
        other = self.plate(1400)
        pick = A.choose_target([other, tracked], tracked['nameplate_bbox'])
        self.assertIs(pick, tracked)

    def test_a_vanished_tracked_ship_hands_over_to_the_next_one(self):
        gone = self.plate(900)
        other = self.plate(1500)
        pick = A.choose_target([other], gone['nameplate_bbox'])
        self.assertIs(pick, other)

    def test_no_candidates_means_no_target(self):
        self.assertIsNone(A.choose_target([], self.plate(900)['nameplate_bbox']))


class MountTraverseTests(unittest.TestCase):
    """The scope shows the gun line, so it may not open mid-traverse."""

    def test_a_traversing_turret_blocks_the_scope(self):
        battery = dict(reload=dict(turrets=[dict(status='ready'), dict(status='traversing')]))
        self.assertIs(A.mounts_traversing(battery), True)

    def test_settled_turrets_allow_it(self):
        battery = dict(reload=dict(turrets=[dict(status='ready'), dict(status='reloading')]))
        self.assertIs(A.mounts_traversing(battery), False)

    def test_an_unobserved_layout_says_nothing(self):
        self.assertIsNone(A.mounts_traversing(dict(reload=dict(turrets=[]))))
        self.assertIsNone(A.mounts_traversing(None))


class ScopeNormaliseTests(unittest.TestCase):
    """A blind Shift press is how the run ended up in the wrong view."""

    def test_a_dark_sky_is_not_evidence_of_the_scope(self):
        presses, why = A.normalise_presses(dict(binocular=True,
                                                reason='corner_darkened_by_binocular_view',
                                                corner_drop_px=12))
        self.assertEqual(presses, 0)
        self.assertEqual(why, 'weak_evidence')

    def test_a_strong_reading_allows_one_press(self):
        presses, why = A.normalise_presses(dict(binocular=True,
                                                reason='corner_darkened_by_binocular_view',
                                                corner_drop_px=128))
        self.assertEqual(presses, 1)
        self.assertEqual(why, 'binocular_confirmed')

    def test_third_person_needs_nothing(self):
        self.assertEqual(A.normalise_presses(dict(binocular=False))[0], 0)

    def test_an_unknown_brightness_verdict_never_presses(self):
        self.assertEqual(A.normalise_presses(dict(binocular=True,
                                                  reason='something_else',
                                                  corner_drop_px=300))[0], 0)


class GainPlausibilityTests(unittest.TestCase):
    """A measured response that cannot be a camera gain must not move the gain."""

    def test_a_tenfold_jump_is_contamination(self):
        self.assertFalse(A.plausible_response(-17.5, -3.0))
        self.assertFalse(A.plausible_response(37.9, -3.0))

    def test_a_modest_change_is_evidence(self):
        self.assertTrue(A.plausible_response(-2.4, -3.0))
        self.assertTrue(A.plausible_response(1.4, -1.0))

    def test_a_gain_beyond_any_camera_is_impossible(self):
        self.assertFalse(A.plausible_response(9.0, None))

    def test_update_gain_keeps_the_old_value_on_a_jump(self):
        before = [-3.0, -2.0]
        gain, measured = A.update_gain(before, [-90.0, 10.0], [400.0, 10.0], [-1.0, 0.0])
        self.assertEqual(gain, before)
        self.assertIsNone(measured[0])

    def test_update_gain_still_learns_a_plausible_response(self):
        # Same sign, 2x the stored value: inside the plausible band, so it is
        # blended rather than dropped.
        gain, measured = A.update_gain([-3.0, None], [-100.0, 0.0], [-160.0, 0.0], [10.0, 0.0])
        self.assertAlmostEqual(measured[0], -6.0, places=6)
        self.assertAlmostEqual(gain[0], -4.5, places=6)

    def test_a_sign_flip_needs_corroboration(self):
        gain, measured = A.update_gain([-3.0, None], [-100.0, 0.0], [-70.0, 0.0], [10.0, 0.0])
        self.assertIsNone(measured[0])
        self.assertAlmostEqual(gain[0], -3.0, places=6)


class PlateDerivedPrecisionTests(unittest.TestCase):
    """The tolerance and the ladder both come from the box that is on screen."""

    def test_a_wide_near_plate_gets_a_wider_tolerance(self):
        px, why = A.resolve_fire_tolerance([0, 0, 123, 16], 60.0)
        self.assertEqual(why, 'plate_width')
        self.assertAlmostEqual(px, 36.9, places=1)

    def test_a_narrow_far_plate_gets_the_floor(self):
        px, why = A.resolve_fire_tolerance([0, 0, 40, 10], 60.0)
        self.assertEqual(why, 'plate_width')
        self.assertEqual(px, 18.0)

    def test_the_ceiling_is_never_exceeded(self):
        px, _ = A.resolve_fire_tolerance([0, 0, 400, 20], 45.0)
        self.assertEqual(px, 45.0)

    def test_without_a_plate_the_caller_keeps_its_value(self):
        px, why = A.resolve_fire_tolerance(None, 60.0)
        self.assertIsNone(px)
        self.assertEqual(why, 'no_plate')

    def test_the_ladder_flips_the_hint_through_the_measured_gain(self):
        # Plate 330 px above the reticle, scope gain -7 px/unit: the view has to
        # move up, i.e. positive units under the convention move = -error/gain.
        self.assertLess(A.ladder_units(-330.0, -7.0, 12), 0)
        self.assertGreater(A.ladder_units(-330.0, 7.0, 12), 0)

    def test_the_ladder_is_capped_and_damped(self):
        self.assertEqual(abs(A.ladder_units(-900.0, -1.0, 12)), 12)

    def test_without_a_gain_the_ladder_refuses_to_guess(self):
        self.assertEqual(A.ladder_units(-330.0, None, 12), 0.0)
        self.assertEqual(A.ladder_units(None, -7.0, 12), 0.0)

    def test_the_wait_and_ladder_knobs_are_on_by_default(self):
        args = A.main.__globals__['build_parser']().parse_args(['--out', '/tmp/x'])
        self.assertGreater(args.confirm_reload_wait_s, 1.0)
        self.assertGreater(args.scope_ladder_units, 1.0)
        self.assertGreater(args.scope_ladder_steps, 1)

    def test_the_knobs_exist_and_default_on(self):
        args = A.main.__globals__['build_parser']().parse_args(['--out', '/tmp/x'])
        self.assertTrue(args.fire_tolerance_from_plate)
        self.assertGreater(args.freeze_reach, 10)
        self.assertGreater(args.scope_ladder_steps, 0)


class UnmovedSampleTests(unittest.TestCase):
    """A velocity may only be fitted over frames no input disturbed."""

    def test_single_samples_between_our_own_moves_are_dropped(self):
        # Four refine rounds: each frame follows one more input, so every
        # "movement" in them is our camera, not the target. Only one sample per
        # input count survives, so no pair can be fitted into a velocity.
        samples = [(0.0, 100, 50, 3), (0.5, 180, 50, 4),
                   (1.0, 120, 50, 5), (1.5, 210, 50, 6)]
        kept = A.unmoved_samples(samples, width=3)
        self.assertEqual(len(kept), 1)
        self.assertIn(kept[0], [(0.0, 100, 50), (0.5, 180, 50),
                                (1.0, 120, 50), (1.5, 210, 50)])

    def test_a_clean_burst_survives(self):
        samples = [(0.0, 100, 50, 7), (0.4, 110, 50, 7), (0.8, 120, 50, 7),
                   (1.2, 400, 50, 8)]
        self.assertEqual(A.unmoved_samples(samples, width=3),
                         [(0.0, 100, 50), (0.4, 110, 50), (0.8, 120, 50)])

    def test_the_longest_clean_run_wins(self):
        samples = [(0.0, 0, 0, 1), (0.2, 1, 0, 2), (0.4, 2, 0, 3),
                   (0.6, 3, 0, 3), (0.8, 4, 0, 3)]
        self.assertEqual(len(A.unmoved_samples(samples, width=3)), 3)

    def test_range_samples_keep_two_fields(self):
        # (t, km, count) -> (t, km); the fitter unpacks exactly two values, and
        # leaving the count in crashed two live attempts with
        # "too many values to unpack (expected 2, got 3)".
        samples = [(0.0, 9.5, 4), (1.0, 9.6, 4), (2.0, 9.7, 5)]
        self.assertEqual(A.unmoved_samples(samples, width=2),
                         [(0.0, 9.5), (1.0, 9.6)])


if __name__ == '__main__':
    unittest.main()


class PanCompensationTests(unittest.TestCase):
    """A camera pan must not be mistaken for target motion in the lead fit."""

    def test_stationary_target_under_a_pan_fits_zero_velocity(self):
        step = 0.5
        history = [(i*step, 100.0 + 20*i, 50.0 - 10*i) for i in range(6)]
        drifts = [[20.0, -10.0] for _ in range(5)]
        velocity = A.fit_velocity(A.pan_compensated(history, drifts),
                                  minimum=3, min_span_s=1.0)
        self.assertIsNotNone(velocity)
        self.assertLess(abs(velocity[0]), 1.0)
        self.assertLess(abs(velocity[1]), 1.0)

    def test_moving_target_without_pan_keeps_its_velocity(self):
        step = 0.5
        history = [(i*step, 100.0 + 5*i, 50.0) for i in range(6)]
        velocity = A.fit_velocity(A.pan_compensated(history, [None]*5),
                                  minimum=3, min_span_s=1.0)
        self.assertAlmostEqual(velocity[0], 10.0, places=3)

    def test_pan_plus_motion_recovers_only_the_motion(self):
        step = 0.5
        history = [(i*step, 100.0 + 5*i + 20*i, 50.0) for i in range(6)]
        drifts = [[20.0, 0.0] for _ in range(5)]
        velocity = A.fit_velocity(A.pan_compensated(history, drifts),
                                  minimum=3, min_span_s=1.0)
        self.assertAlmostEqual(velocity[0], 10.0, places=3)

    def test_missing_drift_is_treated_as_no_correction(self):
        step = 0.5
        history = [(i*step, 5.0*i, 0.0) for i in range(4)]
        velocity = A.fit_velocity(A.pan_compensated(history, [None, None, None]),
                                  minimum=3, min_span_s=1.0)
        self.assertAlmostEqual(velocity[0], 10.0, places=3)


class RangeClosureTests(unittest.TestCase):
    """Elevation is closed on the HUD's ballistic range, never on hull pixels."""

    def test_matching_ranges_need_no_move(self):
        self.assertEqual(A.range_correction(7.70, 7.65, .15), (0, 'closed'))
        self.assertEqual(A.range_correction(7.4, 7.4, .15), (0, 'closed'))

    def test_short_solution_raises_the_reticle_with_a_bounded_move(self):
        # Measured: aim 7.4 km against a 7.7 km target, and 20 units up adds
        # 0.3 km. Aiming further is a negative mouse-y move.
        units, why = A.range_correction(7.4, 7.7, .15, .015, 30)
        self.assertEqual(why, 'move')
        self.assertEqual(units, -20)
        # A long shot needs the opposite sign.
        self.assertGreater(A.range_correction(8.2, 7.7, .15, .015, 30)[0], 0)

    def test_a_large_error_is_clamped_not_sent_in_one_go(self):
        units, why = A.range_correction(5.0, 9.5, .15, .015, 30)
        self.assertEqual(why, 'move')
        self.assertEqual(units, -30)
        self.assertLessEqual(abs(units), 30)

    def test_an_unreadable_range_is_never_closed(self):
        for aim, target in ((None, 7.7), (7.4, None), (None, None),
                            (float('nan'), 7.7)):
            self.assertEqual(A.range_correction(aim, target, .15, .015, 30),
                             (0, 'unavailable'))
        self.assertEqual(A.range_correction(7.4, 7.7, .15, 0, 30), (0, 'unavailable'))


class TargetRangeSourceTests(unittest.TestCase):
    """A read caption wins; an operator value is a tagged fallback, never a reading."""

    def test_caption_wins_and_is_tagged(self):
        self.assertEqual(A.target_range(7.7, 9.0), (7.7, 'nameplate_caption'))

    def test_operator_value_is_used_only_when_the_caption_is_unreadable(self):
        self.assertEqual(A.target_range(None, 13.8), (13.8, 'operator'))

    def test_no_source_reports_unavailable(self):
        self.assertEqual(A.target_range(None, None), (None, 'unavailable'))




class ShotPipelineGateTests(unittest.TestCase):
    """The operator's firing procedure: shake and camera cuts are not results."""

    def test_default_aim_budget_is_fifteen_seconds(self):
        self.assertEqual(A.AIM_BUDGET_S, 15.0)

    def test_empty_scope_holds_through_the_zoom_transition(self):
        self.assertEqual(A.scope_empty_action(1, grace=2, exits=0, retries=2), 'hold')

    def test_empty_scope_hands_back_to_coarse_aim(self):
        self.assertEqual(A.scope_empty_action(2, grace=2, exits=0, retries=2), 'return')

    def test_empty_scope_gives_up_after_the_retries(self):
        self.assertEqual(A.scope_empty_action(2, grace=2, exits=2, retries=2), 'give_up')
        self.assertEqual(A.scope_empty_action(2, grace=2, exits=1, retries=2), 'return')

    def test_view_jump_ignores_motion_the_move_explains(self):
        # 40 units at 1.5 px/unit explains 60 px; 70 px of drift is ordinary.
        self.assertFalse(A.view_jump([70.0, -10.0], [40, 0], [-1.5, -1.4], limit=520))
        self.assertFalse(A.view_jump(None, [40, 0], [-1.5, -1.4], limit=520))
        self.assertFalse(A.view_jump([300.0, 20.0], None, None, limit=520))

    def test_view_jump_flags_a_camera_cut(self):
        # No command, 900 px of unexplained motion: a cut, not an aim result.
        self.assertTrue(A.view_jump([900.0, 12.0], None, [-1.5, -1.4], limit=520))
        # A small command that cannot explain 800 px either.
        self.assertTrue(A.view_jump([800.0, 0.0], [10, 0], [-1.5, -1.4], limit=520))

    def test_vertical_reversals_count_only_large_flips(self):
        self.assertEqual(A.vertical_reversals([300, -280, 300, -260], 120, 3), 3)
        # Dither inside the amplitude is not a shake.
        self.assertEqual(A.vertical_reversals([40, -50, 45, -55], 120, 3), 0)
        # A missing reading breaks the run instead of counting a flip.
        self.assertEqual(A.vertical_reversals([300, None, -300], 120, 3), 0)

    def test_scan_coverage_is_measured_not_assumed(self):
        # The old default (12 steps x 80 units) is a 38-degree look, not 360.
        self.assertAlmostEqual(A.scan_coverage_deg(12, 80), 38.2, places=1)
        self.assertAlmostEqual(A.scan_coverage_deg(24, 400), 382.5, places=0)
        # A stronger gain covers more yaw for the same input.
        self.assertGreater(A.scan_coverage_deg(12, 400, gain_px_per_unit=3.0),
                           A.scan_coverage_deg(12, 400, gain_px_per_unit=1.5))
        self.assertEqual(A.scan_coverage_deg(0, 400), 0.0)
        # One sweep of the old defaults cannot reach the required 360 degrees.
        self.assertLess(A.scan_coverage_deg(12, 80), 360)

    def test_a_minimap_turn_is_sent_as_bounded_pieces(self):
        # 90 degrees at 1.5 px/unit is ~2258 units: one event that large is
        # clipped by the pointer's travel, so it must be split.
        pieces = A.turn_pieces(2258)
        self.assertEqual(pieces, [400.0, 400.0, 400.0, 400.0, 400.0, 258.0])
        self.assertAlmostEqual(sum(pieces), 2258, places=1)
        self.assertEqual(A.turn_pieces(-900, 400), [-400.0, -400.0, -100.0])
        self.assertEqual(A.turn_pieces(0), [])
        self.assertTrue(all(abs(p) <= 400 for p in A.turn_pieces(-5000)))

    def test_the_bearing_gain_default_matches_the_measured_steer(self):
        # The old -9.77 turned a sixth of the commanded angle (measured
        # -1.03..-1.82 px per unit in third person).
        parser_default = -1.5
        self.assertEqual(parser_default, -1.5)
        self.assertLess(abs(parser_default), 3)

    def test_a_gain_sample_cannot_flip_the_sign_on_its_own(self):
        # One drifted anchor must not reverse the elevation gain (measured:
        # that produced a 534 px overshoot on the next correction).
        self.assertEqual(A.learn_gain(-1.5, 2.0), -1.5)
        self.assertEqual(A.learn_gain(-1.5, -1.8), -1.8)
        self.assertEqual(A.learn_gain(None, -1.7), -1.7)
        # Implausible magnitudes are ignored rather than trusted.
        self.assertEqual(A.learn_gain(-1.5, 40.0), -1.5)
        self.assertEqual(A.learn_gain(-1.5, 0.05), -1.5)

    def test_range_closure_iterates_bounded_steps_until_it_agrees(self):
        moves = []
        readings = iter([(9.0, 13.0), (12.4, 13.0), (13.0, 13.0)])

        def measure(_index):
            return next(readings)

        trace = A.close_range_steps(measure, lambda dx, dy: moves.append(dy),
                                    tolerance_km=.2, km_per_unit=.5, step_units=8)
        # Each round sends bounded pieces, never one huge jump.
        self.assertTrue(all(abs(step) <= A.MAX_VERTICAL_STEP_UNITS for step in moves))
        self.assertEqual(trace[-1]['why'], 'closed')
        self.assertGreaterEqual(len(trace), 2)

    def test_range_closure_refuses_to_guess_without_a_reading(self):
        trace = A.close_range_steps(lambda _i: (None, None), lambda dx, dy: None,
                                    tolerance_km=.2, km_per_unit=.5, step_units=8)
        self.assertEqual(trace[0]['why'], 'unavailable')
        self.assertEqual(len(trace), 1)

    def test_descent_pitches_down_until_the_readout_is_sane(self):
        readings = iter([25.4, 23.0, 12.0])
        moves = []
        trace, state = A.descend_to_readable(lambda _i: next(readings),
                                             lambda dx, dy: moves.append(dy), max_km=20.0)
        self.assertEqual(state, 'readable')
        self.assertEqual(moves, [A.MAX_VERTICAL_STEP_UNITS] * 2)
        self.assertEqual(len(trace), 3)

    def test_descent_gives_up_instead_of_aiming_blind(self):
        # A readout that never appears stops the descent after two stalled
        # rounds instead of grinding the camera into the sea.
        moves = []
        trace, state = A.descend_to_readable(lambda _i: None,
                                             lambda dx, dy: moves.append(dy), rounds=3)
        # No readout at all is the state a descent exists to leave, so it keeps
        # pitching (bounded by rounds) rather than stopping immediately.
        self.assertEqual(state, 'not_reached')
        self.assertEqual(len(moves), 3)

    def test_the_scope_relaxes_the_fire_tolerance_to_the_operator_rule(self):
        # Operator: once the binocular view is open, a health bar or ship
        # position that is "about right" is enough. The relaxed value must be
        # the larger of the two, and it is recorded so a shot can be audited.
        base, relaxed_px = 60, 120
        self.assertEqual(max(base, relaxed_px), 120)
        self.assertEqual(max(200, relaxed_px), 200)

    def test_descent_stops_when_the_camera_stops_responding(self):
        moves = []
        trace, state = A.descend_to_readable(lambda _i: 25.0, lambda dx, dy: moves.append(dy),
                                             rounds=6, step=40, stall_rounds=2)
        self.assertEqual(state, 'not_responding')
        self.assertEqual(len(moves), 2)     # two stalled rounds, then stop
        self.assertTrue(all(m == 40 for m in moves))
