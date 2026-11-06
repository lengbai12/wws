"""Exercise the real servo executor with hardware replaced, including fire revocation."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import aim_servo as A
import test_aim_servo as servo_tests


class ActionGateTests(unittest.TestCase):
    def execute(self, *, oneshot=False, scope=False, lose_after_lock=False,
                bad_battle=False, missing_reticle=False, scope_missing=False,
                target_gone=False, ambiguous=False, expire_during_focus=False,
                scope_after=False, battle_lost_on_lock=False, target_offset=0., one_step=False,
                final_weapon='gun', final_weapon_status='observed',
                final_reload_ready=True, final_reload_status='observed'):
        with tempfile.TemporaryDirectory() as tmp:
            args = servo_tests.RunFlowTests().parse_args(tmp, ['--settle', '0', '--aim-nudge', '0',
                '--recenter-margin', '0', '--no-lead-horizon'] +
                (['--oneshot'] if oneshot else []) + (['--scope'] if scope else []) +
                (['--scope-after', '80'] if scope_after else []))
            args.live_source = True
            # The loop opens the scope only after two coarse horizontal frames
            # (or the step budget), so this path needs one more step than the
            # old one-frame rule did: transition, then two converged frames.
            args.max_steps = 1 if one_step else (4 if scope_after else 6)
            frame = np.random.default_rng(74).integers(0, 255, (1600, 2560, 3), dtype=np.uint8)
            state = dict(name='', locked=False, reads=0, now=10_000_000_000)
            events = []

            class Probe:
                def __init__(self, seconds, report):
                    self.report = report
                    self.started = time.monotonic()
                    self.action_deadline = self.started + 30

                def x(self, *argv, **kwargs):
                    events.append(argv)
                    if argv[0] == 'search':
                        return '42'
                    if argv[0] == 'getwindowgeometry':
                        return 'WIDTH=2560\nHEIGHT=1600'
                    return ''

                def verify_window(self, window):
                    return {}

                def command(self, argv):
                    path = Path(argv[-1])
                    state['name'] = path.stem
                    state['reads'] += 1
                    state['now'] += 20_000_000
                    Path(str(path) + '.json').write_text(json.dumps(
                        {'captured_monotonic_ns': state['now']}))
                    return ''

                def focus(self):
                    if expire_during_focus and state['name'].startswith('pre-fire'):
                        state['now'] += 800_000_000

                def key(self, key):
                    state['locked'] |= key == 'x'
                    self.report['inputs'].append({'key': key})

                def move(self, dx, dy):
                    self.report['inputs'].append({'relative': [dx, dy], 'pointer_delta': [dx, dy]})

                def snapshot(self, name):
                    return Path(tmp) / (name + '.png')

                def check_identity(self):
                    pass

                def release(self):
                    self.report['unreleased_own_keys'] = []

            def reticle(_):
                step = state['name'].startswith('step-1')
                if missing_reticle:
                    # A frame with no drawn reticle: the servo may only offer an
                    # assumed centre, and the gate must refuse to fire on that.
                    return (None if step else [1280., 800.],
                            not ((bad_battle and step)
                                 or (battle_lost_on_lock and state['locked'])),
                            'observed', 'native_band' if step else 'scope_screen_center')
                return ([1280., 800.],
                        not ((bad_battle and step) or (battle_lost_on_lock and state['locked'])),
                        'observed', 'native_band')

            def enemies(*_, **__):
                if (lose_after_lock and state['locked']) or (target_gone and state['reads'] > 1):
                    return []
                candidate = dict(nameplate_bbox=[1250, 740, 60, 18], body_roi=[1240, 770, 80, 40],
                                 aim_px=[1280. + target_offset, 800.], trackable=True, confidence=.9,
                                 hp=39100, red_px=1500,
                                 distance_from_reticle_px=abs(target_offset))
                return [candidate, dict(candidate)] if ambiguous else [candidate]

            with contextlib.ExitStack() as stack, contextlib.redirect_stdout(io.StringIO()):
                for name, value in dict(Probe=Probe, game_identity=lambda: {'pid': 1, 'display': ':0'},
                        pointer_state=lambda _: {'x': 1280, 'y': 800, 'mask': 0},
                        read_frame=lambda _: frame, reticle_centre=reticle, find_enemy=enemies,
                        hud_flight_seconds=lambda _: None,
                        hud_scope_evidence=lambda _: {'observed': not scope_missing},
                        main_battery_fire_evidence=lambda _: {
                            'weapon': {'kind': final_weapon, 'status': final_weapon_status},
                            'reload': {'ready': final_reload_ready, 'status': final_reload_status}},
                        own_ship_drift=lambda *_: None,
                        track_patch=lambda *_args, **_kw: {
                            # The tracked patch is the nameplate the candidate
                            # names; reporting a body-sized box here made the
                            # servo add aim_offset to a centre it had already
                            # included, i.e. a 51 px phantom error.
                            'bbox': [1250 + target_offset, 740, 60, 18], 'score': 1.},
                        plan_move=lambda *_args, **_kw: (1, 0, 'probe')).items():
                    stack.enter_context(patch.object(A, name, value))
                stack.enter_context(patch.object(A.time, 'monotonic_ns', lambda: state['now']))
                code = A.run(args)
            report = json.loads((args.out / 'report.json').read_text())
            return code, report, events

    def assert_no_fire(self, **kwargs):
        code, report, events = self.execute(**kwargs)
        self.assertFalse(report.get('fired', False), report)
        self.assertNotIn(('mousedown', 1), events)
        return code, report

    def test_battle_loss_revokes_even_with_trackable_texture(self):
        self.assert_no_fire(bad_battle=True)

    def test_missing_reticle_cannot_reuse_cached_position(self):
        self.assert_no_fire(missing_reticle=True)

    def test_target_disappears_during_lock_wait(self):
        self.assert_no_fire(lose_after_lock=True)

    def test_battle_disappears_during_lock_wait(self):
        self.assert_no_fire(battle_lost_on_lock=True)

    def test_scope_transition_rebuilds_target_and_convergence(self):
        code, report, events = self.execute(scope_after=True)
        self.assertEqual(code, 0, report.get('error'))
        # The transition lands on the step whose coarse pass completed; the
        # point is that the row records the FOV change and drops convergence.
        reset_rows = [row for row in report['steps'] if row.get('convergence_reset')]
        self.assertTrue(reset_rows, report['steps'])
        self.assertEqual(reset_rows[0]['convergence_reset'], 'scope_changed_fov')
        self.assertTrue(report.get('fired'))
        self.assertIn(('mousedown', 1), events)

    def test_oneshot_scope_requires_post_transition_target(self):
        self.assert_no_fire(oneshot=True, scope=True, lose_after_lock=True)

    def test_oneshot_requires_scope_evidence(self):
        self.assert_no_fire(oneshot=True, scope=True, scope_missing=True)

    def test_texture_match_without_enemy_confirmation_cannot_fire(self):
        self.assert_no_fire(target_gone=True)

    def test_ambiguous_current_targets_cannot_fire(self):
        self.assert_no_fire(ambiguous=True)

    def test_focus_check_cannot_outlive_frame_lease(self):
        code, report = self.assert_no_fire(expire_during_focus=True)
        self.assertNotEqual(code, 0)
        rejection = report['frame_rejection']
        self.assertEqual(rejection['reason'], 'expired')
        self.assertEqual(rejection['checkpoint'], 'after_final_focus_check')
        self.assertTrue(rejection['frame'].startswith('pre-fire'))
        self.assertEqual(rejection['age_ns'], 800_000_000)
        self.assertEqual(rejection['checked_monotonic_ns'] - rejection['captured_monotonic_ns'],
                         rejection['age_ns'])
        self.assertEqual(rejection['max_age_ns'], 750_000_000)

    def test_many_loose_observations_never_reach_the_fire_button(self):
        code, report = self.assert_no_fire(target_offset=55.)
        self.assertEqual(code, 0, report.get('error'))
        self.assertEqual(len(report['steps']), 6)

    def test_oneshot_loose_reread_cannot_bypass_convergence(self):
        self.assert_no_fire(oneshot=True, scope=True, target_offset=55.)

    def test_oneshot_cannot_fire_after_only_one_tight_confirmation(self):
        code, report, events = self.execute(oneshot=True, one_step=True)
        self.assertEqual(code, 0, report.get('error'))
        self.assertFalse(report.get('fired', False))
        self.assertNotIn(('mousedown', 1), events)

    def test_final_weapon_and_reload_must_be_observed_in_both_modes(self):
        for oneshot in (False, True):
            for evidence in (
                    {'final_weapon': 'torpedo'},
                    {'final_weapon': None},
                    {'final_weapon_status': 'unknown'},
                    {'final_reload_ready': False},
                    {'final_reload_ready': None},
                    {'final_reload_status': 'unknown'}):
                with self.subTest(oneshot=oneshot, evidence=evidence):
                    code, report = self.assert_no_fire(oneshot=oneshot, **evidence)
                    self.assertNotEqual(code, 0)
                    self.assertIn('fire revoked:', report['error'])

    def test_current_confirmed_target_reaches_button_in_both_modes(self):
        for oneshot in (False, True):
            with self.subTest(oneshot=oneshot):
                code, report, events = self.execute(oneshot=oneshot, scope=True)
                self.assertEqual(code, 0, report.get('error'))
                self.assertTrue(report.get('fired'))
                self.assertIn(('mousedown', 1), events)
                self.assertFalse(report['unreleased_button'])


if __name__ == '__main__':
    unittest.main()
