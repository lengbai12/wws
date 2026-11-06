"""The aim-run verifier must fail the states it claims to catch.

Its whole purpose is to reject a run that fired without evidence, so each
rejection case is exercised here, along with the gain recomputation that makes
the measurement checkable from the frames the run itself saved.
"""
import json
import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import verify_aim_report as V


def write_ppm(path, rgb):
    import cv2
    cv2.imwrite(str(path), np.ascontiguousarray(rgb[:, :, ::-1]))


def textured(seed=1):
    """A low-frequency scene: the shift estimator works on features, not noise."""
    import cv2
    rng = np.random.default_rng(seed)
    coarse = rng.integers(40, 200, (1600 // 16, 2560 // 16, 3)).astype(np.uint8)
    smooth = cv2.resize(coarse, (2560, 1600), interpolation=cv2.INTER_LINEAR)
    return smooth


def shifted(base, dx):
    """Translate without wrapping, filling the vacated band with the border."""
    moved = np.empty_like(base)
    if dx >= 0:
        moved[:, dx:] = base[:, :-dx] if dx else base
        moved[:, :dx] = base[:, :1]
    else:
        moved[:, :dx] = base[:, -dx:]
        moved[:, dx:] = base[:, -1:]
    return moved


class FreshnessTests(unittest.TestCase):
    def test_monotonic_capture_times_pass(self):
        report = {'steps': [{'step': 0, 'captured_s': 10.0}, {'step': 1, 'captured_s': 10.6}]}
        self.assertEqual(V.check_freshness(report)['status'], 'pass')

    def test_the_same_frame_twice_fails(self):
        report = {'steps': [{'step': 0, 'captured_s': 10.0}, {'step': 1, 'captured_s': 10.0}]}
        got = V.check_freshness(report)
        self.assertEqual(got['status'], 'fail')
        self.assertEqual(got['same_frame_as_previous'], [1])

    def test_a_backwards_clock_fails(self):
        report = {'steps': [{'step': 0, 'captured_s': 12.0}, {'step': 1, 'captured_s': 11.0}]}
        self.assertEqual(V.check_freshness(report)['status'], 'fail')

    def test_live_frame_age_must_be_present_and_within_deadline(self):
        for age in (None, -.01, .750001, float('nan')):
            report = {'live_source': True, 'steps': [
                {'step': 0, 'captured_s': 10., 'frame_age_s': .1},
                {'step': 1, 'captured_s': 11., 'frame_age_s': age}]}
            with self.subTest(age=age):
                self.assertEqual(V.check_freshness(report)['status'], 'fail')
        report['steps'][1]['frame_age_s'] = .75
        self.assertEqual(V.check_freshness(report)['status'], 'pass')


class FireGateTests(unittest.TestCase):
    def steps(self, states):
        return [dict(step=i, state=state) for i, state in enumerate(states)]

    def confirmation(self):
        return dict(confirmed=True, battle_active=True, reticle=[1280., 800.],
                    candidates=[{'aim_px': [1280., 800.]}], frame_age_s=.1)

    def test_firing_without_convergence_fails(self):
        report = {'fired': True, 'fire_min_steps': 2, 'steps': self.steps(['aim', 'aim'])}
        got = V.check_fire(report)
        self.assertEqual(got['status'], 'fail')
        self.assertIn('0 consecutive converged steps', got['reasons'][0])

    def test_two_converged_steps_allow_the_shot(self):
        report = {'fired': True, 'fire_min_steps': 2,
                  'fire_confirmation': self.confirmation(),
                  'steps': self.steps(['aim', 'converged', 'converged'])}
        self.assertEqual(V.check_fire(report)['status'], 'pass')
    def test_observation_minimum_is_not_the_convergence_count(self):
        report = dict(fired=True, fire_min_steps=4,
                      fire_confirmation=self.confirmation(),
                      steps=self.steps(['aim', 'aim', 'aim', 'converged', 'converged']))
        self.assertEqual(V.check_fire(report)['status'], 'pass')

    def test_early_convergence_does_not_authorize_a_later_unstable_shot(self):
        report = dict(fired=True, fire_min_steps=1,
                      fire_confirmation=self.confirmation(),
                      steps=self.steps(['converged', 'converged', 'aim']))
        self.assertEqual(V.check_fire(report)['status'], 'fail')

    def test_convergence_does_not_bypass_the_observation_minimum(self):
        report = dict(fired=True, fire_min_steps=4,
                      fire_confirmation=self.confirmation(),
                      steps=self.steps(['converged', 'converged']))
        self.assertEqual(V.check_fire(report)['status'], 'fail')


    def test_a_requested_scope_needs_evidence(self):
        report = {'fired': True, 'fire_min_steps': 2, 'scope_after_px': 80,
                  'fire_confirmation': self.confirmation(),
                  'steps': self.steps(['aim', 'converged', 'converged'])}
        got = V.check_fire(report)
        self.assertEqual(got['status'], 'fail')
        self.assertIn('without HUD scope evidence', got['reasons'][0])
        report['steps'][0]['scope_evidence'] = {'observed': True}
        self.assertEqual(V.check_fire(report)['status'], 'fail')
        report['fire_confirmation']['scope_evidence'] = {'observed': True}
        self.assertEqual(V.check_fire(report)['status'], 'pass')

    def test_current_confirmation_is_required_even_after_convergence(self):
        report = dict(fired=True, steps=self.steps(['converged', 'converged']))
        self.assertEqual(V.check_fire(report)['status'], 'fail')
        report['fire_confirmation'] = self.confirmation()
        self.assertEqual(V.check_fire(report)['status'], 'pass')
        report['fire_confirmation']['candidates'].append({'aim_px': [1280., 800.]})
        self.assertEqual(V.check_fire(report)['status'], 'fail')

    def test_live_fire_age_is_checked_at_the_final_confirmation(self):
        report = dict(fired=True, live_source=True, fire_confirmation=self.confirmation(),
                      steps=self.steps(['converged', 'converged']))
        for age in (None, -.01, .750001, float('nan')):
            report['fire_confirmation']['frame_age_s'] = age
            with self.subTest(age=age):
                self.assertEqual(V.check_fire(report)['status'], 'fail')
        report['fire_confirmation']['frame_age_s'] = .75
        self.assertEqual(V.check_fire(report)['status'], 'pass')


class LeadTests(unittest.TestCase):
    def test_the_lead_must_add_up(self):
        ok = {'steps': [{'lead_total_s': 11.3, 'lead_horizon_s': 0.3,
                         'flight_time': {'value': 11.0}}]}
        self.assertEqual(V.check_lead(ok)['status'], 'pass')
        bad = {'steps': [{'lead_total_s': 20.0, 'lead_horizon_s': 0.3,
                          'flight_time': {'value': 11.0}}]}
        self.assertEqual(V.check_lead(bad)['status'], 'fail')


class GainRecomputationTests(unittest.TestCase):
    """The verifier must recompute the gain from the frames, not trust the report."""

    def test_a_known_scene_shift_is_recovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            base = textured()
            shift = 40
            moved = shifted(base, shift)
            write_ppm(run / 'gain-base.ppm', base)
            write_ppm(run / 'gain-0-plus.ppm', moved)
            write_ppm(run / 'gain-0-back.ppm', base)
            write_ppm(run / 'gain-1-plus.ppm', shifted(base, 0))
            write_ppm(run / 'gain-1-back.ppm', shifted(base, 0))
            report = {'measured_gain': {'units': 20, 'gain': [shift / 20.0, 0.0]}}
            got = V.check_gain(run, report)
            self.assertEqual(got['status'], 'pass', got)
            self.assertLessEqual(got['max_delta'], 0.25)
            self.assertAlmostEqual(got['rows'][0]['recomputed'], 2.0, places=1)

    def test_a_wrong_recorded_gain_is_caught(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            base = textured(3)
            moved = shifted(base, 40)
            write_ppm(run / 'gain-base.ppm', base)
            write_ppm(run / 'gain-0-plus.ppm', moved)
            write_ppm(run / 'gain-0-back.ppm', base)
            write_ppm(run / 'gain-1-plus.ppm', shifted(base, 0))
            write_ppm(run / 'gain-1-back.ppm', shifted(base, 0))
            report = {'measured_gain': {'units': 20, 'gain': [9.9, 0.0]}}
            got = V.check_gain(run, report)
            self.assertEqual(got['status'], 'fail', got)

    def test_vertical_gain_uses_its_own_probe_distance(self):
        from unittest.mock import patch
        report = {'measured_gain': {'units': 20, 'axis_units': [20, 3], 'gain': [2., 2.]}}
        with patch.object(V, 'read_frame', return_value=object()), \
                patch.object(V.A, 'estimate_camera_shift',
                             side_effect=([40., 0.], [0., 0.], [0., 6.], [0., 0.])):
            got = V.check_gain(Path('/unused'), report)
        self.assertEqual(got['status'], 'pass', got)
        self.assertEqual(got['rows'][1]['recomputed'], 2.)


if __name__ == '__main__':
    unittest.main()
