#!/usr/bin/env python3
"""Regression suite for firing at a moving target.

The game's target lock makes the camera follow the ship's bearing: a locked
ship does not move on screen while its range keeps changing (operator-reported
mechanic, 2026-10-27). A screen-space lead is therefore meaningless under lock,
and the shot has to be aimed at the range the ship will have when the shells
arrive. These tests pin that contract offline; the live pass is recorded in
state/match-* runs.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/capture'))
sys.path.insert(0, str(ROOT / 'tools/replay'))

import aim_servo as A  # noqa: E402


class PredictedRangeTests(unittest.TestCase):
    def test_a_closing_target_is_led_closer(self):
        samples = [(0.0, 13.0), (1.0, 12.98), (2.0, 12.96), (3.0, 12.94)]
        out = A.predicted_range_km(samples, 13.0)
        self.assertTrue(out['applied'])
        self.assertAlmostEqual(out['rate_km_s'], -0.02, places=3)
        self.assertAlmostEqual(out['shift_km'], -0.26, places=2)
        self.assertAlmostEqual(out['predicted_km'], 12.68, places=2)

    def test_a_receding_target_is_led_further(self):
        samples = [(0.0, 10.0), (1.0, 10.02), (2.0, 10.04), (3.0, 10.06)]
        out = A.predicted_range_km(samples, 10.0)
        self.assertTrue(out['applied'])
        self.assertGreater(out['predicted_km'], 10.06)

    def test_the_shift_is_bounded(self):
        samples = [(0.0, 10.0), (1.0, 10.07), (2.0, 10.14), (3.0, 10.21)]
        out = A.predicted_range_km(samples, 20.0, max_shift_km=0.5)
        self.assertTrue(out['applied'])
        self.assertTrue(out['clamped'])
        self.assertEqual(out['shift_km'], 0.5)

    def test_refusals_are_named(self):
        self.assertEqual(A.predicted_range_km([(0.0, 10.0)], 5.0)['reason'],
                         'insufficient_samples')
        self.assertEqual(A.predicted_range_km([(0.0, 10.0), (0.2, 10.01), (0.4, 10.02)], 5.0,
                                              min_span_s=1.5)['reason'],
                         'baseline_too_short')
        self.assertEqual(A.predicted_range_km([(0.0, 10.0), (1.0, 10.1), (2.0, 10.2)], None)['reason'],
                         'no_flight_time')

    def test_an_absurd_rate_is_refused(self):
        samples = [(0.0, 10.0), (1.0, 11.0), (2.0, 12.0)]
        self.assertEqual(A.predicted_range_km(samples, 5.0)['reason'], 'rate_out_of_range')


class LeadPlanTests(unittest.TestCase):
    """The mechanic: which geometry the observed target calls for."""

    SCREEN = dict(applied=True, velocity_px_s=[20.0, 0.0], lead_px=[40.0, 0.0])
    RANGE = dict(applied=True, rate_km_s=-0.02, shift_km=-0.26, predicted_km=12.68,
                 last_km=12.94, flight_s=13.0, samples=4, span_s=3.0, clamped=False)

    def test_a_ship_that_slides_on_screen_gets_a_screen_lead(self):
        plan = A.lead_plan(self.SCREEN, self.RANGE, 20.0)
        self.assertEqual(plan['mode'], 'screen')
        self.assertEqual(plan['lead_px'], [40.0, 0.0])

    def test_a_locked_ship_gets_a_range_lead(self):
        # still on screen (the lock follows the bearing) but the range moves
        plan = A.lead_plan(self.SCREEN, self.RANGE, 0.4)
        self.assertEqual(plan['mode'], 'range')
        self.assertAlmostEqual(plan['predicted_km'], 12.68, places=2)
        self.assertNotIn('lead_px', plan)

    def test_a_still_ship_with_a_steady_range_gets_no_lead(self):
        steady = dict(self.RANGE, rate_km_s=0.0)
        plan = A.lead_plan(self.SCREEN, steady, 0.4)
        self.assertEqual(plan['mode'], 'none')
        self.assertFalse(plan['applied'])
        self.assertEqual(plan['reason'], 'target_effectively_still')

    def test_without_any_evidence_there_is_no_guess(self):
        plan = A.lead_plan(None, None, None)
        self.assertEqual(plan['mode'], 'none')
        self.assertEqual(plan['reason'], 'no_lead_evidence')

    def test_screen_series_missing_falls_back_to_the_range_evidence(self):
        plan = A.lead_plan(None, self.RANGE, None)
        self.assertEqual(plan['mode'], 'range')


class FireGeometryTests(unittest.TestCase):
    """The shot is judged against the predicted point, never the old one."""

    def test_a_led_shot_is_judged_at_the_predicted_point(self):
        judged = A.judged_residual([-100.0, 0.0], [100.0, 0.0])
        self.assertEqual(judged, [0.0, 0.0])

    def test_without_lead_the_same_residual_is_outside(self):
        judged = A.judged_residual([-100.0, 0.0], None)
        self.assertEqual(judged, [-100.0, 0.0])
        self.assertGreater(abs(judged[0]), 50)

    def test_no_residual_means_no_judgement(self):
        self.assertIsNone(A.judged_residual(None, [10.0, 0.0]))


if __name__ == '__main__':
    unittest.main()
