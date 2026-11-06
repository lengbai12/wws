#!/usr/bin/env python3
"""The smoothness metrics must separate a clean approach from a shaking one."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/capture'))

import aim_smoothness as S  # noqa: E402


def approach(start=600.0, fractions=(0.6, 0.5, 0.4, 0.3, 0.2, 0.1)):
    errors, value = [], start
    for f in fractions:
        errors.append([value, value / 2])
        value *= f
    return errors


class AnalyseTests(unittest.TestCase):
    def test_a_clean_approach_is_smooth(self):
        m = S.analyse(approach())
        self.assertEqual(m['smoothness'], 'smooth')
        self.assertEqual(m['reversals'], [0, 0])
        self.assertEqual(m['monotonic_fraction'], 1.0)

    def test_alternating_errors_without_progress_are_oscillating(self):
        errs = [[400, 0], [-380, 0], [370, 0], [-360, 0], [350, 0], [-340, 0],
                [330, 0], [-320, 0], [310, 0]]
        m = S.analyse(errs)
        self.assertEqual(m['smoothness'], 'oscillating')
        self.assertEqual(m['reason'], 'sign_flips_without_progress')

    def test_one_huge_step_is_jerky(self):
        m = S.analyse([[900, 0], [100, 0], [-800, 0]])
        self.assertEqual(m['smoothness'], 'jerky')
        self.assertEqual(m['reason'], 'single_step_too_large')

    def test_overshoot_is_measured_after_arriving(self):
        errs = [[400, 0], [200, 0], [50, 0], [40, 0], [180, 0], [90, 0]]
        m = S.analyse(errs)
        self.assertEqual(m['overshoot_px'], 180.0)
        self.assertEqual(m['tolerance_steps'], 2)

    def test_missing_readings_are_skipped_not_guessed(self):
        m = S.analyse([[400, 0], None, [200, 0], None, [100, 0]])
        self.assertEqual(m['valid'], 3)
        self.assertEqual(m['steps'], 5)
        self.assertEqual(m['smoothness'], 'smooth')

    def test_one_sample_is_not_a_trajectory(self):
        m = S.analyse([[100, 0]])
        self.assertEqual(m['smoothness'], 'unknown')
        self.assertEqual(m['reason'], 'too_few_steps')


class GuardTests(unittest.TestCase):
    def test_a_short_run_keeps_going(self):
        self.assertEqual(S.guard([[300, 0], [200, 0]])[0], 'continue')

    def test_shaking_stops_the_loop(self):
        errs = [[400, 0], [-380, 0], [370, 0], [-360, 0], [350, 0], [-340, 0],
                [330, 0], [-320, 0]]
        action, reason = S.guard(errs)
        self.assertEqual(action, 'stop')
        self.assertEqual(reason, 'sign_flips_without_progress')

    def test_a_jump_asks_for_damping_first(self):
        action, reason = S.guard([[900, 0], [100, 0], [-800, 0], [-700, 0],
                                  [-600, 0], [-500, 0], [-400, 0], [-300, 0]])
        self.assertEqual(action, 'damp')
        self.assertEqual(reason, 'single_step_too_large')


if __name__ == '__main__':
    unittest.main()
