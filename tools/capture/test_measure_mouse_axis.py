"""Axis-response gate: a response is only claimed when it beats the control pair.

The camera also drifts while the ship moves, so the analyser must compare the
driven pair against a same-interval no-input pair, refuse flat/lost landmarks,
and report reversibility instead of assuming it.
"""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure_mouse_axis as M


def rows(values, axis='y'):
    key = 'dy' if axis == 'y' else 'dx'
    return [dict(point=[0, 0], state='tracked', score=.9, **{key: value}) for value in values]


class AxisSummaryTests(unittest.TestCase):
    def test_a_response_larger_than_the_control_is_reported(self):
        out = M.summarize(rows([0, 1, -1]), rows([100, 104, 96]), rows([-100, -98, -102]), 'y')
        self.assertEqual(out['state'], 'response')
        self.assertEqual(out['driven_median'], 100.0)
        self.assertTrue(out['reversible'])

    def test_drift_the_size_of_the_input_is_not_a_response(self):
        out = M.summarize(rows([120, -118]), rows([110, -112]), rows([0, 2]), 'y')
        self.assertEqual(out['state'], 'no_response_beyond_drift')

    def test_flat_and_lost_landmarks_are_ignored(self):
        mixed = [dict(point=[0, 0], state='flat', score=None),
                 dict(point=[1, 1], state='lost', score=None),
                 dict(point=[2, 2], state='tracked', score=.9, dy=90)]
        out = M.summarize(rows([0]), mixed, rows([-90]), 'y')
        self.assertEqual(out['landmarks_tracked'], 1)
        self.assertEqual(out['driven_median'], 90.0)

    def test_low_score_landmarks_are_ignored(self):
        weak = [dict(point=[0, 0], state='tracked', score=.4, dy=90)]
        out = M.summarize(rows([0]), weak, rows([-90]), 'y')
        self.assertEqual(out['state'], 'no_trackable_landmark')

    def test_the_x_axis_uses_dx(self):
        out = M.summarize(rows([0, 0], 'x'), rows([-60, -62], 'x'), rows([60, 58], 'x'), 'x')
        self.assertEqual(out['state'], 'response')
        self.assertEqual(out['driven_median'], -61.0)

    def test_an_irreversible_response_is_marked(self):
        out = M.summarize(rows([0, 0]), rows([100, 100]), rows([-10, -12]), 'y')
        self.assertEqual(out['state'], 'response')
        self.assertFalse(out['reversible'])


class AxisCliTests(unittest.TestCase):
    def test_units_and_gap_are_bounded(self):
        from unittest.mock import patch
        for argv in (['prog', '--units', '5', '--out', '/tmp/x1'],
                     ['prog', '--gap', '9', '--out', '/tmp/x2']):
            with patch.object(sys, 'argv', argv), self.assertRaises(SystemExit):
                M.main()


if __name__ == '__main__':
    unittest.main()
