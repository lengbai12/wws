#!/usr/bin/env python3
"""The minimap shot maths, as pure functions, with the measured constants."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/capture'))

import minimap_shot as MS  # noqa: E402


class YawTests(unittest.TestCase):
    def test_degrees_become_units_at_the_measured_gain(self):
        # 37.5 px per degree at 1.5 px/unit => 25 units per degree.
        self.assertAlmostEqual(MS.yaw_units(10.0, 1.5), 250.0, places=3)
        self.assertAlmostEqual(MS.yaw_units(-10.0, 1.5), -250.0, places=3)

    def test_an_unknown_bearing_moves_nothing(self):
        self.assertEqual(MS.yaw_units(None), 0.0)


class ElevationTests(unittest.TestCase):
    def test_a_nearer_target_looks_up_with_a_negative_move(self):
        # aim short of the target => negative dy (look further).
        self.assertLess(MS.elevation_units(10.0, 13.0), 0)

    def test_a_farther_target_looks_down(self):
        self.assertGreater(MS.elevation_units(15.0, 13.0), 0)

    def test_the_move_is_bounded(self):
        self.assertLessEqual(abs(MS.elevation_units(0.1, 20.0, km_per_unit=0.004)), 120.0)

    def test_a_missing_reading_moves_nothing(self):
        self.assertEqual(MS.elevation_units(None, 13.0), 0.0)
        self.assertEqual(MS.elevation_units(13.0, None), 0.0)


class ShotTests(unittest.TestCase):
    def test_a_solution_inside_both_tolerances_is_allowed(self):
        allowed, why = MS.shot_allowed(13.1, 13.2, 1.0)
        self.assertTrue(allowed, why)

    def test_a_range_gap_refuses_and_names_the_number(self):
        allowed, why = MS.shot_allowed(10.0, 13.0, 0.5)
        self.assertFalse(allowed)
        self.assertIn('range_gap', why)
        self.assertIn('3.00', why)

    def test_a_bearing_gap_refuses(self):
        allowed, why = MS.shot_allowed(13.0, 13.0, 8.0)
        self.assertFalse(allowed)
        self.assertIn('bearing_gap', why)

    def test_any_missing_number_refuses(self):
        for aim, target, bearing in ((None, 13.0, 1.0), (13.0, None, 1.0),
                                     (13.0, 13.0, None)):
            allowed, why = MS.shot_allowed(aim, target, bearing)
            self.assertFalse(allowed)
            self.assertEqual(why, 'reading_missing')


class ContactTests(unittest.TestCase):
    def test_a_missing_contact_returns_none(self):
        def contacts_fn(_frame):
            return None, {}
        self.assertIsNone(MS.contact_from_frame(object(), contacts_fn))

    def test_range_comes_from_the_pixel_distance(self):
        def contacts_fn(_frame):
            return (dict(own_pixel=[0.0, 0.0],
                         nearest=dict(pixel=[200.0, 0.0], relative_bearing_deg=12.0)), {})
        km, bearing, own = MS.contact_from_frame(object(), contacts_fn)
        self.assertAlmostEqual(km, 200.0 * 0.0501, places=4)
        self.assertEqual(bearing, 12.0)
        self.assertEqual(own, [0.0, 0.0])


if __name__ == '__main__':
    unittest.main()
