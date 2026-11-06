#!/usr/bin/env python3
"""Diagnostic only: the flight-time estimate is NOT used as a gate.

The HUD draws the target range and the shell-impact range as text; a flight
time multiplied by a nominal speed overstates it (measured 15 s for 9.9 km,
i.e. ~660 m/s average), so this helper stays a diagnostic.
"""
import unittest
from unittest.mock import patch

import aim_servo as A


class ImpactRangeTests(unittest.TestCase):
    def test_flight_time_times_speed_is_the_impact_range(self):
        with patch.object(A, 'hud_flight_seconds', lambda frame: {'value': 15.0}):
            self.assertEqual(A.impact_range_km(None, 800.0), 12.0)

    def test_the_two_measured_frames_reproduce_their_gaps(self):
        # dk.png: 15.0 s against an 11.8 km target -> inside the 2 km rule.
        with patch.object(A, 'hud_flight_seconds', lambda frame: {'value': 15.0}):
            self.assertLessEqual(abs(11.8 - A.impact_range_km(None)), 2.0)
        # after-burst: 8.1 s against a 16.2 km target -> 9.7 km short, refused.
        with patch.object(A, 'hud_flight_seconds', lambda frame: {'value': 8.1}):
            self.assertGreater(abs(16.2 - A.impact_range_km(None)), 2.0)

    def test_absent_or_absurd_flight_time_reads_nothing(self):
        for reading in (None, {}, {'value': None}, {'value': 0.0}, {'value': 40.0}):
            with patch.object(A, 'hud_flight_seconds', lambda frame, r=reading: r):
                self.assertIsNone(A.impact_range_km(None))

    def test_shell_speed_scales_it(self):
        with patch.object(A, 'hud_flight_seconds', lambda frame: {'value': 10.0}):
            self.assertEqual(A.impact_range_km(None, 900.0), 9.0)


if __name__ == '__main__':
    unittest.main()
