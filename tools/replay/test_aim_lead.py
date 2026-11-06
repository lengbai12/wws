import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aim_lead as L


class AimLead(unittest.TestCase):
    def test_stationary_target_needs_no_lead(self):
        aim, seconds, residual = L.solve_aim((100.0, 50.0), (0.0, 0.0), lambda _: 8.0)
        self.assertEqual(aim, (100.0, 50.0))
        self.assertEqual(seconds, 8.0)
        self.assertLess(residual, 1e-9)

    def test_moving_target_is_led_ahead(self):
        # 2 px/s for ~10 s means the aim point sits about 20 px ahead.
        aim, seconds, residual = L.solve_aim((0.0, 0.0), (2.0, 0.0), lambda _: 10.0)
        self.assertAlmostEqual(aim[0], 20.0, places=6)
        self.assertAlmostEqual(aim[1], 0.0, places=6)
        self.assertLess(residual, 1e-6)

    def test_flight_time_that_grows_with_distance_still_settles(self):
        # Shorter range means shorter flight; the fixed point must converge.
        aim, seconds, residual = L.solve_aim((0.0, 0.0), (1.0, 0.0),
                                             lambda p: 2.0 + math.hypot(*p) / 100.0)
        self.assertGreater(aim[0], 0.0)
        self.assertLess(residual, 1e-3)
        self.assertGreater(seconds, 2.0)

    def test_flight_seconds_refuses_nonsense(self):
        self.assertEqual(L.flight_seconds(0, 800), 0.0)
        self.assertEqual(L.flight_seconds(-5, 800), 0.0)
        self.assertAlmostEqual(L.flight_seconds(8000, 800), 10.0)

    def test_map_velocity_needs_positive_dt(self):
        self.assertEqual(L.map_velocity((0, 0), (10, 10), 0.0), (0.0, 0.0))
        self.assertEqual(L.map_velocity((0, 0), (10, 5), 2.0), (5.0, 2.5))

    def test_velocity_sign_flip_resets_instead_of_smoothing(self):
        smoothed = L.smooth_velocity((10.0, 0.0), (-8.0, 0.0))
        self.assertEqual(smoothed, (-8.0, 0.0))
        blended = L.smooth_velocity((10.0, 0.0), (12.0, 0.0), blend=.5)
        self.assertAlmostEqual(blended[0], 11.0)

    def test_clamp_lead_bounds_the_offset(self):
        out = L.clamp_lead((300.0, 400.0), 100.0)
        self.assertAlmostEqual(math.hypot(*out), 100.0, places=6)
        self.assertEqual(L.clamp_lead((3.0, 4.0), 100.0), (3.0, 4.0))


if __name__ == '__main__':
    unittest.main()
