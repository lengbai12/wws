#!/usr/bin/env python3
"""Smoothness guarantees for the camera: no jumps, no shake.

Measured failures on 2026-10-21 that these tests pin down:

* a single vertical action of 30-160 units threw the view at the sky and then
  slammed it onto the sea;
* a sweep of 19 x 140 units ran the pointer into the screen edge, after which
  every further move in that direction was silently clipped and the servo kept
  re-reading the same frame (reported as "pending mouse move has no observed
  response"), i.e. seek - pause - seek jitter;
* rapid up/down alternation shook the camera in place.
"""
import importlib.util
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / 'replay'))

spec = importlib.util.spec_from_file_location('aim_servo', HERE / 'aim_servo.py')
A = importlib.util.module_from_spec(spec)
sys.modules['aim_servo'] = A
spec.loader.exec_module(A)


class Probe:
    def __init__(self):
        self.report = {'inputs': []}

    def move(self, dx, dy):
        self.report['inputs'].append({'relative': [dx, dy]})


class CameraSmoothnessTests(unittest.TestCase):
    def setUp(self):
        A.reset_vertical_guard()

    def tearDown(self):
        A.reset_vertical_guard()

    def test_a_huge_vertical_request_is_clamped_to_the_hard_step(self):
        probe = Probe()
        sent = A.bounded_move(probe, 0, 160, 160)
        self.assertEqual(sent, (0, A.MAX_VERTICAL_STEP_UNITS))
        row = probe.report['inputs'][-1]
        self.assertEqual(row['relative'], [0, A.MAX_VERTICAL_STEP_UNITS])
        self.assertEqual(row['requested_relative'], [0, 160])
        self.assertTrue(row['vertical_limited'])

    def test_the_hard_step_is_at_most_six_units(self):
        self.assertLessEqual(A.MAX_VERTICAL_STEP_UNITS, 6)

    def test_a_small_request_is_passed_through_untouched(self):
        probe = Probe()
        self.assertEqual(A.bounded_move(probe, 40, -2, 3), (40, -2))
        self.assertNotIn('vertical_limited', probe.report['inputs'][-1])

    def test_alternating_vertical_moves_stop_instead_of_shaking(self):
        probe = Probe()
        for index in range(8):
            A.bounded_move(probe, 0, 3 if index % 2 == 0 else -3, 6)
        self.assertTrue(probe.report.get('vertical_oscillation_stopped'))
        self.assertEqual(probe.report['inputs'][-1]['relative'], [0, 0])

    def test_an_ordinary_correction_is_not_mistaken_for_shaking(self):
        probe = Probe()
        for dy in (3, 3, -3, 3, -3, 3):
            A.bounded_move(probe, 0, dy, 6)
        self.assertNotIn('vertical_oscillation_stopped', probe.report)
        self.assertEqual(probe.report['inputs'][-1]['relative'], [0, 3])

    def test_reset_clears_the_history(self):
        probe = Probe()
        for index in range(6):
            A.bounded_move(probe, 0, 3 if index % 2 == 0 else -3, 6)
        A.reset_vertical_guard()
        A.bounded_move(probe, 0, 3, 6)
        self.assertEqual(probe.report['inputs'][-1]['relative'], [0, 3])
        self.assertNotIn('vertical_oscillation_stopped', probe.report)

    def test_a_sweep_is_a_turn_not_an_oscillation(self):
        # The operator asks for 360 degrees; 24 steps of 400 units must walk
        # monotonically to 9200 units, not reverse every 700 units.
        path = A.scan_path(24, 1, 400, 3, reach_y=3)
        self.assertTrue(all(dx > 0 for dx, _ in path))
        self.assertEqual(sum(dx for dx, _ in path), 23 * 400)

    def test_an_explicit_budget_still_bounds_the_walk(self):
        # A caller that must stay inside a small arc can still ask for it.
        path = A.scan_path(20, 1, 140, 3, reach_y=3, budget=A.SCAN_BUDGET_UNITS)
        position = 0
        for dx, _dy in path:
            position += dx
            self.assertLessEqual(abs(position), A.SCAN_BUDGET_UNITS + 140)

    def test_rows_alternate_so_the_second_sweep_comes_back(self):
        path = A.scan_path(4, 2, 100, 3, reach_y=3)
        first = [dx for dx, _ in path if dx]
        self.assertEqual(first[:3], [100, 100, 100])
        self.assertEqual(first[3:], [-100, -100, -100])

    def test_one_row_means_no_pitch_sweep_at_all(self):
        path = A.scan_path(12, 1, 80, 3, reach_y=3)
        self.assertTrue(all(dy == 0 for _dx, dy in path))

    def test_default_sweep_fits_inside_a_screen(self):
        path = A.scan_path(12, 1, 80, 3, reach_y=3)
        total = sum(abs(dx) for dx, _ in path)
        self.assertLessEqual(total, 1400,
                             'default sweep must stay deliverable within 1920 px')

    def test_no_single_scan_step_exceeds_the_budget(self):
        path = A.scan_path(40, 2, 200, 3, reach_y=3)
        self.assertTrue(all(abs(dx) <= 200 for dx, _ in path))


if __name__ == '__main__':
    unittest.main()


class ClosedLoopSmoothnessTests(unittest.TestCase):
    """Drive the real controller against a synthetic camera.

    The camera model is deliberately unfriendly: a fixed gain, the pointer
    clipping at the screen edges, and a pitch that gets coarser near the
    horizon.  The controller must still converge without jumping.
    """

    class Camera:
        # The measured gain is negative: +80 mouse units moved the plate 91 px
        # to the left, so the error (target - reticle) changes by gain * move.
        def __init__(self, error, gain=(-1.3, -1.3), screen=(1920, 1080),
                     pointer=(960, 540), clip=True):
            self.error = list(error)
            self.gain = gain
            self.screen = screen
            self.pointer = list(pointer)
            self.clip = clip
            self.sent = []
            self.report = {'inputs': []}

        def move(self, dx, dy):
            self.report['inputs'].append({'relative': [dx, dy]})
            delivered = [dx, dy]
            if self.clip:
                for axis in (0, 1):
                    room_low = self.pointer[axis]
                    room_high = self.screen[axis] - self.pointer[axis]
                    if delivered[axis] < -room_low:
                        delivered[axis] = -room_low
                    if delivered[axis] > room_high:
                        delivered[axis] = room_high
                    self.pointer[axis] += delivered[axis]
            self.sent.append([dx, dy])
            self.error[0] += self.gain[0] * delivered[0]
            self.error[1] += self.gain[1] * delivered[1]

    def test_a_converging_aim_never_jumps_or_shakes(self):
        cam = self.Camera([240.0, -90.0])
        probe = cam
        previous = None
        for _step in range(20):
            dx, dy, _ = A.plan_move(cam.error, [-1.3, -1.3], probe_y=3,
                                    reach=A.MAX_VERTICAL_STEP_UNITS * 15,
                                    reach_y=A.MAX_VERTICAL_STEP_UNITS,
                                    probe=20)
            A.bounded_move(probe, dx, dy,
                           min(90, A.MAX_VERTICAL_STEP_UNITS))
            self.assertLessEqual(abs(probe.report['inputs'][-1]['relative'][1]),
                                 A.MAX_VERTICAL_STEP_UNITS)
            if previous is not None:
                self.assertLessEqual(abs(cam.error[1]), abs(previous) + 1e-6,
                                     'vertical error must not grow')
            previous = cam.error[1]
            if abs(cam.error[0]) <= 12 and abs(cam.error[1]) <= 12:
                break
        self.assertLessEqual(abs(cam.error[0]), 12)
        self.assertLessEqual(abs(cam.error[1]), 12)
        self.assertNotIn('vertical_oscillation_stopped', probe.report)

    def test_a_pointer_parked_in_the_corner_still_moves_in_bounds(self):
        cam = self.Camera([0.0, -80.0], pointer=(0, 1080 - 1))
        probe = cam
        for _ in range(5):
            dx, dy, _ = A.plan_move(cam.error, [-1.3, -1.3], probe_y=3,
                                    reach=A.MAX_VERTICAL_STEP_UNITS * 15,
                                    reach_y=A.MAX_VERTICAL_STEP_UNITS,
                                    probe=20)
            A.bounded_move(probe, dx, dy, A.MAX_VERTICAL_STEP_UNITS)
        self.assertTrue(all(abs(row['relative'][1]) <= A.MAX_VERTICAL_STEP_UNITS
                            for row in probe.report['inputs']))


class SharedLimitTests(unittest.TestCase):
    """Every camera-driving tool must obey the same numbers."""

    def setUp(self):
        import camera_limits
        self.limits = camera_limits

    def test_the_hard_vertical_cap_is_six(self):
        self.assertEqual(self.limits.MAX_VERTICAL_STEP_UNITS, 6)

    def test_clamp_vertical_bounds_every_request(self):
        for request, expected in ((160, 6), (-160, -6), (6, 6), (-6, -6),
                                  (0, 0), (None, 0), (3, 3)):
            self.assertEqual(self.limits.clamp_vertical(request), expected)

    def test_clamp_vertical_never_widens_a_smaller_limit(self):
        self.assertEqual(self.limits.clamp_vertical(20, limit=3), 3)
        self.assertEqual(self.limits.clamp_vertical(20, limit=60), 6)

    def test_clamp_horizontal_is_bounded(self):
        self.assertEqual(self.limits.clamp_horizontal(1000), 400)
        self.assertEqual(self.limits.clamp_horizontal(-1000), -400)
        self.assertEqual(self.limits.clamp_horizontal(50), 50)

    def test_aim_servo_uses_the_shared_numbers(self):
        import aim_servo
        self.assertEqual(aim_servo.MAX_VERTICAL_STEP_UNITS,
                         self.limits.MAX_VERTICAL_STEP_UNITS)
        self.assertEqual(aim_servo.SCAN_BUDGET_UNITS,
                         self.limits.SCAN_BUDGET_UNITS)


class ScenarioMatrixTests(unittest.TestCase):
    """Deterministic pseudo-random scenarios: the invariants must always hold."""

    def _run(self, seed):
        import random
        rng = random.Random(seed)
        gain = (-rng.uniform(0.6, 2.0), -rng.uniform(0.6, 2.0))
        error = [rng.uniform(-400, 400), rng.uniform(-200, 200)]
        screen = (1920, 1080)
        pointer = [rng.choice([0, 5, 960, 1915]), rng.choice([0, 5, 540, 1075])]
        cam = ClosedLoopSmoothnessTests.Camera(error, gain=gain, screen=screen,
                                               pointer=pointer)
        start = list(cam.error)
        for _step in range(30):
            dx, dy, _ = A.plan_move(cam.error, list(gain), probe_y=3,
                                    reach=A.MAX_VERTICAL_STEP_UNITS * 15,
                                    reach_y=A.MAX_VERTICAL_STEP_UNITS, probe=20)
            A.bounded_move(cam, dx, dy, A.MAX_VERTICAL_STEP_UNITS)
            self.assertLessEqual(abs(cam.report['inputs'][-1]['relative'][1]),
                                 A.MAX_VERTICAL_STEP_UNITS)
            self.assertLessEqual(abs(cam.error[1]), abs(start[1]) + 6 + 40,
                                 'vertical error diverged (seed %s)' % seed)
            if abs(cam.error[0]) <= 12 and abs(cam.error[1]) <= 12:
                break
        self.assertNotIn('vertical_oscillation_stopped', cam.report,
                         'the guard fired on a legitimate run (seed %s)' % seed)

    def test_forty_seeded_scenarios_stay_smooth(self):
        for seed in range(40):
            with self.subTest(seed=seed):
                A.reset_vertical_guard()
                self._run(seed)

    def test_a_clipped_start_position_is_still_bounded(self):
        for seed in range(8):
            with self.subTest(seed=seed):
                A.reset_vertical_guard()
                import random
                rng = random.Random(1000 + seed)
                cam = ClosedLoopSmoothnessTests.Camera(
                    [rng.uniform(-300, 300), rng.uniform(-150, 150)],
                    gain=(-rng.uniform(0.6, 2.0), -rng.uniform(0.6, 2.0)),
                    pointer=[0, 0])
                for _ in range(20):
                    dx, dy, _ = A.plan_move(cam.error, [-1.2, -1.2], probe_y=3,
                                            reach=90,
                                            reach_y=A.MAX_VERTICAL_STEP_UNITS,
                                            probe=20)
                    A.bounded_move(cam, dx, dy, A.MAX_VERTICAL_STEP_UNITS)
                self.assertTrue(all(abs(r['relative'][1]) <= A.MAX_VERTICAL_STEP_UNITS
                                    for r in cam.report['inputs']))
