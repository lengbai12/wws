"""The lead must prefer the client's own flight time over a nominal speed.

Fixture `main_view_fixtures/flight-time-15.9-11s.png` is a real 2560x1600 frame
captured 2026-10-17 from the running client; the HUD prints "11s" beside the
reticle and the native-font reader decodes it.
"""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'capture'))
import aim_servo as A

FIXTURE = Path(__file__).resolve().parent / 'main_view_fixtures' / 'flight-time-15.9-11s.png'


class FlightTimeTests(unittest.TestCase):
    def test_the_fixture_yields_the_client_number(self):
        import cv2
        image = cv2.imread(str(FIXTURE))
        if image is None:
            self.skipTest('fixture missing')
        frame = np.ascontiguousarray(image[:, :, ::-1])
        flight = A.hud_flight_seconds(frame)
        if flight is None:
            self.skipTest('the current HUD reader does not decode this frame')
        self.assertEqual(flight['source'], 'hud_flight_time')
        self.assertGreater(flight['value'], 0)
        self.assertLessEqual(flight['value'], 25)
        self.assertIsNotNone(flight['text'])

    def test_values_outside_the_sane_range_are_refused(self):
        """A misread number must not be used as a lead."""
        class Reader:
            def __init__(self, payload):
                self.payload = payload

            def detect(self, frame):
                return {'flight_time': self.payload}

        def nested(status, value, confidence=.86):
            return {'status': 'unknown', 'value': None,
                    'raw': {'numbers': [{'status': status, 'value': value, 'text': 'x',
                                         'confidence': confidence}]}}

        original = A._flight_reader
        try:
            for payload in (nested('observed', 0.0), nested('observed', 40.0),
                            nested('unknown', 11.0), nested('observed', None)):
                A._flight_reader = Reader(payload)
                self.assertIsNone(A.hud_flight_seconds(np.zeros((1600, 2560, 3), np.uint8)), payload)
            A._flight_reader = Reader(nested('observed', 11.0))
            got = A.hud_flight_seconds(np.zeros((1600, 2560, 3), np.uint8))
            self.assertEqual(got['value'], 11.0)
            self.assertEqual(got['source'], 'hud_flight_time')
        finally:
            A._flight_reader = original

    def test_the_nested_number_is_what_is_used(self):
        """The top level carries no value; the decoded number does."""
        class Reader:
            def detect(self, frame):
                return {'flight_time': {'status': 'unknown', 'value': None,
                                        'raw': {'numbers': [
                                            {'status': 'unknown', 'value': None},
                                            {'status': 'observed', 'value': 11.0, 'text': '11',
                                             'confidence': .857}]}}}

        original = A._flight_reader
        try:
            A._flight_reader = Reader()
            got = A.hud_flight_seconds(np.zeros((1600, 2560, 3), np.uint8))
            self.assertEqual(got['value'], 11.0)
            self.assertEqual(got['text'], '11')
        finally:
            A._flight_reader = original


class ScopeEvidenceTests(unittest.TestCase):
    def test_wanting_the_scope_is_not_having_it(self):
        class Reader:
            def __init__(self, payload):
                self.payload = payload

            def detect(self, frame):
                return {'aiming_view': self.payload}

        original = A._flight_reader
        try:
            A._flight_reader = Reader({'status': 'unknown', 'raw': {'reason': 'scope_ruler_unconfirmed'}})
            got = A.hud_scope_evidence(np.zeros((1600, 2560, 3), np.uint8))
            self.assertFalse(got['observed'])
            self.assertEqual(got['reason'], 'scope_ruler_unconfirmed')
            A._flight_reader = Reader({'status': 'observed', 'mode': 'main_guns',
                                       'zoom_confirmed': True, 'raw': {}})
            got = A.hud_scope_evidence(np.zeros((1600, 2560, 3), np.uint8))
            self.assertTrue(got['observed'])
        finally:
            A._flight_reader = original

    def test_the_scope_gate_decision_is_pure(self):
        """Wanting the scope is not having it: only evidence opens the gate."""
        self.assertFalse(A.scope_blocks_fire(False, False))
        self.assertFalse(A.scope_blocks_fire(False, True))
        self.assertTrue(A.scope_blocks_fire(True, False))
        self.assertTrue(A.scope_blocks_fire(True, None))
        self.assertFalse(A.scope_blocks_fire(True, True))


class GainMeasurementTests(unittest.TestCase):
    """The per-session gain probe, driven by a synthetic scene shift."""

    def _run(self, response):
        calls = []

        class Probe:
            def move(self, dx, dy):
                calls.append((dx, dy))

        frames = {'n': 0}

        def frame_getter(name):
            frames['n'] += 1
            return frames['n']

        original = A.estimate_camera_shift
        try:
            A.estimate_camera_shift = lambda base, now: response(now, calls)
            return A.measure_gain(Probe(), frame_getter, settle=0.0, units=20, min_shift=3.0), calls
        finally:
            A.estimate_camera_shift = original

    def test_a_responding_camera_yields_the_gain(self):
        # forward frames are the 2nd and the 5th reads; each axis moves 40 px
        state = {'x': 0, 'y': 0}

        def response(now, calls):
            if not calls:
                return [0.0, 0.0]
            dx, dy = calls[-1]
            return [40.0 if dx > 0 else (-40.0 if dx < 0 else 0.0),
                    1.2 * dy]

        got, calls = self._run(response)
        self.assertTrue(got['ok'])
        self.assertAlmostEqual(got['gain'][0], 2.0, places=6)
        self.assertAlmostEqual(got['gain'][1], 1.2, places=6)
        self.assertIn((20, 0), calls)
        self.assertIn((-20, 0), calls)          # every probe is reverted
        self.assertIn((0, 3), calls)
        self.assertIn((0, -3), calls)
        self.assertEqual(got['axis_units'], [20, 3])

    def test_a_dead_axis_is_refused(self):
        got, _ = self._run(lambda now, calls: [0.0, 0.0])
        self.assertFalse(got['ok'])
        self.assertIsNone(got['gain'][0])
        self.assertEqual(got['reason'], 'an axis did not move the scene')


class TargetContinuityTests(unittest.TestCase):
    """A sweep must keep the ship it was aiming at, not hop to the nearest."""

    def near(self, bbox, dist):
        return dict(nameplate_bbox=list(bbox), aim_px=[bbox[0], bbox[1]], trackable=True,
                    distance_from_reticle_px=dist)

    def test_the_previous_target_wins_over_a_closer_one(self):
        previous = [1500, 700, 90, 16]              # the plate we were tracking
        staying = self.near([1520, 705, 90, 16], 380)   # same area, far from reticle
        closer = self.near([1200, 800, 90, 16], 40)     # a different ship, near the reticle
        pick = A.choose_target([closer, staying], previous)
        self.assertIs(pick, staying)

    def test_without_history_the_nearest_wins(self):
        closer = self.near([1200, 800, 90, 16], 40)
        far = self.near([1500, 700, 90, 16], 380)
        self.assertIs(A.choose_target([far, closer], None), closer)

    def test_no_candidates_is_none(self):
        self.assertIsNone(A.choose_target([], [1, 2, 3, 4]))


class LeadHorizonTests(unittest.TestCase):
    """The lead is the client's flight time plus the measured latency horizon."""

    def test_horizon_is_half_the_step_period(self):
        total, horizon = A.lead_with_horizon(11.0, 0.0, 0.6)
        self.assertAlmostEqual(horizon, 0.3, places=6)
        self.assertAlmostEqual(total, 11.3, places=6)

    def test_the_cap_bounds_the_horizon(self):
        total, horizon = A.lead_with_horizon(2.0, 0.0, 8.0, cap_s=.5)
        self.assertAlmostEqual(horizon, 0.5, places=6)
        self.assertAlmostEqual(total, 2.5, places=6)

    def test_no_period_means_no_horizon(self):
        total, horizon = A.lead_with_horizon(11.0, 0.0, None)
        self.assertEqual(horizon, 0.0)
        self.assertAlmostEqual(total, 11.0, places=6)

    def test_without_a_flight_time_the_configured_lead_is_used(self):
        total, horizon = A.lead_with_horizon(None, 3.0, 0.0)
        self.assertAlmostEqual(total, 3.0, places=6)


class StaleFrameTests(unittest.TestCase):
    """A move with no visible effect means the frame predates the move."""

    def test_a_real_move_with_an_identical_error_is_stale(self):
        self.assertTrue(A.moved_but_unchanged([300.0, 40.0], [300.4, 40.2], [-150, -20]))

    def test_a_move_the_scene_answered_is_not_stale(self):
        self.assertFalse(A.moved_but_unchanged([300.0, 40.0], [140.0, 40.0], [-150, 0]))

    def test_a_tiny_move_is_never_called_stale(self):
        self.assertFalse(A.moved_but_unchanged([300.0, 40.0], [300.0, 40.0], [-5, 0]))

    def test_missing_history_cannot_be_stale(self):
        self.assertFalse(A.moved_but_unchanged(None, [1.0, 2.0], [200, 0]))
        self.assertFalse(A.moved_but_unchanged([1.0, 2.0], [1.0, 2.0], None))


class PatchPredictionTests(unittest.TestCase):
    """The search must start where the target is going, not where it was."""

    def test_velocity_moves_the_search_seed(self):
        got = A.predict_patch([100, 200, 40, 20], [50.0, -20.0], 0.6)
        self.assertEqual(got[:2], [130.0, 188.0])
        self.assertEqual(got[2:], [40, 20])

    def test_a_missing_velocity_leaves_the_box_alone(self):
        self.assertEqual(A.predict_patch([100, 200, 40, 20], None, 0.6), [100, 200, 40, 20])
        self.assertEqual(A.predict_patch([100, 200, 40, 20], [50.0, 0.0], 0), [100, 200, 40, 20])

    def test_a_wild_velocity_is_clamped(self):
        got = A.predict_patch([0, 0, 10, 10], [10000.0, 0.0], 1.0, max_shift=160.0)
        self.assertLessEqual(abs(got[0]), 160.0)


if __name__ == '__main__':
    unittest.main()
