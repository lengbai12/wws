"""Current-frame hazards must disappear or associate on the next measured frame."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import main_view_observation as M
from main_view_observation import ContinuousMainViewRecognizer
from minimap_observation import FrameStamp

SECOND = 1_000_000_000


class HazardCadenceTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.zeros((M.FRAME_H, M.FRAME_W, 3), np.uint8)
        self.calls = {'aircraft': 0, 'torpedo': 0}
        patches = [
            patch.object(M, 'detect_aircraft', self._aircraft),
            patch.object(M, 'detect_torpedoes', self._torpedo),
            patch.object(M.FireHudReader, 'detect', lambda reader, rgb: {
                'battle': {'active': True}, 'weapon': {}, 'reload': {}, 'reticle': {},
                'flight_time': {}, 'aiming_view': {}}),
            patch.object(M, 'perceive_frame', self._perceive),
            patch.object(M, 'confirm_target', lambda rgb, detections: dict(
                confirmed=False, state='unknown', target_association='unknown')),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    def _aircraft(self, rgb):
        self.calls['aircraft'] += 1
        return [dict(class_='aircraft', bbox=[600, 900, 24, 24], score=0.9,
                     evidence='cadence_fixture', team='enemy',
                     marker_family='central_dangers')]

    def _torpedo(self, rgb):
        self.calls['torpedo'] += 1
        return [dict(bbox=[800, 900, 24, 24], score=0.9, evidence='cadence_fixture',
                     **{'class': 'torpedo'}, team='enemy', marker_type='normal')]

    def _perceive(self, rgb, previous=None, templates=None, dt_s=None, *,
                  previous_view=None, aircraft_job=None, torpedo_job=None):
        markers = list(torpedo_job.result()) + list(aircraft_job.result())
        return dict(detections=[], markers=markers, observations=[], camera_shift=None,
                    body_motion=[], dt_s=dt_s)

    def observe(self, consumer, seq, captured, epoch=0):
        with patch.object(M.time, 'monotonic_ns',
                          side_effect=[captured, captured + 5_000_000]):
            return consumer.observe(self.rgb, FrameStamp(epoch, seq, captured))

    def test_default_still_measures_both_hazards_on_every_frame(self):
        consumer = ContinuousMainViewRecognizer()
        first = self.observe(consumer, 0, SECOND)
        second = self.observe(consumer, 1, SECOND + 100_000_000)
        self.assertTrue(first['fresh'] and second['fresh'])
        self.assertEqual(self.calls, {'aircraft': 2, 'torpedo': 2})
        for hit in second['view']['markers']:
            self.assertNotIn('hazard_age_ns', hit)

    def test_disappeared_torpedo_is_not_republished_on_next_frame(self):
        consumer = ContinuousMainViewRecognizer()
        self.observe(consumer, 0, SECOND)
        with patch.object(M, 'detect_torpedoes', return_value=[]):
            second = self.observe(consumer, 1, SECOND + 100_000_000)
        self.assertFalse(any(hit.get('class') == 'torpedo'
                             for hit in second['view']['markers']))

    def test_measured_torpedo_keeps_identity_across_adjacent_frames(self):
        consumer = ContinuousMainViewRecognizer()
        first = self.observe(consumer, 0, SECOND)
        second = self.observe(consumer, 1, SECOND + 100_000_000)
        before = next(hit for hit in first['view']['markers'] if hit.get('class') == 'torpedo')
        after = next(hit for hit in second['view']['markers'] if hit.get('class') == 'torpedo')
        self.assertIsNotNone(before['track_id'])
        self.assertEqual(after['track_id'], before['track_id'])
        self.assertEqual(after['track_state'], 'tracked')


if __name__ == '__main__':
    unittest.main()
