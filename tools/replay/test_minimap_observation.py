import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from minimap_observation import FrameStamp, MinimapObservations


def detections(own=True):
    return dict(own=dict(u=.2, v=.3) if own else None,
                enemies=[dict(u=.5, v=.5, score=.9, species='Destroyer')])


class FreshnessContract(unittest.TestCase):
    def test_age_bound_expiry_revokes_own_and_enemy(self):
        consumer = MinimapObservations(max_age_s=.75)
        consumer.update(detections(), FrameStamp(0, 1, 1_000_000_000), 1_000_000_000)
        obs = consumer.update(detections(), FrameStamp(0, 2, 1_100_000_000, 100_000_000), 1_100_000_000)
        self.assertTrue(obs['enemies'][0]['usable'])
        self.assertEqual(1_750_000_000, obs['valid_until_ns'])
        self.assertTrue(consumer.snapshot(1_749_999_999)['fresh'])
        expired = consumer.snapshot(1_750_000_000)
        self.assertFalse(expired['fresh'])
        self.assertIsNone(expired['own'])
        self.assertEqual([], expired['enemies'])
        self.assertFalse(expired['fire_authorized'])

    def test_missing_own_does_not_hold_last_position(self):
        consumer = MinimapObservations()
        consumer.update(detections(), FrameStamp(0, 0, 0), 0)
        obs = consumer.update(detections(False), FrameStamp(0, 1, 100_000_000), 100_000_000)
        self.assertIsNone(obs['own'])
        self.assertTrue(obs['fresh'])

    def test_out_of_order_revokes_and_cannot_roll_back_sequence(self):
        consumer = MinimapObservations()
        consumer.update(detections(), FrameStamp(0, 5, 100), 100)
        obs = consumer.update(detections(), FrameStamp(0, 4, 90), 110)
        self.assertEqual('out_of_order', obs['reason'])
        self.assertIsNone(obs['own'])
        self.assertEqual(5, obs['seq'])
        self.assertFalse(consumer.update(detections(), FrameStamp(0, 5, 120), 120)['fresh'])
        self.assertTrue(consumer.update(detections(), FrameStamp(0, 6, 130), 130)['fresh'])

    def test_source_epoch_reset_and_late_old_source_revoke(self):
        consumer = MinimapObservations()
        first = consumer.update(detections(), FrameStamp(1, 10, 100), 100)['enemies'][0]['track_id']
        obs = consumer.update(detections(), FrameStamp(2, 0, 200), 200)
        self.assertNotEqual(first, obs['enemies'][0]['track_id'])
        self.assertFalse(obs['enemies'][0]['usable'])
        self.assertEqual('old_epoch', consumer.update(detections(), FrameStamp(1, 11, 300), 300)['reason'])

    def test_future_and_slow_inference_rejected(self):
        for captured, now, reason in [(100, 99, 'future_frame'), (0, 750_000_001, 'stale_frame')]:
            obs = MinimapObservations().update(detections(), FrameStamp(0, 0, captured), now)
            self.assertEqual(reason, obs['reason'])
            self.assertIsNone(obs['own'])

    def test_focus_loss_and_invalid_detection_clear_state(self):
        consumer = MinimapObservations()
        consumer.update(detections(), FrameStamp(0, 0, 100), 100)
        consumer.revoke('focus_lost')
        self.assertEqual('focus_lost', consumer.snapshot(101)['reason'])
        with self.assertRaises(ValueError):
            consumer.update(dict(own=dict(u=float('nan'), v=.3), enemies=[]), FrameStamp(0, 1, 200), 200)
        self.assertIsNone(consumer.snapshot(201)['own'])

    def test_stamp_rejects_invalid_values(self):
        for value in [-1, 1.2, True]:
            with self.assertRaises(ValueError):
                FrameStamp(0, value, 0)

    def test_output_mutation_cannot_change_later_snapshot(self):
        consumer = MinimapObservations()
        obs = consumer.update(detections(), FrameStamp(0, 0, 100), 100)
        obs['own']['u'] = .9
        obs['enemies'][0]['u'] = .9
        self.assertEqual(.2, consumer.snapshot(100)['own']['u'])
        self.assertEqual(.5, consumer.snapshot(100)['enemies'][0]['u'])

    def test_invalid_clocks_revoke_in_update_and_snapshot(self):
        for value in (True, -1, 1.5, float('nan'), float('inf'), '100'):
            for method in ('update', 'snapshot'):
                with self.subTest(value=value, method=method):
                    consumer = MinimapObservations()
                    consumer.update(detections(), FrameStamp(0, 0, 100), 100)
                    with self.assertRaises(ValueError):
                        if method == 'update':
                            consumer.update(detections(), FrameStamp(0, 1, 200), value)
                        else:
                            consumer.snapshot(value)
                    obs = consumer.snapshot(200)
                    self.assertEqual('invalid_clock', obs['reason'])
                    self.assertFalse(obs['fresh'])
                    self.assertEqual([], obs['enemies'])

    def test_clock_regression_cannot_restore_observations_or_change_epoch(self):
        consumer = MinimapObservations()
        consumer.update(detections(), FrameStamp(0, 0, 100), 200)
        obs = consumer.update(detections(), FrameStamp(1, 0, 150), 150)
        self.assertEqual('clock_regressed', obs['reason'])
        self.assertEqual(0, obs['epoch'])
        self.assertFalse(consumer.snapshot(199)['fresh'])
        self.assertFalse(consumer.snapshot(200)['fresh'])
        self.assertTrue(consumer.update(detections(), FrameStamp(1, 0, 201), 201)['fresh'])

    def test_invalid_stamp_and_boolean_own_revoke_prior_state(self):
        for stamp, own in ((None, dict(u=.2, v=.3)),
                           (object(), dict(u=.2, v=.3)),
                           (FrameStamp(0, 1, 200), dict(u=True, v=.3)),
                           (FrameStamp(0, 1, 200), dict(u=.2, v=False))):
            with self.subTest(stamp=stamp, own=own):
                consumer = MinimapObservations()
                consumer.update(detections(), FrameStamp(0, 0, 100), 100)
                with self.assertRaises(ValueError):
                    consumer.update(dict(own=own, enemies=[]), stamp, 200)
                obs = consumer.snapshot(200)
                self.assertFalse(obs['fresh'])
                self.assertIsNone(obs['own'])
                self.assertEqual([], obs['enemies'])

    def test_capture_regression_revokes_without_rolling_back_watermark(self):
        consumer = MinimapObservations()
        consumer.update(detections(), FrameStamp(0, 0, 200), 200)
        obs = consumer.update(detections(), FrameStamp(0, 1, 199), 201)
        self.assertEqual('out_of_order', obs['reason'])
        self.assertFalse(obs['fresh'])
        self.assertFalse(consumer.update(detections(), FrameStamp(0, 2, 200), 202)['fresh'])
        obs = consumer.update(detections(), FrameStamp(0, 3, 203), 203)
        self.assertEqual('tentative', obs['enemies'][0]['state'])

    def test_freshness_gap_restarts_confirmation_without_snapshot_call(self):
        consumer = MinimapObservations()
        consumer.update(detections(), FrameStamp(0, 0, 0), 0)
        old = consumer.update(detections(), FrameStamp(0, 1, 100_000_000), 100_000_000)
        self.assertTrue(old['enemies'][0]['usable'])
        obs = consumer.update(detections(), FrameStamp(0, 2, 850_000_000), 850_000_000)
        self.assertNotEqual(old['enemies'][0]['track_id'], obs['enemies'][0]['track_id'])
        self.assertFalse(obs['enemies'][0]['usable'])
        self.assertEqual(1, obs['enemies'][0]['hits'])

    def test_missing_enemy_frame_cannot_hold_confirmed_current_target(self):
        consumer = MinimapObservations()
        consumer.update(detections(), FrameStamp(0, 0, 0), 0)
        consumer.update(detections(), FrameStamp(0, 1, 100_000_000), 100_000_000)
        obs = consumer.update(dict(own=None, enemies=[]), FrameStamp(0, 2, 200_000_000), 200_000_000)
        self.assertTrue(obs['fresh'])
        self.assertTrue(all(t['state'] == 'lost' and not t['observed'] and not t['usable']
                            for t in obs['enemies']))


if __name__ == '__main__':
    unittest.main()
