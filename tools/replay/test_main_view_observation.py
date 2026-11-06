"""Consumer-visible expiry/source transitions using actual image perception."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from main_view_observation import ContinuousMainViewRecognizer
from minimap_observation import FrameStamp
from fire_control import FireControl

ROOT = Path(__file__).resolve().parents[2]


class MainViewLeaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with Image.open(ROOT/'state/aimprobe-live-scope/key-action.png') as image:
            cls.rgb = np.asarray(image.convert('RGB'))

    def observe(self, consumer, seq=1, captured=1_000_000_000, epoch=0, completion=None):
        completion = captured+10_000_000 if completion is None else completion
        with patch('main_view_observation.time.monotonic_ns', side_effect=[captured, completion]):
            return consumer.observe(self.rgb, FrameStamp(epoch, seq, captured))

    def test_unique_ui_lock_is_revoked_on_expiry_and_never_authorizes_hull(self):
        c = ContinuousMainViewRecognizer()
        observed = self.observe(c)
        self.assertTrue(observed['lock']['confirmed'])
        self.assertEqual('unverified', observed['target']['body_association'])
        self.assertIsNone(observed['target']['anchor_px'])
        self.assertEqual(observed['valid_until_ns'], observed['target']['valid_until_ns'])
        self.assertFalse(FireControl().decide(observed,1_010_000_000,True)['fire_authorized'])
        tracked = self.observe(c, seq=2, captured=1_100_000_000)
        self.assertEqual('tracked', tracked['target']['state'])
        self.assertEqual('hull_anchor_unconfirmed',
            FireControl().decide(tracked, 1_110_000_000, True)['reason'])
        expired = c.snapshot(tracked['valid_until_ns'])
        self.assertFalse(expired['fresh'])
        self.assertFalse(expired['lock']['confirmed'])
        self.assertIsNone(expired['view'])
        self.assertIsNone(expired['hud'])
        self.assertIsNone(expired['target'])

    def test_slow_inference_cannot_refresh_capture_age(self):
        c = ContinuousMainViewRecognizer()
        out = self.observe(c, completion=1_750_000_000)
        self.assertEqual('processing_expired',out['reason'])
        self.assertFalse(out['lock']['confirmed'])
        self.assertIsNone(out['view'])

    def test_source_change_and_disconnect_clear_history(self):
        c = ContinuousMainViewRecognizer()
        self.observe(c)
        out = self.observe(c, seq=0, captured=1_100_000_000, epoch=1)
        self.assertTrue(out['fresh'])
        self.assertIsNone(out['view']['dt_s'])
        c.revoke()
        out = c.snapshot(1_120_000_000)
        self.assertFalse(out['lock']['confirmed'])
        self.assertIsNone(out['view'])
        self.assertIsNone(c.previous)

    def test_out_of_order_frame_and_clock_regression_revoke(self):
        c = ContinuousMainViewRecognizer()
        self.observe(c)
        with patch('main_view_observation.time.monotonic_ns',return_value=1_020_000_000):
            out = c.observe(self.rgb,FrameStamp(0,1,1_020_000_000))
        self.assertEqual('out_of_order',out['reason'])
        self.assertFalse(out['lock']['confirmed'])
        out = c.snapshot(1_010_000_000)
        self.assertEqual('clock_regressed',out['reason'])
        self.assertIsNone(out['view'])

    def test_snapshot_mutation_does_not_change_later_lock(self):
        c = ContinuousMainViewRecognizer()
        out = self.observe(c)
        out['lock']['confirmed']=False
        out['view']['detections'].clear()
        later = c.snapshot(1_030_000_000)
        self.assertTrue(later['lock']['confirmed'])
        self.assertTrue(later['view']['detections'])

    def test_capture_age_bound_at_deadline_revokes_cached_evidence(self):
        c = ContinuousMainViewRecognizer()
        self.observe(c)
        captured = 1_100_000_000
        with patch('main_view_observation.time.monotonic_ns',
                   side_effect=[captured, captured+650_000_000]):
            out = c.observe(self.rgb, FrameStamp(0, 2, captured, 100_000_000))
        self.assertEqual('processing_expired', out['reason'])
        self.assertFalse(out['fresh'])
        self.assertIsNone(out['view'])
        self.assertIsNone(out['hud'])
        self.assertIsNone(out['target'])
        self.assertFalse(out['lock']['confirmed'])
        self.assertFalse(out['fire_authorized'])
        self.assertIsNone(c.previous_view)

    def test_caller_image_mutation_cannot_change_cached_previous_frame(self):
        c = ContinuousMainViewRecognizer()
        rgb = self.rgb.copy()
        with patch('main_view_observation.time.monotonic_ns',
                   side_effect=[1_000_000_000, 1_010_000_000]):
            c.observe(rgb, FrameStamp(0, 1, 1_000_000_000))
        rgb[:] = 0
        self.assertTrue(np.array_equal(c.previous, self.rgb))
        later = c.snapshot(1_020_000_000)
        self.assertTrue(later['lock']['confirmed'])
        self.assertEqual('screen_track_only', later['target']['identity_association'])
        self.assertFalse(later['fire_authorized'])

    def test_expired_previous_frame_does_not_supply_motion_history(self):
        c = ContinuousMainViewRecognizer()
        self.observe(c)
        out = self.observe(c, seq=2, captured=1_750_000_000)
        self.assertTrue(out['fresh'])
        self.assertIsNone(out['view']['dt_s'])
        self.assertIsNone(out['view']['camera_shift'])
        self.assertTrue(all(item['screen_velocity_px_s'] is None
                            for item in out['view']['observations']))
        self.assertFalse(out['fire_authorized'])

    def test_healthbar_track_is_current_and_not_reused_after_expiry(self):
        c = ContinuousMainViewRecognizer()
        first = self.observe(c)
        track = first['lock']['target_track_id']
        self.assertIsInstance(track, int)
        self.assertEqual('tentative', first['lock']['track_state'])
        expired = c.snapshot(first['valid_until_ns'])
        self.assertFalse(expired['lock']['confirmed'])
        later = self.observe(c, seq=2, captured=first['valid_until_ns']+1)
        self.assertNotEqual(track, later['lock']['target_track_id'])
        self.assertEqual('tentative', later['lock']['track_state'])
        self.assertEqual(later['lock']['target_track_id'], later['target']['id'])
        self.assertEqual('tentative', later['target']['state'])
        self.assertFalse(later['fire_authorized'])

    def test_worker_failure_revokes_previous_lock_and_history(self):
        c = ContinuousMainViewRecognizer()
        self.observe(c)
        with patch('main_view_observation.detect_aircraft', side_effect=RuntimeError('detector failed')):
            with self.assertRaisesRegex(RuntimeError, 'detector failed'):
                self.observe(c, seq=2, captured=1_100_000_000)
        out = c.snapshot(1_120_000_000)
        self.assertEqual('perception_error', out['reason'])
        self.assertIsNone(out['view'])
        self.assertIsNone(out['hud'])
        self.assertIsNone(c.previous)
        self.assertFalse(out['lock']['confirmed'])
        self.assertFalse(out['fire_authorized'])

    def test_detector_failure_on_settlement_revokes_prior_evidence(self):
        with Image.open(Path(__file__).with_name('main_view_fixtures')/'post-battle.png') as image:
            settlement = np.asarray(image.convert('RGB'))
        c = ContinuousMainViewRecognizer()
        self.observe(c)
        with patch('main_view_observation.detect_aircraft', side_effect=RuntimeError('detector failed')):
            with self.assertRaisesRegex(RuntimeError, 'detector failed'):
                with patch('main_view_observation.time.monotonic_ns', return_value=1_100_000_000):
                    c.observe(settlement, FrameStamp(0, 2, 1_100_000_000))
        out = c.snapshot(1_120_000_000)
        self.assertEqual('perception_error', out['reason'])
        self.assertFalse(out['fresh'])
        self.assertIsNone(out['view'])
        self.assertIsNone(out['hud'])
        self.assertIsNone(c.previous)
        self.assertFalse(out['lock']['confirmed'])

    def test_torpedo_failure_on_battle_or_menu_revokes_previous_evidence(self):
        with Image.open(Path(__file__).with_name('main_view_fixtures')/'post-battle.png') as image:
            settlement = np.asarray(image.convert('RGB'))
        for rgb in (self.rgb, settlement):
            with self.subTest(battle=rgb is self.rgb):
                c = ContinuousMainViewRecognizer()
                self.observe(c)
                with patch('main_view_observation.detect_torpedoes',
                           side_effect=RuntimeError('torpedo detector failed')):
                    with self.assertRaisesRegex(RuntimeError, 'torpedo detector failed'):
                        with patch('main_view_observation.time.monotonic_ns', return_value=1_100_000_000):
                            c.observe(rgb, FrameStamp(0, 2, 1_100_000_000))
                out = c.snapshot(1_120_000_000)
                self.assertEqual('perception_error', out['reason'])
                self.assertIsNone(out['view'])
                self.assertIsNone(out['hud'])
                self.assertIsNone(c.previous)
                self.assertFalse(out['lock']['confirmed'])
                self.assertFalse(out['fire_authorized'])

    def test_settlement_revokes_battle_tracks_and_reentry_has_no_menu_history(self):
        with Image.open(Path(__file__).with_name('main_view_fixtures')/'post-battle.png') as image:
            settlement = np.asarray(image.convert('RGB'))
        c = ContinuousMainViewRecognizer()
        first = self.observe(c)
        first_track = first['lock']['target_track_id']
        self.assertTrue(first['lock']['confirmed'])
        with patch('main_view_observation.time.monotonic_ns',
                   side_effect=[1_100_000_000, 1_110_000_000]):
            menu = c.observe(settlement, FrameStamp(0, 2, 1_100_000_000))
        self.assertEqual('battle_hud_unconfirmed', menu['reason'])
        self.assertFalse(menu['fresh'])
        self.assertIsNone(menu['view'])
        self.assertIsNone(menu['hud'])
        self.assertIsNone(c.previous)
        self.assertFalse(menu['lock']['confirmed'])
        self.assertEqual('revoke', FireControl().decide(menu, 1_110_000_000, True)['action'])
        later = self.observe(c, seq=3, captured=1_200_000_000)
        self.assertTrue(later['fresh'])
        self.assertIsNone(later['view']['dt_s'])
        self.assertNotEqual(first_track, later['lock']['target_track_id'])
        self.assertFalse(later['fire_authorized'])

    def test_malformed_frame_clears_previous_lock(self):
        c = ContinuousMainViewRecognizer()
        self.observe(c)
        with patch('main_view_observation.time.monotonic_ns',return_value=1_020_000_000):
            with self.assertRaises(ValueError):
                c.observe(np.zeros((100,100,3),np.uint8),FrameStamp(0,2,1_020_000_000))
        self.assertIsNone(c.snapshot(1_030_000_000)['view'])
        self.assertFalse(c.snapshot(1_030_000_000)['lock']['confirmed'])


if __name__ == '__main__':
    unittest.main()
