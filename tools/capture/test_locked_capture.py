import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import locked_capture as C

BOOT_ID = '12345678-1234-5678-1234-567812345678'
WALL_OFFSET = 1_700_000_000_000_000_000


def clock_sample(before, after, offset=WALL_OFFSET, boot_id=BOOT_ID):
    return dict(boot_id=boot_id, monotonic_before_ns=before,
                wall_ns=(before + after) // 2 + offset,
                monotonic_after_ns=after, bracket_uncertainty_ns=after - before)


class ClockCorrelationTests(unittest.TestCase):
    def setUp(self):
        self.before = clock_sample(950_000_000, 950_000_100)
        self.after = clock_sample(990_000_000, 990_000_100)
        self.stamp = 900_000_000

    def test_wall_read_is_actually_bracketed_and_boot_id_recorded(self):
        events = []

        def monotonic():
            events.append('monotonic')
            return 100 if len(events) == 1 else 106

        def wall():
            events.append('wall')
            return WALL_OFFSET + 103

        with patch.object(C.Path, 'read_text', return_value=BOOT_ID + '\n'), \
                patch.object(C.time, 'monotonic_ns', side_effect=monotonic), \
                patch.object(C.time, 'time_ns', side_effect=wall):
            sample = C.sample_clock_correlation()
        self.assertEqual(events, ['monotonic', 'wall', 'monotonic'])
        self.assertEqual(sample, clock_sample(100, 106))

    def test_missing_boot_id_does_not_create_clock_evidence(self):
        with patch.object(C.Path, 'read_text', side_effect=OSError('unavailable')):
            sample = C.sample_clock_correlation()
        self.assertIsNone(sample['boot_id'])
        result = C.capture_wall_time(self.stamp, sample, self.after)
        self.assert_unknown(result, 'boot ID')

    def assert_unknown(self, result, reason):
        self.assertEqual(result['clock_correlation']['status'], 'unknown')
        self.assertIn(reason, result['clock_correlation']['reason'])
        for key in ('captured_wall_ns', 'captured_wall_bounds_ns',
                    'captured_wall_uncertainty_ns'):
            self.assertIsNone(result[key])

    def test_capture_wall_estimate_uses_capture_pts_and_preserves_brackets(self):
        result = C.capture_wall_time(self.stamp, self.before, self.after)
        evidence = result['clock_correlation']
        self.assertEqual(evidence['status'], 'correlated')
        self.assertEqual(evidence['before'], self.before)
        self.assertEqual(evidence['after'], self.after)
        self.assertEqual(evidence['clock_rate_allowance_ppm'], 1000)
        self.assertEqual(result['captured_wall_ns'], WALL_OFFSET + self.stamp)
        low, high = result['captured_wall_bounds_ns']
        self.assertLess(low, WALL_OFFSET + self.stamp)
        self.assertGreater(high, WALL_OFFSET + self.stamp)
        self.assertEqual(result['captured_wall_uncertainty_ns'],
                         max(result['captured_wall_ns'] - low, high - result['captured_wall_ns']))
        self.assertEqual(evidence['semantics'], C.CAPTURE_TIME_SEMANTICS)
        self.assertNotIn('anchors', evidence)
        self.assertNotIn('replay_clock_s', evidence)

    def test_missing_malformed_or_regressing_evidence_is_unknown(self):
        invalid = (None, {}, dict(self.before, boot_id=None),
                   dict(self.before, bracket_uncertainty_ns=99),
                   dict(self.before, wall_ns=True),
                   dict(self.before, monotonic_before_ns=950_000_101))
        for before in invalid:
            with self.subTest(before=before):
                self.assert_unknown(C.capture_wall_time(self.stamp, before, self.after),
                                    'invalid clock correlation')
        self.assert_unknown(C.capture_wall_time(self.stamp, self.after, self.before), 'regress')
        self.assert_unknown(C.capture_wall_time(None, self.before, self.after), 'timestamp')
        self.assert_unknown(C.capture_wall_time(1_000_000_000, self.before, self.after), 'follows')

    def test_forward_and_backward_wall_jumps_are_unknown(self):
        for jump in (-1_000_000_000, 1_000_000_000):
            with self.subTest(jump=jump):
                after = dict(self.after, wall_ns=self.after['wall_ns'] + jump)
                self.assert_unknown(C.capture_wall_time(self.stamp, self.before, after), 'clock jump')

    def test_clock_jump_between_frames_and_boot_change_are_unknown(self):
        previous = clock_sample(800_000_000, 800_000_100, offset=WALL_OFFSET - 1_000_000_000)
        result = C.capture_wall_time(self.stamp, self.before, self.after, previous)
        self.assert_unknown(result, 'clock jump')
        self.assertEqual(result['clock_correlation']['previous'], previous)
        after = dict(self.after, boot_id='87654321-4321-8765-4321-876543218765')
        self.assert_unknown(C.capture_wall_time(self.stamp, self.before, after), 'boot ID changed')

    def test_small_wall_clock_slew_is_explicitly_bounded(self):
        after = dict(self.after, wall_ns=self.after['wall_ns'] + 1_000)
        result = C.capture_wall_time(self.stamp, self.before, after)
        self.assertEqual(result['clock_correlation']['status'], 'correlated')
        low, high = result['captured_wall_bounds_ns']
        self.assertLessEqual(low, WALL_OFFSET + self.stamp)
        self.assertGreaterEqual(high, WALL_OFFSET + self.stamp + 1_000)


class FreshnessTests(unittest.TestCase):
    def test_future_expired_missing_and_duplicate_frames_refuse(self):
        with patch.object(C.time, 'monotonic_ns', return_value=1_000_000_000):
            for stamp in (None, True, 0, 1_000_000_001, 249_999_999):
                with self.subTest(stamp=stamp), self.assertRaises(C.ProbeError):
                    C.require_fresh(dict(captured_monotonic_ns=stamp,
                                         capture_age_semantics=C.CAPTURE_SEMANTICS))
            meta = dict(captured_monotonic_ns=900_000_000,
                        capture_age_semantics=C.CAPTURE_SEMANTICS)
            self.assertEqual(C.require_fresh(meta), 900_000_000)
            with self.assertRaises(C.ProbeError):
                C.require_fresh(meta, 900_000_000)
            with self.assertRaises(C.ProbeError):
                C.require_fresh(dict(meta, capture_age_semantics='read_completed'))


class PinnedSourceTests(unittest.TestCase):
    def setUp(self):
        self.capture = C.LockedCapture.__new__(C.LockedCapture)
        self.capture.identity = dict(pid=3, display=':2', start_ticks='100')
        self.capture.window = '7'
        self.capture.socket = Path('/unused/control.sock')
        self.capture.node = dict(id=2, serial=80, process_id=1, name='gamescope')
        self.capture.previous_ns = None
        self.capture._temporary = None

    def check_source(self, **changes):
        result = dict(status='completed', target_resolved=True, game_pids=[3],
                      node=dict(self.capture.node))
        result.update(changes)
        with patch.object(C, 'game_identity', return_value=dict(self.capture.identity)), \
                patch.object(C, 'ancestor_pids', return_value={1, 2}), \
                patch.object(C, 'request', return_value=result):
            return self.capture.check()

    def test_new_serial_or_pid_cannot_rebind_existing_capture(self):
        for node in (dict(self.capture.node, serial=81), dict(self.capture.node, process_id=2)):
            with self.subTest(node=node), self.assertRaises(C.ProbeError):
                self.check_source(node=node)
        self.assertEqual(self.check_source()['node'], self.capture.node)

    def test_missing_ambiguous_or_unrelated_game_refuses(self):
        for change in (dict(target_resolved=False), dict(game_pids=[3, 4]),
                       dict(node=dict(self.capture.node, process_id=4))):
            with self.subTest(change=change), self.assertRaises(C.ProbeError):
                self.check_source(**change)

    def test_changed_game_generation_refuses(self):
        with patch.object(C, 'game_identity', return_value=dict(self.capture.identity, start_ticks='101')):
            with self.assertRaises(C.ProbeError):
                self.capture.check()

    def test_source_lost_after_pixels_is_not_published(self):
        with tempfile.TemporaryDirectory() as directory:
            self.capture._temporary = type('Directory', (), dict(name=directory))()
            path = Path(directory) / 'current.ppm'
            Image.new('RGB', (2560, 1600)).save(path)
            Path(str(path)+'.json').write_text(json.dumps(dict(
                clock_basis='gst_segment_running_time', captured_monotonic_ns=900_000_000)))
            with patch.object(self.capture, 'check', side_effect=[None, C.ProbeError('source changed')]), \
                    patch.object(C, 'request', return_value=dict(status='completed', node=2, process_id=1)):
                with self.assertRaisesRegex(C.ProbeError, 'source changed'):
                    self.capture.acquire()
            self.assertIsNone(self.capture.previous_ns)

    def test_rgb_and_capture_time_preserved_not_replaced_with_read_time(self):
        with tempfile.TemporaryDirectory() as directory:
            self.capture._temporary = type('Directory', (), dict(name=directory))()
            path = Path(directory) / 'current.ppm'
            Image.new('RGB', (2560, 1600), (170, 22, 9)).save(path)
            Path(str(path)+'.json').write_text(json.dumps(dict(
                clock_basis='gst_segment_running_time', captured_monotonic_ns=900_000_000)))
            samples = [clock_sample(950_000_000, 950_000_100),
                       clock_sample(990_000_000, 990_000_100)]
            with patch.object(self.capture, 'check'), patch.object(C.time, 'monotonic_ns', return_value=1_000_000_000), \
                    patch.object(C, 'sample_clock_correlation', side_effect=samples), \
                    patch.object(C, 'request', return_value=dict(status='completed', node=2, process_id=1)):
                pixels, meta = self.capture.acquire()
            self.assertEqual(pixels.getpixel((10, 10)), (170, 22, 9))
            self.assertEqual(meta['captured_monotonic_ns'], 900_000_000)
            self.assertFalse(meta['source_render_age_verified'])
            self.assertEqual(meta['capture_time_semantics'], C.CAPTURE_TIME_SEMANTICS)
            self.assertEqual(meta['captured_wall_ns'], WALL_OFFSET + 900_000_000)
            self.assertEqual(meta['clock_correlation']['before'], samples[0])
            self.assertEqual(meta['clock_correlation']['after'], samples[1])
            self.assertEqual(meta['replay_clock_correlation']['status'], 'unknown')
            self.assertNotIn('anchors', meta)
            self.assertNotIn('replay_clock_s', meta)
            self.assertEqual(self.capture._previous_clock_sample, samples[1])
            saved = Path(directory) / 'frame.png'
            self.capture.save(saved, pixels, meta)
            persisted = json.loads(Path(str(saved) + '.json').read_text())
            for key in ('captured_wall_ns', 'captured_wall_bounds_ns',
                        'captured_wall_uncertainty_ns', 'clock_correlation',
                        'capture_time_semantics', 'replay_clock_correlation'):
                self.assertEqual(persisted[key], meta[key])

    def test_acquire_clock_jump_publishes_unknown_wall_time(self):
        with tempfile.TemporaryDirectory() as directory:
            self.capture._temporary = type('Directory', (), dict(name=directory))()
            path = Path(directory) / 'current.ppm'
            Image.new('RGB', (2560, 1600)).save(path)
            # A supplied wall value is not trusted in place of actual sampling.
            Path(str(path) + '.json').write_text(json.dumps(dict(
                clock_basis='gst_segment_running_time', captured_monotonic_ns=900_000_000,
                captured_wall_ns=123, captured_wall_bounds_ns=[122, 124])))
            self.capture._previous_clock_sample = clock_sample(
                800_000_000, 800_000_100, offset=WALL_OFFSET - 1_000_000_000)
            samples = [clock_sample(950_000_000, 950_000_100),
                       clock_sample(990_000_000, 990_000_100)]
            with patch.object(self.capture, 'check'), \
                    patch.object(C.time, 'monotonic_ns', return_value=1_000_000_000), \
                    patch.object(C, 'sample_clock_correlation', side_effect=samples), \
                    patch.object(C, 'request', return_value=dict(status='completed', node=2, process_id=1)):
                _, meta = self.capture.acquire()
            self.assertEqual(meta['captured_monotonic_ns'], 900_000_000)
            self.assertEqual(meta['clock_correlation']['status'], 'unknown')
            self.assertIn('clock jump', meta['clock_correlation']['reason'])
            self.assertIsNone(meta['captured_wall_ns'])
            self.assertIsNone(meta['captured_wall_bounds_ns'])
            self.assertIsNone(meta['captured_wall_uncertainty_ns'])


if __name__ == '__main__':
    unittest.main()
