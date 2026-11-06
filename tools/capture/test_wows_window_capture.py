"""Window capture evidence contracts without an X server or game input."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

import wows_window_capture as W


PPM = b'P6\n1 1\n255\n\x00\x10\xff'


def evidence(lower_bound=200, semantics=W.DRAWABLE_UPDATE_SEMANTICS):
    return json.dumps(dict(drawable_update_lower_bound_monotonic_ns=lower_bound,
                           drawable_update_semantics=semantics)).encode()


class WindowCaptureEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.capture = W.WindowCapture.__new__(W.WindowCapture)
        self.capture.identity = dict(pid=123, display=':test')
        self.capture.env = {'DISPLAY': ':test'}
        self.capture.binary = Path('/test/window_pixels')
        self.capture.window = '0x123'
        self.capture.check = Mock()

    def grab(self, path, stderr=b'', *, wait_update=False, failure=None):
        with patch.object(W.subprocess, 'run',
                          return_value=SimpleNamespace(stdout=PPM, stderr=stderr),
                          side_effect=failure) as run, \
                patch.object(W.time, 'monotonic_ns', side_effect=[100, 900]):
            metadata = self.capture.grab(path, wait_update=wait_update)
        return metadata, run

    def test_wait_records_drawable_bound_without_render_age_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'frame.png'
            stderr = json.loads(evidence())
            stderr.update(source_render_monotonic_ns=300, render_age_verified=True)
            metadata, run = self.grab(path, json.dumps(stderr).encode(), wait_update=True)
            self.assertEqual(metadata['drawable_update_lower_bound_monotonic_ns'], 200)
            self.assertEqual(metadata['drawable_update_semantics'], W.DRAWABLE_UPDATE_SEMANTICS)
            self.assertIsNone(metadata['source_render_monotonic_ns'])
            self.assertIs(metadata['render_age_verified'], False)
            self.assertEqual(metadata['acquisition_started_monotonic_ns'], 100)
            self.assertEqual(metadata['acquisition_completed_monotonic_ns'], 900)
            self.assertEqual(json.loads(Path(str(path) + '.json').read_text()), metadata)
            self.assertTrue(path.is_file())
        self.assertEqual(self.capture.check.call_count, 2)
        run.assert_called_once_with(['/test/window_pixels', '0x123', '--wait-update'],
                                    env=self.capture.env, capture_output=True, timeout=5, check=True)

    def test_default_ignores_stderr_and_adds_no_update_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            metadata, run = self.grab(Path(tmp) / 'frame.png', evidence())
        self.assertNotIn('drawable_update_lower_bound_monotonic_ns', metadata)
        self.assertNotIn('drawable_update_semantics', metadata)
        self.assertIsNone(metadata['source_render_monotonic_ns'])
        self.assertIs(metadata['render_age_verified'], False)
        run.assert_called_once_with(['/test/window_pixels', '0x123'],
                                    env=self.capture.env, capture_output=True, timeout=5, check=True)

    def test_acquire_returns_rgb_and_evidence_before_any_encoding_or_write(self):
        with patch.object(W.subprocess, 'run',
                          return_value=SimpleNamespace(stdout=PPM, stderr=evidence())), \
                patch.object(W.time, 'monotonic_ns', side_effect=[100, 900]), \
                patch.object(W.Image.Image, 'save') as encode, \
                patch.object(W.Path, 'open') as open_path:
            image, metadata = self.capture.acquire(wait_update=True)
        self.assertEqual(image.mode, 'RGB')
        self.assertEqual(image.size, (1, 1))
        self.assertEqual(image.tobytes(), b'\x00\x10\xff')
        self.assertEqual(metadata['raw_pixel_sha256'],
                         hashlib.sha256(b'\x00\x10\xff').hexdigest())
        self.assertNotIn('sha256', metadata)
        self.assertEqual(metadata['drawable_update_lower_bound_monotonic_ns'], 200)
        self.assertIs(metadata['render_age_verified'], False)
        self.assertIsNone(metadata['source_render_monotonic_ns'])
        encode.assert_not_called()
        open_path.assert_not_called()

    def test_save_preserves_acquisition_evidence_without_recapture(self):
        image = W.Image.new('RGB', (1, 1), (0, 16, 255))
        original = dict(acquisition_started_monotonic_ns=100,
                        acquisition_completed_monotonic_ns=900,
                        drawable_update_lower_bound_monotonic_ns=200,
                        drawable_update_semantics=W.DRAWABLE_UPDATE_SEMANTICS,
                        raw_pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                        source_render_monotonic_ns=None, render_age_verified=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'frame.png'
            with patch.object(W.subprocess, 'run') as run, \
                    patch.object(W.time, 'monotonic_ns') as clock:
                saved = self.capture.save(path, image, original)
            self.assertEqual(saved['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(json.loads(Path(str(path) + '.json').read_text()), saved)
        self.assertNotIn('sha256', original)
        self.assertEqual({key: saved[key] for key in original}, original)
        self.capture.check.assert_not_called()
        run.assert_not_called()
        clock.assert_not_called()

    def test_missing_or_malformed_wait_evidence_fails_before_writing(self):
        for stderr in (b'', b'{}', b'null', b'[]', b'not JSON', b'\xff',
                       b'diagnostic\n' + evidence(), evidence() + b'\n{}'):
            with self.subTest(stderr=stderr), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'frame.png'
                with self.assertRaisesRegex(W.ProbeError, 'evidence'):
                    self.grab(path, stderr, wait_update=True)
                self.assertFalse(path.exists())
                self.assertFalse(Path(str(path) + '.json').exists())

    def test_wait_evidence_requires_integer_clock_bounds_and_exact_semantics(self):
        invalid = [evidence(value) for value in (None, True, '200', 200.0, 0, 99, 901)]
        invalid.append(evidence(semantics='source_render_timestamp'))
        invalid.append(json.dumps(dict(drawable_update_lower_bound_monotonic_ns=200)).encode())
        for stderr in invalid:
            with self.subTest(stderr=stderr), self.assertRaisesRegex(W.ProbeError, 'evidence'):
                W.drawable_update_evidence(stderr, 100, 900)
        for value in (100, 900):
            self.assertEqual(W.drawable_update_evidence(evidence(value), 100, 900)
                             ['drawable_update_lower_bound_monotonic_ns'], value)

    def test_native_update_timeout_is_explicit_and_writes_no_frame(self):
        failure = subprocess.CalledProcessError(
            1, ['window_pixels'], stderr=b'timed out after 500 ms waiting for a new drawable update\n')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'frame.png'
            with self.assertRaisesRegex(W.ProbeError, 'timed out after 500 ms'):
                self.grab(path, wait_update=True, failure=failure)
            self.assertFalse(path.exists())

    def test_process_timeout_is_explicit_and_writes_no_frame(self):
        failure = subprocess.TimeoutExpired(['window_pixels'], 5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'frame.png'
            with self.assertRaisesRegex(W.ProbeError, '5 second timeout'):
                self.grab(path, wait_update=True, failure=failure)
            self.assertFalse(path.exists())

    def test_default_preserves_native_errors_and_existing_file_refusal(self):
        failures = (subprocess.TimeoutExpired(['window_pixels'], 5),
                    subprocess.CalledProcessError(1, ['window_pixels'], stderr=b'failed'))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'frame.png'
            for failure in failures:
                with self.subTest(failure=failure), self.assertRaises(type(failure)):
                    self.grab(path, failure=failure)
            path.write_bytes(b'existing')
            with patch.object(self.capture, 'acquire') as acquire:
                with self.assertRaises(FileExistsError):
                    self.capture.grab(path, wait_update=True)
                acquire.assert_not_called()
            self.assertEqual(path.read_bytes(), b'existing')

    def test_cli_passes_wait_update_for_each_frame(self):
        capture = SimpleNamespace(identity={}, window='0x123', grab=Mock(return_value={}))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'capture'
            with patch.object(W, 'WindowCapture', return_value=capture), \
                    patch('sys.argv', ['wows_window_capture.py', '--out', str(path),
                                       '--frames', '2', '--interval', '.05', '--wait-update']), \
                    patch.object(W.time, 'sleep'), patch('builtins.print'):
                W.main()
            self.assertEqual(capture.grab.call_args_list,
                             [call(path / 'frame-000.png', wait_update=True),
                              call(path / 'frame-001.png', wait_update=True)])


if __name__ == '__main__':
    unittest.main()
