"""Replay surface coordinate and owned-button failure boundaries."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import replay_view_action as action


class ReplayViewActionTests(unittest.TestCase):
    def test_capture_content_maps_to_game_window_without_bars(self):
        self.assertEqual((960, 998), action.window_point(1280, 1410, 1920, 1080))
        self.assertEqual((0, 0), action.window_point(0, 80, 1920, 1080))
        for values in ((0, 79, 1920, 1080), (2560, 900, 1920, 1080),
                       (100, 1520, 1920, 1080), (1280, 900, 1000, 1000)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                action.window_point(*values)

    def exercise(self, directory, mask=0, fail_down=False, mode='click',
                 button_hold_ms=0, interrupt_hold=False):
        calls = []
        def x(probe, *args, **kwargs):
            calls.append(args)
            if args[0] == 'search': return '9'
            if args[0] == 'getwindowgeometry': return 'WIDTH=1920\nHEIGHT=1080'
            if args[0] == 'mousedown' and fail_down:
                raise action.ProbeError('timeout after possible delivery')
            return ''
        args = argparse.Namespace(out=Path(directory)/'out', action=mode, execute=True,
                                  point=[1280,1410] if mode == 'click' else None, seconds=15.)
        args.button_hold_ms = button_hold_ms
        def sleep(seconds):
            if interrupt_hold:
                raise KeyboardInterrupt('interrupted while holding menu button')
        with patch.object(action, 'game_identity', return_value=dict(pid=12,display=':2',start_ticks='1')), \
             patch.object(action.Probe, 'x', x), patch.object(action.Probe, 'verify_window'), \
             patch.object(action.Probe, 'focus'), patch.object(action.Probe, 'check_identity'), \
             patch.object(action.Probe, 'snapshot'), \
             patch.object(action, 'pointer_state', return_value=dict(x=10,y=20,mask=mask)), \
             patch.object(action.time, 'sleep', side_effect=sleep), \
             contextlib.redirect_stdout(io.StringIO()):
            code = action.run(args)
        return code, calls, json.loads((args.out/'report.json').read_text())

    def test_preheld_button_is_never_acquired_or_released(self):
        with tempfile.TemporaryDirectory() as directory:
            code, calls, report = self.exercise(directory, mask=256)
        self.assertEqual(1, code)
        self.assertFalse(any(c[0] in ('mousedown','mouseup') for c in calls))
        self.assertEqual([], report['inputs'])

    def test_possible_button_delivery_is_released_on_transport_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            code, calls, report = self.exercise(directory, fail_down=True)
        self.assertEqual(1, code)
        self.assertEqual([('mousedown',1),('mouseup',1)], [c for c in calls if c[0] in ('mousedown','mouseup')])
        self.assertFalse(report['unreleased_button'])
        self.assertEqual('failed',report['status'])

    def test_interrupted_wheel_releases_only_acquired_button(self):
        for mode, button in (('zoom-in', 4), ('zoom-out', 5)):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                code, calls, report = self.exercise(directory, fail_down=True, mode=mode)
            self.assertEqual(1, code)
            self.assertEqual([('mousedown', button), ('mouseup', button)],
                             [c for c in calls if c[0] in ('mousedown', 'mouseup')])
            self.assertFalse(report['unreleased_button'])

    def test_preheld_button_blocks_wheel_without_releasing_it(self):
        with tempfile.TemporaryDirectory() as directory:
            code, calls, report = self.exercise(directory, mask=2048, mode='zoom-in')
        self.assertEqual(1, code)
        self.assertFalse(any(c[0] in ('mousedown', 'mouseup') for c in calls))
        self.assertEqual([], report['inputs'])

    def test_interrupted_click_hold_releases_acquired_button_and_restores_pointer(self):
        with tempfile.TemporaryDirectory() as directory:
            code, calls, report = self.exercise(directory, button_hold_ms=120,
                                                 interrupt_hold=True)
        self.assertEqual(1, code)
        self.assertEqual([('mousedown', 1), ('mouseup', 1)],
                         [c for c in calls if c[0] in ('mousedown', 'mouseup')])
        self.assertIn(('mousemove', 10, 20), calls)
        self.assertFalse(report['unreleased_button'])
        self.assertEqual([], report['inputs'])
        self.assertEqual('failed', report['status'])

    def test_held_button_remains_unowned_with_requested_click_hold(self):
        with tempfile.TemporaryDirectory() as directory:
            code, calls, report = self.exercise(directory, mask=256, button_hold_ms=120)
        self.assertEqual(1, code)
        self.assertFalse(any(c[0] in ('mousedown', 'mouseup') for c in calls))
        self.assertEqual([], report['inputs'])


class WindowCaptureActionTests(unittest.TestCase):
    """Desktop game pixels must stay identity-bound and never claim render age."""

    def probe(self, directory, identity, window):
        report = dict(out=str(Path(directory) / 'out'), snapshots=[])
        (Path(directory) / 'out').mkdir()
        probe = action.WindowActionProbe(15., report)
        probe.identity, probe.window = identity, window
        return probe, report

    def test_snapshot_records_window_pixels_without_render_age_claim(self):
        identity = dict(pid=12, display=':2', start_ticks='1')
        meta = dict(source='xcomposite_window_pixmap', identity=identity, window='9',
                    render_age_verified=False, source_render_monotonic_ns=None)

        class Capture:
            def __init__(self):
                self.identity, self.window = dict(identity), '9'

            def grab(self, path, wait_update=False):
                Path(path).write_bytes(b'png')
                return meta

        with tempfile.TemporaryDirectory() as directory, \
                patch('wows_window_capture.WindowCapture', Capture), \
                patch.object(action.Probe, 'check_identity'):
            probe, report = self.probe(directory, identity, '9')
            path = probe.snapshot('before')
            self.assertEqual(Path(path).read_bytes(), b'png')
        self.assertEqual(report['snapshots'][0]['source'], 'xcomposite_window_pixmap')
        self.assertFalse(report['snapshots'][0]['render_age_verified'])

    def test_snapshot_refuses_a_different_game_generation_or_window(self):
        identity = dict(pid=12, display=':2', start_ticks='1')

        class Capture:
            def __init__(self):
                self.identity = dict(identity, pid=99)
                self.window = '9'

        with tempfile.TemporaryDirectory() as directory, \
                patch('wows_window_capture.WindowCapture', Capture):
            probe, _ = self.probe(directory, identity, '9')
            with self.assertRaisesRegex(action.ProbeError, 'identities differ'):
                probe.snapshot('before')

    def test_window_source_is_explicit_and_never_a_silent_host_fallback(self):
        with tempfile.TemporaryDirectory() as directory, \
                contextlib.redirect_stdout(io.StringIO()):
            args = argparse.Namespace(out=Path(directory)/'out', action='key', execute=False,
                                      seconds=15., key_name='Escape', capture_source='window',
                                      point=None)
            args.key_hold_ms, args.button_hold_ms = 0, 0
            action.run(args)
            plan = json.loads((args.out/'plan.json').read_text())
        self.assertEqual(plan['capture_source'], 'window')


if __name__ == '__main__':
    unittest.main()
