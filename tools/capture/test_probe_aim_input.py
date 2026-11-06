"""Offline regressions: synthetic evidence never establishes world calibration."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import probe_aim_input as probe


class ProbeTests(unittest.TestCase):
    def test_dry_run_is_offline_and_never_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'new'
            with patch.object(probe, 'game_identity', side_effect=AssertionError('offline')), patch.object(probe.subprocess, 'run', side_effect=AssertionError('no transport')), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(probe.main(['--out', str(out)]), 0)
            report = json.loads((out/'report.json').read_text())
            self.assertEqual(report['status'], 'dry-run')
            self.assertFalse(report['identity_verified'])
            self.assertFalse(report['calibration_valid'])
            self.assertEqual(report['inputs'], [])
            self.assertEqual(report['plan'][1]['relative'], [40, 0])
            self.assertEqual(report['plan'][3]['relative'], [-40, 0])

    def test_camera_link_before_baseline_and_movement_plan(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            out = Path(directory)/'new'
            with patch.object(probe, 'game_identity', side_effect=AssertionError('offline')):
                self.assertEqual(probe.main(['--dry-run', '--key-action', 'camera-link',
                                             '--key-phase', 'before', '--baseline',
                                             '--seconds', '30', '--out', str(out)]), 0)
            report = json.loads((out/'report.json').read_text())
            self.assertEqual(report['plan'], [
                {'snapshot': 'initial'}, {'key': 'c'}, {'snapshot': 'key-action'},
                {'snapshot': 'baseline'}, {'snapshot': 'before'}, {'relative': [40, 0]},
                {'snapshot': 'after'}, {'relative': [-40, 0]}, {'snapshot': 'rollback'}])
            self.assertTrue(report['limits']['cleanup_within_seconds'])
            self.assertFalse(report['world_calibration_valid'])

    def test_execute_orders_camera_action_before_baseline_and_mouse(self):
        import numpy as np
        events = []
        identity = {'pid': 12, 'display': ':17', 'start_ticks': '123'}
        def snapshot(p, name):
            events.append(('snapshot', name))
            return Path(name)
        def move(p, dx, dy):
            events.append(('move', dx, dy))
            p.report['movement_delivery_attempted'] = True
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            out = Path(directory)/'new'
            with patch.object(probe, 'game_identity', return_value=identity), patch.object(probe.Probe, 'x', return_value='99'), patch.object(probe.Probe, 'verify_window'), patch.object(probe.Probe, 'focus'), patch.object(probe.Probe, 'snapshot', snapshot), patch.object(probe.Probe, 'move', move), patch.object(probe.Probe, 'key', lambda p, key: events.append(('key', key))), patch.object(probe.signal, 'signal'), patch.object(probe.signal, 'setitimer', return_value=(0.0, 0.0)), patch('cv2.imread', return_value=np.zeros((4, 4, 3), np.uint8)):
                self.assertEqual(probe.main(['--execute', '--key-action', 'camera-link',
                                             '--key-phase', 'before', '--baseline',
                                             '--out', str(out)]), 0)
            self.assertEqual(events, [('snapshot', 'initial'), ('key', 'c'),
                                      ('snapshot', 'key-action'), ('snapshot', 'baseline'),
                                      ('snapshot', 'before'), ('move', 40, 0),
                                      ('snapshot', 'after'), ('move', -40, 0),
                                      ('snapshot', 'rollback')])

    def test_invalid_bounds_rejected_before_output(self):
        for args in (['--seconds', '31'], ['--seconds', 'nan'], ['--dx', '201'], ['--dx', '0', '--dy', '0'], ['--key-phase', 'before']):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                probe.main(args)

    def test_exact_unique_process_and_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def process(pid, comm):
                entry = root/str(pid)
                entry.mkdir()
                (entry/'comm').write_text(comm)
                (entry/'environ').write_bytes(b'DISPLAY=:17\0')
                (entry/'stat').write_text(f'{pid} (space in name) '+ ' '.join(['S', '1']+['0']*17+['123']))
            process(1, 'WorldOfWarships-helper')
            with self.assertRaises(probe.ProbeError):
                probe.game_identity(root)
            process(2, 'WorldOfWarships')
            self.assertEqual(probe.game_identity(root), {'pid': 2, 'display': ':17', 'start_ticks': '123'})
            process(3, 'WorldOfWarships')
            with self.assertRaises(probe.ProbeError):
                probe.game_identity(root)

    def test_wrong_focus_sends_no_move(self):
        report = {'focus_checks': [], 'inputs': []}
        p = probe.Probe(2, report)
        p.identity = {'pid': 12}
        p.window = '99'
        with patch.object(p, 'check_identity'), patch.object(p, 'verify_window', return_value={}), patch.object(p, 'x', return_value='88') as x:
            with self.assertRaises(probe.ProbeError):
                p.move(40, 0)
            x.assert_called_once_with('getwindowfocus')
        self.assertEqual(report['inputs'], [])

    def test_failed_transport_keeps_attempt_and_uncertain_delivery(self):
        p = probe.Probe(2, {'inputs': []})
        p.identity = {'display': ':17'}
        with patch.object(p, 'focus'), \
                patch.object(probe, 'pointer_state', return_value={'x': 100, 'y': 200, 'mask': 0}), \
                patch.object(p, 'x', side_effect=probe.ProbeError('delivery uncertain')):
            with self.assertRaisesRegex(probe.ProbeError, 'delivery uncertain'):
                p.move(4, 0)
        self.assertTrue(p.report['movement_delivery_attempted'])
        event = p.report['inputs'][0]
        self.assertEqual(event['transport_status'], 'attempted')
        self.assertIsNone(event['pointer_after'])
        self.assertIsNone(event['pointer_delta'])

    def test_failed_post_read_preserves_completed_command_without_delta(self):
        p = probe.Probe(2, {'inputs': []})
        p.identity = {'display': ':17'}
        with patch.object(p, 'focus'), patch.object(p, 'x'), \
                patch.object(probe, 'pointer_state', side_effect=[
                    {'x': 100, 'y': 200, 'mask': 0}, probe.ProbeError('lost display')]):
            with self.assertRaisesRegex(probe.ProbeError, 'lost display'):
                p.move(4, 0)
        event = p.report['inputs'][0]
        self.assertEqual(event['transport_status'], 'command_completed')
        self.assertIsNone(event['pointer_delta'])
        self.assertIn('lost display', event['error'])

    def test_failed_pre_read_does_not_attempt_input(self):
        p = probe.Probe(2, {'inputs': []})
        p.identity = {'display': ':17'}
        with patch.object(p, 'focus'), patch.object(p, 'x') as transport, \
                patch.object(probe, 'pointer_state', side_effect=probe.ProbeError('lost display')):
            with self.assertRaisesRegex(probe.ProbeError, 'lost display'):
                p.move(4, 0)
        transport.assert_not_called()
        self.assertFalse(p.report.get('movement_delivery_attempted', False))
        self.assertEqual(p.report['inputs'], [])

    def test_preheld_key_is_never_released(self):
        p = probe.Probe(2, {'inputs': []})
        with patch.object(p, 'focus'), patch.object(p, 'key_is_down', return_value=True), patch.object(p, 'x') as x:
            with self.assertRaises(probe.ProbeError):
                p.key('Shift_L')
            p.release()
            x.assert_not_called()
        self.assertEqual(p.held, {})

    def test_failed_keydown_releases_only_own_key(self):
        p = probe.Probe(2, {'inputs': []})
        p.identity = {'pid': 12, 'display': ':17', 'start_ticks': '123'}
        p.window = '99'
        with patch.object(p, 'focus'), patch.object(p, 'key_is_down', return_value=False), patch.object(p, 'x', side_effect=[probe.ProbeError('delivery uncertain'), '']) as x, patch.object(probe, 'process_identity', return_value=p.identity):
            with self.assertRaises(probe.ProbeError):
                p.key('x')
            p.release()
            self.assertEqual(x.call_args_list[-1].args, ('keyup', 'x'))
            self.assertEqual(x.call_args_list[-1].kwargs,
                             {'cleanup': True, 'identity': p.identity})
        self.assertEqual(p.held, {})

    def test_focus_loss_releases_owned_key_on_original_display(self):
        p = probe.Probe(2, {'inputs': []})
        original = {'pid': 12, 'display': ':17', 'start_ticks': '123'}
        p.identity = dict(original)
        p.window = '99'
        with patch.object(p, 'focus', side_effect=[None, None, probe.ProbeError('focus lost')]) as focus, patch.object(p, 'key_is_down', return_value=False), patch.object(p, 'x') as x, patch.object(probe, 'process_identity', return_value=original):
            with self.assertRaises(probe.ProbeError):
                p.key('Shift_L')
            p.identity = {'pid': 13, 'display': ':18', 'start_ticks': '456'}
            p.window = '100'
            deadline = p.deadline
            p.release()
            self.assertEqual(focus.call_count, 3)  # Cleanup never checks/changes focus.
            self.assertEqual(x.call_args_list[-1].args, ('keyup', 'Shift_L'))
            self.assertEqual(x.call_args_list[-1].kwargs,
                             {'cleanup': True, 'identity': original})
            self.assertEqual(p.deadline, deadline)
        self.assertEqual(p.held, {})
        self.assertEqual(p.report['own_key_releases'][0]['window'], '99')
        self.assertFalse(p.report['own_key_releases'][0]['window_targeted'])

    def test_reused_original_pid_prevents_cleanup_injection(self):
        p = probe.Probe(2, {'inputs': []})
        original = {'pid': 12, 'display': ':17', 'start_ticks': '123'}
        p.held['x'] = {'identity': original, 'window': '99'}
        with patch.object(probe, 'process_identity', return_value={**original, 'start_ticks': '124'}), patch.object(p, 'x') as x:
            p.release()
            x.assert_not_called()
        self.assertEqual(p.report['unreleased_own_keys'], ['x'])

    def test_cleanup_uses_reserved_total_bound_without_extending(self):
        with patch.object(probe.time, 'monotonic', return_value=100):
            p = probe.Probe(30, {'inputs': []})
        self.assertEqual(p.deadline, 130)
        self.assertEqual(p.action_deadline, 128)
        result = probe.subprocess.CompletedProcess([], 0, stdout='', stderr='')
        original = {'pid': 12, 'display': ':17', 'start_ticks': '123'}
        with patch.object(probe.time, 'monotonic', return_value=129), patch.object(probe.subprocess, 'run', return_value=result) as run:
            with self.assertRaises(probe.ProbeError):
                p.command(['xdotool', 'keydown', 'x'])
            p.command(['xdotool', 'keyup', 'x'], cleanup=True, identity=original)
            self.assertEqual(run.call_args.kwargs['timeout'], 1)
            self.assertEqual(run.call_args.kwargs['env']['DISPLAY'], ':17')
        with patch.object(probe.time, 'monotonic', return_value=130), patch.object(probe.subprocess, 'run') as run:
            with self.assertRaises(probe.ProbeError):
                p.command(['xdotool', 'keyup', 'x'], cleanup=True, identity=original)
            run.assert_not_called()

    def test_matching_title_cannot_substitute_for_original_pid(self):
        p = probe.Probe(2, {})
        p.identity = {'pid': 12}
        with patch.object(p, 'x', side_effect=['World of Warships', '13']):
            with self.assertRaises(probe.ProbeError):
                p.verify_window('99')

    def test_insufficient_scene_and_real_feature_flow(self):
        import cv2
        import numpy as np
        blank = np.zeros((1600, 2560, 3), np.uint8)
        self.assertFalse(probe.scene_motion(blank, blank)['sufficient'])
        self.assertFalse(probe.scene_motion(blank[:100], blank[:100])['sufficient'])
        scene = blank.copy()
        rng = np.random.default_rng(91)
        for x, y in rng.integers([430, 360], [2100, 1070], size=(150, 2)):
            cv2.rectangle(scene, (int(x), int(y)), (int(x+8), int(y+8)), (255, 255, 255), -1)
        shifted = cv2.warpAffine(scene, np.float32([[1, 0, 12], [0, 1, -6]]), (2560, 1600))
        motion = probe.scene_motion(scene, shifted)
        self.assertTrue(motion['sufficient'])
        self.assertAlmostEqual(motion['median_scene_dx_px'], 12, delta=1)
        self.assertAlmostEqual(motion['median_scene_dy_px'], -6, delta=1)
        self.assertFalse(motion['world_calibration_valid'])
        self.assertFalse(probe.scene_motion(scene, scene)['sufficient'])
    def test_wrong_capture_compositor_refuses_snapshot(self):
        p = probe.Probe(2, {'out': '/unused', 'snapshots': []})
        p.identity = {'pid': 12}
        preflight = {'target_resolved': True, 'game_pids': [12],
                     'node': {'process_id': 77, 'id': 9}}
        with patch.object(p, 'check_identity'), patch.object(p, 'command', return_value=json.dumps(preflight)) as command, patch.object(probe, 'ancestor_pids', return_value={12, 10}):
            with self.assertRaises(probe.ProbeError):
                p.snapshot('before')
            self.assertEqual(command.call_count, 1)



if __name__ == '__main__':
    unittest.main()


class PreflightCacheTests(unittest.TestCase):
    """The capture source cannot change mid-run; re-checking it every frame cost ~1 s."""

    def _probe(self):
        p = probe.Probe(30.0, {'out': '/tmp', 'snapshots': [], 'focus_checks': []})
        p.identity = {'pid': 1, 'display': ':0', 'start_ticks': '1'}
        return p

    def _fake(self, calls):
        def run(argv, cleanup=False, identity=None):
            calls.append(argv)
            return json.dumps({'target_resolved': True, 'game_pids': [1],
                               'node': {'id': 9, 'process_id': 1}})
        return run

    def test_preflight_runs_once_inside_its_ttl(self):
        p = self._probe(); calls = []
        with patch.object(p, 'command', side_effect=self._fake(calls)):
            for _ in range(3):
                p.preflight()
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][-1], 'preflight')

    def test_preflight_refreshes_when_stale_or_forced(self):
        p = self._probe(); calls = []
        with patch.object(p, 'command', side_effect=self._fake(calls)):
            p.preflight()
            p._preflight_at = 0.0                      # pretend the TTL expired
            p.preflight()
            p.preflight(force=True)
        self.assertEqual(len(calls), 3)

    def test_a_bad_preflight_is_never_cached(self):
        p = self._probe(); calls = []
        answers = [json.dumps({'target_resolved': False, 'game_pids': [1], 'node': {}}),
                   json.dumps({'target_resolved': True, 'game_pids': [1],
                               'node': {'id': 9, 'process_id': 1}})]
        with patch.object(p, 'command', side_effect=lambda *a, **k: (calls.append(a), answers.pop(0))[1]):
            with self.assertRaises(probe.ProbeError):
                p.preflight()
            good = p.preflight()
        self.assertEqual(good['node']['id'], 9)
        self.assertEqual(len(calls), 2)


class EiTransportTests(unittest.TestCase):
    """The EI channel is a subprocess contract; a fake binary stands in for it."""

    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)
        self.socket = self.root / 'gamescope-0-ei'
        self.socket.write_text('')
        self.log = self.root / 'argv.txt'
        self.binary = self.fake_binary()

    def fake_binary(self, name='ei_input', exit_code=0, stdout='{"device":true}'):
        path = self.root / name
        path.write_text(
            '#!/bin/sh\n'
            f'printf \'%s\\n\' "$@" > "{self.log}"\n'
            f'printf \'%s\\n\' \'{stdout}\'\n'
            f'exit {exit_code}\n')
        path.chmod(0o755)
        return path

    def probe(self, transport=None):
        p = probe.Probe(30, {'inputs': [], 'focus_checks': []}, relative_transport=transport)
        p.identity = {'pid': 12, 'display': ':17'}
        p.window = '99'
        return p

    def test_default_transport_remains_xtest(self):
        with patch.dict(os.environ, {}, clear=True):
            p = self.probe()
            self.assertEqual(p.relative_transport, 'xtest')
            with patch.object(p, 'focus'), \
                    patch.object(probe, 'pointer_state', return_value={'x': 100, 'y': 200, 'mask': 0}), \
                    patch.object(probe.subprocess, 'run', side_effect=AssertionError('no EI')), \
                    patch.object(p, 'x') as x:
                p.move(40, 0)
            x.assert_called_once_with('mousemove_relative', '--', 40, 0)
        event = p.report['inputs'][0]
        self.assertEqual(event['transport'], 'XTestFakeRelativeMotionEvent')
        self.assertFalse(self.log.exists())

    def test_environment_selects_ei_and_passes_relative_arguments(self):
        with patch.dict(os.environ, {'WW_RELATIVE_TRANSPORT': 'ei',
                                     'WW_EI_SOCKET': str(self.socket)}), \
                patch.object(probe, 'EI_INPUT', self.binary):
            p = self.probe()
            self.assertEqual(p.relative_transport, 'ei')
            parked = {'x': 960, 'y': 540, 'mask': 0}
            with patch.object(p, 'focus'), \
                    patch.object(probe, 'pointer_state', return_value=parked), \
                    patch.object(p, 'x') as x:
                p.move(250, 60)
            x.assert_not_called()
        argv = self.log.read_text().splitlines()
        self.assertEqual(argv[:4], ['--dx', '250', '--dy', '60'])
        self.assertIn(str(self.socket), argv)
        event = p.report['inputs'][0]
        self.assertEqual(event['transport'], 'ei_libei_gamescope')
        self.assertEqual(event['transport_status'], 'command_completed')
        self.assertEqual(event['transport_reply'], {'device': True})

    def test_ei_click_uses_the_same_channel(self):
        with patch.object(probe, 'EI_INPUT', self.binary), \
                patch.object(probe, 'ei_socket', return_value=self.socket):
            p = self.probe('ei')
            parked = {'x': 960, 'y': 540, 'mask': 0}
            with patch.object(p, 'focus'), \
                    patch.object(probe, 'pointer_state', return_value=parked), \
                    patch.object(p, 'x') as x:
                p.move(0, 0, click=True)
            x.assert_not_called()
        argv = self.log.read_text().splitlines()
        self.assertIn('--click', argv)
        event = p.report['inputs'][0]
        self.assertEqual(event['transport'], 'ei_libei_gamescope')
        self.assertTrue(event['click'])

    def test_missing_binary_fails_with_its_path(self):
        missing = self.root / 'absent' / 'ei_input'
        with patch.object(probe, 'EI_INPUT', missing), \
                patch.object(probe, 'ei_socket', return_value=self.socket):
            p = self.probe('ei')
            with patch.object(p, 'focus'), \
                    patch.object(probe, 'pointer_state', return_value={'x': 1, 'y': 2, 'mask': 0}), \
                    patch.object(probe.subprocess, 'run') as run, \
                    patch.object(p, 'x') as x:
                with self.assertRaises(probe.ProbeError) as raised:
                    p.move(40, 0)
            run.assert_not_called()
            x.assert_not_called()
        self.assertIn(str(missing), str(raised.exception))
        event = p.report['inputs'][0]
        self.assertEqual(event['transport_status'], 'attempted')
        self.assertIn(str(missing), event['error'])
        self.assertTrue(p.report['movement_delivery_attempted'])

    def test_missing_socket_fails_before_running_the_binary(self):
        missing = self.root / 'no-such-socket'
        with patch.object(probe, 'EI_INPUT', self.binary), \
                patch.object(probe, 'ei_socket', return_value=missing):
            p = self.probe('ei')
            with patch.object(p, 'focus'), \
                    patch.object(probe, 'pointer_state', return_value={'x': 1, 'y': 2, 'mask': 0}), \
                    patch.object(probe.subprocess, 'run') as run:
                with self.assertRaises(probe.ProbeError) as raised:
                    p.move(40, 0)
            run.assert_not_called()
        self.assertIn(str(missing), str(raised.exception))
        self.assertFalse(self.log.exists())

    def test_nonzero_exit_is_a_probe_error(self):
        failing = self.fake_binary(name='failing_ei', exit_code=7)
        with patch.object(probe, 'EI_INPUT', failing), \
                patch.object(probe, 'ei_socket', return_value=self.socket):
            p = self.probe('ei')
            with patch.object(p, 'focus'), \
                    patch.object(probe, 'pointer_state', return_value={'x': 1, 'y': 2, 'mask': 0}), \
                    patch.object(p, 'x') as x:
                with self.assertRaises(probe.ProbeError) as raised:
                    p.move(40, 0)
            x.assert_not_called()
        self.assertIn('ei_input failed', str(raised.exception))
        self.assertEqual(p.report['inputs'][0]['transport_status'], 'attempted')

    def test_ei_zero_pointer_delta_is_not_reported_as_clipped(self):
        with patch.object(probe, 'EI_INPUT', self.binary), \
                patch.object(probe, 'ei_socket', return_value=self.socket):
            p = self.probe('ei')
            parked = {'x': 2540, 'y': 700, 'mask': 0}
            with patch.object(p, 'focus'), \
                    patch.object(probe, 'pointer_state', return_value=parked):
                p.move(180, 0)
        event = p.report['inputs'][0]
        self.assertNotIn('pointer_clipped', event)
        self.assertIsNone(event['pointer_delta'])
        self.assertEqual(event['pointer_observation']['delta'], [0, 0])
        self.assertEqual(event['pointer_observation']['after'], parked)
        self.assertIn('do not describe this transport', event['interpretation'])

    def test_constructor_argument_overrides_environment(self):
        with patch.dict(os.environ, {'WW_RELATIVE_TRANSPORT': 'uinput'}):
            self.assertEqual(self.probe('ei').relative_transport, 'ei')
            self.assertEqual(self.probe().relative_transport, 'uinput')

    def test_click_requires_ei_transport(self):
        p = self.probe('xtest')
        with patch.object(p, 'focus') as focus:
            with self.assertRaises(probe.ProbeError):
                p.move(0, 0, click=True)
            focus.assert_not_called()
        self.assertEqual(p.report['inputs'], [])

