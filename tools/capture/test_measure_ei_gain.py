"""Offline tests for measure_ei_gain: no capture, no injected input."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure_ei_gain as M


FRAME = np.zeros((1600, 2560, 3), dtype=np.uint8)


def session(**overrides):
    shifts = overrides.pop('shifts', [(48.0, 0.0, 1.0)])
    queue = iter(shifts)
    calls = {'capture': 0, 'run': 0, 'state': 0}

    def capture(_path):
        calls['capture'] += 1
        return FRAME

    def shift(_before, _after):
        return next(queue)

    def run(argv):
        calls['run'] += 1
        calls.setdefault('commands', []).append(argv)

    def state():
        calls['state'] += 1
        return {'identity': {'pid': 7, 'display': ':2'},
                'pointer': {'x': 100, 'y': 200, 'mask': 0}}

    tmp = tempfile.TemporaryDirectory()
    defaults = dict(execute=True, out=Path(tmp.name) / 'report.json',
                    capture=capture, shift=shift, run=run, sleep=lambda _s: None,
                    state_check=state)
    defaults.update(overrides)
    return M.GainSession(**defaults), calls, tmp


class DryRunTests(unittest.TestCase):
    def test_default_session_plans_but_never_touches_input_or_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            def forbidden(*_args, **_kwargs):
                raise AssertionError('dry-run must not capture, run or check')

            out = Path(directory) / 'report.json'
            measure = M.GainSession(out=out, capture=forbidden, run=forbidden,
                                    shift=forbidden, state_check=forbidden,
                                    repeat=60)
            report = measure.run()
            self.assertEqual([], report['commands'])
            self.assertEqual([], measure.commands)
            self.assertIs(report['input_injected'], False)
            self.assertIs(report['execute'], False)
            self.assertEqual('dry_run', report['state'])
            self.assertEqual(60, len(report['plan']))
            self.assertTrue(all('attempt' in row and 'move' in row
                                for row in report['plan']))
            self.assertIs(False, json.loads(out.read_text())['input_injected'])

    def test_cli_defaults_are_the_documented_dry_run(self):
        args = M.build_parser().parse_args([])
        self.assertIs(False, args.execute)
        self.assertEqual(':2', args.display)
        self.assertEqual('ei', args.transport)
        self.assertEqual(400.0, args.units)
        self.assertEqual(60, args.repeat)
        self.assertAlmostEqual(0.35, args.settle)


class TransportCommandTests(unittest.TestCase):
    def test_ei_transport_uses_ei_input_relative_dx(self):
        measure, calls, tmp = session(transport='ei', repeat=1)
        with tmp:
            report = measure.run()
        self.assertEqual(1, len(measure.commands))
        command = measure.commands[0]
        self.assertTrue(command[0].endswith('ei_input'))
        self.assertIn('--dx', command)
        self.assertIn('--dy', command)
        self.assertEqual('400', command[command.index('--dx') + 1])
        self.assertNotIn('--click', command)
        self.assertIs(True, report['input_injected'])
        self.assertTrue(report['released']['released'])

    def test_xtest_transport_uses_xdotool_relative_motion(self):
        measure, calls, tmp = session(transport='xtest', repeat=1)
        with tmp:
            measure.run()
        self.assertEqual(['xdotool', 'mousemove_relative', '--', '400', '0'],
                         measure.commands[0])
        self.assertEqual(1, calls['run'])

    def test_unknown_transport_is_refused(self):
        with self.assertRaises(M.MeasureError):
            M.move_command('uinput', 1)
        with self.assertRaises(M.MeasureError):
            M.GainSession(transport='uinput')


class GainTests(unittest.TestCase):
    def test_pixels_per_unit_for_a_known_shift(self):
        self.assertAlmostEqual(0.12, M.px_per_unit(400, 48), places=12)
        self.assertIsNone(M.px_per_unit(0, 10))

    def test_measurement_reports_px_per_unit(self):
        measure, calls, tmp = session(units=400, repeat=1,
                                      shifts=[(48.0, 0.0, 0.95)])
        with tmp:
            report = measure.run()
        record = report['attempts'][0]
        self.assertEqual([48.0, 0.0], record['shift_px'])
        self.assertAlmostEqual(0.12, record['px_per_unit'], places=4)
        self.assertAlmostEqual(48 / 2560 * 68.2, record['rotation_deg'], places=3)
        self.assertEqual(2, calls['capture'])

    def test_continuous_run_records_first_degraded_attempt(self):
        shifts = [(48.0, 0.0, 1.0)] * 3 + [(0.0, 0.0, 1.0)] * 3
        measure, calls, tmp = session(units=400, repeat=6, shifts=shifts)
        with tmp:
            report = measure.run()
        self.assertEqual(6, report['attempt_count'])
        self.assertEqual(4, report['degraded_from_attempt'])
        self.assertEqual(3, report['healthy_attempts_before_failure'])
        self.assertEqual([False, False, False, True, True, True],
                         [row['degraded'] for row in report['attempts']])
        self.assertEqual(48.0, report['reference_px'])
        # The run keeps going after the first failure, so the limit is located
        # but the remaining attempts still show no recovery.
        self.assertEqual(6, calls['run'])
        self.assertEqual(2400.0, report['cumulative_units'])
        self.assertAlmostEqual(144.0, report['travelled_px'], places=3)

    def test_untracked_flow_counts_as_degraded(self):
        shifts = [(0.0, 0.0, 0.0)] * 2
        measure, calls, tmp = session(units=400, repeat=2, shifts=shifts)
        with tmp:
            report = measure.run()
        self.assertEqual(1, report['degraded_from_attempt'])
        self.assertEqual(['untracked', 'untracked'],
                         [row['state'] for row in report['attempts']])


class SafetyTests(unittest.TestCase):
    def test_execute_refuses_before_injecting_when_state_check_fails(self):
        def broken_state():
            raise M.MeasureError('no live game process')

        measure, calls, tmp = session(state_check=broken_state, repeat=2)
        with tmp:
            report = measure.run()
        self.assertEqual('aborted', report['state'])
        self.assertIn('no live game process', report['aborted_reason'])
        self.assertEqual(0, calls['run'])
        self.assertEqual([], measure.commands)
        self.assertIs(False, report['input_injected'])

    def test_final_pointer_mask_is_reported(self):
        def held_state():
            return {'identity': {'pid': 7}, 'pointer': {'x': 0, 'y': 0, 'mask': 0x100}}

        measure, calls, tmp = session(state_check=held_state, repeat=1)
        with tmp:
            report = measure.run()
        self.assertIs(False, report['released']['released'])
        self.assertEqual('this tool never presses a button or key; the sampled '
                         'button mask proves no press was left behind',
                         report['released']['note'])

    def test_only_relative_motion_commands_are_ever_built(self):
        for transport in ('ei', 'xtest'):
            command = M.move_command(transport, 400)
            text = ' '.join(command)
            self.assertNotIn('click', text)
            self.assertNotIn('key', text)
            self.assertNotIn('button', text)


if __name__ == '__main__':
    unittest.main()
