#!/usr/bin/env python3
"""The match runner's ledger: no shot is a hit unless the plate says so."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/capture'))
sys.path.insert(0, str(ROOT / 'tools/replay'))

import match_runner as M  # noqa: E402


class FakeFrame:
    def __init__(self, n):
        self.n = n


class MatchRunnerTests(unittest.TestCase):
    def setUp(self):
        self.original = (M._snapshot, M._hud, M._target_bbox, M.run_shot)
        self.calls = []

    def tearDown(self):
        M._snapshot, M._hud, M._target_bbox, M.run_shot = self.original

    def _args(self, **over):
        class A:
            out = ''
            execute = True
            max_shots = 3
            seconds = 60.0
            servo_seconds = 5.0
            reload_wait_s = 0.0
            flight_margin_s = 0.0
            poll_s = 0.01
        for k, v in over.items():
            setattr(A, k, v)
        return A()

    def _run(self, args):
        out = Path(tempfile.mkdtemp())
        args.out = str(out)
        code = M.main(['--out', str(out), *(['--execute'] if args.execute else []),
                       '--max-shots', str(args.max_shots),
                       '--seconds', str(args.seconds),
                       '--reload-wait-s', str(args.reload_wait_s),
                       '--flight-margin-s', str(args.flight_margin_s),
                       '--poll-s', str(args.poll_s),
                       '--servo-seconds', str(args.servo_seconds)])
        return code, json.loads((out / 'summary.json').read_text()), \
            json.loads((out / 'ledger.json').read_text())

    def test_a_fired_shot_with_a_hp_drop_is_a_hit(self):
        M._snapshot = lambda path: FakeFrame(1)
        M._hud = lambda frame: dict(battle=True, reload=True, flight=0.01)
        M._target_bbox = lambda frame: [10, 20, 120, 17]
        M.run_shot = lambda out, args, **kw: dict(verdict='fired', servo=dict(mode='oneshot',
                                                                      fire_reason='ok', residuals=[[5.0, 2.0]]),
                                            stages=dict(sense=dict(terrain=dict(clear=True)),
                                                        aim=dict(error_px=[500.0, 300.0])))
        import shot_accounting as SA
        original = SA.classify_shot
        SA.classify_shot = lambda before, after, bbox=None: dict(hit=True, damage=1234,
                                                                 hp_before=20000, hp_after=18766)
        try:
            code, summary, ledger = self._run(self._args(max_shots=1))
        finally:
            SA.classify_shot = original
        self.assertEqual(code, 0)
        self.assertEqual(summary['shots'], 1)
        self.assertEqual(summary['hits'], 1)
        self.assertEqual(summary['damage'], 1234)
        self.assertEqual(ledger[0]['hit'], True)

    def test_a_fired_shot_without_a_hp_drop_is_not_a_hit(self):
        M._snapshot = lambda path: FakeFrame(1)
        M._hud = lambda frame: dict(battle=True, reload=True, flight=0.01)
        M._target_bbox = lambda frame: [10, 20, 120, 17]
        M.run_shot = lambda out, args, **kw: dict(verdict='fired', servo=dict(mode='oneshot',
                                                                      fire_reason='ok'),
                                            stages=dict(sense={}, aim={}))
        import shot_accounting as SA
        original = SA.classify_shot
        SA.classify_shot = lambda before, after, bbox=None: dict(hit=False, damage=0,
                                                                 reason='unchanged')
        try:
            code, summary, ledger = self._run(self._args(max_shots=1))
        finally:
            SA.classify_shot = original
        self.assertEqual(summary['hits'], 0)
        self.assertEqual(summary['damage'], 0)
        self.assertIs(ledger[0]['hit'], False)

    def test_battle_end_stops_the_run(self):
        M._snapshot = lambda path: FakeFrame(1)
        M._hud = lambda frame: dict(battle=None, reload=None, flight=None)
        M._target_bbox = lambda frame: None
        M.run_shot = lambda out, args, **kw: {}
        code, summary, ledger = self._run(self._args(max_shots=5))
        self.assertEqual(ledger[0]['event'], 'battle_ended')
        self.assertEqual(summary['shots'], 0)

    def test_a_dry_run_never_shoots(self):
        M._snapshot = lambda path: FakeFrame(1)
        M._hud = lambda frame: dict(battle=True, reload=True, flight=1.0)
        M._target_bbox = lambda frame: [10, 20, 120, 17]
        M.run_shot = lambda out, args, **kw: self.calls.append('shot')
        code, summary, ledger = self._run(self._args(execute=False))
        self.assertEqual(self.calls, [])
        self.assertEqual(ledger[0]['event'], 'dry_run')

    def test_a_reloading_gun_waits_then_times_out(self):
        M._snapshot = lambda path: FakeFrame(1)
        M._hud = lambda frame: dict(battle=True, reload=False, flight=None)
        M._target_bbox = lambda frame: [10, 20, 120, 17]
        M.run_shot = lambda out, args, **kw: self.calls.append('shot')
        code, summary, ledger = self._run(self._args(reload_wait_s=0.03))
        self.assertEqual(self.calls, [])
        self.assertIn(ledger[0]['event'], ('reload_timeout',))


class NoInformationTests(unittest.TestCase):
    """With no target on screen and none on the map the chain must stop, not spin."""

    def setUp(self):
        self.original = (M._snapshot, M._hud, M._target_bbox, M.run_shot)

    def tearDown(self):
        M._snapshot, M._hud, M._target_bbox, M.run_shot = self.original

    def test_it_stops_after_the_bounded_attempts(self):
        import tempfile
        out = Path(tempfile.mkdtemp())
        M._snapshot = lambda path: FakeFrame(1)
        M._hud = lambda frame: dict(battle=True, reload=True, flight=1.0)
        M._target_bbox = lambda frame: None
        calls = []
        M.run_shot = lambda out_dir, args, **kw: (calls.append(1) or
                                            dict(verdict='refused',
                                                 servo=dict(error='no enemy healthbar found'),
                                                 stages=dict(sense=dict(minimap=None))))
        code = M.main(['--out', str(out), '--execute', '--max-shots', '5',
                       '--max-no-contact', '2', '--seconds', '30',
                       '--reload-wait-s', '0', '--flight-margin-s', '0', '--poll-s', '0.01',
                       '--servo-seconds', '5'])
        ledger = json.loads((out / 'ledger.json').read_text())
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 2)
        self.assertEqual(ledger[-1]['kind'], 'stopped')
        self.assertEqual(ledger[-1]['reason'], 'no_information')


class ReloadWaitTests(unittest.TestCase):
    """A gun that is still loading is a wait, not a failed shot."""

    def setUp(self):
        self.original = (M._snapshot, M._hud, M._target_bbox, M.run_shot)

    def tearDown(self):
        M._snapshot, M._hud, M._target_bbox, M.run_shot = self.original

    def test_reload_only_refusals_do_not_consume_the_shot_budget(self):
        import tempfile
        out = Path(tempfile.mkdtemp())
        M._snapshot = lambda path: FakeFrame(1)
        M._hud = lambda frame: dict(battle=True, reload=True, flight=1.0)
        M._target_bbox = lambda frame: (100, 200, 100, 16)
        calls = []
        M.run_shot = lambda out_dir, args, **kw: (calls.append(1) or
                                            dict(verdict='refused',
                                                 servo=dict(error='fire revoked: reload_not_ready',
                                                            mode='oneshot'),
                                                 stages=dict(sense=dict(minimap=dict(nearest=1),
                                                                        terrain=dict(clear=True)))))
        code = M.main(['--out', str(out), '--execute', '--max-shots', '2',
                       '--max-reload-waits', '2', '--seconds', '30',
                       '--reload-wait-s', '0', '--flight-margin-s', '0', '--poll-s', '0.01',
                       '--servo-seconds', '5'])
        ledger = json.loads((out / 'ledger.json').read_text())
        events = [r.get('event') for r in ledger]
        self.assertEqual(code, 0)
        # Waits are not appended (that is the point: they must not eat the
        # shot budget), so the proof is the extra calls plus the marker that
        # the budget ran out.
        self.assertIn('reload_never_confirmed', events)
        self.assertNotIn('no_information', events)
        self.assertGreater(len(calls), 2)


class RefusalClassificationTests(unittest.TestCase):
    """A seen-but-unconfirmed target must not be mistaken for no information."""

    def setUp(self):
        self.original = (M._snapshot, M._hud, M._target_bbox, M.run_shot)

    def tearDown(self):
        M._snapshot, M._hud, M._target_bbox, M.run_shot = self.original

    def test_reticle_refusals_are_refusals_not_starvation(self):
        import tempfile
        out = Path(tempfile.mkdtemp())
        M._snapshot = lambda path: FakeFrame(1)
        M._hud = lambda frame: dict(battle=True, reload=True, flight=1.0)
        M._target_bbox = lambda frame: (100, 200, 100, 16)
        calls = []
        M.run_shot = lambda out_dir, args, **kw: (calls.append(1) or
                                            dict(verdict='refused',
                                                 servo=dict(error='fire revoked: reticle_not_observed'),
                                                 stages=dict(sense=dict(minimap=None, plates=[1]))))
        code = M.main(['--out', str(out), '--execute', '--max-shots', '2',
                       '--max-no-contact', '1', '--seconds', '30',
                       '--reload-wait-s', '0', '--flight-margin-s', '0', '--poll-s', '0.01',
                       '--servo-seconds', '5'])
        ledger = json.loads((out / 'ledger.json').read_text())
        self.assertEqual(code, 0)
        self.assertEqual([r.get('event') for r in ledger], ['refused', 'refused'])
        self.assertEqual(len(calls), 2)
        self.assertFalse(any(r.get('kind') == 'stopped' for r in ledger))


class SmoothnessSourceTests(unittest.TestCase):
    def test_one_shot_errors_feed_the_smoothness_metric(self):
        report = dict(servo=dict(oneshot_errors=[[120.0, 10.0], [40.0, 4.0], [8.0, 1.0]]))
        out = M.smoothness_of(report)
        self.assertNotEqual(out.get('smoothness'), 'unknown')
        self.assertIn(out.get('smoothness'), ('smooth', 'rough', 'oscillating', 'converging'))

    def test_no_series_is_reported_as_unknown(self):
        self.assertEqual(M.smoothness_of(dict(servo={})).get('smoothness'), 'unknown')


class BattleEndTests(unittest.TestCase):
    def setUp(self):
        self.original = (M._snapshot, M._hud, M._target_bbox, M.run_shot)

    def tearDown(self):
        M._snapshot, M._hud, M._target_bbox, M.run_shot = self.original

    def test_leaving_the_battle_stops_the_match(self):
        import tempfile
        out = Path(tempfile.mkdtemp())
        M._snapshot = lambda path: FakeFrame(1)
        M._hud = lambda frame: dict(battle=None, reload=None, flight=None)
        M._target_bbox = lambda frame: None
        calls = []
        M.run_shot = lambda out_dir, args, **kw: calls.append(1)
        code = M.main(['--out', str(out), '--execute', '--max-shots', '5',
                       '--seconds', '30', '--poll-s', '0.01', '--reload-wait-s', '0'])
        ledger = json.loads((out / 'ledger.json').read_text())
        self.assertEqual(code, 0)
        self.assertEqual(calls, [])
        self.assertEqual(ledger[-1]['event'], 'battle_ended')


class TerrainHoldTests(unittest.TestCase):
    def setUp(self):
        self.original = (M._snapshot, M._hud, M._target_bbox, M.run_shot)

    def tearDown(self):
        M._snapshot, M._hud, M._target_bbox, M.run_shot = self.original

    def test_blocked_line_of_fire_is_recorded_not_fired(self):
        import tempfile
        out = Path(tempfile.mkdtemp())
        M._snapshot = lambda path: FakeFrame(1)
        M._hud = lambda frame: dict(battle=True, reload=True, flight=1.0)
        M._target_bbox = lambda frame: (100, 200, 100, 16)
        M.run_shot = lambda out_dir, args, **kw: dict(
            verdict='refused',
            servo=dict(error='no enemy healthbar found in 3 scan steps'),
            stages=dict(sense=dict(minimap=dict(nearest=1),
                                   plates=[],
                                   terrain=dict(clear=False, land_pixels=42))))
        code = M.main(['--out', str(out), '--execute', '--max-shots', '1',
                       '--max-no-contact', '5', '--seconds', '30',
                       '--reload-wait-s', '0', '--poll-s', '0.01', '--servo-seconds', '5'])
        ledger = json.loads((out / 'ledger.json').read_text())
        terrain_rows = [r for r in ledger if r.get('event') == 'target_hidden_by_terrain']
        self.assertEqual(code, 0)
        self.assertTrue(terrain_rows, ledger)
        self.assertEqual(terrain_rows[0]['terrain']['land_pixels'], 42)
        self.assertFalse(any(r.get('verdict') == 'fired' for r in ledger))


class SmoothnessSampleCountTests(unittest.TestCase):
    def test_single_measurement_reports_its_sample_count(self):
        out = M.smoothness_of(dict(servo=dict(oneshot_errors=[[0.5, -0.8]])))
        self.assertEqual(out['samples'], 1)
        self.assertNotEqual(out['smoothness'], 'smooth')

    def test_empty_series_says_zero(self):
        out = M.smoothness_of(dict(servo={}))
        self.assertEqual(out['samples'], 0)
        self.assertEqual(out['reason'], 'no_error_series')


class SeedGainTests(unittest.TestCase):
    """The measured gain must carry into the next shot, and outliers must not."""

    def test_measured_gain_is_carried(self):
        report = dict(steps=[dict(gain_effective=[-1.42, 1.31])])
        self.assertEqual(M.next_seed_gain(report, None), [-1.42, 1.31])

    def test_absurd_gain_is_ignored_but_the_other_axis_survives(self):
        report = dict(steps=[dict(gain_effective=[-18.0, 1.2])])
        self.assertEqual(M.next_seed_gain(report, [-1.0, 1.0]), [-1.0, 1.2])

    def test_no_measurement_keeps_the_previous_seed(self):
        self.assertEqual(M.next_seed_gain(dict(steps=[dict()]), [-1.5, -1.4]), [-1.5, -1.4])
        self.assertIsNone(M.next_seed_gain({}, None))

    def test_the_used_gain_is_the_fallback(self):
        report = dict(steps=[dict(gain_used=[-1.6, 1.5])])
        self.assertEqual(M.next_seed_gain(report, None), [-1.6, 1.5])


class InProcessReportShapeTests(unittest.TestCase):
    def test_smoothness_reads_a_flat_servo_report(self):
        report = dict(steps=[dict(error_px=[120.0, 9.0]), dict(error_px=[30.0, 2.0])])
        out = M.smoothness_of(report)
        self.assertEqual(out['samples'], 2)


class ServoArgvRoundTripTests(unittest.TestCase):
    """Every runner flag the in-process path reads must really exist.

    Measured 2026-10-23: the first live in-process run died with
    ``AttributeError: fire_tolerance`` because the path had no test at all. The
    check is a round trip: the argv must parse under the controller's own parser.
    """

    def _parse(self, extra=()):
        return M.build_parser().parse_args(['--out', '/tmp/roundtrip'] + list(extra))

    def test_target_in_view_argv_parses_in_the_controller(self):
        import aim_servo as A
        args = self._parse(['--execute'])
        argv = M.servo_argv(Path('/tmp/roundtrip'), args, True, [-1.5, -1.4])
        parsed = A.build_parser().parse_args(argv)
        self.assertTrue(parsed.oneshot)
        self.assertTrue(parsed.fire)
        self.assertEqual(parsed.seed_gain, [-1.5, -1.4])
        self.assertEqual(parsed.aim_budget_s, 10.0)
        self.assertLessEqual(parsed.aim_budget_s, 10.0)

    def test_acquisition_argv_parses_in_the_controller(self):
        import aim_servo as A
        args = self._parse(['--execute', '--scan-steps', '9'])
        argv = M.servo_argv(Path('/tmp/roundtrip'), args, False, None)
        parsed = A.build_parser().parse_args(argv)
        self.assertTrue(parsed.scan)
        self.assertEqual(parsed.scan_steps, 9)
        self.assertEqual(parsed.aim_budget_s, 12.0)

    def test_no_execute_means_no_fire_flag(self):
        args = self._parse([])
        self.assertNotIn('--fire', M.servo_argv(Path('/tmp/roundtrip'), args, True))


class PartialSeedTests(unittest.TestCase):
    def test_a_half_measured_gain_is_dropped_not_guessed(self):
        args = M.build_parser().parse_args(['--out', '/tmp/x', '--execute'])
        argv = M.servo_argv(Path('/tmp/x'), args, True, [None, 1.3])
        self.assertNotIn('--seed-gain', argv)

    def test_a_full_gain_is_passed(self):
        args = M.build_parser().parse_args(['--out', '/tmp/x', '--execute'])
        argv = M.servo_argv(Path('/tmp/x'), args, True, [-1.5, 1.3])
        self.assertIn('--seed-gain', argv)
        self.assertEqual(argv[argv.index('--seed-gain') + 1:][:2], ['-1.500', '1.300'])


class FlatReportVerdictTests(unittest.TestCase):
    def test_a_flat_servo_report_counts_as_an_attempt(self):
        import tempfile
        out = Path(tempfile.mkdtemp())
        original = (M._snapshot, M._hud, M._target_bbox, M.run_shot)
        try:
            M._snapshot = lambda path: FakeFrame(1)
            M._hud = lambda frame: dict(battle=True, reload=True, flight=1.0)
            M._target_bbox = lambda frame: (100, 200, 100, 16)
            M.run_shot = lambda out_dir, args, **kw: dict(fired=False, error='fire revoked: reload_not_ready',
                                                    steps=[], mode='oneshot_open_loop')
            M.main(['--out', str(out), '--execute', '--max-shots', '1',
                    '--max-reload-waits', '0', '--seconds', '10',
                    '--reload-wait-s', '0', '--poll-s', '0.01', '--servo-seconds', '5'])
        finally:
            M._snapshot, M._hud, M._target_bbox, M.run_shot = original
        summary = json.loads((out / 'summary.json').read_text())
        self.assertEqual(summary['attempts'], 1)
        self.assertEqual(summary['shots'], 0)
        row = json.loads((out / 'ledger.json').read_text())[0]
        self.assertEqual(row['verdict'], 'refused')


class FiredIsNotRefusedTests(unittest.TestCase):
    """A flat report says `fired`, not `verdict`: it must still read as a shot."""

    def test_a_fired_flat_report_has_no_refusal_event(self):
        import tempfile
        out = Path(tempfile.mkdtemp())
        original = (M._snapshot, M._hud, M._target_bbox, M.run_shot)
        try:
            M._snapshot = lambda path: FakeFrame(1)
            M._hud = lambda frame: dict(battle=True, reload=True, flight=0.0)
            M._target_bbox = lambda frame: (100, 200, 100, 16)
            M.run_shot = lambda out_dir, args, **kw: dict(fired=True, steps=[], mode='oneshot')
            M.main(['--out', str(out), '--execute', '--max-shots', '1',
                    '--seconds', '10', '--reload-wait-s', '0', '--poll-s', '0.01',
                    '--servo-seconds', '5'])
        finally:
            M._snapshot, M._hud, M._target_bbox, M.run_shot = original
        row = json.loads((out / 'ledger.json').read_text())[0]
        self.assertEqual(row['verdict'], 'fired')
        self.assertIsNone(row.get('event'))


class NoStrayKeysTests(unittest.TestCase):
    """The controller must not press M: in this game M opens the tactical map."""

    def test_the_servo_argv_has_no_recentre_key(self):
        args = M.build_parser().parse_args(['--out', '/tmp/x', '--execute'])
        argv = M.servo_argv(Path('/tmp/x'), args, True, None)
        self.assertNotIn('--recentre-key', argv)
        self.assertNotIn('m', argv)

    def test_lead_is_opt_in(self):
        plain = M.build_parser().parse_args(['--out', '/tmp/x', '--execute'])
        asked = M.build_parser().parse_args(['--out', '/tmp/x', '--execute', '--lead'])
        self.assertFalse(plain.lead)
        self.assertNotIn('--lead', M.servo_argv(Path('/tmp/x'), plain, True))
        self.assertIn('--lead', M.servo_argv(Path('/tmp/x'), asked, True))

    def test_steering_flags_are_gone(self):
        # The shot chain never moves the ship, so the movement flags must not
        # exist at all: accepting them would invite a silent autopilot press.
        for flag in ('--steer', '--no-steer', '--engage-max-km', '--bearing',
                     '--ensure-minimap-max'):
            with self.assertRaises(SystemExit, msg=flag):
                with open('/dev/null', 'w') as quiet:
                    import contextlib
                    with contextlib.redirect_stderr(quiet):
                        M.build_parser().parse_args(['--out', '/tmp/x', flag])


if __name__ == '__main__':
    unittest.main()
