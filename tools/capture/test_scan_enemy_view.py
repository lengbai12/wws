from contextlib import ExitStack
import json
from pathlib import Path
import signal
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import scan_enemy_view as S


IDENTITY = dict(pid=123, display=':test', start_ticks='test')


class ScanBehavior(unittest.TestCase):
    def exercise(self, scenes, **overrides):
        clock = {'ns': 2_000_000_000}
        current = {'scene': {}, 'index': 0}
        handlers = {}
        events = []
        case = self

        class Capture:
            identity = IDENTITY
            window = 'game'

            def __init__(self):
                events.append(('capture',))

            def check(self):
                pass

            def acquire(self, **kwargs):
                if current['index'] >= len(scenes):
                    raise S.ProbeError('fixture frame supply exhausted')
                scene = scenes[current['index']]
                current.update(scene=scene, index=current['index']+1)
                clock['ns'] += int(scene.get('advance', .1)*1e9)
                if scene.get('signal'):
                    handlers[signal.SIGTERM](signal.SIGTERM, None)
                image = np.zeros((1600, 2560, 3), np.uint8)
                image[0, 0, 0] = scene.get('pixel', current['index'])
                return image, dict(source=scene.get('source', 'pinned'),
                    identity=IDENTITY, window='game',
                    captured_monotonic_ns=clock['ns']-int(scene.get('age', .01)*1e9),
                    raw_pixel_sha256=str(scene.get('pixel', current['index'])),
                    capture_age_semantics=S.require_fresh.__globals__['CAPTURE_SEMANTICS'])

            def save(self, path, pixels, meta):
                case.assertIn(('release',), events)
                case.assertTrue((path.parent/'report.json').is_file())
                events.append(('save', path.name))

            def close(self):
                events.append(('close',))

        class Reader:
            def detect(self, frame):
                return dict(battle=dict(active=current['scene'].get('battle', True)))

        def move(probe, dx, dy):
            events.append(('move', dx, dy))
            probe.report['inputs'].append(dict(relative=[dx, dy],
                pointer_delta=current['scene'].get('delivered', [dx, dy])))

        def release(probe):
            events.append(('release',))

        def coarse(args, *, acquisition):
            events.append(('coarse', acquisition['capture']['raw_pixel_sha256']))
            case.assertIn(('release',), events)
            return current['scene'].get('coarse_exit', 0)

        def install(sig, handler):
            previous = handlers.get(sig)
            handlers[sig] = handler
            return previous

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            args = SimpleNamespace(out=Path(directory)/'scan', execute=True, inspect=False,
                training_room=True, seconds=120., steps=4096, transport='xtest', max_range_km=15.,
                yaw_deg_per_unit=1., pitch_deg_per_unit=1., initial_pitch_deg=0.,
                angle_source='explicit fixture mapping', angle_identity=IDENTITY)
            vars(args).update(overrides)
            for target, name, value in (
                (S, 'LockedCapture', Capture), (S, 'FireHudReader', Reader),
                (S, 'view_state', lambda *a: current['scene'].get('view', 'wide')),
                (S, 'detect_nameplates', lambda *a, **k: current['scene'].get('candidates', [])),
                (S.locked_aim, 'run', coarse),
                (S.AcquisitionProbe, 'move', move), (S.AcquisitionProbe, 'release', release),
                (S.time, 'monotonic_ns', lambda: clock['ns']),
                (S.time, 'monotonic', lambda: clock['ns']/1e9),
                (S.time, 'sleep', lambda seconds: clock.update(ns=clock['ns']+int(seconds*1e9))),
                (S.signal, 'signal', install), (S.signal, 'setitimer', lambda *a: None)):
                stack.enter_context(patch.object(target, name, value))
            code = S.run(args)
            report = json.loads((args.out/'report.json').read_text())
        return code, report, events

    def test_actual_range_above_four_allows_handoff(self):
        candidate = dict(nameplate_bbox=[100, 200, 80, 10])
        for maximum in (4.1, 8.0, 15.0):
            with self.subTest(maximum=maximum):
                code, report, events = self.exercise([dict(candidates=[candidate])], max_range_km=maximum)
                self.assertEqual(code, 0)
                self.assertEqual(report['status'], 'coarse_aim_completed')

    def test_missing_nonfinite_or_at_most_four_range_refuses_before_capture(self):
        for maximum in (None, float('nan'), float('inf'), 4.0, 3.9):
            with self.subTest(maximum=maximum):
                code, report, events = self.exercise([], max_range_km=maximum)
                self.assertEqual(code, 1)
                self.assertIn('maximum range above 4', report['reason'])
                self.assertEqual(events, [])

    def test_watcher_requires_actual_range_above_four(self):
        script = Path(S.__file__).with_name('aim_when_battle.sh').read_text()
        guard = script.split('if [[ $ONCE -ne 1 ]]; then', 1)[1].split('check_once() {', 1)[0]
        guard = 'if [[ $ONCE -ne 1 ]]; then' + guard
        guard = guard.replace('.venv/bin/python', shlex.quote(sys.executable))
        for maximum, expected in (('', 2), ('nan', 2), ('inf', 2), ('3.9', 2),
                                  ('4', 2), ('4.1', 0), ('8', 0), ('15', 0)):
            with self.subTest(maximum=maximum):
                result = subprocess.run(['bash', '-c', guard], capture_output=True, text=True,
                                        env={'ONCE': '0', 'TRAINING_ROOM': '1', 'MAX_RANGE_KM': maximum})
                self.assertEqual(expected, result.returncode, result.stderr)

    def test_healthbar_stops_and_hands_off_exact_capture(self):
        candidate = dict(nameplate_bbox=[100, 200, 80, 10], confidence=.9)
        code, report, events = self.exercise([{}, dict(candidates=[candidate])])
        self.assertEqual(code, 0)
        self.assertEqual(report['status'], 'coarse_aim_completed')
        self.assertEqual(len([e for e in events if e[0] == 'move']), 1)
        self.assertEqual(report['handoff']['candidates'], [candidate])
        self.assertEqual(report['handoff']['capture'], report['frames'][-1]['capture'])
        self.assertFalse(report['coverage_verified'])
        self.assertFalse(report['fired'])
        self.assertEqual([e for e in events if e[0] == 'coarse'], [('coarse', '2')])

    def test_enemy_stops_both_axes_at_any_route_position_even_if_aim_refuses(self):
        # First frame, during pitch change, during yaw sweep, and next row.
        for found_at in (0, 2, 5, 8):
            for coarse_exit in (0, 1):
                with self.subTest(found_at=found_at, coarse_exit=coarse_exit):
                    candidate = dict(nameplate_bbox=[100, 200, 80, 10])
                    scenes = [{} for _ in range(found_at)] + [dict(
                        candidates=[candidate], coarse_exit=coarse_exit)] + [{}]*30
                    code, report, events = self.exercise(scenes)
                    moves = [e for e in events if e[0] == 'move']
                    self.assertEqual(len(moves), found_at)
                    handoff = next(i for i,e in enumerate(events) if e[0] == 'coarse')
                    self.assertFalse([e for e in events[handoff:] if e[0] == 'move'])
                    self.assertEqual(len(report['frames']), found_at+1)
                    self.assertEqual(report['frames'][-1]['action'], 'stop_all_scan_for_coarse_aim')
                    self.assertEqual(code, coarse_exit)

    def test_scope_unknown_or_inactive_never_moves(self):
        lookalike = dict(nameplate_bbox=[100, 200, 80, 10], confidence=.9)
        for scene in (dict(view='scope', candidates=[lookalike]),
                      dict(view='unknown', candidates=[lookalike]),
                      dict(battle=False, candidates=[lookalike])):
            with self.subTest(scene=scene):
                code, report, events = self.exercise([scene])
                self.assertEqual(code, 1)
                self.assertFalse(any(e[0] == 'move' for e in events))
                self.assertEqual(report['frames'][0]['candidates'], [])

    def test_missing_mapping_refuses_before_capture_or_input(self):
        code, report, events = self.exercise([], yaw_deg_per_unit=None,
            pitch_deg_per_unit=None, initial_pitch_deg=None,
            angle_source=None, angle_identity=None)
        self.assertEqual(code, 1)
        self.assertEqual(events, [])
        self.assertIn('angle mapping', report['reason'])

    def test_mapping_identity_mismatch_never_moves(self):
        code, report, events = self.exercise([], angle_identity=dict(pid=456))
        self.assertEqual(code, 1)
        self.assertIn('identity', report['reason'])
        self.assertFalse(any(e[0] == 'move' for e in events))

    def test_unchanged_pixels_wait_without_stacked_input_then_resume(self):
        code, report, events = self.exercise([
            dict(pixel=1), dict(pixel=1), dict(pixel=1), dict(pixel=2),
            dict(pixel=3, candidates=[dict(nameplate_bbox=[1, 2, 3, 4])])])
        self.assertEqual(code, 0)
        self.assertEqual(len([e for e in events if e[0] == 'move']), 2)
        self.assertEqual(sum(r.get('action') == 'wait_for_changed_pixels' for r in report['frames']), 2)

    def test_unchanged_pixels_timeout_refuses(self):
        code, report, events = self.exercise([
            dict(pixel=1), dict(pixel=1, advance=1), dict(pixel=1, advance=1.1)])
        self.assertEqual(code, 1)
        self.assertIn('unchanged', report['reason'])
        self.assertEqual(len([e for e in events if e[0] == 'move']), 1)

    def test_clipped_pointer_refuses_without_recenter(self):
        code, report, events = self.exercise([dict(delivered=[0, -2])])
        self.assertEqual(code, 1)
        self.assertIn('clipping', report['reason'])
        self.assertEqual([e for e in events if e[0] == 'move'], [('move', 0, -3)])

    def test_three_rows_command_complete_serpentine(self):
        code, report, events = self.exercise([{} for _ in range(30)])
        self.assertEqual(code, 0)
        self.assertEqual(report['status'], 'route_commanded')
        angles = report['commanded_angles']
        self.assertEqual([sum(a['yaw_delta_deg'] for a in angles if a['row'] == row)
                          for row in range(3)], [360., -360., 360.])
        self.assertEqual([a['pitch_deg'] for a in angles if a['relative'][0]],
                         [-10.]*3+[0.]*3+[5.]*3)
        for angle in angles:
            self.assertLessEqual(abs(angle['relative'][0]), 150)
            self.assertLessEqual(abs(angle['relative'][1]), 3)
            self.assertTrue(-10 <= angle['pitch_deg'] <= 5)
        self.assertFalse(report['coverage_verified'])
        self.assertLessEqual(len([e for e in events if e[0] == 'save']), 2)

    def test_signed_gains_and_domain_endpoint_start(self):
        code, report, _ = self.exercise([{} for _ in range(30)],
            yaw_deg_per_unit=-1., pitch_deg_per_unit=-1., initial_pitch_deg=5.)
        self.assertEqual(code, 0)
        self.assertEqual(report['commanded_angles'][-1]['yaw_deg'], 360.)
        self.assertEqual(report['commanded_angles'][-1]['pitch_deg'], 5.)

    def test_signal_releases_before_reporting_and_saving(self):
        code, report, events = self.exercise([{}, dict(signal=True)])
        self.assertEqual(code, 1)
        self.assertIn('signal', report['reason'])
        self.assertIn(('release',), events)
        self.assertIn(('close',), events)
        self.assertLess(events.index(('release',)), events.index(('save', 'first.png')))

    def test_stale_frame_refused_before_input(self):
        code, report, events = self.exercise([dict(age=.8)])
        self.assertEqual(code, 1)
        self.assertIn('expired', report['reason'])
        self.assertFalse(any(e[0] == 'move' for e in events))

    def test_read_only_inspection(self):
        code, report, events = self.exercise([{}], execute=False, inspect=True,
            yaw_deg_per_unit=None, pitch_deg_per_unit=None, initial_pitch_deg=None,
            angle_source=None, angle_identity=None)
        self.assertEqual(code, 0)
        self.assertEqual(report['status'], 'inspected')
        self.assertFalse(any(e[0] == 'move' for e in events))

    def test_dry_run_has_route_without_capture_or_mapping(self):
        code, report, events = self.exercise([], execute=False, yaw_deg_per_unit=None,
            pitch_deg_per_unit=None, initial_pitch_deg=None, angle_source=None, angle_identity=None)
        self.assertEqual(code, 0)
        self.assertEqual(report['angle_route'], S.ANGLE_ROUTE)
        self.assertEqual(events, [])

    def test_invalid_bounds_and_mapping_rejected_before_capture(self):
        changes = [dict(steps=0), dict(steps=4097), dict(seconds=121),
                   dict(seconds=float('nan')), dict(yaw_deg_per_unit=0),
                   dict(pitch_deg_per_unit=float('inf')), dict(initial_pitch_deg=-11),
                   dict(initial_pitch_deg=6), dict(angle_source=' '),
                   dict(angle_identity=[]), dict(yaw_deg_per_unit=7),
                   dict(training_room=False)]
        for change in changes:
            with self.subTest(change=change):
                code, report, events = self.exercise([], **change)
                self.assertEqual(code, 1)
                self.assertEqual(events, [])

    def test_step_budget_bounds_inputs(self):
        code, report, events = self.exercise([{}, {}], steps=1)
        self.assertEqual(code, 1)
        self.assertIn('step budget', report['reason'])
        self.assertEqual(len([e for e in events if e[0] == 'move']), 1)


if __name__ == '__main__':
    unittest.main()
