"""Behavior regressions for the Operations route report."""
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ops_route_report as R


def route(points, step=5.0):
    """[(x, z), ...] -> route records at 5 s spacing."""
    return [dict(t=index * step, x=float(x), z=float(z), yaw=0.0)
            for index, (x, z) in enumerate(points)]


def line(count, dx=10.0, dz=0.0):
    return route([(index * dx, index * dz) for index in range(count)])


class TurnDetection(unittest.TestCase):
    def test_corner_reported_at_the_vertex(self):
        points = [(0, 0), (50, 0), (100, 0), (150, 0), (150, 50), (150, 100), (150, 150)]
        turns = R.key_turns(route(points), tolerance=10.0)
        self.assertEqual(1, len(turns))
        self.assertAlmostEqual(-90.0, turns[0]['deg'], delta=1e-6)
        self.assertAlmostEqual(150.0, turns[0]['x'])
        self.assertAlmostEqual(15.0, turns[0]['t_peak'])

    def test_paddle_noise_is_not_a_turn(self):
        points = [(index * 50.0, (index % 2) * 6.0) for index in range(12)]
        self.assertEqual([], R.key_turns(route(points), tolerance=40.0))

    def test_straight_run_has_no_turns(self):
        self.assertEqual([], R.key_turns(line(20), tolerance=10.0))

    def test_backtracking_through_merge_keeps_one_signed_angle(self):
        points = [(0, 0), (40, 0), (80, 0), (120, 0), (80, 40), (40, 0)]
        turns = R.key_turns(route(points), tolerance=20.0)
        self.assertTrue(turns)
        self.assertTrue(all(math.isfinite(turn['deg']) for turn in turns))


class Geometry(unittest.TestCase):
    def test_simplify_keeps_endpoints_and_corners(self):
        points = [(0, 0), (50, 0), (100, 0), (150, 0), (150, 60), (150, 120)]
        simple = R.simplify(route(points), 10.0)
        self.assertEqual((0.0, 0.0), (simple[0]['x'], simple[0]['z']))
        self.assertEqual((150.0, 120.0), (simple[-1]['x'], simple[-1]['z']))
        self.assertLess(len(simple), len(points))
        self.assertTrue(any(abs(point['x'] - 150.0) < 1e-6 and abs(point['z']) < 1e-6
                            for point in simple))

    def test_hausdorff_is_symmetric_and_matches_known_offset(self):
        first = line(5)
        second = ([dict(p, x=p['x'] + 30.0) for p in first])
        self.assertAlmostEqual(30.0, R.hausdorff(first, second), places=9)
        self.assertAlmostEqual(30.0, R.hausdorff(second, first), places=9)
        self.assertAlmostEqual(0.0, R.hausdorff(first, first), places=9)

    def test_truncate_interpolates_the_cut(self):
        cut = R.truncate(line(6), 7.0)
        self.assertEqual(7.0, cut[-1]['t'])
        self.assertAlmostEqual(14.0, cut[-1]['x'])
        self.assertAlmostEqual(0.0, cut[-1]['z'])
        self.assertEqual(3, len(cut))

    def test_truncate_leaves_short_routes_untouched(self):
        short = line(3)
        self.assertEqual(short, R.truncate(short, 100.0))

    def test_corridor_distance_uses_the_perpendicular(self):
        path = [dict(x=0.0, z=0.0), dict(x=100.0, z=0.0)]
        self.assertAlmostEqual(12.0, R.corridor_distance(path, 40.0, 12.0))
        self.assertAlmostEqual(20.0, R.corridor_distance(path, 120.0, 0.0))

    def test_consensus_median_and_spread(self):
        rows = R.consensus([line(4), [dict(p, x=p['x'] + 20.0) for p in line(4)]])
        self.assertEqual(4, len(rows))
        self.assertAlmostEqual(20.0, rows[1]['x'])
        self.assertAlmostEqual(10.0, rows[1]['sd'])
        self.assertEqual(2, rows[1]['n'])

    def test_consensus_single_route_has_no_sd(self):
        rows = R.consensus([line(3)])
        self.assertTrue(all('sd' not in row for row in rows))


class Arrivals(unittest.TestCase):
    def test_bins_are_half_open(self):
        arrivals = [dict(t=0.0, x=0.0, z=0.0), dict(t=59.9, x=0.0, z=0.0),
                    dict(t=60.0, x=0.0, z=0.0), dict(t=360.0, x=0.0, z=0.0)]
        self.assertEqual({'0-60s': 2, '60-120s': 1, '120-240s': 0, '240-360s': 0,
                          '>360s': 1}, R.arrival_bins(arrivals))

    def test_session_stats_reports_arrival_timing(self):
        stats = R.session_stats(line(10), [dict(t=20.0, x=5.0, z=5.0)])
        self.assertEqual(10, stats['samples'])
        self.assertAlmostEqual(45.0, stats['duration_s'])
        self.assertAlmostEqual(90.0, stats['path_units'])
        self.assertEqual(20.0, stats['arrival_first_s'])
        self.assertEqual(1, stats['arrival_bins']['0-60s'])


class Fixtures(unittest.TestCase):
    def roots(self):
        base = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__('shutil').rmtree(base, ignore_errors=True))
        routes = base / 'routes'
        routes.mkdir()
        for name, points in (('a', [(0, 0), (50, 0), (100, 0)]),
                             ('b', [(0, 0), (50, 5), (100, 0)])):
            (routes / ('%s.json' % name)).write_text(json.dumps(dict(
                session=name, map='m1', route=route(points),
                arrivals=[dict(t=30.0, x=10.0, z=10.0)], aim=[])))
        patterns = base / 'patterns.json'
        patterns.write_text(json.dumps({'m1': {'patterns': [
            {'recordings': ['a', 'b'], 'count': 2}]}}))
        return base, routes, patterns

    def test_package_joins_and_scores(self):
        base, routes, patterns = self.roots()
        report = R.package(R.load_routes(routes), R.load_patterns(patterns))
        self.assertEqual(2, report['totals']['route_files_matched'])
        self.assertEqual(2, report['maps']['m1']['patterns'][0]['with_routes'])
        self.assertEqual(2, len(report['maps']['m1']['patterns'][0]['recordings']))
        self.assertEqual([], report['sessions_without_pattern'])
        self.assertEqual(2, len(report['leave_one_out']))
        self.assertIn('m1/p0/b.json', report['leave_one_out'])
        self.assertIsNotNone(report['maps']['m1']['patterns'][0]['dispersion']['median_sd'])

    def test_unclustered_session_is_reported_not_dropped(self):
        base, routes, patterns = self.roots()
        (routes / 'c.json').write_text(json.dumps(dict(session='c', map='m9',
                                                     route=line(3), arrivals=[])))
        report = R.package(R.load_routes(routes), R.load_patterns(patterns))
        self.assertEqual([dict(session='c', map='m9')], report['unclustered'])
        self.assertEqual(['c'], report['sessions_without_pattern'])

    def test_missing_file_is_reported(self):
        base, routes, patterns = self.roots()
        patterns.write_text(json.dumps({'m1': {'patterns': [
            {'recordings': ['a', 'b', 'ghost'], 'count': 3}]}}))
        report = R.package(R.load_routes(routes), R.load_patterns(patterns))
        self.assertEqual(['ghost'], report['maps']['m1']['missing'])
        self.assertEqual(['ghost'], report['sessions_without_file'])

    def test_pattern_count_mismatch_is_rejected(self):
        base, routes, patterns = self.roots()
        patterns.write_text(json.dumps({'m1': {'patterns': [
            {'recordings': ['a', 'b'], 'count': 1}]}}))
        with self.assertRaises(ValueError):
            R.load_patterns(patterns)

    def test_empty_pattern_list_is_rejected(self):
        base, routes, patterns = self.roots()
        patterns.write_text(json.dumps({'m1': {'patterns': []}}))
        with self.assertRaises(ValueError):
            R.load_patterns(patterns)

    def test_map_disagreement_is_rejected(self):
        base, routes, patterns = self.roots()
        (routes / 'a.json').write_text(json.dumps(dict(session='a', map='m2',
                                                     route=line(3), arrivals=[])))
        with self.assertRaises(ValueError):
            R.package(R.load_routes(routes), R.load_patterns(patterns))

    def test_leave_one_out_is_zero_for_identical_routes(self):
        sessions = {'a': dict(map='m1', route=line(4), arrivals=[], source='a.json'),
                    'b': dict(map='m1', route=line(4), arrivals=[], source='b.json')}
        patterns = {'m1': [dict(recordings=['a', 'b'], count=2)]}
        out = R.leave_one_out(sessions, patterns)
        self.assertEqual(2, len(out))
        for value in out.values():
            self.assertEqual(0.0, value['clock_aligned']['p95'])
            self.assertEqual(1.0, value['within_50_corridor'])

    def test_leave_one_out_sees_a_time_shift_that_the_corridor_does_not(self):
        held = [dict(p, t=p['t'] * 2.0) for p in line(8)]
        sessions = {'a': dict(map='m1', route=line(8), arrivals=[], source='a.json'),
                    'b': dict(map='m1', route=line(8), arrivals=[], source='b.json'),
                    'c': dict(map='m1', route=held, arrivals=[], source='c.json')}
        patterns = {'m1': [dict(recordings=['a', 'b', 'c'], count=3)]}
        out = R.leave_one_out(sessions, patterns)['m1/p0/c.json']
        self.assertGreater(out['clock_aligned']['p50'], 5.0)
        self.assertEqual(0.0, out['corridor']['p50'])


class Signature(unittest.TestCase):
    def test_report_separates_and_reproduces(self):
        import numpy as np
        signatures = {'a': np.array([1.0, 0.0, 0.0]), 'b': np.array([0.98, 0.02, 0.0]),
                      'c': np.array([0.0, 1.0, 0.0]), 'd': np.array([0.0, 0.99, 0.01])}
        patterns = {'m1': [dict(recordings=['a', 'b'], count=2),
                           dict(recordings=['c', 'd'], count=2)]}
        report = R.signature_report(patterns, signatures, 0.30)
        self.assertTrue(report['matches_published'])
        self.assertEqual(2, report['per_map']['m1']['clusters'])
        self.assertLess(report['per_map']['m1']['intra_max'], 0.01)
        self.assertGreater(report['per_map']['m1']['cross_min'], 0.9)

    def test_report_flags_a_grouping_that_does_not_reproduce(self):
        import numpy as np
        signatures = {'a': np.array([1.0, 0.0]), 'b': np.array([0.0, 1.0])}
        patterns = {'m1': [dict(recordings=['a', 'b'], count=2)]}
        self.assertFalse(R.signature_report(patterns, signatures, 0.30)['matches_published'])


class RealCorpus(unittest.TestCase):
    """The joined report over the products that exist today, not a fixture."""

    @classmethod
    def setUpClass(cls):
        cls.report = R.build()

    def test_every_listed_session_has_a_route_file(self):
        self.assertEqual(18, self.report['totals']['sessions_with_files'])
        self.assertEqual(self.report['totals']['listed_sessions'],
                         self.report['totals']['route_files_matched'])
        self.assertEqual([], self.report['sessions_without_file'])
        self.assertEqual([], self.report['maps']['s02_Naval_Defense']['missing'])

    def test_known_map_shape(self):
        self.assertEqual(8, self.report['totals']['maps'])
        self.assertEqual(13, self.report['totals']['patterns'])
        counts = {map_id: len(entry['patterns'])
                  for map_id, entry in self.report['maps'].items()}
        self.assertEqual({'s02_Naval_Defense': 2, 's03_Labyrinth': 1, 's06_Atoll': 1,
                          's07_Advance': 1, 's10_USS_CL': 1, 's12_WW2_OP1': 2,
                          's13_WW2_OP2': 2, 's14_WW2_OP3': 3}, counts)

    def test_leave_one_out_only_covers_patterns_with_two_sessions(self):
        keys = sorted(self.report['leave_one_out'])
        self.assertEqual(8, len(keys))
        self.assertTrue(all(key.split('/')[0] in ('s02_Naval_Defense', 's06_Atoll',
                                                  's12_WW2_OP1') for key in keys))

    def test_rendered_report_has_every_section(self):
        text = R.render(self.report)
        for marker in ('leave-one-out', 'per pattern', 'enemy first sightings',
                       'sessions: 18 files'):
            self.assertIn(marker, text)

    def test_json_report_is_serialisable(self):
        json.dumps(self.report, allow_nan=False)

    def test_cli_json_matches_build(self):
        out = subprocess.run([sys.executable, str(Path(R.__file__)),
                              '--json'], capture_output=True, text=True, check=True).stdout
        self.assertEqual(self.report['totals'], json.loads(out)['totals'])


if __name__ == '__main__':
    unittest.main()
