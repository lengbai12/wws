"""Autopilot-course gate: point conversion, chunking and CLI bounds.

The tactical map takes a left click at the game's own cursor, so the tool must
convert a capture point into window coordinates, split the relative move into
bounded chunks, and refuse points outside the content area.
"""
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import autopilot_course as A


class PointConversionTests(unittest.TestCase):
    def test_capture_point_maps_into_the_window(self):
        # 2560x1600 capture with 80 px bars -> 1920x1080 window
        self.assertEqual(A.window_point(1280, 800, 1920, 1080), (960, 540))
        self.assertEqual(A.window_point(0, 80, 1920, 1080), (0, 0))
        self.assertEqual(A.window_point(2559, 1519, 1920, 1080), (1919, 1079))

    def test_out_of_content_points_are_refused(self):
        for point in ((-1, 800), (2560, 800), (1280, 79), (1280, 1520)):
            with self.assertRaises(ValueError):
                A.window_point(*point, 1920, 1080)

    def test_a_non_sixteen_nine_window_is_refused(self):
        with self.assertRaises(ValueError):
            A.window_point(100, 800, 1920, 1200)


class ChunkingTests(unittest.TestCase):
    def test_a_large_move_is_split_within_the_bound(self):
        chunks = A.chunk_deltas(-783, -44)
        self.assertEqual(chunks, [[-400, -44], [-383, 0]])
        self.assertTrue(all(abs(dx) <= 400 and abs(dy) <= 400 for dx, dy in chunks))
        self.assertEqual([sum(c[i] for c in chunks) for i in (0, 1)], [-783, -44])

    def test_a_small_move_is_one_chunk(self):
        self.assertEqual(A.chunk_deltas(12, -30), [[12, -30]])
        self.assertEqual(A.chunk_deltas(0, 0), [])


class CliTests(unittest.TestCase):
    def test_point_bounds_are_enforced(self):
        for argv in (['prog', '--point', '10', '10', '--out', '/tmp/wws-ap-1'],
                     ['prog', '--point', '3000', '800', '--out', '/tmp/wws-ap-2'],
                     ['prog', '--point', '100', '800', '--seconds', '3', '--out', '/tmp/wws-ap-3']):
            with patch.object(sys, 'argv', argv), self.assertRaises(SystemExit):
                A.main()

    def test_a_valid_dry_run_reports_the_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'plan'
            with patch.object(sys, 'argv', ['prog', '--point', '1280', '800',
                                            '--out', str(out)]):
                self.assertEqual(A.main(), 0)
            self.assertTrue((out / 'plan.json').is_file())


if __name__ == '__main__':
    unittest.main()
