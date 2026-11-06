import json
from pathlib import Path
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import resolution_map as rm


class MappingTests(unittest.TestCase):
    def test_reference_window_is_identity(self):
        self.assertEqual(rm.capture_to_window((1280, 800), (2560, 1600), (2560, 1440)),
                         (1280, 720))
        self.assertEqual(rm.window_to_capture((1280, 720), (2560, 1600), (2560, 1440)),
                         (1280, 800))

    def test_measured_1924_by_1058_window_is_per_axis(self):
        # Measured live: the client renders 1924x1058 inside a 1920x1080 screen.
        # A single aspect-ratio factor would be wrong by 2.3 % in y.
        sx, sy = rm.scale((2560, 1600), (1924, 1058))
        self.assertAlmostEqual(sx, 1924/2560, places=6)
        self.assertAlmostEqual(sy, 1058/1440, places=6)
        self.assertNotAlmostEqual(sx, sy, places=3)
        self.assertEqual(rm.capture_to_window((0, 80), (2560, 1600), (1924, 1058)), (0, 0))
        # Last real pixel, not the exclusive right edge.
        self.assertEqual(rm.capture_to_window((2559, 1519), (2560, 1600), (1924, 1058)),
                         (1923, 1057))

    def test_1920_by_1080_window_round_trips(self):
        for point in ((0, 80), (640, 440), (2559, 1519)):
            window = rm.capture_to_window(point, (2560, 1600), (1920, 1080))
            back = rm.window_to_capture(window, (2560, 1600), (1920, 1080))
            self.assertLessEqual(abs(back[0]-point[0]), 2)
            self.assertLessEqual(abs(back[1]-point[1]), 2)

    def test_frame_without_bars_is_treated_as_content(self):
        self.assertEqual(rm.content_from_frame((1920, 1080)), (1920, 1080, 0, 0))
        self.assertEqual(rm.capture_to_window((960, 540), (1920, 1080), (1920, 1080)),
                         (960, 540))

    def test_points_outside_content_or_window_are_rejected(self):
        with self.assertRaises(rm.MappingError):
            rm.capture_to_window((100, 40), (2560, 1600), (1920, 1080))   # inside the top bar
        with self.assertRaises(rm.MappingError):
            rm.capture_to_window((2600, 800), (2560, 1600), (1920, 1080))  # right of the frame
        with self.assertRaises(rm.MappingError):
            rm.window_to_capture((2000, 500), (2560, 1600), (1920, 1080))

    def test_resize_rect_scales_both_edges(self):
        rect = rm.resize_rect([1836, 796, 720, 723], (2560, 1600), (1924, 1058))
        self.assertEqual(rect, [1380, 526, 541, 531])

    def test_detect_reports_the_scale_it_will_use(self):
        out = rm.detect(None, (1924, 1058), (2560, 1600))
        self.assertEqual(out['window'], [1924, 1058])
        self.assertEqual(out['content'], [2560, 1440])
        self.assertFalse(out['uniform'])
        self.assertEqual(rm.detect(None, (2560, 1440), (2560, 1600))['uniform'], True)

    def test_cli_maps_a_point_and_a_rect_from_geometry_override(self):
        out = subprocess.run([sys.executable, str(Path(rm.__file__)),
                              '--geometry', '1924x1058', '--point', '1280', '800',
                              '--rect', '1836', '796', '720', '723'],
                             capture_output=True, text=True, timeout=30)
        self.assertEqual(out.returncode, 0, out.stderr)
        data = json.loads(out.stdout)
        self.assertEqual(data['window_point'], [962, 529])
        self.assertEqual(data['window_rect'], [1380, 526, 541, 531])


if __name__ == '__main__':
    unittest.main()
