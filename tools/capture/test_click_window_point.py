"""Pointer math and ownership boundaries; no game process is touched."""
import unittest

import click_window_point as C


class PointerWalkTests(unittest.TestCase):
    def test_content_point_uses_the_measured_window_offset(self):
        self.assertEqual((1282, 118), C.content_point(1282, 72))
        self.assertEqual((0, 46), C.content_point(0, 0))

    def test_points_outside_the_content_area_are_refused(self):
        for point in ((2560, 72), (10, 1440), (float('nan'), 5)):
            with self.subTest(point=point), self.assertRaises(ValueError):
                C.content_point(*point)

    def test_chunks_respect_the_per_axis_bound_and_preserve_direction(self):
        self.assertEqual([[-400, -400], [-400, -23], [-200, 0]],
                         C.chunks(-1000, -423))
        self.assertEqual([], C.chunks(0, 0))
        with self.assertRaises(ValueError):
            C.chunks(1.5, 0)

    def test_reached_tolerates_a_few_pixels_but_not_a_miss(self):
        self.assertTrue(C.reached({'x': 1282, 'y': 122}, (1282, 118)))
        self.assertFalse(C.reached({'x': 1282, 'y': 130}, (1282, 118)))


if __name__ == '__main__':
    unittest.main()
