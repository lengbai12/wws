"""Bearing and range from minimap pixels, with the north-up convention pinned."""
import math
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import minimap_contacts as C


class BearingTests(unittest.TestCase):
    def test_cardinal_directions(self):
        own = (100, 100)
        self.assertAlmostEqual(C.bearing_deg(own, (100, 40)), 0.0)      # north = -v
        self.assertAlmostEqual(C.bearing_deg(own, (160, 100)), 90.0)    # east = +u
        self.assertAlmostEqual(C.bearing_deg(own, (100, 160)), 180.0)   # south = +v
        self.assertAlmostEqual(C.bearing_deg(own, (40, 100)), 270.0)

    def test_relative_bearing_is_signed(self):
        self.assertAlmostEqual(C.relative_bearing_deg(0, 30), 30.0)
        self.assertAlmostEqual(C.relative_bearing_deg(350, 10), 20.0)
        self.assertAlmostEqual(C.relative_bearing_deg(0, 300), -60.0)
        self.assertIsNone(C.relative_bearing_deg(None, 300))

    def test_range_uses_the_map_scale(self):
        self.assertAlmostEqual(C.range_m((0, 0), (3, 4), 10.0), 50.0)
        with self.assertRaises(ValueError):
            C.range_m((0, 0), (1, 1), 0)

    def test_lead_is_distance_over_speed(self):
        self.assertAlmostEqual(C.lead_seconds(8000.0, 800.0), 10.0)
        self.assertAlmostEqual(C.lead_seconds(8000.0, 800.0, latency_s=.4), 10.4)
        with self.assertRaises(ValueError):
            C.lead_seconds(8000.0, 0)
        with self.assertRaises(ValueError):
            C.lead_seconds(-1.0, 800.0)

    def test_the_nearest_contact_wins(self):
        contacts = [dict(pixel=[100, 60]), dict(pixel=[100, 160]), dict(pixel=[100, None])]
        best = C.nearest_contact((100, 100), contacts, 8.0)
        self.assertEqual(best['pixel'], [100, 60])
        self.assertAlmostEqual(best['range_m'], 320.0)
        self.assertAlmostEqual(best['bearing_deg'], 0.0)


if __name__ == '__main__':
    unittest.main()
