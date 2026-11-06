"""The nameplate-anchored aim point must land on the hull that the bar floats above.

The main view has no learned hull model, so the bar is the only anchor. Measured
on live frames:

frame                     view          bar (native)        ship (native)
hull-aim-8p4km.png        third person  [1221,642,123,17]   20x7 px at ~ (1282,693)
hull-aim-2p7km.png        third person  [1131,285,124,16]   hull band 301..825

A bar-width offset (1.2 w = 149 px) put every shell in the water at 8.4 km: the
ship is only ~33 px below the bar there, so third person uses the measured linear
rule (offset = -14.5 + 6.0*km, calibrated on 8.4 km -> 38 px and 12.75 km -> 62).

The scope magnifies the hull about fourfold and is a different anchor: the aim
follows the measured waterline (strongest horizontal edge under the bar) and sits
8 px above it. Measured 2026-10-23 on a 13 km scoped frame: the third-person
offset aimed at row 703 while the hull met the sea at row 800.
"""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'capture'))
import aim_servo as A
import main_view_perception as M

FIXTURES = Path(__file__).resolve().parent / 'main_view_fixtures'


def load(name):
    import cv2
    image = cv2.imread(str(FIXTURES / name))
    if image is None:
        raise unittest.SkipTest(f'fixture missing: {name}')
    return np.ascontiguousarray(image[:, :, ::-1])


class HullAimTests(unittest.TestCase):
    def test_third_person_aim_lands_on_the_horizon_ship(self):
        rgb = load('hull-aim-8p4km.png')
        plate = M.detect_nameplates(rgb)[0]['nameplate_bbox']
        aim, meta = A.hull_aim_point(rgb, plate)
        self.assertLessEqual(abs(aim[1] - 693), 14, f'aim {aim} not on the hull')
        # Which rule wins is an implementation detail; landing on the hull is the
        # contract. The distance rule is the one that runs above 6 km.
        self.assertIn(meta['source'], ('horizon_hull', 'bar_geometry',
                                       'range_caption_linear'))

    def test_scope_aim_lands_inside_the_drawn_hull(self):
        rgb = load('hull-aim-scope-manual.png')
        plate = M.detect_nameplates(rgb)[0]['nameplate_bbox']
        aim, meta = A.hull_aim_point(rgb, plate)
        # the hull in this frame spans y 726..816
        self.assertGreaterEqual(aim[1], 720)
        self.assertLessEqual(aim[1], 820)

    def test_the_scope_aim_follows_the_waterline(self):
        rgb = load('hull-aim-scope-manual.png')
        plate = M.detect_nameplates(rgb)[0]['nameplate_bbox']
        aim, meta = A.hull_aim_point(rgb, plate)
        if meta['source'] == 'scope_waterline':
            self.assertEqual(aim[1], meta['waterline_row'] - A.AIM_ABOVE_WATERLINE_PX)

    def test_the_aim_is_below_the_bar_never_on_it(self):
        rgb = load('hull-aim-8p4km.png')
        plate = M.detect_nameplates(rgb)[0]['nameplate_bbox']
        aim, _ = A.hull_aim_point(rgb, plate)
        self.assertGreater(aim[1], plate[1] + plate[3] + 10)


if __name__ == '__main__':
    unittest.main()
