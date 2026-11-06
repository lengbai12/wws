"""Nameplate range caption: the number the HUD prints under the health bar.

Every frame used here is a saved capture from the live client
(state/*-20261020*.png, state/cadence-*/before-*.png). The expected values were
read off enlarged crops of those same frames, so the test asserts the decoder
against what the HUD actually drew rather than against a constant.
"""
import sys
from pathlib import Path
import unittest

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/capture'))
sys.path.insert(0, str(ROOT / 'tools/replay'))

import aim_servo as A
import main_view_perception as MVP


def plate_of(frame):
    plates = MVP.detect_nameplates(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not plates:
        raise AssertionError('no nameplate detected in the fixture frame')
    return plates[0]['nameplate_bbox']


def load(name):
    frame = cv2.imread(str(ROOT / 'state' / name))
    if frame is None:
        raise unittest.SkipTest('fixture frame %s is missing' % name)
    return frame


class CaptionRangeTests(unittest.TestCase):
    def test_reads_the_printed_range(self):
        # 6.2 公里 under the plate at [936, 749, 124, 16]; 16.7 公里 on a second
        # frame with a two-digit integer part, so a decoder stuck on one value
        # or on a single digit layout fails here.
        for name, expected in (('now4-20261020.png', 6.2),
                               ('final-20261020.png', 16.7)):
            with self.subTest(frame=name):
                frame = load(name)
                value = A.hud_distance_km(frame, plate_of(frame))
                self.assertIsNotNone(value)
                self.assertAlmostEqual(value, expected, delta=0.15)

    def test_no_caption_returns_none_without_raising(self):
        # A real frame with the plate box moved onto open water: the band holds
        # no caption, so the reader must say so instead of inventing a range.
        frame = load('final-20261020.png')
        self.assertIsNone(A.hud_distance_km(frame, [300, 420, 120, 16]))
        # A nameplate whose caption is off screen and a plate at the frame edge
        # leave bands too small for the blur; both used to raise.
        self.assertIsNone(A.hud_distance_km(frame, [2500, 1560, 60, 20]))
        self.assertIsNone(A.hud_distance_km(frame, [-40, -40, 20, 16]))

    def test_erased_digits_return_none(self):
        # Reverse control: with the digits painted out, a reader that keys on
        # the bar or on a remembered value would still answer 6.2.
        frame = load('now4-20261020.png')
        x, y, w, h = plate_of(frame)
        band_top = y + h + 2
        centre = x + w // 2
        sea = frame[band_top + 22:band_top + 30, centre - 70:centre + 70]
        frame[band_top + 2:band_top + 20, centre - 70:centre + 70] = sea.mean(axis=0)
        self.assertIsNone(A.hud_distance_km(frame, [x, y, w, h]))
        # The same crop still reads when the digits are left alone.
        self.assertAlmostEqual(
            A.hud_distance_km(load('now4-20261020.png'), [x, y, w, h]), 6.2,
            delta=0.15)


if __name__ == '__main__':
    unittest.main()
