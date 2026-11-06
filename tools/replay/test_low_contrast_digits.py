"""HUD digits on storm/night frames: timer, range caption and reticle impact.

Every frame is a saved live capture. ``low-contrast-hud-20261020.png`` is the
night-battle frame whose timer reads 24:39 and whose nameplate caption reads
14.7 公里 while both were drawn white on a grey sea; the absolute-brightness
rules in ``fire_hud`` count that sea as ink and return unknown on it.
``battle-timer-fixture-20261020.png`` is the bright frame the existing readers
already resolve (17:55), and ``beyond-range-hud-20261020.png`` is the frame
whose reticle prints the "- - km" placeholder instead of a landing range.

The expected values were read off enlarged crops of these same frames, so the
tests assert the decoder against what the HUD drew rather than against a
constant screen position.
"""
import sys
from pathlib import Path
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/replay'))

import fire_hud
import main_view_perception as MVP


def load(name):
    frame = cv2.imread(str(ROOT / 'state' / name))
    if frame is None:
        raise unittest.SkipTest('fixture frame %s is missing' % name)
    return frame


def plate_of(frame):
    plates = MVP.detect_nameplates(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not plates:
        raise AssertionError('no nameplate detected in the fixture frame')
    return plates[0]['nameplate_bbox']


def background_above(frame, box, gap=6):
    x, y, w, h = box
    return frame[max(0, y - gap):max(0, y - gap) + 2, x:x + w].reshape(-1, 3).mean(axis=0)


def paint(frame, box):
    """Overwrite a box with the background above it: no digits, no edges."""
    x, y, w, h = box
    frame[y:y + h, x:x + w] = background_above(frame, box)


class BattleTimerLowContrastTests(unittest.TestCase):
    def test_dark_frame_reads_the_printed_time(self):
        # 24:39 is what this frame draws; the white-mask path finds the five
        # glyph groups and fails on every digit, so this read is the front
        # end's (capped by the reader's own template and margin rules).
        self.assertEqual(fire_hud.battle_time_left(load('low-contrast-hud-20261020.png')),
                         24 * 60 + 39)

    def test_bright_frame_keeps_its_absolute_threshold_read(self):
        self.assertEqual(fire_hud.battle_time_left(load('battle-timer-fixture-20261020.png')),
                         17 * 60 + 55)

    def test_painting_over_the_digits_returns_none(self):
        # Reverse control: with the digits painted out a front end that keyed
        # on the band rather than on the glyphs would still report 24:39.
        frame = load('low-contrast-hud-20261020.png')
        paint(frame, (2274, 108, 62, 30))
        self.assertIsNone(fire_hud.battle_time_left(frame))

    def test_a_reduced_contrast_crop_still_reads(self):
        frame = load('battle-timer-fixture-20261020.png')
        x, y, w, h = fire_hud.BATTLE_TIMER_BAND
        band = frame[y:y + h, x:x + w].astype(np.float32)
        frame[y:y + h, x:x + w] = np.clip(band * 0.6 + 50, 0, 255).astype(np.uint8)
        self.assertEqual(fire_hud.battle_time_left(frame), 17 * 60 + 55)


class NameplateRangeCaptionTests(unittest.TestCase):
    def test_dark_frame_reads_the_printed_range(self):
        # 14.7 公里 is drawn under the plate of this frame. The absolute
        # brightness reader returns "ambiguous_glyph" here; the local
        # background front end resolves the same digits.
        frame = load('low-contrast-hud-20261020.png')
        km, source = fire_hud.target_range_km(frame, plate_of(frame))
        self.assertAlmostEqual(km, 14.7, delta=0.15)
        self.assertEqual(source, 'low_contrast_caption')

    def test_a_resolved_caption_keeps_the_existing_reader(self):
        # 13.3 公里 on a frame the replay caption reader already resolves: the
        # new path must stay a fallback rather than displace it.
        frame = load('beyond-range-hud-20261020.png')
        km, source = fire_hud.target_range_km(frame, plate_of(frame))
        self.assertAlmostEqual(km, 13.3, delta=0.15)
        self.assertEqual(source, 'absolute_brightness_caption')

    def test_painting_over_the_caption_returns_none(self):
        frame = load('low-contrast-hud-20261020.png')
        plate = plate_of(frame)
        centre = plate[0] + plate[2] // 2
        paint(frame, (centre - 34, plate[1] + plate[3] + 6, 62, 14))
        self.assertEqual(fire_hud.target_range_km(frame, plate), (None, 'unavailable'))

    def test_a_reduced_contrast_caption_still_reads(self):
        frame = load('beyond-range-hud-20261020.png')
        plate = plate_of(frame)
        centre = plate[0] + plate[2] // 2
        box = (centre - 40, plate[1] + 22, 40, 23)
        x, y, w, h = box
        digits = frame[y:y + h, x:x + w].astype(np.float32)
        frame[y:y + h, x:x + w] = np.clip(digits * 0.6 + 55, 0, 255).astype(np.uint8)
        km, source = fire_hud.target_range_km(frame, plate)
        self.assertAlmostEqual(km, 13.3, delta=0.15)
        self.assertEqual(source, 'low_contrast_caption')


class ImpactReadoutTests(unittest.TestCase):
    def test_beyond_maximum_range_is_named_not_estimated(self):
        # This frame draws "- - km": an aim point past maximum range. The two
        # dashes are the only thing that says so, and no range is printed.
        frame = load('beyond-range-hud-20261020.png')
        centre = fire_hud.green_reticle(frame)['centre']
        self.assertEqual(fire_hud.impact_readout(frame, centre), ('beyond', None))

    def test_printed_landing_distance_is_read(self):
        # 24.6 km beside the reticle on the bright frame.
        frame = load('battle-timer-fixture-20261020.png')
        centre = fire_hud.green_reticle(frame)['centre']
        self.assertEqual(fire_hud.impact_readout(frame, centre), ('km', 24.6))

    def test_painting_over_the_placeholder_leaves_unknown(self):
        # Reverse control: without the dashes this frame is "unknown", not
        # "beyond", and never a range the HUD did not draw.
        frame = load('beyond-range-hud-20261020.png')
        centre = fire_hud.green_reticle(frame)['centre']
        paint(frame, (1330, 826, 26, 10))
        self.assertEqual(fire_hud.impact_readout(frame, centre), ('unknown', None))

    def test_erasing_the_number_leaves_unknown(self):
        frame = load('battle-timer-fixture-20261020.png')
        centre = fire_hud.green_reticle(frame)['centre']
        paint(frame, (1333, 820, 36, 20))
        self.assertEqual(fire_hud.impact_readout(frame, centre), ('unknown', None))


if __name__ == '__main__':
    unittest.main()
