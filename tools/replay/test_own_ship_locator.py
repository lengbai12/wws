"""Own-arrow consumer cases: rotation, terrain, ambiguity, and absent evidence."""
from io import BytesIO
import unittest

import cairosvg
import numpy as np
from PIL import Image, ImageDraw

from own_ship_locator import ICON_PATH, OwnShipLocator


def scene(arrows=(), terrain=False):
    canvas = Image.new('RGB', (1600, 1600), (20, 35, 32))
    draw = ImageDraw.Draw(canvas)
    if terrain:
        draw.rectangle((100, 100, 650, 650), fill='white')
        draw.line((50, 1000, 1400, 1000), fill='white', width=12)
        draw.text((900, 120), 'A 10', fill='white', stroke_width=3)
    icon = Image.open(BytesIO(cairosvg.svg2png(bytestring=ICON_PATH.read_bytes(), scale=4))).convert('RGBA')
    for x, y, angle in arrows:
        rotated = icon.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
        canvas.paste(rotated, (x*4-rotated.width//2, y*4-rotated.height//2), rotated)
    return np.asarray(canvas.resize((400, 400), Image.Resampling.LANCZOS))


class OwnArrow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.locator = OwnShipLocator()

    def test_rotation_and_snow_do_not_select_the_largest_white_region(self):
        for angle in (0, 45, 90, 165, 270, 315):
            with self.subTest(angle=angle):
                found = self.locator.detect(scene([(240, 260, angle)], terrain=True))
                self.assertIsNotNone(found)
                self.assertLess(np.hypot(found['pixel_x']-240, found['pixel_y']-260), 3)

    def test_two_equally_plausible_arrows_are_unknown(self):
        self.assertIsNone(self.locator.detect(scene([(220, 200, 0), (300, 300, 0)])))

    def test_absent_arrow_and_terrain_only_do_not_produce_a_position(self):
        self.assertIsNone(self.locator.detect(scene()))
        self.assertIsNone(self.locator.detect(scene(terrain=True)))

    def test_truncated_arrow_does_not_borrow_nearby_white_support(self):
        self.assertIsNone(self.locator.detect(scene([(0, 0, 45)])))

    def test_bad_colour_and_shape_contract_raise(self):
        for rgb in (np.zeros((40,40), dtype=np.uint8), np.zeros((40,40,3), dtype=np.float32)):
            with self.assertRaises(ValueError):
                self.locator.detect(rgb)


if __name__ == '__main__':
    unittest.main()
