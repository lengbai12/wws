"""Native icon consumer boundaries with independently rendered, rotated RGB inputs."""
from io import BytesIO
import sys
from pathlib import Path
import unittest

import cairosvg
import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ship_icon_templates import ICON_ROOT, SPECIES, ShipIconMatcher


def rendered(icon, angle=0, colour=b'#ff4020'):
    source = (ICON_ROOT / ('minimap_' + icon + '.svg')).read_bytes().replace(b'white', colour)
    image = Image.open(BytesIO(cairosvg.svg2png(bytestring=source, scale=4))).convert('RGBA')
    image = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
    canvas = Image.new('RGB', (192, 192), (20, 35, 32))
    canvas.paste(image, ((192-image.width)//2, (192-image.height)//2), image)
    return np.array(canvas.resize((48, 48), Image.Resampling.LANCZOS))


class NativeIcons(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matcher = ShipIconMatcher()

    def test_rotated_cruiser_battleship_and_carrier_are_distinguished(self):
        for icon in ('cruiser', 'battleship', 'aircarrier'):
            for angle in (0, 37, 90, 173, 269):
                with self.subTest(icon=icon, angle=angle):
                    result = self.matcher.classify_patch(rendered(icon, angle))
                    self.assertEqual(SPECIES[icon], result['species'])

    def test_triangle_classes_retain_candidate_but_abstain_when_gap_is_ambiguous(self):
        for icon in ('destroyer', 'submarine'):
            result = self.matcher.classify_patch(rendered(icon))
            self.assertEqual(SPECIES[icon], result['icon_candidate'])
            self.assertEqual('unknown', result['species'])
            # The internal gap is visible enough at a less conservative margin.
            result = ShipIconMatcher(min_margin=.03).classify_patch(rendered(icon))
            self.assertEqual(SPECIES[icon], result['species'])

    def test_blank_white_green_and_red_circle_are_unknown(self):
        patches = [np.zeros((48, 48, 3), dtype=np.uint8),
                   rendered('battleship', colour=b'#ffffff'),
                   rendered('battleship', colour=b'#40ff40')]
        circle = np.zeros((48, 48, 3), dtype=np.uint8)
        cv2.circle(circle, (24, 24), 9, (255, 60, 30), 2)
        patches.append(circle)
        for patch in patches:
            self.assertEqual('unknown', self.matcher.classify_patch(patch)['species'])

    def test_erased_icon_abstains_instead_of_forcing_a_class(self):
        patch = rendered('battleship', 37)
        patch[16:32] = [20, 35, 32]
        self.assertEqual('unknown', self.matcher.classify_patch(patch)['species'])

    def test_crop_edges_are_padded_and_distant_icon_cannot_supply_class(self):
        rgb = np.zeros((80, 80, 3), dtype=np.uint8)
        rgb[32:80, 32:80] = rendered('cruiser')
        for u, v in ((0, 0), (1, 0), (0, 1), (1, 1)):
            self.assertEqual('unknown', self.matcher.classify(rgb, u, v)['species'])
        self.assertEqual('Cruiser', self.matcher.classify(rgb, 56/79, 56/79)['species'])

    def test_invalid_coordinates_and_colour_contract_are_rejected(self):
        rgb = np.zeros((48, 48, 3), dtype=np.uint8)
        for u, v in ((float('nan'), .5), (-.1, 0), (.5, 1.1)):
            with self.assertRaises(ValueError):
                self.matcher.classify(rgb, u, v)
        with self.assertRaises(ValueError):
            self.matcher.classify_patch(rgb.astype(np.float32))


if __name__ == '__main__':
    unittest.main()
