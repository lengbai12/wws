"""Single native gun selection needs a full mount layout and native reticle."""
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from fire_hud import FireHudReader, FRAME_SHAPE, ROOT


class SingleAmmoHudTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros(FRAME_SHAPE, np.uint8)
        crop = np.asarray(Image.open(Path(__file__).with_name('fire_hud_assets') /
                                    'san-martin-selector-training.png').convert('RGB'))
        self.frame[1395:1510, 780:1140] = crop
        self.reader = FireHudReader()

    def test_native_ap_only_selector_is_selected(self):
        weapon = self.reader._weapon(self.frame)
        self.assertEqual((weapon['status'], weapon['kind'], weapon['ammo']),
                         ('observed', 'gun', 'AP'))

    def test_both_distinct_sprites_and_independent_mounts_required(self):
        # Full native four-mount layout, including raised right endcaps.
        for x in (1108, 1204, 1300, 1396):
            self.frame[1367:1371, x:x+60] = (10, 220, 120)
            self.frame[1365:1367, x+58:x+62] = (10, 220, 120)
        self.assertTrue(self.reader.detect_battle(self.frame)['active'])
        self.frame[1395:1510, 1030:1130] = 0
        self.assertIsNone(self.reader.detect_battle(self.frame)['active'])

    def test_sprite_without_selection_cannot_select_main_guns(self):
        self.frame[1395:1510, 880:900] = 0
        self.frame[1395:1510, 964:980] = 0
        self.assertIsNone(self.reader._weapon(self.frame)['kind'])

    def test_selector_without_mounts_or_reticle_is_not_battle(self):
        self.assertIsNone(self.reader.detect_battle(self.frame)['active'])

    def test_real_mixed_color_mount_still_establishes_battle_layout(self):
        crop = np.asarray(Image.open(Path(__file__).with_name('fire_hud_assets') /
                                    'san-martin-mixed-mounts-training.png').convert('RGB'))
        self.frame[1355:1382, 1060:1470] = crop
        result = self.reader.detect(self.frame)
        self.assertTrue(result['battle']['active'])
        # Operator rule: an amber mount is traversing, so a readable amber bar
        # answers "not ready" (False) rather than "unknown".
        self.assertIs(result['reload']['turrets'][2]['ready'], False)
        # Two mounts are still unreadable, so the all-mounts answer stays None.
        self.assertIsNone(result['reload']['all_ready'])
        self.frame[1355:1382, 1298:1362] = 0
        self.assertIsNone(self.reader.detect_battle(self.frame)['active'])

    def test_real_mixed_mounts_without_two_ammo_sprites_are_unknown(self):
        crop = np.asarray(Image.open(Path(__file__).with_name('fire_hud_assets') /
                                    'san-martin-mixed-mounts-training.png').convert('RGB'))
        self.frame[1355:1382, 1060:1470] = crop
        self.frame[1395:1510, 1030:1130] = 0
        self.assertIsNone(self.reader.detect_battle(self.frame)['active'])


if __name__ == '__main__':
    unittest.main()
