"""Battle evidence must follow the measured slot layout, not a second hard-coded list.

The gate used to re-declare its own offsets, so adding a measured layout to
_weapon left the gate closed. These tests pin both halves: _weapon must try the
live client's bar-centring offset, and the battle evidence must derive its
layouts from the slots the weapon reader actually produced.
"""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fire_hud as FH
from fire_hud import FireHudReader, SELECTOR_ROIS

LIVE_OFFSET = 71


class WeaponLayoutOffsetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rgb = np.zeros((FH.FRAME_SHAPE[0], FH.FRAME_SHAPE[1], 3), np.uint8)

    def test_weapon_tries_the_live_bar_centring_offset(self):
        seen = []

        def fake_match(rgb, template, roi, **kwargs):
            seen.append(roi)
            return []

        with patch.object(FH, '_match', fake_match):
            FireHudReader()._weapon(self.rgb)
        for base in SELECTOR_ROIS:
            self.assertIn(base[0] + LIVE_OFFSET, [roi[0] for roi in seen])

    def test_depth_charge_layout_tries_the_same_offset(self):
        seen = []
        reader = FireHudReader()

        def fake_match(rgb, template, roi, **kwargs):
            seen.append(roi[0])
            return []

        with patch.object(FH, '_match', fake_match), \
                patch.object(FireHudReader, '_cooling_depth_charge_icon', lambda *a: False):
            reader._depth_charge_selection(self.rgb)
        self.assertIn(1259 + LIVE_OFFSET, seen)

    def test_battle_evidence_derives_layouts_from_the_slots(self):
        reader = FireHudReader()
        slots = [{"ammo": ammo, "bbox": [0, 0, 10, 10], "icon": {"confidence": .9},
                  "selected_outline": False, "outline_edge_rows": [0, 0],
                  "layout_offset_px": offset}
                 for offset in (7, 91) for ammo in ('HE', 'AP')]
        weapon = FH.evidence(bbox=[0, 0, 10, 10], kind=None, ammo=None, selected=False)
        weapon['raw'] = {"slots": slots}
        with patch.object(FireHudReader, '_weapon', lambda self, rgb: weapon), \
                patch.object(FireHudReader, '_reload', lambda self, rgb, w: {
                    'raw': {'layout_status': 'observed',
                            'colored_mount_layout_observed': False}, 'ready': None}), \
                patch.object(FireHudReader, '_depth_charge_selection', lambda self, rgb: []):
            battle = reader._battle_components(self.rgb)[3]
        self.assertEqual(battle['raw']['ammo_layout_offsets_px'], [7, 91])
        self.assertTrue(battle['active'])
        self.assertEqual(battle['raw']['reason'], 'native_battle_hud')

    def test_edge_rows_counts_rows_and_clamps_to_the_frame(self):
        rgb = np.zeros((FH.FRAME_SHAPE[0], FH.FRAME_SHAPE[1], 3), np.uint8)
        rgb[20:40, 30] = (0, 255, 120)
        self.assertEqual(FH.edge_rows(rgb, 20, 30, 31, 20), 20)
        # A band that starts inside the frame must not raise or count outside it.
        self.assertEqual(FH.edge_rows(rgb, 20, -4, 0, 20), 0)
        self.assertEqual(FH.edge_rows(rgb, 20, 0, 0, 20), 0)

    def test_outline_two_pixels_outside_the_slot_still_selects(self):
        slot_x, slot_w, slot_y, slot_h = SELECTOR_ROIS[0][0] + LIVE_OFFSET, SELECTOR_ROIS[0][2], 1395, 115
        rgb = np.zeros((FH.FRAME_SHAPE[0], FH.FRAME_SHAPE[1], 3), np.uint8)
        for column in (slot_x-2, slot_x-1, slot_x+slot_w-1, slot_x+slot_w-2):
            rgb[slot_y+5:slot_y+slot_h-5, column] = (0, 255, 120)
        reader = FireHudReader()
        with patch.object(FH, '_match', lambda rgb_, template, roi, **kw:
                          [{'confidence': .9, 'bbox': list(roi)}] if roi[0] == slot_x else []):
            result = reader._weapon(rgb)
        selected = [s for s in result['raw']['slots']
                    if s['layout_offset_px'] == LIVE_OFFSET and s['selected_outline'] and s['icon']]
        self.assertEqual([s['ammo'] for s in selected], ['HE'])

    def test_a_distant_outline_never_selects_the_slot(self):
        slot_x, slot_w = SELECTOR_ROIS[0][0] + LIVE_OFFSET, SELECTOR_ROIS[0][2]
        rgb = np.zeros((FH.FRAME_SHAPE[0], FH.FRAME_SHAPE[1], 3), np.uint8)
        for column in (slot_x-40, slot_x-39, slot_x+slot_w+39, slot_x+slot_w+40):
            rgb[1400:1510, column] = (0, 255, 120)
        reader = FireHudReader()
        with patch.object(FH, '_match', lambda rgb_, template, roi, **kw:
                          [{'confidence': .9, 'bbox': list(roi)}] if roi[0] == slot_x else []):
            result = reader._weapon(rgb)
        self.assertEqual([s for s in result['raw']['slots'] if s['selected_outline']], [])

    def test_an_unselected_single_ammo_sprite_never_opens_the_gate(self):
        reader = FireHudReader()
        weapon = FH.evidence(bbox=[0, 0, 10, 10], kind=None, ammo=None, selected=False)
        weapon['raw'] = {"slots": [{"ammo": 'HE', "bbox": [0, 0, 10, 10],
                                    "icon": {"confidence": .9}, "selected_outline": False,
                                    "outline_edge_rows": [0, 0], "layout_offset_px": 71}]}
        with patch.object(FireHudReader, '_weapon', lambda self, rgb: weapon), \
                patch.object(FireHudReader, '_reload', lambda self, rgb, w: {
                    'raw': {'layout_status': 'observed',
                            'colored_mount_layout_observed': False}, 'ready': None}), \
                patch.object(FireHudReader, '_depth_charge_selection', lambda self, rgb: []):
            battle = reader._battle_components(self.rgb)[3]
        self.assertIsNone(battle['active'])
        self.assertEqual(battle['raw']['reason'], 'battle_hud_unconfirmed')

class NativeBattleshipMountTests(unittest.TestCase):
    def test_three_native_mounts_and_missing_bar_is_not_ready(self):
        from PIL import Image
        rgb = np.zeros(FH.FRAME_SHAPE, np.uint8)
        fixture = np.asarray(Image.open(FH.REFERENCES / 'battleship-three-mounts-20261018.png').convert('RGB'))
        rgb[1370:1385, 1065:1475] = fixture
        reader = FireHudReader()
        weapon = FH.evidence('observed', kind='gun', selected=True, ammo='HE')
        reload = reader._reload(rgb, weapon)
        self.assertEqual(reload['raw']['layout_status'], 'observed')
        self.assertTrue(reload['all_ready'])
        rgb[1370:1385, 1360:1475] = 0
        rejected = reader._reload(rgb, weapon)
        self.assertFalse(rejected['raw']['colored_mount_layout_observed'])
        self.assertIsNot(rejected['all_ready'], True)

    def test_green_ring_requires_independent_native_ammunition(self):
        from PIL import Image
        rgb = np.zeros(FH.FRAME_SHAPE, np.uint8)
        icon = np.asarray(Image.open(FH.REFERENCES / 'green-aim-ring-20261018.png').convert('RGBA'))
        rgb[790:832, 1270:1312] = np.where(icon[..., 3:4] > 0, icon[..., :3], 0)
        reader = FireHudReader()
        ring_only = reader.detect(rgb)
        self.assertEqual(ring_only['reticle']['status'], 'observed')
        self.assertIsNone(ring_only['battle']['active'])
        from test_fire_hud import composite
        composite(rgb, reader.icons['he'], (945, 1412))
        composite(rgb, reader.icons['ap'], (1041, 1412))
        self.assertTrue(reader.detect(rgb)['battle']['active'])
        rgb[790:832, 1270:1312] = 0
        self.assertIsNone(reader.detect(rgb)['battle']['active'])


class SanMartinFourMountTests(unittest.TestCase):
    def frame(self):
        from PIL import Image
        frame = np.zeros(FH.FRAME_SHAPE, np.uint8)
        path = Path(__file__).parent / 'main_view_fixtures' / 'san-martin-four-mount-hud-20261018.png'
        frame[1365:1520, 880:1475] = np.asarray(Image.open(path).convert('RGB'))
        return frame

    def test_real_selected_ammo_and_mounts_establish_battle(self):
        from test_fire_hud import composite
        reader = FireHudReader()
        frame = self.frame()
        composite(frame, reader.icons['green_aim_ring'], (1270, 790))
        result = reader.detect(frame)
        self.assertEqual(result['weapon']['kind'], 'gun')
        self.assertTrue(result['battle']['active'])
        self.assertEqual(len(result['reload']['turrets']), 4)
        self.assertTrue(result['reload']['ready'])
        self.assertIsNone(result['reload']['all_ready'])
        for x in (1119, 1209, 1299, 1389):
            missing = frame.copy()
            missing[1371:1382, x-2:x+58] = 0
            with self.subTest(missing_mount=x):
                self.assertIsNone(reader.detect(missing)['battle']['active'])
        missing = frame.copy()
        missing[1395:1510, 912:992] = 0
        self.assertIsNone(reader.detect(missing)['battle']['active'])


    def test_thin_amber_mounts_need_both_caps_and_do_not_claim_ready(self):
        from PIL import Image
        from test_fire_hud import composite
        frame = np.zeros(FH.FRAME_SHAPE, np.uint8)
        path = Path(__file__).parent / 'main_view_fixtures' / 'san-martin-thin-amber-hud-20261018.png'
        frame[1365:1520, 880:1475] = np.asarray(Image.open(path).convert('RGB'))
        reader = FireHudReader()
        composite(frame, reader.icons['green_aim_ring'], (1270, 790))
        result = reader.detect(frame)
        self.assertTrue(result['battle']['active'])
        self.assertTrue(result['reload']['ready'])
        self.assertEqual([t['ready'] for t in result['reload']['turrets']],
                         [True, True, None, None])
        for lo, hi in ((1298, 1301), (1353, 1356), (1320, 1335)):
            missing = frame.copy()
            missing[1372:1380, lo:hi] = 0
            with self.subTest(erased=(lo, hi)):
                self.assertIsNone(reader.detect(missing)['battle']['active'])


class CerberusNativeAmmoTests(unittest.TestCase):
    def test_live_selected_ammo_requires_native_sprite_and_both_edges(self):
        from PIL import Image
        from test_fire_hud import composite
        frame = np.zeros(FH.FRAME_SHAPE, np.uint8)
        path = Path(__file__).parent / 'main_view_fixtures' / 'cerberus-ammo-hud-20261019.png'
        frame[1395:1510, 950:1140] = np.asarray(Image.open(path).convert('RGB'))
        reader = FireHudReader()
        composite(frame, reader.icons['green_aim_ring'], (1270, 790))
        result = reader.detect(frame)
        self.assertTrue(result['battle']['active'])
        self.assertEqual(result['weapon']['kind'], 'gun')
        self.assertEqual(result['weapon']['ammo'], 'HE')
        self.assertTrue(result['weapon']['selected'])
        # The remaining AP sprite alone must not establish battle or a gun selection.
        missing = frame.copy()
        missing[1395:1510, 950:1048] = 0
        rejected = reader.detect(missing)
        self.assertIsNone(rejected['battle']['active'])
        self.assertIsNone(rejected['weapon']['kind'])


if __name__ == '__main__':
    unittest.main()
