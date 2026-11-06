#!/usr/bin/env python3
"""Line-of-fire terrain test: real panels plus a synthetic wall."""
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/replay'))

import minimap_terrain as T  # noqa: E402

FIXTURES = Path(__file__).with_name('minimap_fixtures')


def panel(name):
    return np.asarray(Image.open(FIXTURES / name).convert('RGB'))


class LandMaskTests(unittest.TestCase):
    def test_a_real_island_map_has_land(self):
        mask = T.land_mask(panel('atoll-panel-720x723.png'))
        self.assertGreater(int(mask.sum()), 1000)

    def test_open_ocean_has_no_land(self):
        mask = T.land_mask(panel('ocean-panel-720x723.png'))
        self.assertLess(int(mask.sum()), 200)

    def test_water_and_icons_are_not_land(self):
        # Teal water and the white/red ship marks must not enter the mask.
        water = np.zeros((4, 4, 3), dtype=np.uint8)
        water[..., 1] = 32
        water[..., 2] = 32
        self.assertFalse(T.land_mask(water).any())
        mark = np.zeros((4, 4, 3), dtype=np.uint8)
        mark[..., 0] = 220
        mark[..., 1] = 30
        mark[..., 2] = 30
        self.assertFalse(T.land_mask(mark).any())


class SegmentTests(unittest.TestCase):
    def test_a_wall_between_the_marks_blocks(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[:, 48:52] = True
        row = T.segment_clear(mask, (10, 50), (90, 50))
        self.assertFalse(row['clear'])
        self.assertEqual(row['reason'] if 'reason' in row else 'land_on_the_line',
                         'land_on_the_line')
        self.assertGreater(row['land_samples'], 0)
        self.assertIsNotNone(row['first_hit'])

    def test_a_clear_line_passes(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[0:10, :] = True
        row = T.segment_clear(mask, (10, 50), (90, 50))
        self.assertTrue(row['clear'])
        self.assertEqual(row['land_samples'], 0)

    def test_the_margin_widens_the_obstacle(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[60, 50] = True           # a single pixel, 10 px away from the line
        tight = T.segment_clear(mask, (10, 50), (90, 50), margin=0)
        self.assertTrue(tight['clear'])
        wide = T.segment_clear(mask, (10, 50), (90, 50), margin=11)
        self.assertFalse(wide['clear'])

    def test_own_and_enemy_at_the_same_spot_is_clear(self):
        mask = np.zeros((50, 50), dtype=bool)
        mask[10:20, 10:20] = True
        row = T.segment_clear(mask, (40, 40), (40, 40))
        self.assertTrue(row['clear'])

    def test_a_missing_contact_returns_unknown_not_clear(self):
        row = T.shot_terrain(panel('atoll-panel-720x723.png'), None, (10, 10))
        self.assertIsNone(row['clear'])
        self.assertEqual(row['reason'], 'no_contact')


class RealPanelTests(unittest.TestCase):
    def test_a_line_across_the_atoll_island_is_blocked(self):
        image = panel('atoll-panel-720x723.png')
        mask = T.land_mask(image)
        # Find a column that actually carries a run of land, then aim the
        # segment from above that run to below it. Picking a fixed point would
        # depend on the fixture's exact art.
        column = None
        for x in range(mask.shape[1]):
            rows = np.nonzero(mask[:, x])[0]
            if len(rows) >= 20:
                column, first, last = x, int(rows.min()), int(rows.max())
                break
        self.assertIsNotNone(column, 'fixture has no land column to test')
        row = T.segment_clear(mask, (column, max(0, first - 10)),
                              (column, min(mask.shape[0] - 1, last + 10)))
        self.assertFalse(row['clear'])
        self.assertGreaterEqual(row['land_samples'], 1)

    def test_the_open_ocean_panel_never_blocks(self):
        image = panel('ocean-panel-720x723.png')
        row = T.shot_terrain(image, (360, 240), (330, 490))
        self.assertTrue(row['clear'], row)


class IconIsNotTerrainTests(unittest.TestCase):
    """Measured 2026-10-23: a capture-point icon refused two shots on open water."""

    def _panel(self, blob):
        panel = np.zeros((200, 300, 3), dtype=np.uint8)
        panel[..., 1] = 32
        panel[..., 2] = 32
        x, y, w, h = blob
        panel[y:y + h, x:x + w] = (150, 100, 70)
        return panel

    def test_an_icon_sized_mark_does_not_block(self):
        panel = self._panel((150, 98, 6, 6))  # 36 px, the size of the real icon
        row = T.shot_terrain(panel, (40, 100), (260, 100))
        self.assertTrue(row['clear'], row)
        self.assertEqual(row['reason'], 'clear')
        self.assertIn('ignored', row)
        self.assertLess(row.get('blob_px', 0), 120)
        self.assertLess(row['longest_run_px'], 12)

    def test_a_real_island_blocks(self):
        panel = self._panel((140, 60, 40, 80))  # 3200 px across the line
        row = T.shot_terrain(panel, (40, 100), (260, 100))
        self.assertFalse(row['clear'], row)
        self.assertEqual(row['reason'], 'land_on_the_line')
        self.assertGreaterEqual(row['blob_px'], 120)

    def test_a_short_crossing_is_not_a_coast(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[48:52, 49:52] = True  # a tiny square right on the line
        row = T.segment_clear(mask, (10, 50), (90, 50), min_run_px=12)
        self.assertTrue(row['clear'])
        self.assertLess(row['longest_run_px'], 12)

    def test_a_long_crossing_is_a_coast(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[:, 40:60] = True
        row = T.segment_clear(mask, (10, 50), (90, 50), min_run_px=12)
        self.assertFalse(row['clear'])
        self.assertGreaterEqual(row['longest_run_px'], 12)

    def test_blob_area_counts_a_solid_region(self):
        mask = np.zeros((40, 40), dtype=bool)
        mask[10:20, 10:20] = True
        self.assertEqual(T.blob_area(mask, (12, 12)), 100)
        self.assertEqual(T.blob_area(mask, (0, 0)), 0)


if __name__ == '__main__':
    unittest.main()
