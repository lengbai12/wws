"""Consumer-visible boundaries of optional red confirmation for enemy peaks."""
import sys
from pathlib import Path
import unittest

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_enemy_recognition as C
import train_minimap_heatmap as T
import enemy_marker_classifier as E


class FixedHeatmaps(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = logits

    def forward(self, images):
        return self.logits.clone(), torch.zeros(images.shape[0], 1)


class RedConfirmation(unittest.TestCase):
    def test_red_keeps_enemy_peak_green_rejects_it_other_channels_unchanged(self):
        logits = torch.full((1, 2, 32, 32), -8.)
        logits[0, :, 16, 16] = 8.
        images = torch.zeros(1, 3, 32, 32)
        images[0, 0, 16, 16] = 1.
        model = C.RedConfirmed(FixedHeatmaps(logits), 1)
        red, _ = model(images)
        self.assertEqual([(16., 16.)], [(x, y) for x, y, _ in T.decode(red.sigmoid())[0][1]])
        images[0, 0, 16, 16] = 0.
        images[0, 1, 16, 16] = 1.
        green, _ = model(images)
        self.assertEqual([], T.decode(green.sigmoid())[0][1])
        self.assertTrue(torch.equal(logits[:, 0], green[:, 0]))

    def test_colour_boundary_cannot_create_peak_on_heatmap_slope(self):
        logits = torch.full((1, 2, 32, 32), -8.)
        logits[0, 1, 16, 8:22] = torch.linspace(1., 8., 14)
        images = torch.zeros(1, 3, 32, 32)
        images[0, 0, 16, 8] = 1.
        filtered, _ = C.RedConfirmed(FixedHeatmaps(logits), 1)(images)
        self.assertEqual([], T.decode(filtered.sigmoid())[0][1])

    def test_white_and_dim_red_are_not_confirmation(self):
        images = torch.ones(2, 3, 32, 32)
        images[1] = 0.
        images[1, 0, 16, 16] = 50 / 255
        self.assertFalse(C.RedConfirmed.support(images, (32, 32)).any())


class MarkerInputAndMatching(unittest.TestCase):
    def test_half_crop_preserves_white_and_rounds_half_up(self):
        rgb = np.full((5, 4, 3), 255, dtype=np.uint8)
        self.assertTrue(np.array_equal(E.half_crop(rgb), np.full((2, 2, 3), 255, dtype=np.uint8)))
        rgb[:2, :2] = 0
        rgb[0, 0] = 2
        self.assertTrue(np.array_equal(E.half_crop(rgb)[0, 0], [1, 1, 1]))

    def test_border_patch_has_fixed_size_and_centred_pixel(self):
        rgb = np.zeros((20, 20, 3), dtype=np.uint8)
        rgb[0, 0] = [255, 0, 0]
        patch = E.patches(rgb, [(0, 0, 1)], 20)
        self.assertEqual((1, 32, 32, 3), patch.shape)
        self.assertTrue(np.array_equal(patch[0, 16, 16], [255, 0, 0]))
        self.assertEqual((0, 32, 32, 3), E.patches(rgb, [], 20).shape)

    def test_duplicate_cannot_match_twice_and_rejected_candidate_is_not_predicted(self):
        records = [dict(offset=0, peaks=[(2, 2, 1), (2, 2, 1), (7, 7, 1)], truth=[(2, 2)])]
        metrics = E.score(records, np.asarray([.9, .8, .1]), .5, 1)
        self.assertEqual(1, metrics['matched'])
        self.assertEqual(2, metrics['predicted'])
        self.assertEqual(.5, metrics['precision'])
        self.assertEqual(1., metrics['recall'])


if __name__ == "__main__":
    unittest.main()
