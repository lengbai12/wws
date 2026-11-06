"""Physical glyph identity and close-target behavioral regressions."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from enemy_glyph_dedup import deduplicate
from test_ship_icon_templates import rendered
from ship_icon_templates import ShipIconMatcher


def candidate(x, y, centre=(30, 30), score=.9, heatmap=.95, species='Cruiser', icon_score=.9):
    return dict(u=x/79, v=y/79, score=score, heatmap_score=heatmap,
                species=species, icon_candidate='Cruiser', icon_score=icon_score,
                icon_angle_deg=0, icon_scale=1., icon_offset_px=[centre[0]-x, centre[1]-y])


class GlyphIdentity(unittest.TestCase):
    def setUp(self):
        self.rgb = np.zeros((80, 80, 3), dtype=np.uint8)
        self.rgb[26:35, 27:34] = [255, 60, 30]
        template = np.zeros((25, 25), np.float32)
        template[8:17, 9:16] = 1
        self.matcher = SimpleNamespace(templates=[('Cruiser', 0, 1., template)])

    def test_duplicate_glyph_keeps_strongest_original_coordinates(self):
        weak = candidate(28, 30, score=.8)
        strong = candidate(32, 30)
        self.assertEqual([strong], deduplicate(self.rgb, [weak, strong], self.matcher))
        self.assertEqual([strong], deduplicate(self.rgb, [strong, weak], self.matcher))

    def test_two_close_ships_survive_even_with_identical_offsets_and_class(self):
        self.rgb[26:35, 39:46] = [255, 60, 30]
        left = candidate(30, 30)
        right = candidate(42, 30, centre=(42, 30))
        self.assertCountEqual([left, right], deduplicate(self.rgb, [right, left], self.matcher))

    def test_offsets_pointing_at_neighbor_cannot_merge_distinct_components(self):
        self.rgb[26:35, 39:46] = [255, 60, 30]
        left = candidate(30, 30)
        right = candidate(42, 30, centre=(30, 30))
        self.assertEqual(2, len(deduplicate(self.rgb, [left, right], self.matcher)))

    def test_unknown_species_can_share_strong_pixel_identity(self):
        first = candidate(28, 30, species='unknown')
        second = candidate(32, 30, species='unknown', score=.8)
        self.assertEqual([first], deduplicate(self.rgb, [second, first], self.matcher))

    def test_weak_unknown_offsets_do_not_supply_identity(self):
        first = candidate(28, 30, species='unknown', icon_score=.4)
        second = candidate(32, 30, species='unknown', icon_score=.4)
        self.assertEqual(2, len(deduplicate(self.rgb, [first, second], self.matcher)))

    def test_touching_ships_with_same_refined_centre_remain_ambiguous(self):
        self.rgb[26:35, 34:42] = [255, 60, 30]
        first = candidate(30, 30)
        second = candidate(38, 30)
        self.assertEqual(2, len(deduplicate(self.rgb, [first, second], self.matcher)))

    def test_multiple_components_near_candidate_remain_ambiguous(self):
        self.rgb[29:32, 24] = [255, 60, 30]
        first = candidate(27, 30)
        second = candidate(28, 30)
        self.assertEqual(2, len(deduplicate(self.rgb, [first, second], self.matcher)))

    def test_sparse_fragments_cannot_impersonate_two_template_lobes(self):
        self.rgb[:] = 0
        self.rgb[27:30, 27:30] = [255, 60, 30]
        self.rgb[32:35, 32:35] = [255, 60, 30]
        template = self.matcher.templates[0][3]
        template[:] = 0
        template[7:12, 7:12] = 1
        template[14:19, 14:19] = 1
        first = candidate(30, 30)
        second = candidate(31, 30)
        self.assertEqual(2, len(deduplicate(self.rgb, [first, second], self.matcher)))

    def test_conflicting_template_species_do_not_merge(self):
        template = self.matcher.templates[0][3]
        self.matcher.templates.append(('Destroyer', 0, 1., template))
        first = candidate(28, 30)
        second = candidate(32, 30)
        second['icon_candidate'] = 'Destroyer'
        self.assertEqual(2, len(deduplicate(self.rgb, [first, second], self.matcher)))

    def test_centre_tolerance_does_not_chain_across_one_component(self):
        self.matcher.templates[0][3][:] = 1
        first = candidate(29, 30, centre=(29, 30), score=.99)
        second = candidate(30, 30, centre=(31, 30), score=.98)
        third = candidate(31, 30, centre=(33, 30), score=.97)
        self.assertCountEqual([first, third], deduplicate(self.rgb, [third, second, first], self.matcher))

    def test_score_ties_use_heatmap_then_coordinates_independent_of_input_order(self):
        first = candidate(28, 30, heatmap=.96)
        second = candidate(32, 30, heatmap=.97)
        self.assertEqual([second], deduplicate(self.rgb, [first, second], self.matcher))
        second['heatmap_score'] = .96
        self.assertEqual([first], deduplicate(self.rgb, [second, first], self.matcher))

    def test_clipped_glyph_is_not_identity_evidence(self):
        rgb = np.zeros_like(self.rgb)
        rgb[26:35, :5] = [255, 60, 30]
        first = candidate(0, 30, centre=(1, 30))
        second = candidate(2, 30, centre=(1, 30))
        self.assertEqual(2, len(deduplicate(rgb, [first, second], self.matcher)))

    def test_empty_and_singleton_return_without_needing_template_evidence(self):
        self.assertEqual([], deduplicate(self.rgb, [], self.matcher))
        first = candidate(30, 30)
        self.assertEqual([first], deduplicate(self.rgb, [first], self.matcher))


class NativeGlyph(unittest.TestCase):
    def test_independently_rendered_native_cruiser_duplicate(self):
        matcher = ShipIconMatcher()
        rgb = np.zeros((80, 80, 3), np.uint8)
        rgb[16:64, 16:64] = rendered('cruiser', 37)
        detections = []
        for x, score in ((38, .8), (42, .9)):
            detections.append(dict(u=x/79, v=40/79, score=score, heatmap_score=.99,
                                   **matcher.classify(rgb, x/79, 40/79)))
        self.assertEqual([detections[1]], deduplicate(rgb, detections, matcher))



if __name__ == '__main__':
    unittest.main()
