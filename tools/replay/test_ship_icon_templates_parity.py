"""Exact-output regression for concurrent native ship-template matching."""
import sys
from pathlib import Path
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ship_icon_templates import ShipIconMatcher, red_signal
from test_ship_icon_templates import rendered


def sequential_classify_patch(matcher, rgb):
    """Original ordered NCC path; compare every returned field without tolerance."""
    search = red_signal(rgb)[4:45, 4:45]
    winners = {}
    for species, angle, scale, template in matcher.templates:
        result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, location = cv2.minMaxLoc(result)
        if species not in winners or score > winners[species]['score']:
            winners[species] = dict(species=species, score=float(score), angle=angle,
                                    scale=scale, offset=[location[0]-8, location[1]-8])
    ranked = sorted(winners.values(), key=lambda entry: entry['score'], reverse=True)
    best, second = ranked[:2]
    margin = best['score'] - second['score']
    accepted = best['score'] >= matcher.min_score and margin >= matcher.min_margin
    return dict(species=best['species'] if accepted else 'unknown',
                icon_score=best['score'], icon_margin=margin,
                icon_candidate=best['species'], icon_angle_deg=best['angle'],
                icon_scale=best['scale'], icon_offset_px=best['offset'])


class ExactTemplateParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matcher = ShipIconMatcher()

    def test_every_output_field_matches_sequential_native_ncc(self):
        for icon, angle in (('cruiser', 37), ('battleship', 173),
                            ('aircarrier', 269), ('destroyer', 0), ('submarine', 90)):
            with self.subTest(icon=icon, angle=angle):
                patch = rendered(icon, angle)
                self.assertEqual(sequential_classify_patch(self.matcher, patch),
                                 self.matcher.classify_patch(patch))
        erased = rendered('battleship', 37)
        erased[16:32] = [20, 35, 32]
        self.assertEqual(sequential_classify_patch(self.matcher, erased),
                         self.matcher.classify_patch(erased))

    def test_equal_scores_keep_original_species_template_and_location(self):
        patch = np.zeros((48, 48, 3), dtype=np.uint8)
        result = self.matcher.classify_patch(patch)
        self.assertEqual(sequential_classify_patch(self.matcher, patch), result)
        self.assertEqual(dict(species='unknown', icon_score=0., icon_margin=0.,
                              icon_candidate='Destroyer', icon_angle_deg=0,
                              icon_scale=.85, icon_offset_px=[-8, -8]), result)


if __name__ == '__main__':
    unittest.main()
