"""Focused policy tests; the parent runs these with the final integration suite."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from target_range_selection import inspect_candidates, read_distance_km, SCHEMA

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).with_name("main_view_fixtures")


def candidate(name: str, distance, *, track_id: int, **extra):
    row = {
        "nameplate_bbox": [100 + track_id * 130, 300, 120, 16],
        "track_id": track_id,
        "track_state": "tracked",
        "body_status": "confirmed",
        "body_association": "confirmed",
        "identity": name,
    }
    if distance is not None:
        row["distance_km"] = distance
    row.update(extra)
    return row


class TargetRangeSelectionTests(unittest.TestCase):
    def test_decimal_joined_to_following_digit_keeps_real_punctuation(self):
        with Image.open(FIXTURES/'range-9p2km-left-diagonal-decimal.png') as image:
            rgb = np.asarray(image.convert('RGB'))
        result = read_distance_km(rgb, [20, 30, 123, 17])
        self.assertEqual(result['status'], 'observed')
        self.assertEqual(result['raw']['candidate_text'], '9.2')
        self.assertEqual(result['value'], 9.2)
        self.assertGreaterEqual(result['confidence'], .74)
        np.testing.assert_allclose(result['interval_km'], [9.15, 9.25])
        self.assertEqual([c['char'] for c in result['raw']['components']], ['9', '.', '2'])
        # Removing only the dot cannot leave a manufactured decimal behind.
        erased = rgb.copy()
        erased[62:65, 61:64] = rgb[75, 61]
        self.assertIsNone(read_distance_km(erased, [20, 30, 123, 17])['value'])

    def test_diagonally_joined_decimal_is_separated_without_inventing_punctuation(self):
        rgb = np.asarray(Image.open(FIXTURES/'range-4p7km-diagonal-decimal.png').convert('RGB'))
        result = read_distance_km(rgb, [20,30,122,17])
        self.assertEqual(result['value'], 4.7)
        self.assertEqual(result['raw']['candidate_text'], '4.7')
        self.assertGreaterEqual(result['confidence'], .74)
        np.testing.assert_allclose(result['interval_km'], [4.65,4.75])
        # Erasing the actual dot must not manufacture another decimal value.
        erased = rgb.copy()
        erased[62:66,60:63] = erased[69,60]
        changed = read_distance_km(erased, [20,30,122,17])
        self.assertNotEqual(changed['value'], 4.7)

    def test_native_thin_one_keeps_ten_km_caption_readable(self):
        rgb = np.asarray(Image.open(FIXTURES/'range-10p2km-thin-one.png').convert('RGB'))
        result = read_distance_km(rgb, [20,30,120,17])
        self.assertEqual(result['value'], 10.2)
        erased = rgb.copy()
        erased[50:70,49:80] = 90
        self.assertEqual(read_distance_km(erased, [20,30,120,17])['status'], 'unknown')

    def test_wide_native_nine_is_located_without_absorbing_chinese_units(self):
        rgb = np.asarray(Image.open(FIXTURES/'range-9p5km-wide-nine.png').convert('RGB'))
        result = read_distance_km(rgb,[20,30,122,17])
        self.assertEqual(result['status'],'observed')
        self.assertEqual(result['value'],9.5)

    def test_native_small_five_keeps_distance_readable(self):
        rgb = np.asarray(Image.open(FIXTURES/'range-9p5km-small-five.png').convert('RGB'))
        result = read_distance_km(rgb,[20,30,124,17])
        self.assertEqual(result['status'],'observed')
        self.assertEqual(result['value'],9.5)
        self.assertGreaterEqual(result['confidence'],.74)

    def test_native_open_four_caption_uses_rasterized_font_not_closed_counter(self):
        rgb = np.asarray(Image.open(FIXTURES/'range-14p0km-open-four.png').convert('RGB'))
        result = read_distance_km(rgb,[20,26,124,17])
        self.assertEqual(result['status'],'observed')
        self.assertEqual(result['value'],14.0)
        self.assertEqual(result['raw']['candidate_text'],'14.0')
        self.assertGreaterEqual(result['confidence'],.74)
        blank = rgb.copy()
        blank[48:68] = 0
        self.assertEqual(read_distance_km(blank,[20,26,124,17])['status'],'unknown')

    def test_native_nine_caption_uses_enclosed_upper_counter(self):
        with Image.open(FIXTURES / 'range-9p4km-native.png') as image:
            rgb = np.asarray(image.convert('RGB'))
        result = read_distance_km(rgb, [1, 30, 123, 17])
        self.assertEqual(result['status'], 'observed')
        self.assertEqual(result['value'], 9.4)
        np.testing.assert_allclose(result['interval_km'], [9.35, 9.45])
        selected = inspect_candidates(rgb, [{'nameplate_bbox': [1, 30, 123, 17],
                                            'trackable': True}], max_range_km=15.8)
        self.assertIsNotNone(selected['range_selected'])
        self.assertIsNone(selected['selected'])

    def test_counter_match_does_not_override_ambiguous_second_digit(self):
        with Image.open(FIXTURES / 'range-9p6km-ambiguous.png') as image:
            rgb = np.asarray(image.convert('RGB'))
        result = read_distance_km(rgb, [16, 30, 123, 17])
        self.assertEqual(result['status'], 'unknown')
        self.assertIsNone(result['value'])
        self.assertIsNone(result['interval_km'])
        self.assertEqual(result['raw']['reason'], 'ambiguous_glyph')
        self.assertEqual(result['raw']['components'][0]['char'], '9')
        self.assertEqual(result['raw']['components'][-1]['char'], '6')

    def test_native_font_counters_distinguish_zero_six_eight_nine(self):
        from PIL import ImageDraw, ImageFont
        from fire_hud import GUI, NativeNumericReader
        reader = NativeNumericReader()
        font = ImageFont.truetype(str(GUI / 'fonts/Warhelios_Regular.ttf'), 20)
        for char in '03689':
            with self.subTest(char=char):
                image = Image.new('RGB', (45, 45))
                ImageDraw.Draw(image).text((3, -2), char, font=font, fill='white')
                result = reader.read(np.asarray(image), [0, 0, 45, 45], match_holes=True)
                self.assertEqual(result['status'], 'observed')
                self.assertEqual(result['text'], char)

    def test_real_numeric_caption_preserves_native_ocr_provenance(self):
        path = FIXTURES / "hull-aim-2p7km.png"
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
        evidence = read_distance_km(rgb, [1131, 285, 124, 16])
        self.assertEqual("observed", evidence["status"])
        self.assertAlmostEqual(2.7, evidence["value"], places=1)
        self.assertEqual("native_font_stroke_ocr", evidence["provenance"]["method"])
        self.assertEqual("fire_hud.NativeNumericReader", evidence["provenance"]["reader"])
        self.assertAlmostEqual(2.65, evidence["interval_km"][0], places=2)
        self.assertAlmostEqual(2.75, evidence["interval_km"][1], places=2)

    def test_real_distant_caption_drives_coarse_selection(self):
        with Image.open(FIXTURES / "hull-aim-8p4km.png") as image:
            rgb = np.asarray(image.convert("RGB"))
        from main_view_perception import detect_nameplates
        row = detect_nameplates(rgb)[0]
        row["track_state"] = "observed"
        self.assertEqual([1221, 642, 123, 17], row["nameplate_bbox"])
        out = inspect_candidates(rgb, [row], max_range_km=20)
        evidence = out["range_selected"]["distance_evidence"]
        self.assertEqual("observed", evidence["status"])
        self.assertAlmostEqual(8.4, evidence["value"])
        self.assertEqual("8.4", evidence["raw"]["candidate_text"])
        self.assertEqual("minimum_channel_neutral_caption", evidence["provenance"]["preprocessing"])
        self.assertIsNone(out["selected"])
        self.assertFalse(out["candidates"][0]["eligible"])
        self.assertTrue(out["candidates"][0]["distance_eligibility"]["eligible"])
        from main_view_perception import verify_ship_bodies
        occluded = verify_ship_bodies(rgb, [row])
        blocked = inspect_candidates(rgb, occluded, max_range_km=20)
        self.assertTrue(blocked["candidates"][0]["distance_eligibility"]["eligible"])
        self.assertIsNone(blocked["range_selected"])
        self.assertIsNone(blocked["selected"])

    def test_missing_caption_keeps_unknown(self):
        rgb = np.zeros((1600, 2560, 3), dtype=np.uint8)
        evidence = read_distance_km(rgb, [1221, 642, 123, 17])
        self.assertEqual("unknown", evidence["status"])
        self.assertIsNone(evidence["value"])
        self.assertIsNone(evidence["interval_km"])

    def test_ocr_last_decimal_controls_range_interval_and_uncertain_text_refuses(self):
        rgb = np.zeros((1600, 2560, 3), dtype=np.uint8)
        located = ([[0, 0, 30, 12]], 100, 100)
        for text, value, half in [('8.40', 8.4, .005), ('8.4', 8.4, .05), ('9', 9., .5)]:
            reading = dict(status='observed', value=value, text=text,
                           raw={'candidate_text': text}, confidence=.9)
            with self.subTest(text=text), patch('target_range_selection._caption_groups', return_value=located), \
                    patch('fire_hud.NativeNumericReader.read', return_value=reading):
                result = read_distance_km(rgb, [100, 80, 120, 16])
            self.assertEqual(result['status'], 'observed')
            np.testing.assert_allclose(result['interval_km'], [value-half, value+half])
        for text in ('8..4', '8.4km', '8.41', ''):
            reading = dict(status='observed', value=8.4, text=text,
                           raw={'candidate_text': text}, confidence=.9)
            with self.subTest(text=text), patch('target_range_selection._caption_groups', return_value=located), \
                    patch('fire_hud.NativeNumericReader.read', return_value=reading):
                result = read_distance_km(rgb, [100, 80, 120, 16])
            self.assertEqual(result['status'], 'unknown')
            self.assertIsNone(result['interval_km'])

    def test_closed_four_to_fifteen_boundaries_and_unknown(self):
        out = inspect_candidates(None, [
            candidate("lower", 4.0, track_id=1),
            candidate("upper", 15.0, track_id=2),
            candidate("below", 3.9, track_id=3),
            candidate("above", 15.1, track_id=4),
            candidate("unknown", None, track_id=5),
        ], max_range_km=15)
        by_name = {row["identity"]: row for row in out["candidates"]}
        for name in ("lower", "upper"):
            self.assertTrue(by_name[name]["eligible"])
        for name, reason in (("below", "range_below_min"),
                             ("above", "range_above_max"), ("unknown", "distance_unknown")):
            self.assertFalse(by_name[name]["distance_eligibility"]["eligible"])
            self.assertIn(reason, by_name[name]["reasons"])
        self.assertEqual("lower", out["selected"]["identity"])

    def test_caption_interval_crossing_lower_boundary_uses_center(self):
        out = inspect_candidates(None, [candidate("rounded", [3.95, 4.05], track_id=1)], max_range_km=15)
        row = out["selected"]
        self.assertIsNotNone(row)
        self.assertEqual([3.95, 4.05], row["distance_evidence"]["interval_km"])
        self.assertEqual(4.0, row["distance_eligibility"]["center_km"])
        self.assertFalse(row["distance_eligibility"]["strict_lower_bound"])
        self.assertEqual("inclusive_interval_center", row["distance_eligibility"]["boundary_semantics"])
        self.assertEqual("inclusive_interval_center", out["provenance"]["distance_policy"])

    def test_ambiguous_or_unconfirmed_dark_structure_cannot_be_hull(self):
        rows = [
            candidate("ambiguous", 9.0, track_id=1, body_status="ambiguous_silhouettes"),
            candidate("dark-structure", 9.1, track_id=2,
                      body_status="visible_silhouette_candidate",
                      body_association="unknown"),
            candidate("bow-confirmed", 9.2, track_id=3,
                      body_status="visible_silhouette_candidate",
                      body_association="unknown",
                      bow_confirmation={"confirmed": True, "current": True}),
        ]
        out = inspect_candidates(None, rows, max_range_km=20)
        self.assertEqual("dark-structure", out["range_selected"]["identity"])
        self.assertEqual("bow-confirmed", out["selected"]["identity"])
        self.assertIn("ambiguous_target", out["candidates"][0]["reasons"])
        self.assertIn("hull_unconfirmed", out["candidates"][1]["reasons"])
        self.assertTrue(out["candidates"][2]["eligible"])

    def test_occlusion_and_stale_bow_confirmation_fail_closed(self):
        rows = [candidate("occluded", 9.0, track_id=1,
                          body_status="occluded_or_unresolved", body_association="unknown"),
                candidate("stale", 9.1, track_id=2, body_status="visible_silhouette_candidate",
                          body_association="unknown", bow_confirmation={"confirmed": True, "current": False}),
                candidate("current", 9.2, track_id=3, body_status="occluded_or_unresolved",
                          body_association="unknown", bow_confirmation={"confirmed": True, "current": True})]
        out = inspect_candidates(None, rows, max_range_km=20)
        self.assertFalse(out["candidates"][0]["range_eligible"])
        self.assertFalse(out["candidates"][1]["eligible"])
        self.assertEqual("current", out["selected"]["identity"])

    def test_overlapping_distances_cannot_choose_nearest(self):
        rows = [candidate("a", [8.9, 9.1], track_id=1),
                candidate("b", [9.0, 9.2], track_id=2)]
        out = inspect_candidates(None, rows, max_range_km=20)
        self.assertIsNone(out["range_selected"])
        self.assertIsNone(out["selected"])
        self.assertEqual("nearest_distance_ambiguous", out["provenance"]["selection_policy"])

    def test_maximum_boundary_preserves_interval_and_uses_center(self):
        for distance in (15.0, [14.95, 15.05]):
            with self.subTest(distance=distance):
                out = inspect_candidates(None, [candidate("upper", distance, track_id=1)], max_range_km=15)
                self.assertEqual("upper", out["selected"]["identity"])
                self.assertEqual(15.0, out["selected"]["distance_eligibility"]["center_km"])
        self.assertEqual([14.95, 15.05], out["selected"]["distance_evidence"]["interval_km"])

    def test_previous_identity_is_retained_only_when_unique(self):
        previous = candidate("old", 12.0, track_id=7)
        rows = [candidate("old", 12.5, track_id=7), candidate("near", 9.5, track_id=8)]
        out = inspect_candidates(None, rows, max_range_km=20, previous=previous)
        self.assertEqual("old", out["selected"]["identity"])
        self.assertEqual("retained_unambiguous_identity", out["provenance"]["selection_policy"])

        duplicated = [candidate("old-a", 12.5, track_id=7), candidate("old-b", 9.5, track_id=7)]
        out = inspect_candidates(None, duplicated, max_range_km=20, previous=previous)
        self.assertIsNone(out["selected"])
        self.assertEqual("previous_identity_ambiguous", out["provenance"]["selection_policy"])

    def test_nearest_eligible_is_used_without_previous_identity(self):
        out = inspect_candidates(None, [candidate("far", 14.0, track_id=1),
                                        candidate("near", 9.0, track_id=2)], max_range_km=20)
        self.assertEqual(SCHEMA, out["schema"])
        self.assertEqual("near", out["selected"]["identity"])
        self.assertEqual("nearest_eligible", out["provenance"]["selection_policy"])


if __name__ == "__main__":
    unittest.main()
