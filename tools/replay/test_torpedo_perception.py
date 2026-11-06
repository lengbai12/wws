"""Consumer regressions: synthetic native composites are not real positives.

normal-captured-236374.png contains untouched RGB pixels [140:195,730:775]
from state/tracking-mainview-236374.png. Its positive triangle is near (164,751).
The saved screenshot has no trustworthy recording timestamp. Two additional
fixtures preserve clean RGB crops from recordings/236374/capture.mkv at seek
410.000s: normal-recording [1760,770,70,50], alert-lookalike [1590,470,60,80].
The latter is the spotted-alert exclamation mark that the previous single-sprite
matcher falsely labeled torpedo. alert-lookalike-aimprobe.png preserves untouched
RGB [1590:1650,470:550] from state/aimprobe-live-scope/key-action.png; its timestamp
is unknown. Its detached dot matched not_dangerous without the upper glyph.
Other four families lack real positive samples. No raw wake/trajectory labels
are available.
"""
import sys
from pathlib import Path
import unittest

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from torpedo_perception import ASSET_ROOT, FAMILIES, SCALES, asset_manifest, detect
from torpedo_perception import _color_feasible, _templates

HERE = Path(__file__).resolve().parent


def blank():
    return np.full((1600, 2560, 3), 80, dtype=np.uint8)


def rendered(family, scale=1.0):
    with Image.open(ASSET_ROOT / f"icon_torpedo_{family}_enemy.png") as image:
        rgba = np.asarray(image.convert("RGBA")).astype(np.float32)
    alpha = rgba[..., 3:4] / 255
    rgb = rgba[..., :3] * alpha + 80 * (1-alpha)
    return cv2.resize(rgb, (round(rgba.shape[1]*scale), round(rgba.shape[0]*scale)),
                      interpolation=cv2.INTER_LINEAR).astype(np.uint8)


def stamp(frame, image, x=1100, y=700):
    h, w = image.shape[:2]
    frame[y:y+h, x:x+w] = image
    return [x, y, w, h]


class TorpedoConsumer(unittest.TestCase):
    def test_all_native_enemy_families_and_capture_scales_have_marker_evidence(self):
        for family in FAMILIES:
            for scale in SCALES:
                with self.subTest(family=family, scale=scale):
                    frame = blank()
                    box = stamp(frame, rendered(family, scale))
                    found = detect(frame)
                    self.assertEqual(1, len(found))
                    item = found[0]
                    self.assertEqual("torpedo", item["class"])
                    self.assertEqual(box, item["bbox"])
                    self.assertIn(family, item["marker_type_candidates"])
                    self.assertTrue(np.isfinite(item["score"]))
                    self.assertIn("sprite", item["evidence"])
                    self.assertEqual("marker_only", item["geometry_status"])
                    self.assertIsNone(item["wake_geometry"])
                    self.assertIsNone(item["world_trajectory"])
                    self.assertIsNone(item["world_position"])

    def test_saved_actual_frame_pixels_supply_normal_marker_evidence(self):
        with Image.open(HERE / "torpedo_fixtures/normal-captured-236374.png") as image:
            crop = np.asarray(image.convert("RGB"))
        frame = blank()
        stamp(frame, crop, 140, 730)
        found = detect(frame)
        target = [item for item in found if item["bbox"][0] <= 164 < item["bbox"][0]+item["bbox"][2]
                  and item["bbox"][1] <= 751 < item["bbox"][1]+item["bbox"][3]]
        self.assertEqual(1, len(target))
        self.assertIn("normal", target[0]["marker_type_candidates"])
        self.assertIn(target[0]["template_scale"], (1.25, 1.5))

    def test_clean_recording_resolves_two_markers_but_rejects_alert_icon(self):
        frame = blank()
        with Image.open(HERE / "torpedo_fixtures/normal-recording-236374-410s.png") as image:
            stamp(frame, np.asarray(image.convert("RGB")), 1760, 770)
        with Image.open(HERE / "torpedo_fixtures/alert-lookalike-236374-410s.png") as image:
            stamp(frame, np.asarray(image.convert("RGB")), 1590, 470)
        found = detect(frame)
        self.assertEqual(2, len(found))
        for x, y in ((1785, 790), (1804, 792)):
            self.assertTrue(any(item["bbox"][0] <= x < item["bbox"][0]+item["bbox"][2]
                                and item["bbox"][1] <= y < item["bbox"][1]+item["bbox"][3]
                                for item in found))
        self.assertTrue(all("normal" in item["marker_type_candidates"] for item in found))
        self.assertFalse(any(item["bbox"][1] < 600 for item in found))

    def test_complete_actual_detected_glyph_rejects_dot_match_at_new_position(self):
        with Image.open(HERE / "torpedo_fixtures/alert-lookalike-aimprobe.png") as image:
            crop = np.asarray(image.convert("RGB"))
        frame = blank()
        stamp(frame, crop, 1100, 600)
        # A real normal marker beside the alert must remain independently visible.
        with Image.open(HERE / "torpedo_fixtures/normal-recording-236374-410s.png") as image:
            stamp(frame, np.asarray(image.convert("RGB")), 1200, 700)
        found = detect(frame)
        self.assertEqual(2, len(found))
        self.assertTrue(all("normal" in item["marker_type_candidates"] for item in found))
        self.assertTrue(all(item["bbox"][1] >= 700 for item in found))

    def test_circular_marker_requires_visible_unexcluded_upper_context(self):
        native = rendered("not_dangerous", 1.5)
        frame = blank()
        stamp(frame, native)
        self.assertEqual(1, len(detect(frame)))
        # Beyond the old three-pixel border, but within the complete glyph context.
        mask = np.zeros(frame.shape[:2], bool)
        mask[680, 1103] = True
        self.assertEqual([], detect(frame, mask))
        frame = blank()
        stamp(frame, native, 1100, 200)
        self.assertEqual([], detect(frame))

    def test_partial_boundary_marker_and_excluded_support_abstain(self):
        native = rendered("deepwater", 2)
        for x, y in ((0, 700), (2542, 700), (1200, 80)):
            with self.subTest(x=x, y=y):
                frame = blank()
                stamp(frame, native, x, y)
                self.assertEqual([], detect(frame))
        frame = blank()
        box = stamp(frame, rendered("normal", 1.5))
        mask = np.zeros(frame.shape[:2], bool)
        mask[box[1]+4, box[0]+5] = True
        self.assertEqual([], detect(frame, mask))

    def test_exclusion_is_additive_and_includes_transparent_margin(self):
        native = rendered("normal")
        for x, y in ((1900, 900), (2200, 500), (1200, 1400), (100, 100),
                     (100, 1300), (1250, 110), (1826, 800)):
            with self.subTest(x=x, y=y):
                frame = blank()
                stamp(frame, native, x, y)
                self.assertEqual([], detect(frame, np.zeros(frame.shape[:2], bool)))
        frame = blank()
        stamp(frame, native)
        mask = np.zeros(frame.shape[:2], bool)
        mask[697, 1097] = True  # exact first pixel of the padded match window
        self.assertEqual([], detect(frame, mask))

    def test_unrelated_local_exclusion_preserves_complete_marker_output(self):
        # This exclusion is inside the component search window but outside the
        # complete padded sprite, including the circular marker's upper context.
        # Empty and nonempty local exclusion windows must have identical output
        # when the exclusion cannot invalidate the resolved placement.
        for family in ("normal", "not_dangerous"):
            with self.subTest(family=family):
                frame = blank()
                stamp(frame, rendered(family, 2.0))
                expected = detect(frame)
                self.assertEqual(1, len(expected))
                mask = np.zeros(frame.shape[:2], bool)
                mask[700, 1080] = True
                self.assertEqual(expected, detect(frame, mask))

    def test_ambiguous_family_blend_never_claims_precise_weapon_type(self):
        normal, acoustic = rendered("normal", 1.25), rendered("acoustic", 1.25)
        h = max(normal.shape[0], acoustic.shape[0])
        n = np.full((h, normal.shape[1], 3), 80, np.float32)
        c = n.copy()
        n[:normal.shape[0]] = normal
        c[:acoustic.shape[0]] = acoustic
        frame = blank()
        stamp(frame, (.47*n+.53*c).astype(np.uint8))
        found = detect(frame)
        # Resolved class is allowed; a close normal/acoustic family contest must
        # remain unknown rather than use arbitrary loop/asset order as a label.
        self.assertEqual(1, len(found))
        for item in found:
            self.assertEqual("unknown", item["marker_type"])
            self.assertGreater(len(item["marker_type_candidates"]), 1)

    def test_uniform_shapes_text_wakes_and_wrong_team_colors_are_negative(self):
        frame = blank()
        frame[500:510, 1000:1010] = (255, 65, 0)
        cv2.putText(frame, "7 11 V", (1000, 580), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 65, 0), 1)
        cv2.line(frame, (1000, 650), (1110, 670), (245, 245, 245), 2)
        cv2.line(frame, (1000, 690), (1100, 690), (255, 65, 0), 2)
        for index, team in enumerate(("ally", "own")):
            # Preserve the native shape while making the palette white/green.
            image = rendered("normal", 1.5)
            excess = np.maximum(0, image[..., 0].astype(np.int16)-image[..., 1]).astype(np.uint8)
            wrong = np.full_like(image, 80)
            if team == "ally":
                wrong[:] = np.minimum(255, 80+excess.astype(np.int16))[..., None]
            else:
                wrong[..., 1] = np.minimum(255, 80+excess.astype(np.int16))
            stamp(frame, wrong, 1200+index*100, 700)
        self.assertEqual([], detect(frame))
        self.assertEqual([], detect(blank()))

    def test_scale_and_frame_contract_never_silently_resizes(self):
        with self.assertRaises(ValueError):
            detect(np.zeros((800, 1280, 3), np.uint8))
        with self.assertRaises(ValueError):
            detect(blank().astype(np.float32))
        with self.assertRaises(ValueError):
            detect(blank(), np.zeros((1600, 2560), np.uint8))
        with self.assertRaises(ValueError):
            detect(blank(), np.zeros((800, 1280), bool))
        frame = blank()
        stamp(frame, rendered("normal", 3))
        self.assertEqual([], detect(frame))

    def test_manifest_has_independent_native_asset_provenance(self):
        manifest = asset_manifest()
        self.assertEqual(set(FAMILIES), {row["family"] for row in manifest})
        self.assertEqual(5, len({row["sha256"] for row in manifest}))
        for row in manifest:
            self.assertEqual(64, len(row["sha256"]))
            self.assertGreater(min(row["native_size"]), 0)

    def test_color_gate_preserves_exact_eighty_percent_at_valid_domain_edges(self):
        rng = np.random.default_rng(20261016)
        for template in _templates():
            th, tw = template.shape.shape
            ys, xs = np.where(template.support)
            required = (4 * len(xs) + 4) // 5
            # Include the last valid placement of the full padded template;
            # the circular marker's upper context must not shift this domain.
            for x, y in ((0, 0), (7, 0), (0, 9), (7, 9)):
                with self.subTest(family=template.family, scale=template.scale, x=x, y=y):
                    colored = np.zeros((th+9, tw+7), np.uint8)
                    chosen = rng.choice(len(xs), required, replace=False)
                    colored[y+ys[chosen], x+xs[chosen]] = 1
                    self.assertTrue(_color_feasible(colored, cv2.integral(colored),
                                                    cv2.countNonZero(colored), template))
                    # One fewer orange pixel cannot meet any translation's
                    # required support, even if its rectangular bound fits.
                    colored[y+ys[chosen[0]], x+xs[chosen[0]]] = 0
                    self.assertFalse(_color_feasible(colored, cv2.integral(colored),
                                                     cv2.countNonZero(colored), template))


if __name__ == "__main__":
    unittest.main()
