import sys
from pathlib import Path
import unittest
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from main_view_perception import (
    FRAME_H, FRAME_W, body_motion, detect_markers, detect_nameplates,
    detect_target_lock, hud_mask, perceive_frame, verify_ship_bodies, load_native_templates,
    estimate_camera_shift,
)


def blank():
    return np.full((FRAME_H, FRAME_W, 3), 96, dtype=np.uint8)


def rect(a, x, y, w, h, color):
    a[y:y+h, x:x+w] = color

def healthbar(a, x, y, w, h, fill=None):
    rect(a, x-1, y-1, w+2, h+2, (30, 30, 30))
    rect(a, x, y, w if fill is None else fill, h, (220, 30, 30))



class FrameContract(unittest.TestCase):
    def test_orange_detected_caption_is_not_an_enemy_healthbar(self):
        from PIL import Image
        crop = np.asarray(Image.open(Path(__file__).with_name('main_view_fixtures') /
                          'detected-caption-20261019.png').convert('RGB'))
        for x, y in ((700,500), (1250,650)):
            with self.subTest(position=(x,y)):
                frame = blank()
                frame[y:y+80,x:x+160] = crop
                healthbar(frame,x-250,y+90,123,17)
                found = detect_nameplates(frame)
                self.assertEqual(len(found), 1)
                bx, by, bw, bh = found[0]['nameplate_bbox']
                self.assertLessEqual(bx, x-250)
                self.assertGreaterEqual(bx+bw, x-127)
                self.assertEqual((by,bh), (y+90,17))

    def test_rejects_wrong_shape(self):
        with self.assertRaises(ValueError): detect_nameplates(np.zeros((100, 100, 3), np.uint8))

    def test_minimap_and_hud_red_ribbons_are_excluded(self):
        a = blank()
        rect(a, 1900, 900, 180, 8, (220, 25, 25))
        rect(a, 2100, 100, 180, 8, (220, 25, 25))
        rect(a, 2376, 630, 40, 6, (220, 25, 25))  # floating destroyed ribbon
        rect(a, 1300, 1366, 60, 4, (220, 25, 25))  # turret reload indicator
        self.assertEqual([], detect_nameplates(a))

    def test_objective_status_bars_do_not_become_ship_targets(self):
        a = blank()
        healthbar(a, 1158, 158, 96, 17)
        healthbar(a, 1306, 158, 96, 17)
        healthbar(a, 1158, 250, 96, 17)
        found = detect_nameplates(a)
        self.assertEqual(1, len(found))
        self.assertGreater(found[0]["nameplate_bbox"][1], 210)

    def test_nameplate_is_not_body_confirmation(self):
        a = blank(); healthbar(a, 800, 550, 150, 8)
        found = detect_nameplates(a)
        self.assertEqual(1, len(found)); self.assertFalse(found[0]["body_visible"])
        self.assertEqual("occluded_or_unresolved", verify_ship_bodies(a, found)[0]["body_status"])

    def test_nameplate_body_stays_unverified_without_native_hull(self):
        a = blank(); healthbar(a, 800, 550, 150, 8)
        for y in range(565, 640, 4): rect(a, 770 + (y % 7), y, 210, 2, (30, 150, 180))
        found = verify_ship_bodies(a, detect_nameplates(a))
        self.assertEqual(1, len(found)); self.assertFalse(found[0]["body_visible"])
        self.assertEqual("occluded_or_unresolved", found[0]["body_status"])
        x, y, w, h = found[0]["body_roi"]
        self.assertGreaterEqual(x, 0); self.assertGreaterEqual(y, 0)
        self.assertLessEqual(x+w, FRAME_W); self.assertLessEqual(y+h, FRAME_H)

    def test_markers_require_shapes_and_reject_solid_rectangles(self):
        a = blank()
        rect(a, 1200, 500, 16, 16, (240, 120, 20)); rect(a, 1300, 500, 16, 16, (30, 220, 220))
        self.assertEqual([], detect_markers(a))
        with Image.open(Path(__file__).parent / "torpedo_assets/icon_torpedo_normal_enemy.png") as image:
            native = np.asarray(image.convert("RGBA"))
        h, w = native.shape[:2]
        alpha = native[..., 3:4].astype(np.float32)/255
        a[600:600+h, 1200:1200+w] = (native[..., :3]*alpha + 96*(1-alpha)).astype(np.uint8)
        found = detect_markers(a)
        self.assertTrue(any(x["class"] == "torpedo" for x in found))
        self.assertFalse(any(x["class"] == "aircraft" for x in found))
        self.assertTrue(all(np.isfinite(x["score"]) for x in found))
        rect(a, 1200, 600, w, h, (96, 96, 96))
        a[900:900+h, 2200:2200+w] = (native[..., :3]*alpha + 96*(1-alpha)).astype(np.uint8)
        self.assertEqual([], detect_markers(a))

    def test_lock_absence_is_unknown(self):
        result = detect_target_lock(blank())
        self.assertFalse(result["confirmed"]); self.assertEqual("unknown", result["state"])

    def test_native_lock_requires_both_aligned_sprites(self):
        templates = load_native_templates()
        arc, serif = templates['target_lock_arc'], templates['target_lock_serif']
        def stamp(frame, sprite, x, y):
            h, w = sprite.shape[:2]
            alpha = sprite[..., 3:4].astype(np.float32)/255
            patch = frame[y:y+h, x:x+w]
            patch[:] = (sprite[..., :3]*alpha + patch*(1-alpha)).astype(np.uint8)
        for mode in ('arc', 'serif', 'separate', 'paired'):
            with self.subTest(mode=mode):
                frame = blank()
                if mode != 'serif':
                    stamp(frame, arc, 900, 500)
                if mode != 'arc':
                    x = 900+arc.shape[1]-serif.shape[1]
                    y = 500+arc.shape[0]-serif.shape[0]
                    stamp(frame, serif, x+(100 if mode == 'separate' else 0), y)
                result = detect_target_lock(frame, templates)
                self.assertEqual(mode == 'paired', result['confirmed'])
                self.assertEqual('confirmed' if mode == 'paired' else 'unknown', result['state'])

    def test_motion_reports_screen_semantics_and_lost(self):
        a = blank(); b = blank(); rect(a, 800, 550, 150, 8, (220, 30, 30)); rect(b, 800, 550, 150, 8, (220, 30, 30))
        result = body_motion(a, b, [700, 570, 300, 100], (0, 0))
        self.assertIn("velocity_semantics", result)
        self.assertIn(result["state"], ("tracked", "ambiguous", "lost"))
        self.assertEqual("unknown", body_motion(a, b, [-10, -10, 2, 2])["state"])

    def test_per_frame_contract_contains_unknown_lock(self):
        result = perceive_frame(blank())
        self.assertEqual([FRAME_H, FRAME_W, 3], result["frame_shape"])
        self.assertFalse(result["lock_confirmation"]["confirmed"])
        self.assertEqual({"torpedo": "unknown", "aircraft": "unknown"}, result["class_status"])

    def test_invalid_dt_abstains_from_motion(self):
        result = perceive_frame(blank(), blank(), {}, dt_s=float("nan"))
        self.assertIsNone(result["dt_s"])
        self.assertIsNone(result["camera_shift"])
        self.assertEqual([], result["body_motion"])
        self.assertEqual("unknown", result["lock_confirmation"]["target_association"])

    def test_central_aircraft_warning_never_acquires_entity_motion(self):
        from unittest.mock import patch
        marker = {"class": "aircraft", "team": "unknown", "bbox": [900, 600, 30, 30],
                  "marker_family": "central_dangers", "world_velocity": None,
                  "observation_semantics": "screen_hud_danger_indicator_only"}
        with patch("main_view_perception.detect_markers", return_value=[marker]), \
             patch("main_view_perception.estimate_camera_shift", return_value=(0, 0, 1)):
            result = perceive_frame(blank(), blank(), {}, dt_s=.25)
        warning = result["markers"][0]
        self.assertIsNone(warning["screen_velocity_px_s"])
        self.assertIsNone(warning["track_id"])
        self.assertEqual("observed", warning["track_state"])
        self.assertIn("entity_motion_unknown", warning["velocity_semantics"])

    def test_partial_fill_uses_complete_backing_without_confirming_body(self):
        a = blank(); healthbar(a, 800, 550, 150, 8, fill=38)
        found = verify_ship_bodies(a, detect_nameplates(a))
        self.assertEqual(1, len(found))
        self.assertEqual([800, 550, 38, 8], found[0]["red_fill_bbox"])
        x, y, w, h = found[0]["nameplate_bbox"]
        self.assertLessEqual(x, 800); self.assertGreaterEqual(x+w, 950)
        self.assertLessEqual(y, 550); self.assertGreaterEqual(y+h, 558)
        self.assertFalse(found[0]["body_visible"])

    def test_red_tracer_with_only_one_dark_edge_is_not_a_healthbar(self):
        a = blank()
        rect(a, 471, 724, 116, 4, (220, 30, 30))
        rect(a, 470, 723, 118, 1, (30, 30, 30))
        self.assertEqual([], detect_nameplates(a))

    def test_real_tracer_with_misaligned_dark_edges_is_rejected(self):
        from PIL import Image
        crop = np.asarray(Image.open(Path(__file__).with_name('main_view_fixtures') /
                                    'tracer-and-enemy-bars-394108.png').convert('RGB'))
        for dx, dy in ((0, 0), (160, 90)):
            with self.subTest(offset=(dx, dy)):
                frame = blank()
                frame[650+dy:770+dy, 350+dx:1250+dx] = crop
                bars = [d['nameplate_bbox'] for d in detect_nameplates(frame)]
                # The same untouched crop contains a shell tracer and two real
                # enemy bars, including a short numeric fill. Translation must
                # preserve both positives without accepting the tracer.
                self.assertEqual(2, len(bars))
                self.assertTrue(all(x > 900+dx for x, y, w, h in bars))
                self.assertTrue(any(w > 110 and h >= 15 for x, y, w, h in bars))

    def test_numeric_bar_recovers_backing_despite_white_text_over_short_fill(self):
        import cv2
        a = blank(); healthbar(a, 800, 550, 125, 17, fill=29)
        cv2.putText(a, "16 / 70", (812, 564), cv2.FONT_HERSHEY_SIMPLEX, .45,
                    (240, 240, 240), 1)
        found = detect_nameplates(a)
        self.assertEqual(1, len(found))
        x, y, w, h = found[0]["nameplate_bbox"]
        self.assertLessEqual(x, 800); self.assertGreaterEqual(x+w, 924)
        self.assertLessEqual(y, 551); self.assertGreaterEqual(y+h, 565)
        self.assertFalse(found[0]["body_visible"])

    def test_compact_fill_without_long_paired_edges_is_rejected(self):
        a = blank(); healthbar(a, 800, 550, 29, 17)
        self.assertEqual([], detect_nameplates(a))
        healthbar(a, 800, 550, 125, 17, fill=29)
        rect(a, 829, 565, 97, 5, (96, 96, 96))
        self.assertEqual([], detect_nameplates(a))

    def test_red_numeric_text_with_outline_is_not_a_healthbar(self):
        import cv2
        a = blank()
        for thickness, color in ((3, (30, 30, 30)), (1, (220, 30, 30))):
            cv2.putText(a, "16 135 / 70 600", (800, 550), cv2.FONT_HERSHEY_SIMPLEX,
                        .6, color, thickness)
        self.assertEqual([], detect_nameplates(a))

    def test_actual_adjacent_frames_track_bar_and_reject_shell_tracers(self):
        from verify_main_view import read_frame
        root = Path(__file__).resolve().parents[2]
        video = str(root / "recordings/236387/capture.mkv")
        frames = [read_frame(video, t) for t in (394.0, 394.25, 394.5)]
        # Two distinct enemy UI bars: thin segmented and numeric short-fill.
        # Neither shell tracers to the left nor the red class glyph is a bar.
        for i, frame in enumerate(frames):
            with self.subTest(frame=i):
                result = perceive_frame(frame, frames[i-1] if i else None, {}, dt_s=.25 if i else None)
                bars = [d["nameplate_bbox"] for d in result["detections"]]
                self.assertEqual(2, len(bars))
                self.assertEqual(1, sum(x < 1050 and 705 <= y <= 715 for x, y, w, h in bars))
                self.assertEqual(1, sum(1080 <= x <= 1110 and 690 <= y <= 705 and w > 110
                                        for x, y, w, h in bars))
                for ship in result["detections"]:
                    self.assertEqual("tracked" if i else "observed", ship["track_state"])
                    self.assertFalse(ship["body_visible"])
                    self.assertIn("not_world_velocity", ship["velocity_semantics"])

    def test_actual_island_contour_is_rejected_as_body_localization(self):
        from verify_main_view import read_frame
        video = str(Path(__file__).resolve().parents[2] / "recordings/236374/capture.mkv")
        previous, current = (read_frame(video, t) for t in (700.0, 700.25))
        result = perceive_frame(current, previous, {}, dt_s=.25)
        # Numeric recovery can also find the separate left-hand ship. Gamma's
        # island shadow must not become a localized body or supply body motion.
        ships = [d for d in result["detections"] if 450 < d["nameplate_bbox"][0] < 650]
        self.assertEqual(1, len(ships))
        ship = ships[0]
        self.assertEqual("occluded_or_unresolved", ship["body_status"])
        self.assertEqual("unknown", ship["body_association"])
        self.assertFalse(ship["body_visible"])
        self.assertIsNone(ship["body_bbox"])
        index = result["detections"].index(ship)
        self.assertFalse(any(m["detection_index"] == index for m in result["body_motion"]))

    def test_enclosed_dark_component_localizes_without_hull_confirmation(self):
        a = blank()
        rect(a, 820, 625, 60, 12, (30, 30, 30))
        plate = {"body_roi": [760, 570, 250, 160]}
        body = verify_ship_bodies(a, [plate])[0]
        self.assertEqual([820, 625, 60, 12], body["body_bbox"])
        self.assertEqual("visible_silhouette_candidate", body["body_status"])
        self.assertEqual("unknown", body["body_association"])
        self.assertFalse(body["body_visible"])
        self.assertIsNone(body["body_confidence"])

    def test_dark_terrain_connected_beyond_body_window_is_rejected(self):
        a = blank()
        # An interior dark ridge sits inside brighter, still-dark terrain
        # extending outside the verification window.
        rect(a, 720, 625, 350, 55, (40, 40, 40))
        rect(a, 820, 625, 60, 12, (30, 30, 30))
        body = verify_ship_bodies(a, [{"body_roi": [760, 570, 250, 160]}])[0]
        self.assertIsNone(body["body_bbox"])
        self.assertEqual("unknown", body["body_association"])
        self.assertFalse(body["body_visible"])

class CameraTranslationTests(unittest.TestCase):
    def frames(self, dx=8, dy=4, scene=True):
        import cv2
        previous = blank()
        if scene:
            rng = np.random.default_rng(13)
            for x, y in rng.integers((100, 300), (1700, 720), size=(45, 2)):
                rect(previous, int(x), int(y), 9, 9, (180, 180, 180))
        current = cv2.warpAffine(previous, np.float32([[1, 0, dx], [0, 1, dy]]),
                                 (FRAME_W, FRAME_H), borderMode=cv2.BORDER_REFLECT)
        # Dense stationary UI must not overwhelm sparse moving scene features.
        for frame in (previous, current):
            for y in range(760, 820, 15):
                for x in range(450, 2100, 15):
                    rect(frame, x, y, 4, 8, (250, 250, 250))
            for y in range(90, 220, 15):
                for x in range(600, 1900, 15):
                    rect(frame, x, y, 4, 8, (250, 250, 250))
            for x in (790, 1610):
                cv2.putText(frame, 'WARNING', (x, 490), cv2.FONT_HERSHEY_SIMPLEX,
                            .4, (250, 250, 250), 1)
        return previous, current

    def test_stationary_hud_cannot_hide_scene_translation(self):
        previous, current = self.frames()
        dx, dy, score = estimate_camera_shift(previous, current, hud_mask(previous.shape))
        self.assertAlmostEqual(8, dx, delta=.15)
        self.assertAlmostEqual(4, dy, delta=.15)
        self.assertGreater(score, .35)

    def test_stationary_scene_does_not_gain_motion(self):
        previous, current = self.frames(0, 0)
        dx, dy, score = estimate_camera_shift(previous, current)
        self.assertAlmostEqual(0, dx, delta=.05)
        self.assertAlmostEqual(0, dy, delta=.05)
        self.assertGreater(score, .35)

    def test_fixed_hud_alone_cannot_supply_camera_confidence(self):
        previous, current = self.frames(scene=False)
        self.assertEqual((0., 0., 0.), estimate_camera_shift(previous, current))



if __name__ == "__main__":
    unittest.main()
