import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from main_view_perception import detect_nameplates, verify_ship_bodies
from target_confirmation import confirm_target

ROOT = Path(__file__).resolve().parents[2]


def blank():
    return np.full((1600, 2560, 3), 50, np.uint8)


def actual(relative):
    return np.asarray(Image.open(ROOT / relative).convert('RGB'))


class TargetConfirmationTests(unittest.TestCase):
    def test_real_hough_radius_error_requires_same_radius_quadrant_support(self):
        crop = np.asarray(Image.open(Path(__file__).parent/'main_view_fixtures'/
                          'short-haus-lock-radius-20261019.png').convert('RGB'))
        for x,y in ((700,500),(1100,600)):
            with self.subTest(position=(x,y)):
                frame = blank()
                frame[y:y+145,x:x+330] = crop
                bar = {'nameplate_bbox':[x+50,y+99,124,17]}
                self.assertTrue(confirm_target(frame,[bar])['confirmed'])
                frame[y+60:y+100,x+65:x+165] = 50
                self.assertFalse(confirm_target(frame,[bar])['confirmed'])

    def test_real_x_toggle_locked_unlocked_locked(self):
        for name, expected in (('locked_before', True), ('unlocked', False),
                               ('locked_restored', True)):
            crop = np.asarray(Image.open(Path(__file__).parent / 'main_view_fixtures' /
                                        f'lock-state-{name}-20261019.png').convert('RGB'))
            frame = blank()
            frame[400:530,700:924] = crop
            bar = {'nameplate_bbox': [750,510,124,17]}
            result = confirm_target(frame,[bar])
            with self.subTest(name=name):
                self.assertEqual(result['confirmed'],expected)
                if expected:
                    self.assertEqual(result['detection_index'],0)
                # A circle without a corresponding bar cannot confirm a target.
                self.assertFalse(confirm_target(frame,[])['confirmed'])

    def test_native_uint8_rgb_contract(self):
        for frame in (np.zeros((800, 1280, 3), np.uint8),
                      np.zeros((1600, 2560, 4), np.uint8),
                      np.zeros((1600, 2560, 3), np.float32)):
            with self.assertRaises(ValueError):
                confirm_target(frame, [])

    def test_ring_without_healthbar_stays_unknown(self):
        frame = blank()
        cv2.circle(frame, (850, 650), 33, (245, 245, 245), 2)
        result = confirm_target(frame, [])
        self.assertFalse(result['confirmed'])
        self.assertEqual('unknown', result['state'])
        self.assertEqual([], result['candidates'])

    def test_partial_arc_does_not_confirm(self):
        frame = blank()
        cv2.ellipse(frame, (850, 650), (33, 33), 0, 0, 90, (245, 245, 245), 2)
        result = confirm_target(frame, [{'nameplate_bbox': [800, 720, 100, 8]}])
        self.assertFalse(result['confirmed'])

    def test_full_ring_support_and_overlap_association(self):
        frame = blank()
        # The lower part lies below the Hough region's bottom (healthbar y).
        center = (850.5, 694.5)
        radius = np.float32(58)
        angles = np.linspace(0, 2*np.pi, 128, endpoint=False)
        for delta in (-2, -1, 0, 1, 2):
            xx = np.rint(center[0]+(radius+delta)*np.cos(angles)).astype(int)
            yy = np.rint(center[1]+(radius+delta)*np.sin(angles)).astype(int)
            frame[yy, xx] = (245, 245, 245)
        detection = {'nameplate_bbox': [800, 720, 100, 8]}
        circles = np.array([[[150.5, 149.5, radius]]], dtype=np.float32)
        for count in (1, 2):
            with self.subTest(healthbars=count), \
                    patch('target_confirmation.cv2.HoughCircles', return_value=circles):
                result = confirm_target(frame, [detection] * count)
            self.assertEqual(count == 1, result['confirmed'])
            self.assertEqual(1, len(result['candidates']))
            ring = result['candidates'][0]
            self.assertEqual(list(center), ring['center'])
            self.assertEqual(1.0, ring['support'])
            self.assertEqual([1.0] * 4, ring['quarters'])
            self.assertEqual(list(range(count)), ring['compatible_detection_indices'])

    def test_frame_edge_preserves_whole_offset_bounds_check(self):
        frame = np.full((1600, 2560, 3), 245, np.uint8)
        detection = {'nameplate_bbox': [0, 150, 100, 8]}
        # At 31.5, the radius-2 contour fits but larger contours cross the edge.
        # At 20.5, all contours cross it, so even white pixels cannot confirm.
        for cx, confirmed in ((31.5, True), (20.5, False)):
            circles = np.array([[[cx, 80.5, 33]]], dtype=np.float32)
            with self.subTest(cx=cx), \
                    patch('target_confirmation.cv2.HoughCircles', return_value=circles):
                result = confirm_target(frame, [detection])
            self.assertEqual(confirmed, result['confirmed'])
            self.assertEqual(int(confirmed), len(result['candidates']))

    def test_sampled_white_predicate_preserves_strict_thresholds(self):
        detection = {'nameplate_bbox': [800, 720, 100, 8]}
        circles = np.array([[[150.5, 105.5, 33]]], dtype=np.float32)
        for color, confirmed in (((156, 210, 180), True),
                                 ((156, 211, 180), False),
                                 ((155, 180, 180), False),
                                 ((255, 0, 255), False)):
            frame = np.empty((1600, 2560, 3), np.uint8)
            frame[:] = color
            with self.subTest(color=color), \
                    patch('target_confirmation.cv2.HoughCircles', return_value=circles):
                result = confirm_target(frame, [detection])
            self.assertEqual(confirmed, result['confirmed'])

    def test_actual_single_target_ui_does_not_confirm_hull_or_identity(self):
        frame = actual('state/aimprobe-live-scope/key-action.png')
        detections = verify_ship_bodies(frame, detect_nameplates(frame))
        result = confirm_target(frame, detections)
        self.assertTrue(result['confirmed'])
        self.assertEqual('unique_healthbar', result['target_association'])
        self.assertEqual(0, result['detection_index'])
        self.assertEqual('unknown', result['identity_association'])
        self.assertEqual('unverified', result['body_association'])
        self.assertFalse(detections[0]['body_visible'])
        ring = result['ring']
        self.assertEqual([0], ring['compatible_detection_indices'])
        self.assertAlmostEqual(1217.5, ring['center'][0], delta=2)
        self.assertAlmostEqual(695.5, ring['center'][1], delta=2)
        x, y, w, h = ring['bbox']
        self.assertLessEqual(x, ring['center'][0]-ring['radius_px'])
        self.assertGreaterEqual(x+w-1, ring['center'][0]+ring['radius_px'])
        self.assertLessEqual(y, ring['center'][1]-ring['radius_px'])
        self.assertGreaterEqual(y+h-1, ring['center'][1]+ring['radius_px'])


    def test_actual_ring_removed_is_unknown(self):
        frame = actual('state/aimprobe-live-scope/key-action.png').copy()
        detections = detect_nameplates(frame)
        frame[640:733, 1130:1320] = (50, 50, 50)
        result = confirm_target(frame, detections)
        self.assertFalse(result['confirmed'])
        self.assertEqual('unknown', result['state'])


if __name__ == '__main__':
    unittest.main()
