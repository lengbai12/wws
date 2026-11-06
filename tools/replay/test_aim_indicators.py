from pathlib import Path
import unittest
import cv2
import numpy as np
from PIL import Image
from aim_indicators import read_aim_indicators

FIX = Path(__file__).parents[1]/'capture/aim_fixtures'

class AimIndicatorContracts(unittest.TestCase):
    def test_real_white_point_does_not_follow_colored_gun_state(self):
        results = []
        for color in ('green','orange'):
            rgb = np.asarray(Image.open(FIX/f'white-aim-{color}-guns-20261019.png').convert('RGB'))
            result = read_aim_indicators(rgb)
            results.append(result)
            np.testing.assert_allclose(result['aim_point']['center'],[1279.5,804],atol=1)
            self.assertEqual(result['gun_direction']['ring']['color'],color)
            self.assertEqual(result['gun_direction']['ready'],color=='green')
        self.assertEqual(results[0]['aim_point']['center'],results[1]['aim_point']['center'])

    def test_scale_and_translation_are_measured_not_fixed_coordinates(self):
        rgb = np.asarray(Image.open(FIX/'white-aim-green-guns-20261019.png').convert('RGB'))
        # Synthetic geometry transforms test search/scale behavior only, not
        # native client rendering or cross-machine acceptance.
        for factor in (.75,1.25):
            frame = cv2.resize(rgb,None,fx=factor,fy=factor)
            result = read_aim_indicators(frame)
            self.assertIsNotNone(result['aim_point'])
            np.testing.assert_allclose(result['aim_point']['center'],np.array([1279.5,804])*factor,atol=2)
        frame = cv2.warpAffine(rgb,np.float32([[1,0,55],[0,1,-30]]),(2560,1600))
        result = read_aim_indicators(frame)
        np.testing.assert_allclose(result['aim_point']['center'],[1334.5,774],atol=1)

    def test_green_ring_permits_partial_battery_with_other_orange_rings(self):
        rgb = np.asarray(Image.open(FIX/'white-aim-green-guns-20261019.png').convert('RGB'))
        mixed = rgb.copy()
        # Keep the real white point and green ring, add unavailable turrets.
        for center in ((1330,800),(1230,800)):
            cv2.circle(mixed,center,19,(240,170,30),2)
        result = read_aim_indicators(mixed)
        self.assertTrue(any(r['color']=='orange' for r in result['gun_rings']))
        self.assertTrue(result['gun_direction']['ready'])
        self.assertEqual(result['gun_direction']['ring']['color'],'green')
        orange_only = np.asarray(Image.open(FIX/'white-aim-orange-guns-20261019.png').convert('RGB'))
        self.assertFalse(read_aim_indicators(orange_only)['gun_direction']['ready'])

    def test_green_ring_not_vetoed_by_orange_at_white_point(self):
        rgb = np.array(Image.open(FIX/'white-aim-orange-guns-20261019.png').convert('RGB'))
        cv2.circle(rgb,(1380,800),19,(30,230,160),2)
        result = read_aim_indicators(rgb)
        self.assertTrue(result['gun_direction']['ready'])
        self.assertEqual(result['gun_direction']['ring']['color'],'green')
        self.assertGreater(abs(result['gun_direction']['ring']['center'][0]-
                               result['aim_point']['center'][0]),30)

    def test_missing_white_or_green_ring_never_ready(self):
        rgb = np.asarray(Image.open(FIX/'white-aim-green-guns-20261019.png').convert('RGB'))
        frame = rgb.copy()
        frame[794:815,1268:1292] = 0
        self.assertIsNone(read_aim_indicators(frame)['aim_point'])
        self.assertFalse(read_aim_indicators(frame)['gun_direction']['ready'])
        blank = np.zeros_like(rgb)
        cv2.line(blank,(1270,800),(1290,800),(255,255,255),2)
        self.assertIsNone(read_aim_indicators(blank)['aim_point'])
        self.assertFalse(read_aim_indicators(blank)['gun_direction']['ready'])

if __name__=='__main__':
    unittest.main()
