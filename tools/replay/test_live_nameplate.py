"""The live 15.9 plate draws its top edge longer than the bar.

Real frame `main_view_fixtures/live-15.9-nameplate-20261016.png` (2560x1600,
captured 2026-10-16 from the running client) shows an enemy plate whose dark top
edge extends past the bar, so endpoint-matching alone found nothing. The bar is
the interval both edges support, and a thin red line on water is still not one.
"""
import json
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import main_view_perception as M

FIXTURE = Path(__file__).resolve().parent / 'main_view_fixtures' / 'live-15.9-nameplate-20261016.png'
# A 3.2 km live plate whose fill is 120x17: too wide for the compact numeric
# rule and without the paired top/bottom edges the broad rule wants.
WIDE_FIXTURE = (Path(__file__).resolve().parent / 'main_view_fixtures'
                / 'live-15.9-wide-numeric-bar-20261017.png')


def load_fixture():
    import cv2
    image = cv2.imread(str(FIXTURE))
    if image is None:
        raise unittest.SkipTest('fixture missing')
    return np.ascontiguousarray(image[:, :, ::-1]), image


class LiveNameplateTests(unittest.TestCase):
    def test_a_live_bar_under_a_longer_top_edge_is_found(self):
        rgb, _ = load_fixture()
        found = M.detect_nameplates(rgb)
        self.assertTrue(found, 'the live enemy health bar must be detected')
        bar = sorted(found, key=lambda row: row['red_pixels'], reverse=True)[0]['nameplate_bbox']
        self.assertGreaterEqual(bar[3], 8)
        self.assertLessEqual(bar[3], 26)
        self.assertGreaterEqual(bar[2], 48)

    def test_the_detected_bar_still_needs_both_dark_edges(self):
        """Removing one edge must remove the detection, whatever the pairing allows."""
        rgb, bgr = load_fixture()
        baseline = M.detect_nameplates(rgb)
        self.assertTrue(baseline)
        bar = sorted(baseline, key=lambda row: row['red_pixels'], reverse=True)[0]['nameplate_bbox']
        x, y, w, h = bar
        edited = rgb.copy()
        edited[y + h - 2:y + h + 3, max(0, x - 6):x + w + 420] = 90    # erase the bottom edge
        gone = [row for row in M.detect_nameplates(edited)
                if M._rect_intersects(M.Rect(*row['nameplate_bbox']), M.Rect(x, y, w, h))]
        self.assertEqual(gone, [], 'a bar without its lower edge is not evidence')

    def test_a_wide_numeric_live_bar_is_found(self):
        """Measured 2026-10-17: a 120x17 live fill passed neither width rule.

        The compact path stops at a 2.4 width ratio and the broad path needs
        paired dark edges, so this plate produced zero candidates while it was
        plainly drawn on screen. The numeric backing proves it instead.
        """
        import cv2
        image = cv2.imread(str(WIDE_FIXTURE))
        if image is None:
            self.skipTest('fixture missing')
        rgb = np.ascontiguousarray(image[:, :, ::-1])
        found = M.detect_nameplates(rgb)
        wide = [row for row in found
                if M._rect_intersects(M.Rect(*row['nameplate_bbox']), M.Rect(1598, 712, 124, 16))]
        self.assertEqual(len(wide), 1, found)
        self.assertGreaterEqual(wide[0]['confidence'], .9)

    def test_the_wide_bar_still_needs_its_complete_backing(self):
        import cv2
        image = cv2.imread(str(WIDE_FIXTURE))
        if image is None:
            self.skipTest('fixture missing')
        rgb = np.ascontiguousarray(image[:, :, ::-1])
        edited = rgb.copy()
        edited[726:733, 1590:1728] = 200      # erase the numeric bar's lower edge
        gone = [row for row in M.detect_nameplates(edited)
                if M._rect_intersects(M.Rect(*row['nameplate_bbox']), M.Rect(1598, 712, 124, 16))]
        self.assertEqual(gone, [], 'the numeric rule must still require a complete backing')

    def test_a_thin_red_line_on_water_is_not_a_healthbar(self):
        frame = np.full((M.FRAME_H, M.FRAME_W, 3), 90, np.uint8)
        frame[800:802, 400:900] = (220, 40, 40)
        self.assertEqual(M.detect_nameplates(frame), [])

    def test_bright_corner_scope_uses_current_scope_evidence(self):
        from PIL import Image
        source = Path(__file__).resolve().parent / 'main_view_fixtures/bright-corner-scoped-healthbar-20261018.png'
        rgb = np.asarray(Image.open(source).convert('RGB'))
        target = M.Rect(1204,859,112,15)
        self.assertFalse(any(M._rect_intersects(M.Rect(*p['nameplate_bbox']),target)
                             for p in M.detect_nameplates(rgb)))
        plates = M.detect_nameplates(rgb, scope_observed=True)
        self.assertTrue(any(M._rect_intersects(M.Rect(*p['nameplate_bbox']),target)
                            for p in plates))
        edited = rgb.copy()
        edited[855:878,1198:1322] = (130,160,170)
        self.assertFalse(any(M._rect_intersects(M.Rect(*p['nameplate_bbox']),target)
                             for p in M.detect_nameplates(edited, scope_observed=True)))


if __name__ == '__main__':
    unittest.main()
