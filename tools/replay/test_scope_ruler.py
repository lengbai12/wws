"""The binocular ruler centre is aim evidence only when it is symmetric.

Measured 2026-10-17 on a live armoured-training battle: the scoped frame draws
80 bright stroke columns spanning 705..1853, midpoint 1279 against a 1280 frame
centre. Two third-person frames draw asymmetric shell-fall rulers (midpoints
1342 and 1530), so symmetry and span are the discriminators. The zoom level is
never inferred, and a ruler alone is not battle or weapon evidence.
"""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fire_hud as F

HERE = Path(__file__).resolve().parent
SCOPED = HERE / 'main_view_fixtures' / 'scope-ruler-20261017.png'
THIRD = HERE / 'main_view_fixtures' / 'live-15.9-wide-numeric-bar-20261017.png'
# Byte-for-byte saved native captures: scoped inspect v4 and wide guard v8.
EMPTY_SEA = HERE / 'main_view_fixtures' / 'scope-ruler-empty-sea-20261019.png'
WIDE_SEA = HERE / 'main_view_fixtures' / 'scope-ruler-wide-sea-20261019.png'


def load(path):
    import cv2
    image = cv2.imread(str(path))
    if image is None:
        raise unittest.SkipTest(f'fixture missing: {path.name}')
    return np.ascontiguousarray(image[:, :, ::-1])


def synthetic_ruler(midpoint, span=900, ticks=30):
    frame = np.zeros((1600, 2560, 3), np.uint8)
    frame[500:1100] = 60
    xs = np.linspace(midpoint - span / 2, midpoint + span / 2, ticks).astype(int)
    for x in xs:
        frame[795:820, x:x + 3] = 250
    return frame


class ScopeRulerTests(unittest.TestCase):
    def test_hull_tails_reanchor_only_real_on_grid_baseline_strokes(self):
        rgb = np.zeros((1600, 2560, 3), np.uint8)
        rgb[755:820, 420:1836] = load(HERE / 'main_view_fixtures' /
                                   'scope-ruler-trailing-hull-20261019.png')
        reader = F.FireHudReader()
        reticle = dict(status='observed', center=[1291., 811.])
        result = reader._aiming_view(rgb, dict(kind='gun'), reticle, dict(active=True))
        self.assertEqual(result['status'], 'observed')
        right = result['raw']['sides'][1]
        self.assertEqual(right['ticks'], 16)
        self.assertEqual(right['span_px'], 480)
        self.assertGreaterEqual(right['regular_fraction'], .85)
        xs = [t[0] for t in right['ticks_xy']]
        for x in (1426.5, 1506.5, 1613.):
            self.assertIn(x, xs)
        for x in (1490., 1587., 1609., 1638., 1667., 1694., 1792.5, 1820., 1832.5):
            self.assertNotIn(x, xs)
        for lo, hi in ((420, 1240), (1341, 1836)):
            missing = rgb.copy()
            missing[755:820, lo:hi] = 0
            self.assertEqual(reader._aiming_view(missing, dict(kind='gun'), reticle,
                                                dict(active=True))['status'], 'unknown')
        # Keep the ship clutter but remove ruler ink: grid fitting cannot
        # manufacture the missing side from hull edges.
        clutter = rgb.copy()
        for x in np.arange(1346.5, 1830, 80/3):
            clutter[755:820, int(x)-2:int(x)+3] = 0
        self.assertEqual(reader._aiming_view(clutter, dict(kind='gun'), reticle,
                                            dict(active=True))['status'], 'unknown')

    def test_off_grid_tails_and_edge_strokes_do_not_add_ruler_evidence(self):
        reader = F.FireHudReader()
        reticle = dict(status='observed', center=[1280., 811.])
        rgb = np.zeros((1600, 2560, 3), np.uint8)
        for x in np.arange(480, 1830, 27):
            rgb[776:801, x:x+2] = 250
        reference = reader._aiming_view(rgb, dict(kind='gun'), reticle, dict(active=True))
        self.assertEqual(reference['status'], 'observed')
        # Trailing off-grid hull line, same-baseline end line, and a truncated
        # on-grid stroke above the baseline must all remain rejected.
        rgb[785:814, 1492:1494] = 250
        rgb[794:801, 1810:1812] = 250
        rgb[776:801, 1587:1589] = 0
        rgb[776:792, 1587:1589] = 250
        result = reader._aiming_view(rgb, dict(kind='gun'), reticle, dict(active=True))
        xs = [t[0] for t in result['raw']['sides'][1]['ticks_xy']]
        for x in (1493., 1811., 1588.):
            self.assertNotIn(x, xs)
        self.assertEqual(result['raw']['sides'][1]['ticks'],
                         reference['raw']['sides'][1]['ticks']-1)

    def test_short_hull_stroke_between_regular_ticks_does_not_break_scope(self):
        rgb = np.zeros((1600, 2560, 3), np.uint8)
        patch = load(HERE / 'main_view_fixtures' /
                     'scope-ruler-interstitial-hull-20261019.png')
        rgb[755:820, 420:1836] = patch
        reader = F.FireHudReader()
        reticle = dict(status='observed', center=[1291., 811.])
        result = reader._aiming_view(rgb, dict(kind='gun'), reticle, dict(active=True))
        self.assertEqual(result['status'], 'observed')
        for lo, hi in ((420, 1240), (1341, 1836)):
            missing = rgb.copy()
            missing[755:820, lo:hi] = 0
            self.assertEqual(reader._aiming_view(missing, dict(kind='gun'), reticle,
                                                dict(active=True))['status'], 'unknown')

    def test_a_live_scoped_frame_reports_the_ruler_centre(self):
        ruler = F.scope_ruler_center(load(SCOPED))
        self.assertIsNotNone(ruler, 'the live binocular ruler must be found')
        self.assertLessEqual(abs(ruler['centre_offset_px']), 6)
        self.assertGreaterEqual(ruler['span_px'], 800)
        self.assertGreaterEqual(ruler['ticks'], 20)

    def test_an_asymmetric_third_person_ruler_is_refused(self):
        self.assertIsNone(F.scope_ruler_center(load(THIRD)),
                          'the shell-fall ruler is not a scope caret')

    def test_an_off_centre_synthetic_ruler_is_refused(self):
        self.assertIsNotNone(F.scope_ruler_center(synthetic_ruler(1280)))
        self.assertIsNone(F.scope_ruler_center(synthetic_ruler(1500)))
        self.assertIsNone(F.scope_ruler_center(synthetic_ruler(1280, span=600)))

    def test_the_view_read_records_the_caret_source(self):
        hud = F.FireHudReader().detect(load(SCOPED))
        raw = hud['aiming_view']['raw']
        self.assertIn('scope_ruler_center', raw)
        self.assertEqual(raw['scope_ruler_center']['status'], 'observed')
        # The scoped frame selects the main battery, so the bilateral ruler plus
        # the ruler-centre caret is a complete view read. The zoom stays unknown.
        self.assertEqual(hud['weapon'].get('kind'), 'gun')
        self.assertEqual(hud['aiming_view']['status'], 'observed')
        self.assertEqual(hud['aiming_view']['mode'], 'scope_ruler')
        self.assertIsNone(hud['aiming_view']['zoom_confirmed'])
        self.assertEqual(raw['caret_source'], 'scope_ruler_center')

    def test_real_hull_clutter_is_not_part_of_the_tick_baseline(self):
        from PIL import Image
        rgb = np.zeros((1600, 2560, 3), np.uint8)
        patch = np.asarray(Image.open(HERE / 'main_view_fixtures' /
                           'scope-ruler-hull-clutter-20261018.png').convert('RGB'))
        rgb[755:820, 420:1836] = patch
        reader = F.FireHudReader()
        weapon = dict(kind='gun')
        reticle = dict(status='observed', center=[1291., 811.])
        result = reader._aiming_view(rgb, weapon, reticle, dict(active=True))
        self.assertEqual(result['status'], 'observed')
        self.assertIsNone(result['zoom_confirmed'])
        for lo, hi in ((420, 1240), (1341, 1836)):
            missing = rgb.copy()
            missing[755:820, lo:hi] = 0
            with self.subTest(missing_side=(lo, hi)):
                result = reader._aiming_view(missing, weapon, reticle, dict(active=True))
                self.assertEqual(result['status'], 'unknown')

    def test_native_empty_sea_uses_the_observed_green_ring_and_repaired_ticks(self):
        rgb = load(EMPTY_SEA)
        # Global bright columns include sea highlights at x=492: they cannot
        # replace the strict symmetric-ruler centre evidence.
        self.assertIsNone(F.scope_ruler_center(rgb))
        hud = F.FireHudReader().detect(rgb)
        # Geometry is view-only evidence: it must not grant the native reticle
        # evidence used by the independent battle HUD gate.
        self.assertEqual(hud['reticle']['status'], 'unknown')
        self.assertAlmostEqual(hud['aiming_view']['raw']['caret_center'][0], 1291., places=2)
        self.assertAlmostEqual(hud['aiming_view']['raw']['caret_center'][1], 811.31, places=2)
        self.assertEqual(hud['weapon']['kind'], 'gun')
        self.assertIs(hud['battle']['active'], True)
        view = hud['aiming_view']
        self.assertEqual(view['status'], 'observed')
        self.assertEqual(view['mode'], 'scope_ruler')
        self.assertEqual(view['raw']['caret_source'], 'green_ring')
        self.assertIsNone(view['zoom_confirmed'])
        for side in view['raw']['sides']:
            self.assertTrue(side['accepted'])
            self.assertGreaterEqual(side['ticks'], 12)
            self.assertGreaterEqual(side['span_px'], 450)
            self.assertGreaterEqual(side['regular_fraction'], .85)

    def test_native_wide_sea_with_a_green_ring_has_no_scope_evidence(self):
        rgb = load(WIDE_SEA)
        self.assertIsNotNone(F.green_reticle(rgb))
        self.assertIsNone(F.scope_ruler_center(rgb))
        hud = F.FireHudReader().detect(rgb)
        self.assertEqual(hud['aiming_view']['status'], 'unknown')

    def test_native_empty_sea_requires_each_side_even_with_the_ring(self):
        rgb = load(EMPTY_SEA)
        reader = F.FireHudReader()
        ring = F.green_reticle(rgb)
        reticle = dict(status='observed', center=ring['centre'])
        for bands in (((431, 1242),), ((1341, 1836),),
                      ((431, 1242), (1341, 1836))):
            missing = rgb.copy()
            for lo, hi in bands:
                missing[755:817, lo:hi] = 0
            with self.subTest(missing_sides=bands):
                self.assertIsNotNone(F.green_reticle(missing))
                result = reader._aiming_view(missing, dict(kind='gun'), reticle,
                                             dict(active=True))
                self.assertEqual(result['status'], 'unknown')

    def test_native_empty_sea_still_requires_gun_and_battle_evidence(self):
        rgb = load(EMPTY_SEA)
        reader = F.FireHudReader()
        reticle = dict(status='observed', center=F.green_reticle(rgb)['centre'])
        for kind, active in (('torpedo', True), (None, True), ('gun', None), ('gun', False)):
            with self.subTest(kind=kind, active=active):
                result = reader._aiming_view(rgb, dict(kind=kind), reticle, dict(active=active))
                self.assertEqual(result['status'], 'unknown')

    def test_fragment_repair_requires_an_uninterrupted_vertical_seed(self):
        reader = F.FireHudReader()
        reticle = dict(status='observed', center=[1280., 811.])
        for has_seed in (True, False):
            rgb = np.zeros((1600, 2560, 3), np.uint8)
            for x in np.arange(480, 1830, 27):
                if has_seed:
                    rgb[776:801, x:x+2] = 250
                    rgb[788:790, x:x+2] = 0
                else:
                    # Closing connects these three-pixel fragments, but none
                    # supplies the seven-pixel seed required of a real tick.
                    for y in range(776, 801, 5):
                        rgb[y:y+3, x:x+2] = 250
            with self.subTest(has_seed=has_seed):
                result = reader._aiming_view(rgb, dict(kind='gun'), reticle, dict(active=True))
                self.assertEqual(result['status'], 'observed' if has_seed else 'unknown')



if __name__ == '__main__':
    unittest.main()
