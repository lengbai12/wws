import sys
from pathlib import Path
import unittest

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fire_hud import FireHudReader, NativeNumericReader, ROOT, GUI, RELOAD_ROIS


def blank():
    return np.full((1600, 2560, 3), 40, np.uint8)


def text(frame, value, xy, size=20):
    image = Image.fromarray(frame)
    font = ImageFont.truetype(str(GUI / "fonts/Warhelios_Regular.ttf"), size)
    ImageDraw.Draw(image).text(xy, value, font=font, fill=(210, 210, 210))
    frame[:] = np.asarray(image)


def composite(frame, rgba, xy):
    x, y = xy; h, w = rgba.shape[:2]
    alpha = rgba[..., 3:4].astype(np.float32)/255
    frame[y:y+h, x:x+w] = (rgba[..., :3]*alpha + frame[y:y+h, x:x+w]*(1-alpha)).astype(np.uint8)


class FireHudTests(unittest.TestCase):
    def test_scoped_sea_and_consumables_do_not_steal_ammo_selection(self):
        frame = np.asarray(Image.open(Path(__file__).with_name('main_view_fixtures') /
                           'scoped-sea-selection-20261019.png').convert('RGB'))
        result = self.reader.detect(frame)
        self.assertEqual(result['weapon']['kind'], 'gun')
        self.assertEqual(result['weapon']['ammo'], 'HE')
        self.assertEqual(result['aiming_view']['mode'], 'scope_ruler')
        erased = frame.copy()
        erased[1395:1510, 950:1138] = 40
        self.assertIsNone(self.reader.detect(erased)['weapon']['kind'])

    def test_scaled_native_distance_suffix_requires_unit_and_numeric_strokes(self):
        frame = np.asarray(Image.open(Path(__file__).with_name('main_view_fixtures') /
                           'flight-range-5p3-20261019.png').convert('RGB'))
        distance = self.reader.detect(frame)['flight_time']['distance']
        self.assertEqual(distance['km'], 5.3)
        self.assertEqual(distance['raw']['numbers'][0]['raw']['unit'], 'km')
        for region in ((1361, 819, 27, 22), (1332, 819, 26, 22),
                       (1348, 819, 8, 22), (1303, 815, 29, 29)):
            with self.subTest(region=region):
                erased = frame.copy()
                x, y, w, h = region
                erased[y:y+h, x:x+w] = 90
                self.assertIsNone(self.reader.detect(erased)['flight_time']['distance']['km'])

    def test_real_four_thirds_readouts_require_anchors_units_and_digits(self):
        frame = np.asarray(Image.open(Path(__file__).with_name('main_view_fixtures') /
                           'flight-range-4over3-20261019.png').convert('RGB'))
        result = self.reader.detect(frame)['flight_time']
        self.assertEqual(result['seconds'],3.9)
        self.assertEqual(result['distance']['km'],7.2)
        for field, region in (('seconds',(1217,820,14,21)),('distance',(1361,820,26,21))):
            erased = frame.copy()
            x,y,w,h = region
            erased[y:y+h,x:x+w] = 90
            current = self.reader.detect(erased)['flight_time']
            self.assertIsNone(current['seconds'] if field=='seconds' else current['distance']['km'])
        no_digits = frame.copy()
        no_digits[819:842,1186:1216] = 90
        no_digits[819:842,1332:1359] = 90
        result = self.reader.detect(no_digits)['flight_time']
        self.assertIsNone(result['seconds'])
        self.assertIsNone(result['distance']['km'])

    def test_native_distance_background_recovery_requires_every_readout_part(self):
        frame = np.asarray(Image.open(Path(__file__).with_name('main_view_fixtures') /
                           'flight-range-background-5p8-20261019.png').convert('RGB'))
        distance = self.reader.detect(frame)['flight_time']['distance']
        self.assertEqual(distance['km'], 5.8)
        self.assertEqual(distance['anchor']['raw']['method'], 'native_anchor_median_background')
        self.assertEqual(distance['raw']['numbers'][0]['raw']['unit'], 'km')
        self.assertGreaterEqual(distance['anchor']['raw']['native_contrast'], 16)
        for region in ((1303, 815, 29, 29), (1332, 819, 26, 22),
                       (1348, 819, 10, 22), (1361, 819, 27, 22)):
            with self.subTest(region=region):
                erased = frame.copy()
                x, y, w, h = region
                erased[y:y+h, x:x+w] = 90
                self.assertIsNone(self.reader.detect(erased)['flight_time']['distance']['km'])
        background = frame.copy()
        background[808:849, 1298:1395] = frame[858:899, 1298:1395]
        self.assertIsNone(self.reader.detect(background)['flight_time']['distance']['km'])

    def test_ship_background_does_not_invent_visible_but_unresolved_digits(self):
        frame = np.asarray(Image.open(Path(__file__).with_name('main_view_fixtures') /
                           'flight-range-background-unresolved-20261019.png').convert('RGB'))
        result = self.reader.detect(frame)['flight_time']
        self.assertIsNone(result['seconds'])
        self.assertIsNone(result['distance']['km'])
        self.assertIsNotNone(result['distance']['anchor'])
        self.assertEqual(result['distance']['raw']['reason'], 'anchored_number_unresolved')

    @classmethod
    def setUpClass(cls):
        cls.reader = FireHudReader()

    def test_input_contract_accepts_any_resolution_and_rejects_bad_pixels(self):
        # HUD rectangles come from the projection of the frame that was handed
        # in, so a smaller client is read with its own boxes instead of being
        # refused. The pixel contract itself is unchanged.
        for shape in ((800, 1280, 3), (1080, 1920, 3), (1058, 1924, 3)):
            for method in (self.reader.detect, self.reader.detect_battle):
                with self.subTest(method=method.__name__, shape=shape):
                    method(np.zeros(shape, np.uint8))
        for a in (np.zeros((1600, 2560, 4), np.uint8),
                  np.zeros((1600, 2560, 3), np.float32),
                  np.zeros((1600, 2560), np.uint8),
                  np.zeros((64, 64, 3), np.uint8)):
            for method in (self.reader.detect, self.reader.detect_battle):
                with self.subTest(method=method.__name__, shape=a.shape, dtype=a.dtype):
                    with self.assertRaises(ValueError):
                        method(a)

    def test_blank_does_not_imply_reload_ready(self):
        result = self.reader.detect(blank())
        self.assertIsNone(result["reload"]["ready"])
        self.assertIsNone(result["reload"]["all_ready"])
        self.assertTrue(all(t["ready"] is None for t in result["reload"]["turrets"]))
        self.assertIsNone(result["weapon"]["kind"])
        self.assertIsNone(result["flight_time"]["seconds"])
        self.assertEqual("unknown", result["reticle"]["status"])

    def test_all_readouts_have_evidence_contract(self):
        result = self.reader.detect(blank())
        for field in list(result.values()) + result["reload"]["turrets"] + [result["flight_time"]["distance"]]:
            for name in ("bbox", "raw", "confidence", "status"):
                self.assertIn(name, field)
            self.assertIsInstance(field["raw"], dict)
            self.assertTrue(np.isfinite(field["confidence"]))

    def test_reload_ocr_reports_distinct_numbers_and_absence(self):
        a = blank()
        for roi, value in zip(RELOAD_ROIS[:2], ("20", "7")):
            text(a, value, (roi[0]+4, roi[1]-1))
        result = self.reader.detect(a)["reload"]
        self.assertEqual([20.0, 7.0, None, None], [t["remaining_s"] for t in result["turrets"]])
        self.assertEqual([False, False, None, None], [t["ready"] for t in result["turrets"]])
        self.assertIsNone(result["ready"])

    def test_numeric_reader_rejects_letters_and_out_of_range(self):
        for value in ("X", "20X", "999"):
            a = blank(); text(a, value, (1000, 701))
            result = self.reader.numeric.read(a, (998, 700, 65, 30), maximum=180)
            self.assertIsNone(result["value"], value)
            self.assertEqual("unknown", result["status"], value)

    def test_decimal_numeric_shape_readout(self):
        a = blank(); text(a, "8.5", (1000, 701))
        result = self.reader.numeric.read(a, (998, 700, 65, 30), decimal=True)
        self.assertEqual(8.5, result["value"])
        self.assertEqual("8.5", result["text"])

    def test_digits_without_native_time_anchor_are_unknown(self):
        a = blank(); text(a, "8.5", (1021, 703))
        result = self.reader.detect(a)["flight_time"]
        self.assertIsNone(result["seconds"])
        self.assertEqual("native_anchor_absent", result["raw"]["reason"])

    def test_native_time_anchor_requires_actual_digits(self):
        a = blank(); composite(a, self.reader.anchors["time"], (1000, 700))
        result = self.reader.detect(a)["flight_time"]
        self.assertIsNotNone(result["anchor"])
        self.assertIsNone(result["seconds"])
        self.assertEqual("unknown", result["status"])
        text(a, "8.5", (1021, 701))
        result = self.reader.detect(a)["flight_time"]
        self.assertEqual(8.5, result["seconds"])
        self.assertEqual("observed", result["status"])

    def test_distance_anchor_does_not_supply_flight_time(self):
        a = blank(); composite(a, self.reader.anchors["distance"], (1000, 750))
        text(a, "12.4", (1021, 751))
        result = self.reader.detect(a)["flight_time"]
        self.assertIsNone(result["seconds"])
        self.assertEqual(12.4, result["distance"]["km"])

    def test_two_time_anchors_are_ambiguous(self):
        a = blank()
        for xy in ((1000, 700), (1100, 700)):
            composite(a, self.reader.anchors["time"], xy)
        result = self.reader.detect(a)["flight_time"]
        self.assertIsNone(result["seconds"])
        self.assertEqual("ambiguous_native_anchors", result["raw"]["reason"])

    def test_green_bars_alone_do_not_make_unknown_weapon_ready(self):
        a = blank()
        for x, _, _, _ in RELOAD_ROIS:
            a[1365:1371, x-20:x+52] = (20, 210, 140)
        result = self.reader.detect(a)["reload"]
        self.assertTrue(all(t["raw"]["full_green_bar"] for t in result["turrets"]))
        self.assertTrue(all(t["ready"] is None for t in result["turrets"]))

    def test_explicit_selected_gun_and_green_bars_report_ready(self):
        a = blank(); composite(a, self.reader.icons["ap"], (1041, 1412))
        a[1400:1505, 1025:1028] = (20, 210, 140)
        a[1400:1505, 1105:1108] = (20, 210, 140)
        for x, _, _, _ in RELOAD_ROIS:
            a[1365:1371, x-20:x+52] = (20, 210, 140)
        result = self.reader.detect(a)
        self.assertEqual("gun", result["weapon"]["kind"])
        self.assertTrue(result["reload"]["ready"])
        self.assertTrue(result["reload"]["all_ready"])

    def test_visible_reload_digits_override_green_bars(self):
        a = blank(); composite(a, self.reader.icons["ap"], (1041, 1412))
        a[1400:1505, 1025:1028] = (20, 210, 140)
        a[1400:1505, 1105:1108] = (20, 210, 140)
        x, y, _, _ = RELOAD_ROIS[0]
        a[1365:1371, x-20:x+52] = (20, 210, 140)
        text(a, "20", (x+4, y-1))
        result = self.reader.detect(a)["reload"]["turrets"][0]
        self.assertFalse(result["ready"])
        self.assertEqual(20.0, result["remaining_s"])

    def test_selected_torpedo_cannot_supply_gun_readiness(self):
        a = blank(); composite(a, self.reader.icons["torpedo"], (1175, 1406))
        a[1400:1505, 1172:1175] = (20, 210, 140)
        a[1400:1505, 1255:1258] = (20, 210, 140)
        for x, _, _, _ in RELOAD_ROIS:
            a[1365:1371, x-20:x+52] = (20, 210, 140)
        result = self.reader.detect(a)
        self.assertEqual("torpedo", result["weapon"]["kind"])
        self.assertTrue(all(t["ready"] is None for t in result["reload"]["turrets"]))

    def test_missing_assets_remain_unknown(self):
        reader = FireHudReader(gui_root=ROOT / "state/no-such-fire-hud-assets", reference_root=ROOT / "state/no-such-fire-hud-assets")
        result = reader.detect(blank())
        self.assertIsNone(result["flight_time"]["seconds"])
        self.assertIsNone(result["weapon"]["ammo"])
        self.assertIsNone(result["reload"]["ready"])

    def test_actual_frame_twenty_second_reload_and_empty_turrets(self):
        image = np.asarray(Image.open(ROOT / "state/tracking-mainview-236374.png").convert("RGB"))
        result = self.reader.detect(image)
        self.assertEqual([20.0, 20.0, None, None], [t["remaining_s"] for t in result["reload"]["turrets"]])
        self.assertEqual([False, False, False, None], [t["ready"] for t in result["reload"]["turrets"]])
        self.assertEqual("gun", result["weapon"]["kind"])
        self.assertEqual("AP", result["weapon"]["ammo"])
        self.assertEqual("observed", result["reticle"]["status"])
        self.assertIsNone(result["flight_time"]["seconds"])

    def test_actual_shifted_selectors_three_mounts_and_alt_readouts(self):
        image = np.asarray(Image.open(ROOT / "state/mainview-numeric-bar-fixed-20261015/frame-000-394.000s.ppm").convert("RGB"))
        result = self.reader.detect(image)
        self.assertEqual("AP", result["weapon"]["ammo"])
        self.assertEqual("observed", result["reload"]["raw"]["layout_status"])
        self.assertEqual([True, True, False], [t["ready"] for t in result["reload"]["turrets"]])
        self.assertEqual([108, 108, 108], [t["raw"]["bar_bbox"][2] for t in result["reload"]["turrets"]])
        # Every mount is now a known state (two green, one traversing), so
        # all_ready is a real False rather than unknown -- and ready is still
        # True because a single green mount is enough to fire.
        self.assertIs(result["reload"]["all_ready"], False)
        self.assertIs(result["reload"]["ready"], True)
        self.assertEqual(2.2, result["flight_time"]["seconds"])
        self.assertEqual(5.3, result["flight_time"]["distance"]["km"])
        self.assertEqual("observed", result["reticle"]["status"])
        self.assertEqual("none", result["reticle"]["raw"]["lock_semantics"])

    def test_actual_sixty_pixel_green_bar_is_ready_without_countdown(self):
        image = np.asarray(Image.open(ROOT / "state/aimprobe-axes-v3-restored-playback.png").convert("RGB"))
        result = self.reader.detect(image)
        self.assertEqual("AP", result["weapon"]["ammo"])
        self.assertEqual([19.0, 20.0, None, None], [t["remaining_s"] for t in result["reload"]["turrets"]])
        self.assertEqual([False, False, False, True], [t["ready"] for t in result["reload"]["turrets"]])
        self.assertIsNone(result["flight_time"]["seconds"])

    def test_seconds_on_both_sides_of_anchor_are_ambiguous(self):
        a = blank(); composite(a, self.reader.anchors["time"], (1200, 820))
        text(a, "2.2s", (1154, 818))
        text(a, "8.5", (1221, 818))
        result = self.reader._anchored_number(a, "time")
        self.assertIsNone(result["seconds"])
        self.assertEqual("ambiguous_anchored_numbers", result["raw"]["reason"])

    def test_numeric_units_must_match_the_requested_field(self):
        a = blank(); text(a, "2.2s", (1000, 701))
        observed = self.reader.numeric.read(a, (998, 700, 65, 30), decimal=True, unit="s")
        self.assertEqual(2.2, observed["value"])
        self.assertEqual("s", observed["raw"]["unit"])
        wrong = self.reader.numeric.read(a, (998, 700, 65, 30), decimal=True, unit="km")
        self.assertIsNone(wrong["value"])

    def test_battle_requires_ammo_layout_and_independent_hud_evidence(self):
        a = blank()
        composite(a, self.reader.icons['ap'], (1041, 1412))
        composite(a, self.reader.icons['aim_caret'], (1269, 796))
        self.assertIsNone(self.reader.detect(a)['battle']['active'])
        composite(a, self.reader.icons['he'], (945, 1412))
        result = self.reader.detect(a)
        self.assertTrue(result['battle']['active'])
        self.assertIsNone(result['weapon']['kind'])
        a[750:850, 1200:1350] = 40
        self.assertIsNone(self.reader.detect(a)['battle']['active'])

    def test_gun_and_torpedo_need_same_layout_and_independent_battle_evidence(self):
        a = blank()
        composite(a, self.reader.icons["he"], (945, 1412))
        composite(a, self.reader.icons["torpedo"], (1175, 1406))
        self.assertIsNone(self.reader.detect(a)["battle"]["active"])
        composite(a, self.reader.icons["aim_caret"], (1269, 796))
        self.assertTrue(self.reader.detect(a)["battle"]["active"])
        a[1395:1510, 1170:1260] = 40
        self.assertIsNone(self.reader.detect(a)["battle"]["active"])
        # The shifted torpedo slot cannot complete an unshifted HE layout.
        composite(a, self.reader.icons["torpedo"], (1131, 1406))
        self.assertIsNone(self.reader.detect(a)["battle"]["active"])

    def test_selected_depth_charge_confirms_battle_not_gun_readiness(self):
        for name in ('frame-000.png', 'frame-001.png'):
            with self.subTest(frame=name):
                image = np.asarray(Image.open(ROOT /
                    'state/mainview-resume-live-20261016-s6' / name).convert('RGB'))
                result = self.reader.detect(image)
                self.assertTrue(result['battle']['active'])
                self.assertEqual([-44], result['battle']['raw'][
                    'selected_depth_charge_layout_offsets_px'])
                self.assertIsNone(result['weapon']['kind'])
                self.assertIsNone(result['reload']['ready'])
                self.assertIsNone(result['aiming_view']['mode'])
                for region in ((1215, 1395, 12, 115), (1293, 1395, 12, 115),
                               (1228, 1410, 65, 65), (896, 1395, 320, 115)):
                    with self.subTest(missing=region):
                        damaged = image.copy()
                        x, y, w, h = region
                        damaged[y:y+h, x:x+w] = 40
                        self.assertIsNone(self.reader.detect(damaged)['battle']['active'])

    def test_cooling_depth_charge_preserves_battle_not_weapon_readiness(self):
        for name in ('frame-000.png', 'frame-010.png'):
            with self.subTest(frame=name):
                image = np.asarray(Image.open(ROOT /
                    'state/mainview-target-online-20261016-r3' / name).convert('RGB'))
                result = self.reader.detect(image)
                self.assertTrue(result['battle']['active'])
                self.assertIsNone(result['weapon']['kind'])
                self.assertIsNone(result['reload']['ready'])
                self.assertIsNone(result['aiming_view']['mode'])
                for region in ((1228, 1410, 10, 24), (1283, 1410, 10, 24),
                               (1215, 1395, 12, 115), (1293, 1395, 12, 115),
                               (896, 1395, 320, 115)):
                    with self.subTest(missing=region):
                        damaged = image.copy()
                        x, y, w, h = region
                        damaged[y:y+h, x:x+w] = 40
                        self.assertIsNone(self.reader.detect(damaged)['battle']['active'])
                # The two native wings must describe the same icon origin.
                displaced = image.copy()
                displaced[1410:1434, 1283:1293] = 40
                displaced[1438:1462, 1283:1293] = image[1410:1434, 1283:1293]
                self.assertIsNone(self.reader.detect(displaced)['battle']['active'])

    def test_depth_charge_requires_same_ammo_layout_and_contiguous_outline(self):
        image = blank()
        original = np.asarray(Image.open(ROOT /
            'state/mainview-resume-live-20261016-s6/frame-000.png').convert('RGB'))
        image[1395:1510, 1215:1305] = original[1395:1510, 1215:1305]
        composite(image, self.reader.icons['he'], (945, 1412))
        composite(image, self.reader.icons['ap'], (1041, 1412))
        self.assertIsNone(self.reader.detect(image)['battle']['active'])
        image[1395:1510, 896:1112] = 40
        composite(image, self.reader.icons['he'], (909, 1412))
        composite(image, self.reader.icons['ap'], (997, 1412))
        self.assertTrue(self.reader.detect(image)['battle']['active'])
        image[1442:1447, 1215:1227] = 40
        self.assertIsNone(self.reader.detect(image)['battle']['active'])

    def test_actual_amber_mounts_confirm_battle_without_digits_or_caret(self):
        path = ROOT / 'state/mainview-hotspots-guns-receiver-resumed-20261016/frame-001.png'
        image = np.asarray(Image.open(path).convert('RGB'))
        result = self.reader.detect(image)
        self.assertEqual('unknown', result['reticle']['status'])
        self.assertTrue(result['reload']['raw']['amber_mount_layout_observed'])
        self.assertTrue(result['battle']['active'])
        # Operator rule: an amber mount is traversing, and no green mount means
        # the guns cannot fire yet -- a definite False, not an unknown.
        self.assertIs(result['reload']['ready'], False)
        # Losing a whole mount must remove this independent battle evidence.
        missing = image.copy()
        missing[1361:1372, 1226:1338] = 40
        self.assertIsNone(self.reader.detect(missing)['battle']['active'])
        # Mount layout alone is insufficient without two native ammo sprites.
        missing = image.copy()
        missing[1395:1510, 895:1260] = 40
        self.assertIsNone(self.reader.detect(missing)['battle']['active'])

    def test_actual_partial_amber_mounts_preserve_battle_without_caret(self):
        image = np.asarray(Image.open(ROOT /
            'state/mainview-live-enemy-receiver-20261016-r2/frame-017.png').convert('RGB'))
        result = self.reader.detect(image)
        self.assertEqual('unknown', result['reticle']['status'])
        self.assertTrue(result['battle']['active'])
        self.assertEqual(3, len(result['reload']['turrets']))
        # Operator rule: an amber mount is traversing, so readiness is a known
        # False here rather than an unreadable None.
        self.assertIs(result['reload']['ready'], False)
        self.assertTrue(all(t['ready'] is False for t in result['reload']['turrets']))
        for start in (1084, 1228, 1372):
            for kind in ('missing', 'no_cap', 'broken'):
                with self.subTest(start=start, kind=kind):
                    damaged = image.copy()
                    if kind == 'missing':
                        damaged[1361:1372, start-2:start+110] = 40
                    elif kind == 'no_cap':
                        damaged[1365:1367, start-1:start+2] = 40
                    else:
                        damaged[1367:1371, start+10:start+14] = 40
                    self.assertIsNone(self.reader.detect(damaged)['battle']['active'])
        no_ammo = image.copy()
        no_ammo[1395:1510, 896:1260] = 40
        self.assertIsNone(self.reader.detect(no_ammo)['battle']['active'])

    def test_partial_mount_layouts_never_imply_readiness(self):
        for count, start, stride, width in ((3, 1084, 144, 108), (4, 1108, 96, 60)):
            with self.subTest(count=count):
                image = blank()
                composite(image, self.reader.icons['he'], (945, 1412))
                composite(image, self.reader.icons['ap'], (1041, 1412))
                for i in range(count):
                    x = start+i*stride
                    image[1367:1371, x:x+width//2] = (230, 150, 30)
                    image[1365:1367, x:x+2] = (230, 150, 30)
                result = self.reader.detect(image)
                self.assertTrue(result['battle']['active'])
                self.assertEqual(count, len(result['reload']['turrets']))
                self.assertIsNone(result['reload']['ready'])

    def raised_mount_frame(self):
        # Native RGB crop [880,1320,650,195], unchanged frame-017 from
        # state/target-receiver-resume-20261016-r5; fourth slot selected.
        image = blank()
        crop = np.asarray(Image.open(Path(__file__).with_name('fire_hud_assets') /
                                     'raised-mount-layout.png').convert('RGB'))
        image[1320:1515, 880:1530] = crop
        return image

    def test_actual_raised_mount_layout_preserves_battle_not_readiness(self):
        result = self.reader.detect(self.raised_mount_frame())
        self.assertTrue(result['battle']['active'])
        self.assertEqual(3, len(result['reload']['turrets']))
        self.assertEqual([1349]*3, [t['raw']['bar_bbox'][1]
                                   for t in result['reload']['turrets']])
        self.assertIsNone(result['weapon']['kind'])
        self.assertTrue(all(t['ready'] is False for t in result['reload']['turrets']))
        self.assertIs(result['reload']['all_ready'], False)

    def test_raised_layout_requires_every_mount_endcap_and_ammo(self):
        for start in (1084, 1228, 1372):
            for kind in ('missing', 'no_cap', 'broken'):
                with self.subTest(start=start, kind=kind):
                    image = self.raised_mount_frame()
                    if kind == 'missing':
                        image[1345:1360, start-2:start+110] = 40
                    elif kind == 'no_cap':
                        image[1345:1352, start-2:start+110] = 40
                    else:
                        image[1352:1359, start+10:start+25] = 40
                    self.assertIsNone(self.reader.detect(image)['battle']['active'])
        image = self.raised_mount_frame()
        image[1395:1510, 896:1260] = 40
        self.assertIsNone(self.reader.detect(image)['battle']['active'])

    def test_mounts_at_different_heights_do_not_form_one_layout(self):
        image = self.raised_mount_frame()
        patch = image[1345:1360, 1226:1338].copy()
        image[1345:1360, 1226:1338] = 40
        image[1361:1376, 1226:1338] = patch
        self.assertIsNone(self.reader.detect(image)['battle']['active'])

    def test_flat_amber_stripes_without_endcaps_do_not_confirm_battle(self):
        image = blank()
        composite(image, self.reader.icons['he'], (945, 1412))
        composite(image, self.reader.icons['ap'], (1041, 1412))
        for x in (1084, 1228, 1372):
            image[1367:1370, x:x+108] = (230, 150, 30)
        self.assertIsNone(self.reader.detect(image)['battle']['active'])

    def test_actual_settlement_has_no_supported_ammo_layout(self):
        path = Path(__file__).with_name("main_view_fixtures") / "post-battle.png"
        image = np.asarray(Image.open(path).convert("RGB"))
        result = self.reader.detect(image)
        self.assertEqual("unknown", result["battle"]["status"])
        self.assertIsNone(result["battle"]["active"])
        self.assertEqual([], result["battle"]["raw"]["ammo_layout_offsets_px"])
        self.assertEqual("battle_hud_unconfirmed", result["battle"]["raw"]["reason"])

    def test_battle_only_matches_complete_evidence_on_actual_frames(self):
        fixtures = (
            (ROOT / "state/tracking-mainview-236374.png", True),
            (ROOT / "state/mainview-numeric-bar-fixed-20261015/frame-000-394.000s.ppm", True),
            (Path(__file__).with_name("main_view_fixtures") / "post-battle.png", None),
        )
        for path, active in fixtures:
            with self.subTest(frame=str(path)):
                image = np.asarray(Image.open(path).convert("RGB"))
                battle = self.reader.detect_battle(image)
                self.assertIs(battle["active"], active)
                self.assertEqual("observed" if active else "unknown", battle["status"])
                self.assertEqual(self.reader.detect(image)["battle"], battle)

    def test_real_scope_ruler_is_observed_without_inventing_zoom(self):
        image = np.asarray(Image.open(Path(__file__).with_name('main_view_fixtures') /
                                     'bright-corner-scoped-healthbar-20261018.png').convert('RGB'))
        result = self.reader.detect(image)
        self.assertEqual('scope_ruler', result['aiming_view']['mode'])
        self.assertEqual('gun', result['weapon']['kind'])
        self.assertIsNone(result['aiming_view']['zoom_confirmed'])
        external = np.asarray(Image.open(ROOT / 'state/aimprobe-small-camera-executed-20261015/before.png').convert('RGB'))
        self.assertIsNone(self.reader.detect(external)['aiming_view']['mode'])

    def test_scope_ruler_rejects_minimap_labels_as_tick_evidence(self):
        path = Path(__file__).with_name('fire_hud_assets') / 'scope-ruler-map-columns.png'
        image = blank()
        image[760:810, 420:2116] = np.asarray(Image.open(path).convert('RGB'))
        context = (dict(kind='gun'), dict(status='observed', center=[1280.5, 805]),
                   dict(active=True))
        result = self.reader._aiming_view(image, *context)
        self.assertEqual('scope_ruler', result['mode'])
        self.assertIsNone(result['zoom_confirmed'])
        # Retain map column labels while removing the visible right ruler.
        image[760:810, 1310:1836] = 40
        self.assertIsNone(self.reader._aiming_view(image, *context)['mode'])

    def test_scope_needs_both_regular_sides_and_selected_gun(self):
        image = blank()
        for x in list(range(500, 1230, 27)) + list(range(1340, 2070, 27)):
            image[778:800, x:x+2] = 240
        weapon = dict(kind='gun')
        reticle = dict(status='observed', center=[1280, 805])
        battle = dict(active=True)
        self.assertEqual('scope_ruler', self.reader._aiming_view(image, weapon, reticle, battle)['mode'])
        self.assertIsNone(self.reader._aiming_view(image, dict(kind='torpedo'), reticle, battle)['mode'])
        self.assertIsNone(self.reader._aiming_view(image, weapon, reticle, dict(active=None))['mode'])
        image[:, 1340:2072] = 40
        self.assertIsNone(self.reader._aiming_view(image, weapon, reticle, battle)['mode'])

    def test_reticle_is_never_lock_confirmation(self):
        a = blank(); composite(a, self.reader.icons["aim_caret"], (1269, 796))
        result = self.reader.detect(a)["reticle"]
        self.assertEqual("observed", result["status"])
        self.assertEqual("none", result["raw"]["lock_semantics"])
        self.assertNotIn("locked", result)


if __name__ == "__main__":
    unittest.main()


class ReadBoundsTests(unittest.TestCase):
    """A box outside the frame must be refused, not crash OpenCV."""

    def setUp(self):
        from fire_hud import FireHudReader
        self.reader = FireHudReader()

    def test_a_box_past_the_frame_edge_is_unknown(self):
        frame = np.zeros((1600, 2560, 3), np.uint8)
        # Fully outside: a merely clipped box still yields pixels and is read.
        got = self.reader.numeric_distance.read(frame, [3000, 2000, 40, 40],
                                                decimal=True, maximum=50, unit='km')
        self.assertEqual(got['status'], 'unknown')
        self.assertEqual(got['raw']['reason'], 'bbox_outside_frame')

    def test_a_negative_origin_box_is_unknown(self):
        frame = np.zeros((1600, 2560, 3), np.uint8)
        got = self.reader.numeric_distance.read(frame, [-20, -20, 10, 10],
                                                decimal=True, maximum=50)
        self.assertEqual(got['status'], 'unknown')


class LinhaiLayoutTests(unittest.TestCase):
    """X Linhai's ammo pair sits at offset -85; the gate must depend on it."""

    FRAME = Path(__file__).with_name('main_view_fixtures') / 'linhai-battle-20261020.png'
    BOXES = ((855, 1379, 80, 114), (937, 1379, 90, 114))   # HE and AP at -85

    def setUp(self):
        from fire_hud import FireHudReader
        if not self.FRAME.is_file():
            self.skipTest('linhai fixture missing')
        self.reader = FireHudReader()

    def test_the_live_frame_confirms_battle_at_minus_85(self):
        frame = np.asarray(Image.open(self.FRAME).convert('RGB'))
        raw = self.reader.detect(frame)['battle']['raw']
        self.assertEqual(raw['ammo_layout_offsets_px'], [-85])
        self.assertEqual(raw['reason'], 'native_battle_hud')

    def test_erasing_both_ammo_sprites_closes_the_gate(self):
        frame = np.asarray(Image.open(self.FRAME).convert('RGB')).copy()
        for x, y, w, h in self.BOXES:
            frame[y:y + h, x:x + w] = 90
        raw = self.reader.detect(frame)['battle']['raw']
        self.assertNotIn(-85, raw['ammo_layout_offsets_px'])
        self.assertEqual(raw['reason'], 'battle_hud_unconfirmed')

    def test_one_remaining_pair_still_confirms_but_a_lone_sprite_does_not(self):
        # The gate needs two *distinct* ammo sprites at one offset; this panel
        # draws three (HE, AP, torpedo), so erasing one must keep it open.
        one = np.asarray(Image.open(self.FRAME).convert('RGB')).copy()
        x, y, w, h = self.BOXES[1]
        one[y:y + h, x:x + w] = 90
        self.assertIn(-85, self.reader.detect(one)['battle']['raw']['ammo_layout_offsets_px'])
        # Leave a single sprite: HE and the torpedo sprite erased.
        lone = np.asarray(Image.open(self.FRAME).convert('RGB')).copy()
        for x, y, w, h in (self.BOXES[0], (1085, 1379, 90, 114)):
            lone[y:y + h, x:x + w] = 90
        raw = self.reader.detect(lone)['battle']['raw']
        self.assertNotIn(-85, raw['ammo_layout_offsets_px'])


class LinhaiMountRowTests(unittest.TestCase):
    """X Linhai draws five 60 px mount bars with dim "1".."5" labels."""

    FRAME = Path(__file__).with_name('main_view_fixtures') / 'linhai-battle-20261020.png'
    LABEL_BOXES = tuple((1037 + i * 96, 1347, 20, 22) for i in range(5))
    BAR_BOXES = tuple((1061 + i * 96, 1349, 61, 7) for i in range(5))

    def setUp(self):
        from fire_hud import FireHudReader
        if not self.FRAME.is_file():
            self.skipTest('linhai fixture missing')
        self.reader = FireHudReader()
        self.frame = np.asarray(Image.open(self.FRAME).convert('RGB'))

    def test_the_row_establishes_both_layout_and_readiness(self):
        out = self.reader.detect(self.frame)
        self.assertTrue(out['battle']['raw']['mount_layout_observed'])
        self.assertEqual(out['reload']['status'], 'observed')
        self.assertTrue(out['reload']['ready'])

    def test_erasing_the_bars_leaves_the_layout_unresolved(self):
        # Only one of the five dim labels is readable, so the coloured bars are
        # what establish this row; erasing them must leave it unresolved.
        frame = self.frame.copy()
        for x, y, w, h in self.BAR_BOXES:
            frame[y:y + h, x:x + w] = 80
        out = self.reader.detect(frame)
        self.assertFalse(out['battle']['raw']['mount_layout_observed'])
        self.assertNotEqual(out['reload']['raw']['layout_status'], 'observed')

    def test_erasing_the_labels_alone_keeps_the_row(self):
        frame = self.frame.copy()
        for x, y, w, h in self.LABEL_BOXES:
            frame[y:y + h, x:x + w] = 80
        out = self.reader.detect(frame)
        self.assertTrue(out['battle']['raw']['mount_layout_observed'])

    def test_erasing_every_bar_means_readiness_is_unknown(self):
        frame = self.frame.copy()
        for x, y, w, h in self.BAR_BOXES:
            frame[y:y + h, x:x + w] = 80
        out = self.reader.detect(frame)
        # Labels still prove the layout, but a blank row cannot prove readiness.
        self.assertNotEqual(out['reload'].get('ready'), True)
