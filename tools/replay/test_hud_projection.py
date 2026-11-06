"""Tests for the reference <-> frame HUD resolution map.

Reference space is the 2560x1440 game surface inside the 2560x1600 capture
(the 80 px letterbox is not HUD space). These tests pin the four properties the
readers rely on:

1. on a reference-size frame the projection is the identity, so every readout is
   the same field-for-field as the pre-projection reader;
2. real frames resampled to 1924x1058 and 1920x1080 still land their ROIs on the
   same HUD elements (cropped-pixel agreement plus recognizer hits, reported);
3. the letterbox is measured from the pixels: 80 px bars, 96 px bars and a frame
   with no bars all give the right surface rectangle;
4. a rectangle that leaves the frame is clamped and recorded, never silently
   truncated.

The resampled frames exist only to build samples; the reader path under test is
the coordinate map, not a resized image.
"""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hud_projection as hp
import fire_hud

FIXTURES = HERE / "main_view_fixtures"
REFERENCE_FRAME_ASSETS = (
    "scoped-sea-selection-20261019.png",
    "flight-range-5p3-20261019.png",
    "flight-range-4over3-20261019.png",
)
SCALED_SHAPES = ((1924, 1058), (1920, 1080))
# Every reference-space rectangle the readers use to place a window.
ROI_LISTS = {
    "reload_numbers": tuple(hp.REFERENCE_BAR_ROW["first"] - 10 + 96 * i and
                            (hp.REFERENCE_BAR_ROW["first"] - 10 + 96 * i,
                             hp.REFERENCE_BAR_ROW["bar_y"] - 27, 42, 25)
                            for i in range(4)),
    "ammo_slots": tuple(fire_hud.SELECTOR_ROIS),
    "selector_panel": (fire_hud.REFERENCE_SELECTOR_BBOX,),
    "slot_band": (fire_hud.REFERENCE_SLOT_BAND,),
    "battle_carets": (fire_hud.REFERENCE_BATTLE_ROIS["caret"],
                      fire_hud.REFERENCE_BATTLE_ROIS["caret_wide"],
                      fire_hud.REFERENCE_BATTLE_ROIS["ring"]),
    "anchored_number": (fire_hud.REFERENCE_BATTLE_ROIS["anchored_number"],),
    "minimap": (hp.REFERENCE_MINIMAP,),
}


def load(name):
    return np.asarray(Image.open(FIXTURES / name).convert("RGB"))


def content(frame, projection=None):
    """The game surface of a reference frame, as the projection defines it."""
    width, height, x0, y0 = hp.content_bounds(frame)
    return frame[y0:y0 + height, x0:x0 + width]


def to_reference_crop_rect(rect):
    """Reference canvas rect -> the same rect in reference content coordinates."""
    x, y, w, h = rect
    return x, y - hp.REFERENCE_CONTENT_ORIGIN[1], w, h


def resample(surface, width, height):
    return cv2.resize(surface, (width, height), interpolation=cv2.INTER_AREA)


def inside_reference_content(rect):
    """True when a reference rect's own extent lies inside the game surface.

    The slot band deliberately reaches into the bottom letterbox (it is a
    search band for the whole ammo row plus the plates under it), so it is not
    a rectangle ``to_ref`` can return unchanged.
    """
    x, y, w, h = rect
    top = hp.REFERENCE_CONTENT_ORIGIN[1]
    return (x >= 0 and y >= top
            and x + w <= hp.REFERENCE[0]
            and y + h <= top + hp.REFERENCE[1])


def compare(left, right, drop=()):
    """Differences between two nested evidence structures, with numpy payloads free."""
    diffs = []
    def walk(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                diffs.append((path, sorted(set(a) ^ set(b))))
                return
            for key in a:
                if key in drop:
                    continue
                walk(a[key], b[key], f"{path}.{key}" if path else key)
        elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                diffs.append((path, (len(a), len(b))))
                return
            for index, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{path}[{index}]")
        elif a != b:
            diffs.append((path, (a, b)))
    walk(left, right, "")
    return diffs


class ReferenceIdentityTests(unittest.TestCase):
    """On the reference capture the new map must not move a single window."""

    def test_identity_projection_maps_every_roi_unchanged(self):
        projection = hp.projection_for((1600, 2560, 3))
        report = projection.report()
        self.assertEqual(report["scale"], [1.0, 1.0])
        self.assertEqual(report["offset"], [0.0, 0.0])
        self.assertEqual(report["source"], "declared_shape")
        for name, rects in ROI_LISTS.items():
            for rect in rects:
                with self.subTest(roi=name, rect=rect):
                    self.assertEqual(projection.from_ref(rect), list(rect))
                    self.assertEqual(projection.project_rect(rect), list(rect))
                    if inside_reference_content(rect):
                        self.assertEqual(projection.to_ref(rect), list(rect))
        self.assertEqual(projection.report()["clamped"], [])

    def test_the_slot_band_is_reported_when_it_reaches_into_the_bottom_bar(self):
        # The band is a search window for the ammo row *and* the plate row, so
        # it extends 60 px into the letterbox. to_ref() must clamp it to the
        # surface and record that, not return a rectangle that pretends the bar
        # is HUD space.
        projection = hp.projection_for((1600, 2560, 3))
        out = projection.to_ref(fire_hud.REFERENCE_SLOT_BAND)
        bottom = hp.REFERENCE_CONTENT_ORIGIN[1] + hp.REFERENCE[1]
        self.assertEqual(out[1] + out[3], bottom)
        clamped = projection.report()["clamped"]
        self.assertEqual(len(clamped), 1)
        self.assertEqual(clamped[0]["reference"],
                         [float(v) for v in fire_hud.REFERENCE_SLOT_BAND])
        self.assertEqual(clamped[0]["reason"], "reference_bounds")

    def test_reference_frames_read_the_same_as_the_pre_projection_path(self):
        for name in REFERENCE_FRAME_ASSETS:
            frame = load(name)
            with self.subTest(frame=name):
                legacy = fire_hud.FireHudReader(projection=hp.identity(frame.shape))
                mapped = fire_hud.FireHudReader(projection=hp.projection_for(frame.shape))
                before, after = legacy.detect(frame), mapped.detect(frame)
                diffs = compare(before, after, drop=("hud_projection",))
                self.assertEqual(diffs, [])
                # The ROI windows themselves are numerically identical.
                self.assertEqual(
                    legacy.reload_regions(frame),
                    mapped.reload_regions(frame))
                self.assertEqual(legacy.reload_regions(frame),
                                 [list(rect) for rect in fire_hud.RELOAD_ROIS])


class ScaledFrameTests(unittest.TestCase):
    """The same frame resampled to other client sizes keeps the same reads."""

    @classmethod
    def setUpClass(cls):
        cls.reference = load(REFERENCE_FRAME_ASSETS[0])
        cls.surface = content(cls.reference)
        cls.reader = fire_hud.FireHudReader(projection=hp.identity(cls.reference.shape))
        cls.reference_result = cls.reader.detect(cls.reference)
        cls.reference_slots = fire_hud.slot_selection(
            cls.reference, hp.identity(cls.reference.shape))
        cls.reference_ruler = fire_hud.scope_ruler_center(
            cls.reference, projection=hp.identity(cls.reference.shape))
        cls.reference_ring = fire_hud.green_reticle(
            cls.reference, projection=hp.identity(cls.reference.shape))

    def samples(self):
        for width, height in SCALED_SHAPES:
            yield (width, height), resample(self.surface, width, height)

    def test_rois_cover_the_same_pixels_after_resampling(self):
        """Every ROI must hold the same HUD content, not merely plausible coordinates.

        Two measurements, because resampling twice (surface -> sample -> ROI)
        and once (surface -> ROI) differ by more than a shifted window would:
        the mean absolute difference has to stay small *and* the two crops have
        to correlate, which a ROI aimed at neighbouring water cannot do.
        """
        for (width, height), frame in self.samples():
            projection = hp.projection_for(frame.shape)
            for name, rects in ROI_LISTS.items():
                for rect in rects:
                    x, y, w, h = rect
                    top = hp.REFERENCE_CONTENT_ORIGIN[1]
                    reference_crop = self.surface[max(0, y - top):y - top + h, x:x + w]
                    fx, fy, fw, fh = projection.from_ref(rect)
                    sample_crop = frame[fy:fy + fh, fx:fx + fw]
                    with self.subTest(size=(width, height), roi=name, rect=rect):
                        self.assertTrue(sample_crop.size, "projected ROI is empty")
                        expected = cv2.resize(reference_crop, (fw, fh),
                                              interpolation=cv2.INTER_AREA)
                        self.assertEqual(sample_crop.shape, expected.shape)
                        left = sample_crop.astype(np.float32).mean(axis=2)
                        right = expected.astype(np.float32).mean(axis=2)
                        difference = float(np.abs(left - right).mean())
                        correlation = float(np.corrcoef(left.ravel(), right.ravel())[0, 1])
                        self.assertLess(difference, 12.0,
                                        f"ROI {name} covers different content")
                        self.assertGreater(correlation, .90,
                                           f"ROI {name} does not correlate with its window")

    def test_recognizers_hit_the_same_elements(self):
        rows = []
        for (width, height), frame in self.samples():
            projection = hp.projection_for(frame.shape)
            reader = fire_hud.FireHudReader(projection=projection)
            result = reader.detect(frame)
            slots = fire_hud.slot_selection(frame, projection)
            ruler = fire_hud.scope_ruler_center(frame, projection=projection)
            ring = fire_hud.green_reticle(frame, projection=projection)
            hits = {
                "weapon_ammo": (result["weapon"]["ammo"],
                                self.reference_result["weapon"]["ammo"]),
                "weapon_kind": (result["weapon"]["kind"],
                                self.reference_result["weapon"]["kind"]),
                "battle_active": (result["battle"]["active"],
                                  self.reference_result["battle"]["active"]),
                "reload_layout": (result["reload"]["raw"]["layout_status"],
                                  self.reference_result["reload"]["raw"]["layout_status"]),
                "slot_index": (slots["selected_slot_index"],
                               self.reference_slots["selected_slot_index"]),
            }
            for key, (observed, expected) in hits.items():
                with self.subTest(size=(width, height), field=key):
                    self.assertEqual(observed, expected, f"{key} at {width}x{height}")
            # Positions, converted back to reference space, land on the same
            # element: no check reads an absolute frame coordinate.
            if self.reference_ruler is not None:
                self.assertIsNotNone(ruler, "scope ruler lost at a scaled frame")
                dix, diy = (ruler["center"][0] - self.reference_ruler["center"][0],
                            ruler["center"][1] - self.reference_ruler["center"][1])
                back = projection.to_ref((int(ruler["center"][0]), int(ruler["center"][1]), 1, 1))
                self.assertLessEqual(abs(back[0] - self.reference_ruler["center"][0]), 3)
                self.assertLessEqual(abs(back[1] - self.reference_ruler["center"][1]), 3)
            else:
                dix = diy = None
            if self.reference_ring is not None:
                self.assertIsNotNone(ring, "green reticle lost at a scaled frame")
                origin = projection.to_ref((int(ring["centre"][0]), int(ring["centre"][1]), 1, 1))
                self.assertLessEqual(abs(origin[0] - self.reference_ring["centre"][0]), 4)
                self.assertLessEqual(abs(origin[1] - self.reference_ring["centre"][1]), 4)
            if self.reference_slots["frame_marks"]:
                marks = slots["frame_marks"]
                self.assertIsNotNone(marks, "selection frame lost at a scaled frame")
                back = projection.to_ref((marks["bbox"][0], marks["bbox"][1],
                                          marks["bbox"][2], marks["bbox"][3]))
                expected = to_reference_crop_rect(self.reference_slots["frame_marks"]["bbox"])
                self.assertLessEqual(abs(back[0] - expected[0]), 6)
                self.assertLessEqual(abs(back[1] - expected[1]), 6)
                self.assertLessEqual(abs(back[2] - expected[2]), 8)
                self.assertLessEqual(abs(back[3] - expected[3]), 8)
                frame_diff = [back[0] - expected[0], back[1] - expected[1],
                              back[2] - expected[2], back[3] - expected[3]]
            else:
                frame_diff = None
            rows.append({
                "size": f"{width}x{height}",
                "hits": {k: v[0] for k, v in hits.items()},
                "expected": {k: v[1] for k, v in hits.items()},
                "ruler_centre_delta_px": None if dix is None else [dix, diy],
                "selection_frame_delta_ref_px": frame_diff,
            })
        self.assertEqual(len(rows), len(SCALED_SHAPES))
        ScaledFrameTests.report = rows


class LetterboxTests(unittest.TestCase):
    """The game surface is measured from the frame, not assumed from its size."""

    def test_reference_capture_with_80px_bars(self):
        frame = load(REFERENCE_FRAME_ASSETS[0])
        self.assertEqual(hp.content_bounds(frame), (2560, 1440, 0, 80))
        declared = hp.declared(frame)
        self.assertEqual(declared.report()["scale"], [1.0, 1.0])
        self.assertEqual(declared.report()["offset"], [0.0, 0.0])
        self.assertEqual(declared.source, "declared_content")

    def test_reference_canvas_with_96px_bars_is_measured(self):
        frame = np.asarray(Image.open(fire_hud.ROOT / "state/s_g.png").convert("RGB"))
        self.assertEqual(frame.shape, (1600, 2560, 3))
        self.assertEqual(hp.content_bounds(frame), (2560, 1408, 0, 96))
        declared = hp.declared(frame)
        scale = declared.report()["scale"]
        self.assertAlmostEqual(scale[0], 1.0)
        self.assertAlmostEqual(scale[1], 1408 / 1440, places=5)
        self.assertAlmostEqual(declared.project_y(80), 96, places=3)
        self.assertAlmostEqual(declared.project_y(1520), 1504, places=3)

    def test_frame_without_bars_is_the_whole_surface(self):
        surface = content(load(REFERENCE_FRAME_ASSETS[0]))
        frame = resample(surface, 1920, 1080)
        self.assertEqual(hp.content_bounds(frame), (1920, 1080, 0, 0))
        declared = hp.declared(frame)
        self.assertEqual(declared.report()["scale"], [0.75, 0.75])
        self.assertEqual(declared.report()["offset"], [0.0, -60.0])
        projection = hp.projection_for(frame.shape)
        self.assertEqual(projection.report()["scale"], [0.75, 0.75])
        self.assertEqual(projection.report()["offset"], [0.0, -60.0])
        self.assertEqual(projection.report()["source"], "declared_shape")

    def test_projection_for_uses_declared_bars_only_for_the_reference_canvas(self):
        self.assertEqual(hp.projection_for((1600, 2560)).report()["offset"], [0.0, 0.0])
        # A frame whose shape is not the reference canvas is its own surface.
        self.assertEqual(hp.projection_for((1080, 1920)).report()["offset"], [0.0, -60.0])
        # An explicit content rectangle wins over the shape default.
        custom = hp.projection_for((1080, 1920), content=(0, 60, 1920, 960))
        self.assertAlmostEqual(custom.report()["scale"][0], 0.75, places=5)
        self.assertAlmostEqual(custom.report()["scale"][1], 960 / 1440, places=5)
        self.assertAlmostEqual(custom.project_y(80), 60.0)
        with self.assertRaises(hp.MappingError):
            hp.projection_for((1080, 1920), content=(0, 0, 4000, 4000))


class ClampTests(unittest.TestCase):
    """Out-of-frame rectangles are clamped *and* reported."""

    def setUp(self):
        self.projection = hp.projection_for((1080, 1920, 3))

    def test_from_ref_clamps_and_reports(self):
        rect = (2500, 1450, 400, 300)
        out = self.projection.from_ref(rect)
        self.assertEqual(len(out), 4)
        x, y, w, h = out
        self.assertGreaterEqual(x, 0)
        self.assertGreaterEqual(y, 0)
        self.assertLessEqual(x + w, 1920)
        self.assertLessEqual(y + h, 1080)
        clamped = self.projection.report()["clamped"]
        self.assertEqual(len(clamped), 1)
        self.assertEqual(clamped[0]["reference"], [float(v) for v in rect])
        self.assertEqual(clamped[0]["axis"], "both")
        self.assertEqual(clamped[0]["reason"], "frame_bounds")

    def test_to_ref_clamps_to_the_reference_surface_and_reports(self):
        out = self.projection.to_ref((1900, 1000, 400, 300))
        x, y, w, h = out
        self.assertLessEqual(x + w, hp.REFERENCE[0])
        self.assertLessEqual(y + h, hp.REFERENCE_CONTENT_ORIGIN[1] + hp.REFERENCE[1])
        self.assertGreaterEqual(y, hp.REFERENCE_CONTENT_ORIGIN[1])
        clamped = self.projection.report()["clamped"]
        self.assertEqual(clamped[-1]["reason"], "reference_bounds")

    def test_rect_inside_the_frame_reports_no_clamp(self):
        rect = (940, 1395, 80, 115)
        out = self.projection.from_ref(rect)
        self.assertEqual(out, [705, 986, 60, 86])
        self.assertEqual(self.projection.report()["clamped"], [])

    def test_strict_projection_still_refuses_a_rect_it_cannot_place(self):
        with self.assertRaises(hp.MappingError):
            self.projection.project_rect((2500, 1450, 400, 300))


if __name__ == "__main__":
    unittest.main()
