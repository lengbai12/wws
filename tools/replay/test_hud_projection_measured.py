"""Tests for hud_projection: measured anchors, per-axis affine, strict bounds.

The synthetic frames here are built the way the client actually renders: the
HUD is laid out at 1924x1058 and the host upscales that surface into the
2560x1600 canvas (96 px bars, 2560x1408 content). That makes the vertical
mapping differ from the horizontal one, which is exactly what a uniform-scale
or aspect-ratio assumption gets wrong.
"""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hud_projection as hp

REFERENCE = (2560, 1600)
DOWNSCALE = (1924, 1058)        # what the client really renders
CANVAS_CONTENT = (2560, 1408)   # aspect-preserved upscale, 96 px bars


def blank(canvas=REFERENCE):
    return np.full((canvas[1], canvas[0], 3), 0, np.uint8)


def reference_hud(canvas=REFERENCE):
    """A frame whose HUD sits exactly where the reference constants say.

    The canvas keeps the reference 80 px letterbox, so the frame's measured
    content is 2560x1440 and the projection has to come out as identity.
    """
    frame = blank(canvas)
    frame[80:80 + 1440] = 70
    paint_hud(frame, lambda x, y: (x, y), content=(2560, 1440), bar=80)
    return frame


def scaled_hud():
    """Render the HUD at 1924x1058, then upscale per axis into the canvas."""
    surface = np.full((DOWNSCALE[1], DOWNSCALE[0], 3), 70, np.uint8)

    def downscale(x, y):
        return (x * DOWNSCALE[0] / 2560.0, (y - 80) * DOWNSCALE[1] / 1440.0)

    paint_hud(surface, downscale, content=DOWNSCALE, bar=0)
    content = cv2.resize(surface, CANVAS_CONTENT, interpolation=cv2.INTER_NEAREST)
    frame = blank()
    frame[96:96 + CANVAS_CONTENT[1]] = content
    return frame


def box(transform, x, y, w, h):
    """Transform a reference rect's corners, then measure its frame size."""
    x0, y0 = transform(x, y)
    x1, y1 = transform(x + w, y + h)
    return (int(round(x0)), int(round(y0)),
            max(1, int(round(x1 - x0))), max(1, int(round(y1 - y0))))


def paint_hud(frame, transform, content, bar):
    """Draw the two anchors plus one colour block, through `transform`."""
    px0, py0, pw, ph = box(transform, hp.REFERENCE_MINIMAP[0], hp.REFERENCE_MINIMAP[1],
                           hp.REFERENCE_MINIMAP[2], hp.REFERENCE_MINIMAP[3])
    px1, py1 = px0 + pw, py0 + ph
    # Flat panel fill with a bright 2 px border, like the native panel.
    frame[py0:py1, px0:px1] = (24, 40, 46)
    frame[py0:py0 + 2, px0:px1] = (200, 200, 200)
    frame[py1 - 2:py1, px0:px1] = (200, 200, 200)
    frame[py0:py1, px0:px0 + 2] = (200, 200, 200)
    frame[py0:py1, px1 - 2:px1] = (200, 200, 200)
    # Reference turret bar row: four saturated 60 px bodies on a 96 px stride.
    row = hp.REFERENCE_BAR_ROW
    for i in range(row["count"]):
        bx, by, bw, bh = box(transform, row["first"] + i * row["pitch"], row["bar_y"],
                             row["width"], 6)
        frame[by:by + bh, bx:bx + bw] = (30, 210, 120)
    # A colour block at a reference ROI that must be found again after projection.
    bx, by, bw, bh = box(transform, *TARGET_ROI)
    frame[by:by + bh, bx:bx + bw] = TARGET_COLOR


TARGET_ROI = (1022, 1395, 90, 115)
TARGET_COLOR = (255, 255, 0)


def coverage(frame, rect, color=TARGET_COLOR, tolerance=24):
    x, y, w, h = (int(v) for v in rect)
    patch = frame[max(0, y):y + h, max(0, x):x + w]
    if patch.size == 0:
        return 0.0
    hit = np.all(np.abs(patch.astype(np.int16) - np.asarray(color)) <= tolerance, axis=2)
    return float(hit.mean())


class ProjectionTests(unittest.TestCase):
    def test_reference_frame_projects_as_identity(self):
        frame = reference_hud()
        projection = hp.measure(frame)
        self.assertEqual(projection.source, "measured_anchors")
        self.assertAlmostEqual(projection.x_scale, 1.0, places=2)
        self.assertAlmostEqual(projection.y_scale, 1.0, places=2)
        for rect in (hp.REFERENCE_MINIMAP, TARGET_ROI, (1118, 1338, 42, 25)):
            projected = projection.project_rect(rect)
            for want, got in zip(rect, projected):
                with self.subTest(rect=rect):
                    self.assertLessEqual(abs(want - got), 2)
        self.assertGreaterEqual(coverage(frame, projection.project_rect(TARGET_ROI)), 0.97)

    def test_scaled_synthetic_frame_projects_roi_onto_the_same_element(self):
        frame = scaled_hud()
        projection = hp.measure(frame)
        projected = projection.project_rect(TARGET_ROI)
        # The upscale is not uniform: the vertical factor differs, so a
        # single-scale projection cannot be right for both axes.
        self.assertAlmostEqual(projection.x_scale, 1.0, places=2)
        self.assertAlmostEqual(projection.y_scale, CANVAS_CONTENT[1] / 1440.0, places=2)
        # The identity rect is materially wrong on the vertical axis, while the
        # projected one covers the drawn block.
        self.assertGreaterEqual(abs(projected[1] - TARGET_ROI[1]), 8)
        self.assertGreaterEqual(coverage(frame, projected), 0.95)
        self.assertGreater(coverage(frame, projected), coverage(frame, TARGET_ROI))
        # The panel itself is covered too, on both axes.
        panel = projection.project_rect(hp.REFERENCE_MINIMAP)
        self.assertLessEqual(abs(panel[0] - projection.panel.left), 2)
        self.assertLessEqual(abs(panel[1] + panel[3] - projection.panel.bottom), 2)

    def test_missing_anchors_raise(self):
        for frame in (blank(), np.full((1600, 2560, 3), 70, np.uint8)):
            with self.subTest(mean=float(frame.mean())):
                with self.assertRaises(hp.MappingError):
                    hp.measure(frame)
                with self.assertRaises(hp.MappingError):
                    hp.measure(frame[:900])

    def test_out_of_bounds_reference_coordinates_are_rejected(self):
        projection = hp.measure(reference_hud())
        for point in ((-1, 900), (3000, 900), (100, 40), (100, 1590)):
            with self.subTest(point=point):
                with self.assertRaises(hp.MappingError):
                    projection.project_point(point)
        for rect in ((-10, 900, 50, 50), (2500, 900, 200, 50), (100, 1400, 50, 400)):
            with self.subTest(rect=rect):
                with self.assertRaises(hp.MappingError):
                    projection.project_rect(rect)
        with self.assertRaises(hp.MappingError):
            projection.project_rect((100, 800, 0, 50))

    def test_projection_that_leaves_the_frame_is_rejected(self):
        # A projection wider than the frame must refuse coordinates that fall
        # outside it instead of clamping them.
        projection = hp.Projection(x_scale=1.4, x_offset=0.0, y_scale=1.0,
                                   y_offset=0.0, frame=(2560, 1600))
        with self.assertRaises(hp.MappingError):
            projection.project_point((2000, 900))
        with self.assertRaises(hp.MappingError):
            projection.project_rect((1800, 900, 100, 100))
        self.assertEqual(projection.project_point((1800, 900)), (2520, 900))

    def test_declared_projection_is_labelled_and_content_derived(self):
        frame = np.full((1408, 2560, 3), 80, np.uint8)
        projection = hp.declared(frame)
        self.assertEqual(projection.source, "declared_content")
        self.assertAlmostEqual(projection.y_scale, 1408 / 1440.0, places=4)
        self.assertTrue(any('declared_content' in w for w in projection.warnings))

    def test_resolve_prefers_measurement_and_records_fallback(self):
        measured = hp.resolve(reference_hud())
        self.assertEqual(measured.source, "measured_anchors")
        fallback = hp.resolve(np.full((1600, 2560, 3), 70, np.uint8))
        self.assertEqual(fallback.source, "declared_content")
        with self.assertRaises(hp.MappingError):
            hp.resolve(np.full((1600, 2560, 3), 70, np.uint8), allow_declared=False)

    def test_report_carries_measurements_and_warnings(self):
        frame = scaled_hud()
        report = hp.measure(frame).report()
        self.assertEqual(report["source"], "measured_anchors")
        self.assertEqual(report["anchors"]["reference_bar_row"]["pitch"], 96)
        self.assertEqual(report["anchors"]["bar_row"]["count"], 4)
        self.assertIn("minimap_residual_px", report)
        self.assertIsInstance(report["warnings"], list)

    def test_off_switch_forces_reference_space(self):
        frame = scaled_hud()
        with patch.dict(os.environ, {"WWS_HUD_PROJECTION": "reference"}):
            projection = hp.resolve(frame)
        self.assertEqual(projection.source, "identity_reference")
        self.assertIn("WWS_HUD_PROJECTION=reference", projection.warnings)
        self.assertEqual(projection.project_rect(TARGET_ROI), list(TARGET_ROI))
        self.assertEqual(hp.resolve(frame).source, "measured_anchors")

    def test_mask_projection_never_raises_at_the_frame_edge(self):
        # A live run reported "point [0.0, 80.0] projects outside the frame"
        # while a mask rectangle was being projected. Masks clip; only a real
        # ROI read may raise on a mapping that far off.
        projection = hp.Projection(x_scale=1.0, x_offset=-22.0, y_scale=1.0,
                                   y_offset=-30.0, frame=(2560, 1600))
        self.assertEqual(projection.clip_rect((0, 80, 560, 185)), [0, 50, 538, 185])
        with self.assertRaises(hp.MappingError):
            projection.project_rect((0, 80, 560, 185))

    def test_real_saved_frames_measure_and_keep_the_panel(self):
        root = Path(__file__).resolve().parents[2]
        frames = [root / "state/hud-now-20261019.png", root / "state/aim-src-20261019.png"]
        for path in frames:
            if not path.exists():
                continue
            image = cv2.imread(str(path))
            if image is None:
                continue
            frame = np.ascontiguousarray(image[:, :, ::-1])
            with self.subTest(frame=path.name):
                projection = hp.measure(frame)
                panel = projection.panel
                rect = projection.project_rect(hp.REFERENCE_MINIMAP)
                self.assertLessEqual(abs(rect[0] - panel.left), 4)
                self.assertLessEqual(abs(rect[0] + rect[2] - panel.right), 8)
                self.assertLessEqual(abs(rect[1] + rect[3] - panel.bottom), 8)
                # This client renders a shorter surface, so the measured
                # vertical scale is below 1 and the offset is positive.
                self.assertLess(projection.y_scale, 1.0)


if __name__ == '__main__':
    unittest.main()
