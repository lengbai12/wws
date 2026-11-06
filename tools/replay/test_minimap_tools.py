"""Tests for the minimap calibration, dataset splitting and position decoding.

The minimap layer needs numpy, so unlike the other replay tools this suite runs
under the project environment rather than the system interpreter:

    ./.venv/bin/python tools/replay/test_minimap_tools.py

Every case is chosen so that it can fail: a component finder is checked against
two blobs of different size, a fit against an exactly known transform and
against a deliberately wrong pairing, and the splitter against the exact defect
that made an earlier dataset unusable.
"""

from __future__ import annotations

import struct
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import build_minimap_dataset as dataset
import export_events
import minimap_calibration as mini


def position_packet(entity, x, z, yaw=0.0, clock=1.0, packet_type=0x2C):
    """Build a fake 0x2C packet carrying one position sample."""
    payload = struct.pack("<i7f", entity, 0.0, x, 0.0, z, yaw, 0.0, 0.0)
    return type("Packet", (), {"type": packet_type, "payload": payload, "clock": clock})()


class MarkerDetection(unittest.TestCase):
    def test_the_largest_blob_wins(self):
        mask = np.zeros((40, 40), dtype=bool)
        mask[2:5, 2:5] = True          # 9 px
        mask[20:30, 20:30] = True      # 100 px
        found = mini.largest_component(mask, min_area=5, max_area=200)
        area, (row, column) = found
        self.assertEqual(100, area)
        self.assertAlmostEqual(24.5, row)
        self.assertAlmostEqual(24.5, column)

    def test_a_blob_outside_the_area_bounds_is_ignored(self):
        mask = np.zeros((40, 40), dtype=bool)
        mask[0:1, 0:1] = True          # 1 px, below the floor
        self.assertIsNone(mini.largest_component(mask, min_area=5, max_area=200))

    def test_nothing_present_returns_none(self):
        self.assertIsNone(mini.largest_component(np.zeros((20, 20), dtype=bool)))

    def test_detect_reports_frame_pixels(self):
        # The marker must clear the detector's area floor, which exists to reject
        # text glyphs, so this blob is the smallest realistic size.
        frame = np.zeros((30, 30, 3), dtype=np.uint8)
        frame[10:16, 6:12] = 255
        found = mini.detect_marker(frame, [5, 5, 20, 20])
        self.assertIsNotNone(found)
        x, y, area = found
        self.assertEqual(36, area)
        self.assertAlmostEqual(8.5, x)     # rect origin 5 + column 3.5
        self.assertAlmostEqual(12.5, y)


class AffineFit(unittest.TestCase):
    def setUp(self):
        self.matrix = np.array([[0.2, 0.0], [0.0, 0.3]])
        self.translation = np.array([2300.0, 1200.0])
        self.world = [(0.0, 0.0), (100.0, 0.0), (0.0, 200.0), (500.0, 400.0)]
        self.pixels = [tuple(self.matrix @ np.array(point) + self.translation)
                       for point in self.world]

    def test_a_known_transform_is_recovered_exactly(self):
        matrix, translation, residuals = mini.fit_affine(self.world, self.pixels)
        self.assertTrue(np.allclose(self.matrix, matrix, atol=1e-9))
        self.assertTrue(np.allclose(self.translation, translation, atol=1e-9))
        self.assertLess(float(residuals.max()), 1e-9)

    def test_a_wrong_pairing_cannot_be_hidden(self):
        shuffled = [self.pixels[0], self.pixels[1], self.pixels[3], self.pixels[2]]
        _matrix, _translation, residuals = mini.fit_affine(self.world, shuffled)
        self.assertGreater(float(residuals.max()), 10.0)

    def test_too_few_samples_is_refused(self):
        with self.assertRaises(ValueError):
            mini.fit_affine([(0.0, 0.0), (1.0, 1.0)], [(0.0, 0.0), (1.0, 1.0)])

    def test_the_inverse_round_trips(self):
        matrix, translation, _residuals = mini.fit_affine(self.world, self.pixels)
        inverse, inverse_translation = mini.inverse_affine(matrix, translation)
        for point in self.world:
            pixel = mini.apply_affine(matrix, translation, point[0], point[1])
            back = inverse @ pixel + inverse_translation
            self.assertTrue(np.allclose(np.asarray(point), back, atol=1e-9))

    def test_an_axis_aligned_fit_reports_no_rotation(self):
        matrix, translation, residuals = mini.fit_affine(self.world, self.pixels)
        stats = mini.describe(matrix, translation, residuals)
        self.assertAlmostEqual(0.0, stats["rotation_deg"], places=6)
        self.assertAlmostEqual(0.2, stats["scale_px_per_m_x"], places=9)
        self.assertAlmostEqual(2.0 / 3.0, stats["scale_ratio"], places=9)

    def test_a_rotated_fit_reports_the_rotation(self):
        rotated = np.array([[0.0, -0.25], [0.25, 0.0]])
        pixels = [tuple(rotated @ np.array(point) + np.array([10.0, 10.0]))
                  for point in self.world]
        matrix, translation, residuals = mini.fit_affine(self.world, pixels)
        stats = mini.describe(matrix, translation, residuals)
        self.assertAlmostEqual(90.0, abs(stats["rotation_deg"]), places=6)


class OwnPositionDecoding(unittest.TestCase):
    def test_a_matching_packet_is_decoded(self):
        events = export_events.own_position_events(
            [position_packet(1448698, 420.0, 450.0, 3.1416, clock=2.5)], 1448698)
        self.assertEqual(1, len(events))
        event = events[0]
        self.assertEqual("position", event["event"])
        self.assertEqual(420.0, event["x"])
        self.assertEqual(450.0, event["z"])
        self.assertAlmostEqual(3.1416, event["yaw"])
        self.assertEqual(2.5, event["clock"])

    def test_another_entity_is_ignored(self):
        self.assertEqual([], export_events.own_position_events(
            [position_packet(1448660, 1.0, 2.0)], 1448698))

    def test_other_packet_types_and_lengths_are_ignored(self):
        wrong_type = position_packet(1448698, 1.0, 2.0, packet_type=0x0A)
        short = type("Packet", (), {"type": 0x2C, "payload": b"\x01\x02\x03\x04",
                                    "clock": 0.0})()
        self.assertEqual([], export_events.own_position_events([wrong_type, short], 1448698))


class SplitAssignment(unittest.TestCase):
    def test_adjacent_samples_share_a_block(self):
        """Block boundaries, not the cycle, decide where a split ends."""
        plan = [(float(index), float(index)) for index in range(100)]
        splits = dataset.assign_splits(plan, block_s=10.0)
        self.assertEqual(splits[0], splits[9])          # inside the first block
        self.assertEqual("train", splits[49])           # cycle block 4 is train
        self.assertEqual("validation", splits[50])      # cycle block 5 is validation
        self.assertNotEqual(splits[49], splits[50])

    def test_every_split_appears(self):
        plan = [(float(index), float(index)) for index in range(100)]
        splits = dataset.assign_splits(plan, block_s=10.0)
        self.assertEqual({"train", "validation", "test"}, set(splits))

    def test_a_monotone_path_still_covers_every_split(self):
        """The defect that made a plain thirds split unusable.

        A ship that travels across the map has a monotone position, so cutting
        the battle into thirds makes the thirds position-disjoint and forces the
        model to extrapolate. Cycling blocks must keep the spans overlapping.
        """
        plan = [(float(index), float(index)) for index in range(900)]
        splits = dataset.assign_splits(plan, block_s=12.0)
        spans = {}
        for (video_time, _clock), split in zip(plan, splits):
            low, high = spans.get(split, (video_time, video_time))
            spans[split] = (min(low, video_time), max(high, video_time))
        for split, (low, high) in spans.items():
            # Each split must cover almost the whole battle, which is what keeps
            # the position spans overlapping.
            self.assertGreater(high - low, 700.0,
                               "%s covers only %.0f s of the battle" % (split, high - low))

    def test_an_empty_plan_is_handled(self):
        self.assertEqual([], dataset.assign_splits([]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
