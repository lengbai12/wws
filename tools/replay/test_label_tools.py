"""Tests for the HUD calibration, label derivation and anchor tools.

Run: /usr/bin/python3 tools/replay/test_label_tools.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import anchors_from_pylog as anchors
import derive_labels as labels
import hud_calibration as hud

PARSED = Path(__file__).resolve().parents[2] / "replays/parsed"

GREEN = (16, 240, 176)
DARK = (10, 10, 14)


def frame(width, height, bars, background=DARK):
    """Build a raw RGB24 frame with solid rectangles drawn on it."""
    buffer = bytearray()
    for y in range(height):
        for x in range(width):
            colour = background
            for rect, fill in bars:
                rx, ry, rw, rh = rect
                if rx <= x < rx + rw and ry <= y < ry + rh:
                    colour = fill
                    break
            buffer += bytes(colour)
    return bytes(buffer)


class HudCalibration(unittest.TestCase):
    def test_blank_record_validates(self):
        cal = hud.blank_calibration((100, 50))
        self.assertEqual([], hud.validate_calibration(cal))

    def test_bad_records_are_rejected(self):
        cal = hud.blank_calibration((100, 50))
        cal["schema_version"] = 99
        self.assertIn("schema_version must be 1", hud.validate_calibration(cal))
        cal = hud.blank_calibration((100, 50))
        cal["elements"]["own_hp_bar"] = {"rect": [90, 40, 40, 10]}
        self.assertIn("element 'own_hp_bar' rect falls outside resolution",
                      hud.validate_calibration(cal))
        cal = hud.blank_calibration((100, 50))
        cal["elements"]["nose"] = {"rect": [0, 0, 4, 4]}
        self.assertIn("unknown element 'nose'", hud.validate_calibration(cal))

    def test_saving_an_invalid_record_is_refused(self):
        cal = hud.blank_calibration((100, 50))
        cal["elements"]["own_hp_bar"] = {"rect": [0, 0, 0, 0]}
        with self.assertRaises(ValueError):
            hud.save_calibration("/tmp/never-written.json", cal)
        self.assertFalse(Path("/tmp/never-written.json").exists())

    def test_letterbox_is_measured(self):
        rgb = frame(64, 40, [([0, 8, 64, 24], GREEN)])
        self.assertEqual([0, 8, 64, 24], hud.letterbox_rect(rgb, 64, 40))

    def test_a_full_bar_measures_one(self):
        rgb = frame(32, 20, [([4, 6, 20, 6], GREEN)])
        measured = hud.measure_fill_fraction(rgb, 32, 20, [4, 6, 20, 6])
        self.assertAlmostEqual(1.0, measured, places=6)

    def test_half_a_bar_measures_half(self):
        rgb = frame(32, 20, [([4, 6, 10, 6], GREEN)])
        measured = hud.measure_fill_fraction(rgb, 32, 20, [4, 6, 20, 6])
        self.assertAlmostEqual(0.5, measured, places=2)

    def test_an_empty_bar_is_unreadable_not_full(self):
        rgb = frame(32, 20, [])
        self.assertIsNone(hud.measure_fill_fraction(rgb, 32, 20, [4, 6, 20, 6]))

    def test_a_wrong_frame_size_is_refused(self):
        with self.assertRaises(ValueError):
            hud.frame_size(b"\x00" * 10, 4, 4)

    def test_candidates_find_the_widest_bar_first(self):
        rgb = frame(64, 24, [([2, 4, 30, 6], GREEN), ([2, 14, 12, 6], (240, 200, 16))])
        found = hud.candidate_bars(rgb, 64, 24, [0, 0, 64, 24], min_length=8)
        self.assertGreaterEqual(len(found), 2)
        self.assertEqual([2, 4, 30, 6], found[0]["rect"])


class LabelDerivation(unittest.TestCase):
    def setUp(self):
        self.steps = [(0.0, 100.0), (10.0, 80.0), (20.0, 80.0), (30.0, 10.0)]

    def test_a_value_holds_until_the_next_step(self):
        self.assertEqual(100.0, labels.hp_at(self.steps, 0.0))
        self.assertEqual(100.0, labels.hp_at(self.steps, 9.999))
        self.assertEqual(80.0, labels.hp_at(self.steps, 10.0))
        self.assertEqual(10.0, labels.hp_at(self.steps, 99.0))

    def test_before_the_first_step_there_is_no_value(self):
        self.assertIsNone(labels.hp_at(self.steps, -0.001))

    def test_occupancy_separates_the_three_cases(self):
        rows = labels.build_rows("s.wowsreplay", self.steps, 100.0, 35.0,
                                 1.0, 0.0, [0, 0, 10, 4], 0.0, 40.0, 5.0)
        by_clock = {round(row["replay_clock_s"]): row for row in rows}
        self.assertEqual(labels.PRESENT, by_clock[5]["occupancy"])
        self.assertEqual(1.0, by_clock[5]["value"]["fraction"])
        self.assertEqual(labels.ABSENT, by_clock[40]["occupancy"])
        self.assertIsNone(by_clock[40]["value"])
        self.assertIn("own_ship_dead_hud_bar_absent", by_clock[40]["caveats"])

    def test_a_negative_clock_is_no_information(self):
        rows = labels.build_rows("s.wowsreplay", self.steps, 100.0, None,
                                 1.0, 10.0, [0, 0, 10, 4], 0.0, 0.0, 5.0)
        self.assertEqual(labels.NO_INFORMATION, rows[0]["occupancy"])
        self.assertIn("replay_clock_negative_before_battle_start", rows[0]["caveats"])

    def test_summary_counts_present_rows(self):
        rows = labels.build_rows("s.wowsreplay", self.steps, 100.0, 35.0,
                                 1.0, 0.0, [0, 0, 10, 4], 0.0, 40.0, 5.0)
        summary = labels.summarise(rows)
        self.assertEqual(len(rows), summary["rows"])
        self.assertEqual(summary["present_rows"], summary["occupancy"][labels.PRESENT])
        self.assertLess(summary["fraction_min"], summary["fraction_max"])


class Anchors(unittest.TestCase):
    def test_log_time_is_parsed_to_epoch(self):
        stamp = anchors.parse_log_line_time("2026-10-10 09:22:02.853: INFO: x")
        self.assertIsNotNone(stamp)
        self.assertIsNone(anchors.parse_log_line_time("no timestamp here"))

    def test_pairing_matches_identity_in_order(self):
        replay_deaths = [{"clock": 10.0, "victim": 7, "death_type": 17},
                         {"clock": 20.0, "victim": 7, "death_type": 17}]
        logged = [(100.0, 7, 1, 17), (200.0, 7, 1, 17)]
        rows, unpaired_log, unpaired_replay = anchors.pair_deaths(logged, replay_deaths)
        self.assertEqual([10.0, 20.0], [row[3] for row in rows])
        self.assertEqual([], unpaired_log)
        self.assertEqual([], unpaired_replay)

    def test_an_unknown_death_is_reported_not_guessed(self):
        replay_deaths = [{"clock": 10.0, "victim": 7, "death_type": 17}]
        rows, unpaired_log, unpaired_replay = anchors.pair_deaths(
            [(100.0, 9, 1, 17)], replay_deaths)
        self.assertEqual([], rows)
        self.assertEqual(1, len(unpaired_log))
        self.assertEqual(1, len(unpaired_replay))

    def test_windows_are_offered_in_both_directions(self):
        zero = 1000.0
        starts = [(900.0, "/r/replay.wowsreplay"), (1040.0, "/r/replay.wowsreplay"),
                  (2000.0, "/r/replay.wowsreplay")]
        windows = anchors.candidate_windows(starts, "replay.wowsreplay", zero, 100.0)
        offsets = [round(window["offset_from_zero"]) for window in windows]
        self.assertEqual([40, -100], offsets)
        self.assertEqual(2000.0, windows[0]["end"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
