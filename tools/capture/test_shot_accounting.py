"""Acceptance for shot accounting on saved battle frames.

Every case is a real before/after pair from ``state/``: the 2026-10-17 training
volley that removed 340 HP from a 39 100 HP bot, the r46 volley that removed
990 HP from a 35 500 HP target, and salvos whose frames show the plate's number
unchanged.  Erasing the printed HP must make the shot unknown, never a miss.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _path in (HERE, HERE.parent / "replay"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import shot_accounting as S  # noqa: E402

IMPACT_BEFORE = REPO / "state/goal-static-shot-20261017/impact2.png"
IMPACT_AFTER = REPO / "state/goal-static-shot-20261017/impact4.png"
IMPACT_BEFORE_PLATE = [1221, 642, 123, 17]
IMPACT_AFTER_PLATE = [1215, 748, 124, 16]

R46_DIR = REPO / "state/training-duration-hull-waterline-fire-r46-178"
R46_BEFORE = R46_DIR / "before.png"
R46_AFTER_HIT = REPO / "state/static-training-postfire-r46-3.png"
R46_AFTER_PREIMPACT = R46_DIR / "after.png"
R46_PLATE = [1131, 285, 124, 16]

DYNFIRE1 = REPO / "state/dyn-fire-20261020"
DYNFIRE2 = REPO / "state/dyn-fire2-20261020"


def load(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise unittest.SkipTest("missing capture %s" % path)
    return np.ascontiguousarray(image[:, :, ::-1])


def erase_hp_readout(frame: np.ndarray, bbox) -> np.ndarray:
    """Paint the plate with its own bar colour, deleting the printed HP only."""
    out = frame.copy()
    x, y, w, h = bbox
    patch = out[y:y + h, x:x + w].reshape(-1, 3)
    colours, counts = np.unique(patch, axis=0, return_counts=True)
    out[y:y + h, x:x + w] = colours[counts.argmax()]
    return out


class ShotAccounting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.impact_before = load(IMPACT_BEFORE)
        cls.impact_after = load(IMPACT_AFTER)
        cls.r46_before = load(R46_BEFORE)
        cls.r46_after_hit = load(R46_AFTER_HIT)
        cls.r46_after_preimpact = load(R46_AFTER_PREIMPACT)
        cls.dynfire1_before = load(DYNFIRE1 / "before.png")
        cls.dynfire1_after = load(DYNFIRE1 / "after.png")
        cls.dynfire2_before = load(DYNFIRE2 / "before.png")
        cls.dynfire2_after = load(DYNFIRE2 / "after.png")

    def test_the_340_hp_volley_is_a_hit(self):
        result = S.classify_shot(self.impact_before, self.impact_after)
        self.assertEqual(result["hp_before"], 39100)
        self.assertEqual(result["hp_after"], 38760)
        self.assertEqual(result["damage"], 340)
        self.assertIs(result["hit"], True)
        self.assertEqual(result["reason"], "nameplate_hp_drop")

    def test_the_990_hp_volley_is_a_hit(self):
        result = S.classify_shot(self.r46_before, self.r46_after_hit)
        self.assertEqual(result["hp_before"], 35500)
        self.assertEqual(result["hp_after"], 34510)
        self.assertEqual(result["damage"], 990)
        self.assertIs(result["hit"], True)
        self.assertEqual(result["reason"], "nameplate_hp_drop")

    def test_an_unchanged_full_plate_is_a_miss(self):
        result = S.classify_shot(self.dynfire1_before, self.dynfire1_after)
        self.assertEqual(result["hp_before"], 97200)
        self.assertEqual(result["hp_after"], 97200)
        self.assertEqual(result["damage"], 0)
        self.assertIs(result["hit"], False)
        self.assertEqual(result["reason"], "nameplate_hp_unchanged")

    def test_an_unchanged_damaged_plate_is_a_miss(self):
        result = S.classify_shot(self.r46_before, self.r46_after_preimpact)
        self.assertEqual(result["hp_before"], 35500)
        self.assertEqual(result["hp_after"], 35500)
        self.assertEqual(result["damage"], 0)
        self.assertIs(result["hit"], False)

    def test_a_second_unchanged_pair_is_a_miss(self):
        result = S.classify_shot(self.dynfire2_before, self.dynfire2_after)
        self.assertIs(result["hit"], False)
        self.assertEqual(result["damage"], 0)

    def test_a_nameplate_hint_keeps_the_same_verdict(self):
        result = S.classify_shot(self.impact_before, self.impact_after, IMPACT_BEFORE_PLATE)
        self.assertIs(result["hit"], True)
        self.assertEqual(result["damage"], 340)
        self.assertEqual(result["before"]["source"], "hint")

    def test_erased_hp_readout_makes_the_shot_unknown(self):
        before = erase_hp_readout(self.impact_before, IMPACT_BEFORE_PLATE)
        after = erase_hp_readout(self.impact_after, IMPACT_AFTER_PLATE)
        result = S.classify_shot(before, after, IMPACT_BEFORE_PLATE)
        self.assertIsNone(result["hit"])
        self.assertIsNone(result["damage"])
        self.assertIsNone(result["hp_before"])
        self.assertIsNone(result["hp_after"])
        self.assertTrue(result["reason"].startswith("hp_unreadable:"), result["reason"])
        self.assertIn("no_hp_text", result["reason"])

    def test_erasing_only_the_after_frame_is_unknown_not_a_miss(self):
        after = erase_hp_readout(self.impact_after, IMPACT_AFTER_PLATE)
        result = S.classify_shot(self.impact_before, after, IMPACT_BEFORE_PLATE)
        self.assertIsNone(result["hit"])
        self.assertIsNone(result["damage"])
        self.assertEqual(result["hp_before"], 39100)
        self.assertIsNone(result["hp_after"])
        self.assertTrue(result["reason"].startswith("hp_unreadable:"), result["reason"])

    def test_a_frame_without_a_nameplate_is_unknown(self):
        blank = self.impact_before.copy()
        blank[600:820, 1150:1400] = 90
        result = S.classify_shot(blank, self.impact_after)
        self.assertIsNone(result["hit"])
        self.assertEqual(result["reason"], "no_nameplate")

    def test_the_plate_reader_needs_repeated_threshold_support(self):
        reading = S.read_nameplate_hp(self.impact_before, IMPACT_BEFORE_PLATE)
        self.assertEqual(reading["status"], "observed")
        self.assertEqual((reading["hp"], reading["maximum"]), (39100, 39100))
        self.assertGreaterEqual(reading["support"], S.MIN_READ_SUPPORT)
        self.assertEqual(reading["reason"], "nameplate_hp_text")

    def test_damage_is_the_printed_hp_difference(self):
        result = S.classify_shot(self.impact_before, self.impact_after)
        self.assertEqual(result["damage"], result["hp_before"] - result["hp_after"])
        self.assertEqual(result["before"]["maximum"], result["after"]["maximum"])


if __name__ == "__main__":
    unittest.main()
