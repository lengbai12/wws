import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nameplate_hp import read_hp

STATE = Path(__file__).resolve().parents[2] / "state"

# Nameplate boxes as reported by main_view_perception.detect_nameplates on each
# capture (confidence 0.95).  The red plate carries the digits "97 200 / 97 200".
PLATES = {
    "dyn-hit3-20261020.png": (1069, 753, 123, 16),
    "dyn-hit4-20261020.png": (1078, 747, 124, 17),
    "dyn-hit5-20261020.png": (1185, 746, 123, 17),
    "dyn-now7-20261020.png": (1097, 708, 124, 16),
    "final-check-20261019.png": (1079, 721, 123, 17),
}


def load(name):
    return np.array(Image.open(STATE / name).convert("RGB"))


def plate_fill(frame, box):
    x, y, w, h = box
    return np.median(frame[y + 2:y + h - 2, x + 2:x + 8].reshape(-1, 3), axis=0).astype(np.uint8)


class NameplateHpTests(unittest.TestCase):
    def test_live_plates_read_both_bounds(self):
        for name, box in PLATES.items():
            with self.subTest(frame=name):
                result = read_hp(load(name), box)
                self.assertEqual(result["status"], "ok", result)
                self.assertEqual(result["current"], 97200, result)
                self.assertEqual(result["maximum"], 97200, result)
                self.assertEqual(result["reason"], "ok")
                self.assertEqual(result["text"], "97200/97200")

    def test_erased_digits_are_unknown(self):
        name = "dyn-hit4-20261020.png"
        box = PLATES[name]
        frame = load(name)
        x, y, w, h = box
        frame[y:y + h, x:x + w] = plate_fill(frame, box)
        result = read_hp(frame, box)
        self.assertEqual(result["status"], "unknown", result)
        self.assertIsNone(result["current"], result)
        self.assertIsNone(result["maximum"], result)
        self.assertIn(result["reason"], ("low_contrast", "no_glyphs"))

    def test_erased_separator_is_unknown(self):
        name = "dyn-hit4-20261020.png"
        box = PLATES[name]
        frame = load(name)
        x, y, w, h = box
        fill = plate_fill(frame, box)
        frame[y:y + h, x + 55:x + 68] = fill  # covers only the "/" between the numbers
        result = read_hp(frame, box)
        self.assertEqual(result["status"], "unknown", result)
        self.assertIsNone(result["current"], result)
        self.assertEqual(result["reason"], "separator_missing")

    def test_erased_middle_digit_is_unknown(self):
        name = "dyn-hit4-20261020.png"
        box = PLATES[name]
        frame = load(name)
        x, y, w, h = box
        fill = plate_fill(frame, box)
        frame[y:y + h, x + 31:x + 39] = fill  # removes the "2" of the current value
        result = read_hp(frame, box)
        self.assertEqual(result["status"], "unknown", result)
        self.assertIsNone(result["current"], result)
        self.assertEqual(result["reason"], "glyph_gap")

    def test_scaled_and_recoloured_plate_still_reads(self):
        name = "dyn-hit4-20261020.png"
        x, y, w, h = PLATES[name]
        crop = load(name)[y:y + h, x:x + w]
        big = np.clip(np.asarray(Image.fromarray(crop).resize((w * 2, h * 2), Image.BICUBIC))
                      .astype(np.float32) * np.array([1.0, 0.7, 0.8]), 0, 255).astype(np.uint8)
        frame = np.full((big.shape[0] + 40, big.shape[1] + 40, 3), 30, np.uint8)
        frame[20:20 + big.shape[0], 20:20 + big.shape[1]] = big
        result = read_hp(frame, (20, 20, big.shape[1], big.shape[0]))
        self.assertEqual(result["status"], "ok", result)
        self.assertEqual((result["current"], result["maximum"]), (97200, 97200))

    def test_bbox_outside_frame_is_unknown(self):
        frame = load("dyn-hit4-20261020.png")
        result = read_hp(frame, (frame.shape[1] - 20, 0, 124, 17))
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["reason"], "bbox_outside_frame")
        self.assertIsNone(result["current"])


if __name__ == "__main__":
    unittest.main()
