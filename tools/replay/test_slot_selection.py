"""Structured slot-selection readout: which slot the client's frame is on.

Every case runs on the real 2560x1600 frame state/s_g.png, not on a drawing of
one. The moved and erased cases lift the client's own outline pixels and put
them back elsewhere (or not at all), so the detector sees the same kind of
pixels the game emits.
"""
import sys
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fire_hud import FireHudReader, ROOT, slot_selection

FRAME = ROOT / "state/s_g.png"
# Measured on that frame: the frame the client draws round the first slot and
# the number plates on the row beneath it ("1", "2", "4", "R", "T", "Y", "U").
FRAME_BBOX = (965, 1384, 84, 84)
PLATE_CENTRES = (1093.3, 1192.6, 1379.3, 1465.5, 1551.9)
# Consumable columns (R and T), centres read from the same plate row.
CONSUMABLE_CENTRES = (("R", 1268.4), ("T", 1379.3))


def load():
    return np.asarray(Image.open(FRAME).convert("RGB")).copy()


def green(frame):
    a = frame.astype(np.int16)
    return ((a[..., 1] > a[..., 0] + 25) & (a[..., 1] > a[..., 2] + 25)
            & (a[..., 1] > 90))


def background(frame, box):
    """Water from left of the bar: a clean patch to lift or erase a mark with."""
    x, y, w, h = box
    return frame[y:y + h, x - 140:x - 140 + w].copy()


class SlotSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reader = FireHudReader()

    def moved(self, centre, frame=None):
        """The frame with the client's selection outline moved to ``centre``.

        Only the outline's own pixels move; the slot icon and its number plate
        stay where the game drew them, exactly as they would if the client had
        highlighted another slot.
        """
        frame = load() if frame is None else frame
        x, y, w, h = slot_selection(frame)["frame_marks"]["bbox"]
        mask = green(frame)[y:y + h, x:x + w].copy()
        outline = frame[y:y + h, x:x + w].copy()
        clean = background(frame, (x, y, w, h))
        region = frame[y:y + h, x:x + w]
        region[mask] = clean[mask]
        dx = int(round(centre - (x + w / 2.0)))
        target = frame[y:y + h, x + dx:x + dx + w]
        patched = target.copy()
        patched[mask] = outline[mask]
        target[:] = patched
        return frame

    def test_real_frame_selects_the_first_shell_slot(self):
        frame = load()
        result = slot_selection(frame)
        self.assertEqual(result["status"], "observed")
        self.assertEqual(result["selected_slot_index"], 1)
        self.assertIs(result["main_battery"], True)
        self.assertEqual(tuple(result["frame_marks"]["bbox"]), FRAME_BBOX)
        self.assertEqual(result["frame_marks"]["border_rows"], [1384, 1467])
        self.assertLess(result["frame_marks"]["interior_fill"], 0.01)
        columns = result["slot_columns"]
        self.assertEqual([c["index"] for c in columns], list(range(1, len(columns) + 1)))
        centres = [c["center_x"] for c in columns]
        self.assertEqual(centres, sorted(centres))
        # Seven slots are drawn: two shell columns, the torpedo column "4",
        # then the R/T/Y/U consumables. R's plate is lost against the hull and
        # is restored from the row's own pitch, so six centres are measured.
        self.assertEqual(len(columns), 7)
        # The first column is the highlighted plate, matched to the frame that
        # sits on the same slot; the plate's centroid is within 2 px of the
        # frame centre measured above.
        self.assertEqual(columns[0]["source"], "highlighted_plate")
        self.assertEqual(tuple(columns[0]["selection_frame_bbox"]), FRAME_BBOX)
        self.assertAlmostEqual(centres[0], 1007.0, delta=2.0)
        for centre in PLATE_CENTRES:
            self.assertTrue(any(abs(c["center_x"] - centre) < 1.0 for c in columns), centre)
        weapon = self.reader._weapon(frame)
        self.assertEqual(weapon["kind"], "gun")
        self.assertTrue(weapon["selected"])
        self.assertEqual(weapon["raw"]["selected_slot_index"], 1)
        self.assertEqual(weapon["raw"]["main_battery_evidence"], "selection_frame")
        self.assertEqual(weapon["raw"]["reason"], "selection_frame_main_battery")
        self.assertIsNone(weapon["ammo"])          # the frame names no shell type

    def test_frame_moved_onto_a_consumable_column_is_not_main_battery(self):
        for name, centre in CONSUMABLE_CENTRES:
            with self.subTest(slot=name):
                frame = self.moved(centre)
                result = slot_selection(frame)
                self.assertIsNot(result["main_battery"], True)
                self.assertTrue(result["selected_slot_index"] is None
                                or result["selected_slot_index"] > 2)
                self.assertNotEqual(self.reader._weapon(frame)["kind"], "gun")

    def test_frame_moved_onto_the_torpedo_column_is_not_main_battery(self):
        frame = self.moved(1192.6)
        result = slot_selection(frame)
        self.assertIs(result["main_battery"], False)
        self.assertGreater(result["selected_slot_index"], 2)
        self.assertNotEqual(self.reader._weapon(frame)["kind"], "gun")

    def test_missing_frame_is_unknown_and_does_not_raise(self):
        frame = load()
        box = slot_selection(frame)["frame_marks"]["bbox"]
        x, y, w, h = box
        erased = frame.copy()
        erased[y:y + h, x:x + w] = background(frame, box)
        result = slot_selection(erased)
        self.assertEqual(result["status"], "no_selection")
        self.assertIsNone(result["selected_slot_index"])
        self.assertIsNone(result["main_battery"])
        self.assertIsNone(result["frame_marks"])
        self.assertEqual(result["slot_columns"], [])
        self.assertNotEqual(self.reader._weapon(erased)["kind"], "gun")
        blank = np.zeros((1600, 2560, 3), np.uint8)
        self.assertIsNone(slot_selection(blank)["main_battery"])


if __name__ == "__main__":
    unittest.main()
