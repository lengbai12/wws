#!/usr/bin/env python3
"""The top-right battle timer must be readable, and never invented."""
import unittest
from pathlib import Path

import cv2
import numpy as np

import fire_hud as F

FRAME = Path('state/battle-timer-fixture-20261020.png')


class BattleTimerTests(unittest.TestCase):
    def setUp(self):
        if not FRAME.is_file():
            self.skipTest('timer fixture missing')
        self.rgb = cv2.cvtColor(cv2.imread(str(FRAME)), cv2.COLOR_BGR2RGB)

    def test_reads_the_time_that_is_on_screen(self):
        # The fixture shows 17:55 in the top-right corner.
        self.assertEqual(F.battle_time_left(self.rgb), 17 * 60 + 55)

    def test_a_frame_without_a_timer_returns_none(self):
        blank = np.full_like(self.rgb, 120)
        self.assertIsNone(F.battle_time_left(blank))

    def test_painting_over_the_digits_returns_none(self):
        frame = self.rgb.copy()
        x, y, w, h = F.BATTLE_TIMER_BAND
        frame[y:y + h, x:x + w] = 120
        self.assertIsNone(F.battle_time_left(frame))

    def test_an_out_of_frame_band_raises(self):
        with self.assertRaises(ValueError):
            F.battle_time_left(self.rgb, band=(2500, 95, 260, 70))


if __name__ == '__main__':
    unittest.main()
