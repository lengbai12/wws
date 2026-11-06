"""Offline contracts for the training-room entry recogniser (no game, no input)."""
import json
import unittest
from pathlib import Path

import cv2
import numpy as np

import enter_training_battle as E


class RecogniserTests(unittest.TestCase):
    def setUp(self):
        self.templates = E.load_templates()

    def frame(self):
        return np.full((1058, 1924, 3), 40, np.uint8)

    def paste(self, frame, name, x, y):
        template = cv2.imread(str(E.FIXTURES / f'{name}.png'))
        h, w = template.shape[:2]
        frame[y:y + h, x:x + w] = template

    def test_each_real_template_is_recognised_at_its_native_pixels(self):
        for name in E.BUTTONS:
            frame = self.frame()
            self.paste(frame, name, 300, 400)
            found = E.detect(frame, self.templates)
            self.assertIn(name, found, name)
            self.assertEqual(found[name]['bbox'][:2], [300, 400], name)
            self.assertEqual(E.screen_state(found), E.SCREENS[name], name)

    def test_blank_frame_matches_nothing_and_is_unknown(self):
        found = E.detect(self.frame(), self.templates)
        self.assertEqual(found, {})
        self.assertEqual(E.screen_state(found), 'unknown')

    def test_results_button_wins_over_other_screens(self):
        frame = self.frame()
        self.paste(frame, 'return_port', 800, 900)
        self.paste(frame, 'join_battle', 900, 30)
        found = E.detect(frame, self.templates)
        self.assertEqual(E.screen_state(found), 'results')
        self.assertEqual(E.next_action('results', found, 0), 'return_port')

    def test_lobby_ready_runs_once_then_join(self):
        frame = self.frame()
        self.paste(frame, 'ready', 655, 593)
        self.paste(frame, 'join_battle', 900, 39)
        found = E.detect(frame, self.templates)
        self.assertEqual(E.next_action('lobby', found, 0), 'ready')
        self.assertEqual(E.next_action('lobby', found, 1), 'join_battle')
        self.assertEqual(E.next_action('lobby', found, 2), 'join_battle')

    def test_prebattle_clicks_start_not_ready(self):
        frame = self.frame()
        self.paste(frame, 'start_battle', 894, 1023)
        found = E.detect(frame, self.templates)
        self.assertEqual(E.screen_state(found), 'prebattle')
        self.assertEqual(E.next_action('prebattle', found, 0), 'start_battle')

    def test_join_button_absent_never_guesses_an_action(self):
        found = {'ready': {'score': .9, 'bbox': [0, 0, 10, 10], 'center': [5, 5]}}
        self.assertEqual(E.next_action('lobby', found, 0), None)

    def test_threshold_is_honoured_against_a_scrambled_frame(self):
        frame = self.frame()
        template = cv2.imread(str(E.FIXTURES / 'join_battle.png'))
        scaled = cv2.resize(template, None, fx=.8, fy=.8, interpolation=cv2.INTER_AREA)
        h, w = scaled.shape[:2]
        frame[300:300 + h, 300:300 + w] = scaled
        self.assertEqual(E.detect(frame, self.templates), {})

    def test_real_screens_only_report_their_own_buttons(self):
        cases = {'results': 'state/bow-fresh-battle-check-20261018.png',
                 'lobby': 'state/bow-ready-ok-20261018.png',
                 'prebattle': 'state/bow-prebattle-20261018.png'}
        for expected, relative in cases.items():
            frame = cv2.imread(str(E.ROOT / relative))
            self.assertIsNotNone(frame, relative)
            found = E.detect(frame, self.templates)
            self.assertEqual(E.screen_state(found), expected, relative)
            for name in found:
                self.assertEqual(E.SCREENS[name], expected, f'{relative}:{name}')


class WarpTests(unittest.TestCase):
    class FakeCapture:
        identity = {'pid': 7, 'display': ':test'}
        window = '1234'

        def check(self):
            return None

    class FakeX(E.X11):
        """X server that offsets the first warp by (-12, +22) like the real one."""

        def __init__(self):
            super().__init__(WarpTests.FakeCapture())
            self.pointer_at = [0, 0]
            self.calls = []
            self.limit = 1

        def x(self, *args):
            self.calls.append(args)
            if args[0] == 'getwindowfocus':
                return '1234'
            if args[0] == 'mousemove':
                x, y = int(args[1]), int(args[2])
                self.pointer_at = [x - 12, y + 22] if self.limit else [x, y]
                self.limit -= 1
            elif args[0] == 'mousemove_relative':
                self.pointer_at = [self.pointer_at[0] + int(args[2]),
                                   self.pointer_at[1] + int(args[3])]
            elif args[0] == 'getmouselocation':
                return f'X={self.pointer_at[0]}\nY={self.pointer_at[1]}'
            elif args[0] in ('mousedown', 'mouseup', 'windowactivate', 'click'):
                return ''
            return ''

        def click(self, x, y, hold_ms=90):
            self.warp(x, y)
            self.x('mousedown', 1)
            self.x('mouseup', 1)
            return tuple(self.pointer_at)

    def test_offset_warp_is_corrected_before_the_click(self):
        x11 = self.FakeX()
        landed = x11.click(700, 600)
        self.assertEqual(list(landed), [700, 600])
        kinds = [c[0] for c in x11.calls]
        self.assertIn('mousemove_relative', kinds)
        # The correction must happen before any button press.
        self.assertLess(kinds.index('mousemove_relative'), kinds.index('mousedown'))

    def test_unrecoverable_warp_refuses_instead_of_clicking_blind(self):
        class Stuck(WarpTests.FakeX):
            def x(self, *args):
                if args[0] == 'mousemove':
                    return ''
                if args[0] == 'mousemove_relative':
                    return ''
                return super().x(*args)

        stuck = Stuck()
        with self.assertRaises(E.EntryError):
            stuck.warp(100, 100)


if __name__ == '__main__':
    unittest.main()
