#!/usr/bin/env python3
"""Tests for the recognition annotation tools (no game, no screen needed)."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/capture'))
sys.path.insert(0, str(ROOT / 'tools/replay'))

import recognition_overlay as O  # noqa: E402


class AnnotateTests(unittest.TestCase):
    def test_a_claim_is_drawn_where_it_says_and_the_frame_is_not_resized(self):
        frame = np.zeros((1600, 2560, 3), dtype=np.uint8)
        rows = [dict(kind='enemy_plate', label='ENEMY', confidence=0.95, bbox=[200, 300, 120, 17])]
        image = O.annotate(frame, rows, O.font(20), {'ENEMY': 1}, 1, battle=True)
        self.assertEqual(image.size, (2560, 1600))
        pixels = np.asarray(image)
        border = pixels[300 - O.EXPAND:300 - O.EXPAND + 5, 200:320]
        self.assertGreater(int(border[..., 0].max()), 200, 'the red outline must be there')
        inside = pixels[330:380, 240:300]
        self.assertLess(int(inside.max()), 40, 'the inside of the box must be left alone')

    def test_outside_a_battle_the_claim_is_labelled_raw(self):
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        rows = [dict(kind='enemy_plate', label='ENEMY', confidence=0.7, bbox=[50, 100, 60, 10])]
        image = O.annotate(frame, rows, O.font(18), {'ENEMY': 1}, 7, battle=None)
        grey = np.asarray(image)[100 - O.EXPAND:100 - O.EXPAND + 5, 50:110]
        self.assertLess(int(grey[..., 0].max()), 200, 'not-in-battle claims are grey, not red')

    def test_a_minimap_contact_is_ringed_at_its_pixel(self):
        frame = np.zeros((1600, 2560, 3), dtype=np.uint8)
        rows = [dict(kind='contact', label='UNKNOWN', confidence=0.99, pixel=[2200.0, 1200.0], km=13.1)]
        image = O.annotate(frame, rows, O.font(18), {'UNKNOWN': 1}, 2, battle=True)
        pixels = np.asarray(image)
        ring = pixels[1200 - 20:1200 + 20, 2200 - 20:2200 + 20]
        self.assertGreater(int(ring[..., 0].max()), 150)


class ReviewTests(unittest.TestCase):
    def test_the_summary_counts_frames_and_confidences(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'detections.jsonl'
            rows = [
                dict(frame=1, battle=True, claims=[
                    dict(kind='enemy_plate', label='ENEMY', confidence=0.9),
                    dict(kind='contact', label='CRUISER', confidence=0.8, km=10.0)]),
                dict(frame=2, battle=True, claims=[
                    dict(kind='enemy_plate', label='ENEMY', confidence=0.5)]),
                dict(frame=3, battle=False, claims=[
                    dict(kind='enemy_plate', label='ENEMY', confidence=0.7)]),
            ]
            path.write_text('\n'.join(json.dumps(r) for r in rows))
            out = Path(directory) / 'summary.json'
            subprocess.run([sys.executable, '-B', str(ROOT / 'tools/capture/overlay_review.py'),
                            '--dir', directory, '--out', str(out)], check=True,
                           capture_output=True)
            summary = json.loads(out.read_text())
            self.assertEqual(summary['frames'], 3)
            self.assertEqual(summary['battle_frames'], 2)
            self.assertEqual(summary['labels']['ENEMY']['claims'], 3)
            self.assertEqual(summary['labels']['ENEMY']['confidence_min'], 0.5)
            self.assertEqual(summary['labels']['ENEMY']['confidence_max'], 0.9)
            self.assertEqual(summary['labels']['CRUISER']['km_min'], 10.0)


class VerdictTests(unittest.TestCase):
    """Accuracy is what the operator judged, and it must survive a restart."""

    def test_a_verdict_is_appended_and_scored(self):
        import overlay_verdicts as V
        with tempfile.TemporaryDirectory() as directory:
            payload = dict(frame=12, battle=True,
                           claims=[dict(label='ENEMY', confidence=0.9)])
            V.record_verdict(directory, payload, 'correct')
            V.record_verdict(directory, payload, 'wrong')
            V.record_verdict(directory, payload, 'ambiguous')
            score = V.score_verdicts(directory)
            self.assertEqual(score['overall']['frames'], 3)
            self.assertEqual(score['overall']['correct'], 1)
            self.assertEqual(score['overall']['wrong'], 1)
            self.assertEqual(score['overall']['ambiguous'], 1)
            self.assertEqual(score['overall']['accuracy'], 0.5)
            self.assertEqual(score['per_label']['ENEMY']['accuracy'], 0.5)

    def test_an_unknown_verdict_is_refused(self):
        import overlay_verdicts as V
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                V.record_verdict(directory, dict(frame=1, claims=[]), 'probably')

    def test_an_empty_score_is_none_not_zero(self):
        import overlay_verdicts as V
        with tempfile.TemporaryDirectory() as directory:
            score = V.score_verdicts(directory)
            self.assertIsNone(score['overall']['accuracy'])
            self.assertEqual(score['overall']['decided'], 0)

    def test_the_review_summary_carries_the_verdicts(self):
        with tempfile.TemporaryDirectory() as directory:
            frames = Path(directory) / 'detections.jsonl'
            frames.write_text(json.dumps(dict(frame=1, battle=True,
                                              claims=[dict(kind='enemy_plate', label='ENEMY',
                                                           confidence=0.9)])) + '\n')
            import overlay_verdicts as V
            V.record_verdict(directory, dict(frame=1, battle=True,
                                             claims=[dict(label='ENEMY', confidence=0.9)]),
                             'correct')
            out = Path(directory) / 'summary.json'
            subprocess.run([sys.executable, '-B', str(ROOT / 'tools/capture/overlay_review.py'),
                            '--dir', directory, '--out', str(out)], check=True,
                           capture_output=True)
            summary = json.loads(out.read_text())
            self.assertEqual(summary['verdicts']['overall']['accuracy'], 1.0)


if __name__ == '__main__':
    unittest.main()
