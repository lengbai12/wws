#!/usr/bin/env python3
"""Exercise the viewer's verdict path without a keyboard.

Run with the system interpreter (PyGObject):

    /usr/bin/python3 -B -m unittest tools/capture/test_overlay_viewer.py

The key press itself is GTK's business; what matters here is that the handler the
key is wired to records the frame currently on screen, and that the score it
prints comes from the same file the review reads.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault('GDK_BACKEND', 'x11' if os.environ.get('DISPLAY') else 'wayland')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/capture'))

import overlay_verdicts as V  # noqa: E402


def load_viewer():
    try:
        import overlay_viewer as module
        module.Gtk.init([])
    except Exception as error:  # pragma: no cover - depends on a running session
        raise unittest.SkipTest(f'no GTK session: {error}')
    return module


class VerdictPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_viewer()

    def test_a_key_press_records_the_frame_on_screen(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            payload = dict(frame=41, battle=True,
                           claims=[dict(label='ENEMY', confidence=0.91)])
            (directory / 'boxes.json').write_text(json.dumps(payload))
            (directory / 'latest.png').write_bytes(b'not-really-a-png')
            (directory / 'detections.jsonl').write_text(json.dumps(dict(frame=41, claims=payload['claims'])) + '\n')
            viewer = self.module.Viewer(directory, 10_000, 400, 300)
            viewer.refresh()          # load the payload without needing a mapped window
            viewer.paused = True
            self.assertIsNotNone(viewer.payload)
            viewer.record('wrong')
            rows = [json.loads(line) for line in V.verdict_path(directory).read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['verdict'], 'wrong')
            self.assertEqual(rows[0]['frame'], 41)
            self.assertEqual(rows[0]['claims'][0]['label'], 'ENEMY')
            score = V.score_verdicts(directory)['overall']
            self.assertEqual(score['wrong'], 1)
            self.assertEqual(score['accuracy'], 0.0)

    def test_a_verdict_before_any_frame_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            viewer = self.module.Viewer(Path(directory), 10_000, 400, 300)
            viewer.payload = None
            viewer.record('correct')
            self.assertFalse(V.verdict_path(directory).exists())


if __name__ == '__main__':
    unittest.main()
