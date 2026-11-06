#!/usr/bin/env python3
"""Live viewer for recognition_overlay.py: shows the annotated frame as it updates.

Run with the system interpreter (PyGObject lives there, not in .venv):

    /usr/bin/python3 tools/capture/overlay_viewer.py --dir state/overlay

A game cannot be annotated in place: gamescope keeps its client above every other
X client on its own display (measured), and the host session is Wayland, so there
is no window to draw over the game with. This viewer is the honest substitute --
it shows exactly the frame the perception chain saw, with its identity claims and
confidences drawn on it, so a wrong label is visible as a wrong label.

Keys: q or Esc quits, f toggles fullscreen, space pauses the refresh,
s saves the frame, and 1/2/3 record a verdict on it (correct / wrong / ambiguous).

A verdict is the only ground truth this project has for identity accuracy: the
person sees the annotated frame and says whether the claim was right. The
verdicts are appended to ``verdicts.jsonl`` in the same directory and scored by
``overlay_review.py``.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import overlay_verdicts as V  # noqa: E402

import gi

gi.require_version('Gtk', '3.0')
gi.require_version('Gdk', '3.0')
gi.require_version('GdkPixbuf', '2.0')
from gi.repository import Gdk, GdkPixbuf, GLib, Gtk  # noqa: E402


class Viewer(Gtk.Window):
    def __init__(self, directory, interval_ms, width, height, snapshot=None, exit_after=False):
        super().__init__(title='recognition viewer')
        self.directory = Path(directory)
        self.image_path = self.directory / 'latest.png'
        self.log_path = self.directory / 'detections.jsonl'
        self.stamp = None
        self.paused = False
        self.fullscreen = False
        self.frames = 0
        self.pixbuf = None
        self.scaled_for = None
        self.snapshot = Path(snapshot) if snapshot else None
        self.payload = None
        self.verdicts = []
        self.exit_after_snapshot = exit_after
        self.image = Gtk.Image()
        # The capture is 2560x1600; unscaled it is simply cropped by the window,
        # which is what made the picture impossible to see as a whole.
        self.image.connect('size-allocate', self.on_allocate)
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        box.pack_start(self.image, True, True, 0)
        self.status = Gtk.Label(label='waiting for the annotator…')
        self.status.set_xalign(0.0)
        box.pack_start(self.status, False, False, 4)
        self.add(box)
        self.set_default_size(width, height)
        self.connect('destroy', Gtk.main_quit)
        self.connect('key-press-event', self.on_key)
        GLib.timeout_add(interval_ms, self.refresh)

    def on_key(self, _widget, event):
        key = event.keyval
        if key in (113, 65293 + 1000):  # q
            Gtk.main_quit()
            return True
        if key == 65307:  # Escape
            if self.fullscreen:
                self.unfullscreen()
                self.fullscreen = False
            else:
                Gtk.main_quit()
            return True
        if key == 102:  # f
            self.fullscreen = not self.fullscreen
            (self.fullscreen if self.fullscreen else self.unfullscreen)()
            return True
        if key == 32:  # space
            self.paused = not self.paused
            return True
        if key == 115:  # s
            self.save_current()
            return True
        verdicts = {49: 'correct', 50: 'wrong', 51: 'ambiguous'}
        if key in verdicts:
            self.record(verdicts[key])
            return True
        return False

    def save_current(self):
        target = self.directory / f'saved-{time.strftime("%H%M%S")}.png'
        try:
            target.write_bytes(self.image_path.read_bytes())
        except OSError:
            return
        print(f'saved {target}', flush=True)

    def record(self, verdict):
        if not self.payload:
            print('verdict ignored: no frame loaded yet', flush=True)
            return
        try:
            row = V.record_verdict(self.directory, self.payload, verdict)
        except (ValueError, OSError) as error:
            print(f'verdict failed {error}', flush=True)
            return
        self.verdicts.append(row)
        score = V.score_verdicts(self.directory)['overall']
        print(f"verdict {verdict} frame {row['frame']} | "
              f"correct {score['correct']} wrong {score['wrong']} "
              f"ambiguous {score['ambiguous']} accuracy {score['accuracy']}", flush=True)

    def refresh(self):
        if self.paused:
            return True
        try:
            stamp = self.image_path.stat().st_mtime_ns
        except FileNotFoundError:
            return True
        if stamp != self.stamp:
            self.stamp = stamp
            # The claims come from boxes.json and describe THIS frame, so they are
            # read before the image: an unreadable PNG must not leave the viewer
            # unable to name what the operator is judging.
            try:
                self.payload = json.loads((self.directory / 'boxes.json').read_text())
            except (OSError, ValueError):
                self.payload = None
            try:
                pixbuf = GdkPixbuf.Pixbuf.new_from_file(str(self.image_path))
            except GLib.Error:
                return True
            self.pixbuf = pixbuf
            self.scaled_for = None
            self.apply_scale()
            self.frames += 1
            text = self.describe()
            self.status.set_text(text)
            # A service supervisor watches stdout for readiness; the first frame
            # on screen is the honest moment to say the viewer is up.
            print(f'viewer-ready {text}', flush=True) if self.frames == 1 else None
            if self.frames % 10 == 0:
                print(f'viewer-frame {text}', flush=True)
            if self.snapshot and self.frames == 1:
                GLib.timeout_add(700, self.take_snapshot)
        return True

    def on_allocate(self, *_args):
        self.apply_scale()
        return False

    def apply_scale(self):
        """Fit the frame into whatever the window currently is."""
        if self.pixbuf is None:
            return
        width = self.image.get_allocated_width()
        height = self.image.get_allocated_height()
        if width <= 1 or height <= 1:
            return
        source_w, source_h = self.pixbuf.get_width(), self.pixbuf.get_height()
        scale = min(width / source_w, height / source_h)
        target = (max(1, int(source_w * scale)), max(1, int(source_h * scale)))
        if target == self.scaled_for:
            return
        self.scaled_for = target
        self.image.set_from_pixbuf(self.pixbuf.scale_simple(
            target[0], target[1], GdkPixbuf.InterpType.BILINEAR))

    def take_snapshot(self):
        """Save what this window actually shows; proof the drawing reached pixels."""
        window = self.get_window()
        try:
            pixbuf = Gdk.pixbuf_get_from_window(window, 0, 0, window.get_width(),
                                                window.get_height())
        except Exception as error:  # pragma: no cover - depends on the compositor
            print(f'snapshot-failed {error}', flush=True)
            return False
        if pixbuf is None:
            print('snapshot-failed None', flush=True)
            return False
        pixbuf.savev(str(self.snapshot), 'png', [], [])
        print(f'snapshot {self.snapshot} {pixbuf.get_width()}x{pixbuf.get_height()}', flush=True)
        if self.exit_after_snapshot:
            GLib.timeout_add(200, Gtk.main_quit)
        return False

    def describe(self):
        text = f'frames shown {self.frames}'
        if self.payload:
            counts = {}
            for claim in self.payload.get('claims') or []:
                counts[claim['label']] = counts.get(claim['label'], 0) + 1
            summary = ', '.join(f'{k}×{v}' for k, v in sorted(counts.items())) or 'no claims'
            battle = self.payload.get('battle')
            tag = '' if battle is True else '  [not in battle: RAW]'
            return f"{text}  |  frame {self.payload.get('frame')}  |  {summary}{tag}"
        try:
            with self.log_path.open('rb') as handle:
                handle.seek(0, 2)
                size = handle.tell()
                handle.seek(max(0, size - 4096))
                tail = handle.read().decode('utf-8', 'replace').strip().splitlines()
            last = json.loads(tail[-1]) if tail else None
        except (OSError, ValueError):
            last = None
        if last:
            counts = {}
            for claim in last.get('claims', []):
                counts[claim['label']] = counts.get(claim['label'], 0) + 1
            summary = ', '.join(f'{k}×{v}' for k, v in sorted(counts.items())) or 'no claims'
            text += f"  |  frame {last.get('frame')}  |  {summary}"
        return text


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', required=True, help='the annotator output directory')
    ap.add_argument('--interval-ms', type=int, default=200)
    ap.add_argument('--width', type=int, default=900)
    ap.add_argument('--height', type=int, default=640)
    ap.add_argument('--snapshot', help='save the window pixels after the first frame')
    ap.add_argument('--exit-after-snapshot', action='store_true')
    args = ap.parse_args()
    viewer = Viewer(args.dir, args.interval_ms, args.width, args.height,
                    snapshot=args.snapshot, exit_after=args.exit_after_snapshot)
    viewer.show_all()
    Gtk.main()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
