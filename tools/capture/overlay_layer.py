#!/usr/bin/env python3
"""Draw the recognition claims directly over the game, as a Wayland layer-shell overlay.

Why this and not an X11 window: the game runs inside gamescope, and gamescope's
own compositor keeps its client above every other client on its X server
(measured 2026-10-23: an override-redirect, shaped, ABOVE-hinted window stayed
below the game), while the host session is Wayland with no X window to stack
over. A layer-shell surface on the OVERLAY layer is above normal windows and
above fullscreen clients, which is exactly the "boxes over the game" this is for.

It only reads ``boxes.json`` written by recognition_overlay.py; it never decides
anything, and it takes no input: the surface sets an empty input region and no
keyboard focus, so clicks and keys keep reaching the game.

Run with the system interpreter (PyGObject lives there):

    /usr/bin/python3 tools/capture/overlay_layer.py --dir state/overlay-live
"""
import argparse
import json
import sys
from pathlib import Path

import gi

gi.require_version('Gtk', '4.0')
gi.require_version('Gdk', '4.0')
gi.require_version('Pango', '1.0')
gi.require_version('Gtk4LayerShell', '1.0')
from gi.repository import Gdk, GLib, Graphene, Gtk, Pango  # noqa: E402
from gi.repository import Gtk4LayerShell as Layer  # noqa: E402

COLOURS = {
    'enemy_plate': (1.0, 0.24, 0.24, 1.0),
    'enemy_body': (1.0, 0.75, 0.24, 1.0),
    'own_ship': (0.35, 1.0, 0.47, 1.0),
    'contact': (1.0, 0.47, 1.0, 1.0),
    'hazard': (0.24, 0.78, 1.0, 1.0),
}
EXPAND = 8


class Overlay(Gtk.Widget):
    def __init__(self, directory, interval_ms):
        super().__init__()
        self.directory = Path(directory)
        self.boxes_path = self.directory / 'boxes.json'
        self.claims = []
        self.source = [2560, 1600]
        self.battle = None
        self.stamp = None
        self.frames = 0
        self.set_hexpand(True)
        self.set_vexpand(True)
        GLib.timeout_add(interval_ms, self.reload)

    def reload(self):
        try:
            stamp = self.boxes_path.stat().st_mtime_ns
        except OSError:
            return True
        if stamp != self.stamp:
            self.stamp = stamp
            try:
                payload = json.loads(self.boxes_path.read_text())
            except (OSError, ValueError):
                return True
            self.claims = payload.get('claims') or []
            self.source = payload.get('source_size') or [2560, 1600]
            self.battle = payload.get('battle')
            self.frames += 1
            self.queue_draw()
            if self.frames == 1:
                print(f'overlay-ready {self.describe()}', flush=True)
            elif self.frames % 20 == 0:
                print(f'overlay-frame {self.describe()}', flush=True)
        return True

    def describe(self):
        counts = {}
        for claim in self.claims:
            counts[claim['label']] = counts.get(claim['label'], 0) + 1
        return f'frame {self.frames} battle={self.battle} claims={counts}'

    def do_snapshot(self, snapshot):
        width, height = self.get_width(), self.get_height()
        if width <= 1 or height <= 1:
            return
        sx = width / float(self.source[0] or 2560)
        sy = height / float(self.source[1] or 1600)
        for claim in self.claims:
            colour = COLOURS.get(claim.get('kind'), (1.0, 1.0, 1.0, 1.0))
            if self.battle is not True:
                colour = (0.6, 0.6, 0.6, 1.0)
            box = claim.get('bbox')
            if box:
                x, y, w, h = [float(v) for v in box]
                rect = Graphene.Rect().init(max(0.0, (x - EXPAND) * sx), max(0.0, (y - EXPAND) * sy),
                                            (w + 2 * EXPAND) * sx, (h + 2 * EXPAND) * sy)
                snapshot.append_border(rect, [4, 4, 4, 4], [colour])
                text = f"{claim['label']} {float(claim['confidence']):.2f}"
                self.draw_text(snapshot, text, max(0.0, (x - EXPAND) * sx),
                               max(0.0, (y - EXPAND) * sy - 26), colour)
            pixel = claim.get('pixel')
            if pixel:
                px, py = float(pixel[0]) * sx, float(pixel[1]) * sy
                radius = 18 * max(sx, sy)
                circle = Graphene.Rect().init(px - radius, py - radius, radius * 2, radius * 2)
                snapshot.append_border(circle, [4, 4, 4, 4], [colour])
                text = f"{claim['label']} {float(claim['confidence']):.2f}"
                if claim.get('km') is not None:
                    text += f" {float(claim['km']):.1f}km"
                self.draw_text(snapshot, text, px + radius + 4, py - radius - 26, colour)

    def draw_text(self, snapshot, text, x, y, colour):
        layout = self.create_pango_layout(text)
        attributes = Pango.AttrList()
        attributes.insert(Pango.attr_scale_new(1.1))
        layout.set_attributes(attributes)
        snapshot.save()
        snapshot.translate(Graphene.Point().init(x, y))
        snapshot.append_layout(layout, Gdk.RGBA(*colour))
        snapshot.restore()


def build(directory, interval_ms):
    window = Gtk.Window()
    window.set_decorated(False)
    window.set_resizable(False)
    Layer.init_for_window(window)
    Layer.set_layer(window, Layer.Layer.OVERLAY)
    for edge in (Layer.Edge.TOP, Layer.Edge.BOTTOM, Layer.Edge.LEFT, Layer.Edge.RIGHT):
        Layer.set_anchor(window, edge, True)
    Layer.set_exclusive_zone(window, -1)
    Layer.set_keyboard_mode(window, Layer.KeyboardMode.NONE)
    overlay = Overlay(directory, interval_ms)

    def on_realize(widget):
        surface = widget.get_surface()
        if surface is not None:
            # Empty input region: the overlay must never take a click or a key
            # from the game. Measured need: a layer-shell surface otherwise owns
            # every pointer event inside its bounds.
            empty = Gdk.Region()
            surface.set_input_region(empty)
            print('overlay-input-region empty', flush=True)

    overlay.connect('realize', on_realize)
    window.set_child(overlay)
    return window, overlay


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', required=True, help='recognition_overlay.py output directory')
    ap.add_argument('--interval-ms', type=int, default=150)
    ap.add_argument('--seconds', type=float, default=0.0, help='0 runs until killed')
    args = ap.parse_args()
    if not Layer.is_supported():
        print('layer shell is not supported by this compositor', file=sys.stderr)
        return 3
    window, overlay = build(args.dir, args.interval_ms)
    window.present()
    if args.seconds:
        GLib.timeout_add(int(args.seconds * 1000), lambda: (window.close(), False)[1])
    loop = GLib.MainLoop()
    window.connect('close-request', lambda *_: (loop.quit(), False)[1])
    loop.run()
    print('overlay stopped', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
