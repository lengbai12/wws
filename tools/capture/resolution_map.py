#!/usr/bin/env python3
"""Map reference-frame image coordinates onto whatever resolution is live now.

Every HUD rectangle, nameplate coordinate and click point in this project is
written in the *reference* space: a 2560x1440 game surface inside a 2560x1600
captured canvas (80 px bars above and below, because the canvas is 16:10 and
the game is 16:9).

The client does not always render that size. Measured on this machine:

* 2560x1440 in a 2560x1600 canvas: reference 1:1.
* 1924x1058 window inside a 1920x1080 screen: 0.7516 x 0.7347 — non-uniform,
  so a single "aspect ratio" factor is wrong and bars are not implied.
* 1920x1080 window: 0.75 x 0.75.

This module owns one conversion so tools stop re-deriving it (and stop
rejecting windows whose aspect is not exactly 16:9). It never guesses: a
missing window geometry raises, and a point outside the content area is
rejected instead of being silently clamped.

Detect with ``detect()``; convert with ``capture_to_window`` /
``window_to_capture`` / ``resize_rect``. Pass ``--geometry WxH`` to override
detection when the window cannot be queried.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

REFERENCE_FRAME = (2560, 1600)
CONTENT = (2560, 1440)
BAR_Y = 80


class MappingError(RuntimeError):
    pass


def content_from_frame(frame):
    """Content rectangle inside a captured frame: bars top and bottom at reference size."""
    if not frame or len(frame) != 2:
        raise MappingError('frame size must be (width, height)')
    width, height = (int(v) for v in frame)
    if width <= 0 or height <= 0:
        raise MappingError('frame size must be positive')
    # A canvas that is not the reference canvas may be a direct grab with no
    # bars at all; only strip bars when the proportions match the reference.
    if abs(height/width - REFERENCE_FRAME[1]/REFERENCE_FRAME[0]) < 1e-3:
        bar = round(BAR_Y * height / REFERENCE_FRAME[1])
        return (width, height - 2*bar, 0, bar)
    return (width, height, 0, 0)


def parse_geometry(text):
    """'1924x1058' -> (1924, 1058)."""
    if not isinstance(text, str) or 'x' not in text:
        raise MappingError(f'geometry must look like WxH, got {text!r}')
    left, right = text.split('x', 1)
    try:
        width, height = int(left), int(right)
    except ValueError as exc:
        raise MappingError(f'geometry must look like WxH, got {text!r}') from exc
    if width <= 0 or height <= 0:
        raise MappingError('geometry must be positive')
    return width, height


def scale(frame, window):
    """(sx, sy) mapping content pixels onto the window client area."""
    content_w, content_h, _, _ = content_from_frame(frame)
    win_w, win_h = (int(v) for v in window)
    if content_w <= 0 or content_h <= 0 or win_w <= 0 or win_h <= 0:
        raise MappingError('content and window must be positive')
    return win_w/content_w, win_h/content_h


def capture_to_window(point, frame, window):
    """Reference capture pixel -> window client pixel."""
    x, y = (float(v) for v in point)
    content_w, content_h, _, bar = content_from_frame(frame)
    if not (0 <= x < content_w and 0 <= y < content_w + 2*bar):
        raise MappingError(f'point {point} outside the captured frame')
    content_y = y - bar
    if not (0 <= content_y < content_h):
        raise MappingError(f'point {point} lies in the capture bars, not content')
    sx, sy = scale(frame, window)
    return round(x*sx), round(content_y*sy)


def window_to_capture(point, frame, window):
    """Window client pixel -> reference capture pixel."""
    wx, wy = (float(v) for v in point)
    win_w, win_h = (int(v) for v in window)
    if not (0 <= wx < win_w and 0 <= wy < win_h):
        raise MappingError(f'point {point} outside the window')
    sx, sy = scale(frame, window)
    content_w, _, _, bar = content_from_frame(frame)
    return round(wx/sx), round(wy/sy) + bar


def content_to_window(point, frame, window):
    """Content-area pixel (0..2559, 0..1439) -> window client pixel."""
    x, y = (float(v) for v in point)
    content_w, content_h, _, _ = content_from_frame(frame)
    if not (0 <= x < content_w and 0 <= y < content_h):
        raise MappingError(f'point {point} outside the content area')
    sx, sy = scale(frame, window)
    return round(x*sx), round(y*sy)


def window_to_content(point, frame, window):
    """Window client pixel -> content-area pixel."""
    wx, wy = (float(v) for v in point)
    win_w, win_h = (int(v) for v in window)
    if not (0 <= wx < win_w and 0 <= wy < win_h):
        raise MappingError(f'point {point} outside the window')
    sx, sy = scale(frame, window)
    return round(wx/sx), round(wy/sy)


def resize_rect(rect, frame, window):
    """Reference-space [x, y, w, h] -> window-space rectangle for this window."""
    if len(rect) != 4:
        raise MappingError('rect must be [x, y, w, h]')
    x, y, w, h = (float(v) for v in rect)
    x0, y0 = capture_to_window((x, y), frame, window)
    x1, y1 = capture_to_window((x+max(0, w), y+max(0, h)), frame, window)
    return [x0, y0, max(1, x1-x0), max(1, y1-y0)]


def window_geometry(display, window):
    out = subprocess.run(['xdotool', 'getwindowgeometry', '--shell', str(window)],
                         capture_output=True, text=True, timeout=10,
                         env={'DISPLAY': display, 'PATH': '/usr/bin:/bin'})
    if out.returncode:
        raise MappingError(f'window geometry unavailable: {out.stderr.strip()}')
    values = dict(line.split('=', 1) for line in out.stdout.splitlines())
    return int(values['WIDTH']), int(values['HEIGHT'])


def detect(display=None, window=None, frame=None):
    """Report the live frame, window and derived scale, or raise."""
    frame = tuple(frame) if frame else REFERENCE_FRAME
    if window is None:
        if not display or window is None:
            raise MappingError('detect needs a display and a window id')
        window = window_geometry(display, window)
    content_w, content_h, _, bar = content_from_frame(frame)
    sx, sy = scale(frame, window)
    return dict(reference_frame=list(REFERENCE_FRAME), frame=list(frame),
                content=[content_w, content_h], bar_y=bar,
                window=list(window), scale=[sx, sy],
                uniform=bool(math.isclose(sx, sy, rel_tol=1e-3)),
                semantics='per-axis content-to-window mapping; no aspect assumption')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--display', help='X display of the game (e.g. :2)')
    parser.add_argument('--window', help='window id')
    parser.add_argument('--geometry', help='WxH override when the window cannot be queried')
    parser.add_argument('--frame', help='captured frame WxH (default 2560x1600)')
    parser.add_argument('--point', nargs=2, type=float, help='capture point to map')
    parser.add_argument('--rect', nargs=4, type=float, help='capture rect to map')
    args = parser.parse_args(argv)
    frame = parse_geometry(args.frame) if args.frame else REFERENCE_FRAME
    try:
        window = (parse_geometry(args.geometry) if args.geometry
                  else window_geometry(args.display, args.window))
        out = detect(args.display, window, frame)
        if args.point:
            out['point'] = list(args.point)
            out['window_point'] = list(capture_to_window(args.point, frame, window))
        if args.rect:
            out['rect'] = list(args.rect)
            out['window_rect'] = resize_rect(args.rect, frame, window)
    except MappingError as exc:
        print(json.dumps({'status': 'error', 'reason': str(exc)}))
        return 1
    print(json.dumps(out, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
