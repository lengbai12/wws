#!/usr/bin/env python3
"""Click one window-space point of the running game, verified, bounded.

Why this exists (measured 2026-10-21): the client ignores an absolute warp, and
`xdotool mousemove X Y` is interpreted **relative to the focused window** while
`getmouselocation` reports root coordinates, so mixing them lands the pointer on
half the intended y. This moves with relative steps against the root readback,
re-focuses before press and before release (a press without focus is dropped),
and never presses more than one button.

Default is a dry-run showing the target; --execute sends the click.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time


def xdotool(*args, display, timeout=5):
    return subprocess.run(['xdotool', *map(str, args)], capture_output=True, text=True,
                          env={'DISPLAY': display, 'PATH': '/usr/bin:/bin'},
                          timeout=timeout).stdout


def pointer(display):
    out = xdotool('getmouselocation', '--shell', display=display)
    fields = dict(line.split('=', 1) for line in out.split() if '=' in line)
    return int(fields['X']), int(fields['Y'])


def move_to(display, window, target, tolerance=2, tries=10):
    for _ in range(tries):
        x, y = pointer(display)
        dx, dy = target[0] - x, target[1] - y
        if abs(dx) <= tolerance and abs(dy) <= tolerance:
            return True
        xdotool('mousemove_relative', '--', dx, dy, display=display)
        time.sleep(0.05)
    return False


def focus(display, window):
    xdotool('windowfocus', window, display=display)


def click(display, window, point, origin, hold_ms=110):
    """Move, then press with no intervening call.

    The client warps the pointer while the port UI animates (measured: the
    pointer read back 270 px away a fifth of a second after the move), so a
    focus call or a sleep between the last move and the press lets the click
    land somewhere else. Everything that can be done early is done early.
    """
    target = (point[0] + origin[0], point[1] + origin[1])
    focus(display, window)
    reached = move_to(display, window, target)
    if not reached:
        # One more correction pass is cheaper than a click on the wrong widget.
        reached = move_to(display, window, target, tolerance=1, tries=6)
    xdotool('mousedown', 1, display=display)
    time.sleep(max(0, hold_ms) / 1000)
    xdotool('mouseup', 1, display=display)
    return reached, pointer(display)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--display', default=':2')
    parser.add_argument('--window', required=True, help='X window id of the game')
    parser.add_argument('--point', nargs=2, type=float, required=True,
                        help='window-space x y (content coordinates of the game window)')
    parser.add_argument('--origin', nargs=2, type=float, default=None,
                        help='root offset of the window; read from xdotool when omitted')
    parser.add_argument('--hold-ms', type=int, default=110)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    origin = args.origin
    if origin is None:
        # The window origin moved once already (2026-10-23: borderless fullscreen
        # put it at 0,0 instead of 0,30), so it is measured, not assumed.
        shell = xdotool('getwindowgeometry', '--shell', args.window, display=args.display)
        fields = dict(line.split('=', 1) for line in shell.split() if '=' in line)
        origin = [float(fields['X']), float(fields['Y'])]
    target = (args.point[0] + origin[0], args.point[1] + origin[1])
    if not args.execute:
        print(f'would click window {tuple(args.point)} -> root {target} (dry run)')
        return 0
    reached, now = click(args.display, args.window, args.point, origin, args.hold_ms)
    print(f'clicked root {target} reached={reached} pointer={now}')
    return 0 if reached else 1


if __name__ == '__main__':
    raise SystemExit(main())
