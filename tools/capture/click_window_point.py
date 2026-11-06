#!/usr/bin/env python3
"""Click a lobby/menu point by RELATIVE motion, then verify the real pointer.

Measured on this client: an absolute `xdotool mousemove` warp is ignored while
the game keeps its own cursor, so a click lands wherever the pointer already is
(that is how an earlier "join battle" click opened a ship card instead). Relative
motion does move it, so this tool walks the pointer in bounded chunks, confirms
the final root position, and only then clicks.

Coordinates are native capture pixels: the saved frame is the game's content
area, while the window sits at y=46 on a 2560x1600 screen; `window_point` keeps
that mapping in one place.

Default is a dry run. A click is one 1..250 ms pulse; every path restores nothing
it did not change and reports whether a button was left held.
"""
import argparse
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'tools/capture'))

from probe_aim_input import Probe, ProbeError, game_identity, pointer_state
import resolution_map  # noqa: E402

MAX_CHUNK = 400
TOLERANCE_PX = 4
UINPUT = ROOT / 'state/cn360-tools/uinput_input'


def content_point(x, y, window_y=46, window_width=2560, window_height=1440):
    """Content-area pixel -> root-screen point for this client's window placement."""
    if not all(math.isfinite(v) for v in (x, y)):
        raise ValueError('point must be finite')
    try:
        local = resolution_map.content_to_window((x, y), resolution_map.REFERENCE_FRAME,
                                                 (window_width, window_height))
    except resolution_map.MappingError as exc:
        raise ValueError(str(exc)) from exc
    return local[0], window_y + local[1]


def chunks(dx, dy, limit=MAX_CHUNK):
    if not all(isinstance(v, int) for v in (dx, dy)):
        raise ValueError('relative motion must be integer pixels')
    out = []
    while dx or dy:
        step_x = max(-limit, min(limit, dx))
        step_y = max(-limit, min(limit, dy))
        out.append([step_x, step_y])
        dx -= step_x
        dy -= step_y
    return out


def kwin_cursor(script=None, timeout=2.):
    """Read the compositor's real cursor; the X pointer can be frozen by the game."""
    script = Path(script or (ROOT / 'tools/capture/kwin_cursor.js'))
    qdbus = ['qdbus6', 'org.kde.KWin', '/Scripting']
    subprocess.run([*qdbus, 'org.kde.kwin.Scripting.loadScript', str(script)],
                   capture_output=True, timeout=timeout)
    subprocess.run([*qdbus, 'org.kde.kwin.Scripting.start'],
                   capture_output=True, timeout=timeout)
    time.sleep(.35)
    log = subprocess.run(['journalctl', '--user', '-n', '60', '--no-pager', '-q'],
                         capture_output=True, text=True, timeout=timeout).stdout
    lines = [line for line in log.splitlines() if 'CURSOR {' in line]
    if not lines:
        raise ProbeError('compositor cursor position unavailable')
    return json.loads(lines[-1].split('CURSOR ', 1)[1])


def reached(pointer, target, tolerance=TOLERANCE_PX):
    return (abs(pointer['x'] - target[0]) <= tolerance
            and abs(pointer['y'] - target[1]) <= tolerance)


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    target = content_point(*args.point)
    report = dict(action='click_content_point', execute=args.execute,
                  content_point=list(args.point), screen_point=list(target),
                  out=str(args.out.resolve()),
                  inputs=[], pointer_samples=[], focus_checks=[], snapshots=[],
                  status='dry-run',
                  semantics='relative pointer walk then one click; no menu recognition')
    (args.out / 'plan.json').write_text(json.dumps(report, indent=2) + '\n')
    if not args.execute:
        print(json.dumps(report, indent=2))
        return 0

    probe = Probe(args.seconds, report)
    held_button = None
    handlers = {}

    def stop(signum, _):
        raise ProbeError(f'interrupted by signal {signum}')

    try:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM):
            handlers[sig] = signal.signal(sig, stop)
        signal.setitimer(signal.ITIMER_REAL,
                         max(.001, probe.action_deadline - time.monotonic()))
        probe.identity = game_identity()
        report['identity'] = probe.identity
        windows = []
        for window in probe.x('search', '--onlyvisible', '--name', '.').splitlines():
            try:
                probe.verify_window(window)
                windows.append(window)
            except ProbeError:
                continue
        if len(windows) != 1:
            raise ProbeError('expected exactly one verified game window')
        probe.window = windows[0]
        geometry = dict(line.split('=', 1) for line in
                        probe.x('getwindowgeometry', '--shell', probe.window).splitlines())
        origin = (int(geometry.get('X', 0)), int(geometry.get('Y', 0)))
        size = (int(geometry['WIDTH']), int(geometry['HEIGHT']))
        # The window sits at a measured origin (X=0, Y=30 here), not the old
        # hardcoded y=46; convert content pixels through both.
        local = resolution_map.content_to_window(args.point, resolution_map.REFERENCE_FRAME, size)
        target = (origin[0] + local[0], origin[1] + local[1])
        report['window_geometry'] = dict(origin=list(origin), size=list(size))
        report['screen_point'] = list(target)
        before_focus = probe.x('getwindowfocus')
        probe.x('windowactivate', '--sync', probe.window)
        report['activation'] = dict(window=probe.window, previous_focus=before_focus,
                                    focus_changed=before_focus != probe.window)
        probe.focus()

        for attempt in range(args.steps):
            # Nested gamescope session: the desktop compositor cursor lives in
            # another coordinate space (measured 722,625 vs the X pointer at
            # 962,799), so position and verify on the game's own display.
            pointer = dict(pointer_state(probe.identity['display']))
            report['pointer_samples'].append(pointer)
            mask = pointer_state(probe.identity['display'])['mask']
            if mask & (256 | 512 | 1024 | 2048 | 4096):
                raise ProbeError('pointer button already held; refusing ownership')
            if reached(pointer, target):
                break
            dx, dy = target[0] - pointer['x'], target[1] - pointer['y']
            for step_x, step_y in chunks(dx, dy):
                probe.focus()
                # Measured in this session: the kernel device does not move the
                # seat pointer at all (250 units left the X pointer unchanged),
                # while XTest relative motion does. Try the kernel device first
                # and fall back on evidence, never on faith.
                before = pointer_state(probe.identity['display'])
                result = subprocess.run([str(UINPUT), '--dx', str(step_x), '--dy', str(step_y)],
                                        capture_output=True, text=True,
                                        timeout=max(1., probe.action_deadline - time.monotonic()))
                if result.returncode:
                    raise ProbeError(f'uinput move failed: {result.stderr.strip()}')
                time.sleep(.05)
                after = pointer_state(probe.identity['display'])
                transport = 'uinput_virtual_mouse'
                if (after['x'], after['y']) == (before['x'], before['y']):
                    probe.x('mousemove_relative', '--', str(step_x), str(step_y))
                    transport = 'xtest_relative_fallback'
                    after = pointer_state(probe.identity['display'])
                report['inputs'].append(dict(relative=[step_x, step_y], transport=transport,
                                             pointer=[after['x'], after['y']]))
                if (after['x'], after['y']) == (before['x'], before['y']):
                    raise ProbeError('pointer did not move with either transport')
                time.sleep(.05)
        else:
            raise ProbeError(f'pointer did not reach {target} within {args.steps} steps')

        final = dict(pointer_state(probe.identity['display']))
        report['pointer_samples'].append(final)
        if not reached(final, target):
            raise ProbeError(f'pointer stopped at {final} short of {target}')

        probe.focus()
        # Measured: this kernel device does not deliver the press either (a
        # hover tooltip appears and the button never toggles), while XTest
        # clicks the launcher and the menu buttons. Use the transport that is
        # proven to work on this display, with the same bounded hold.
        held_button = 1
        probe.x('mousedown', 1)
        time.sleep(max(1, args.button_hold_ms)/1000)
        probe.focus()
        probe.x('mouseup', 1)
        held_button = None
        report['inputs'].append(dict(button=1, pulse=True, hold_ms=args.button_hold_ms,
                                     transport='xtest',
                                     screen_point=list(target)))
        probe.snapshot('after')
        report['status'] = 'completed'
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if held_button:
            try:
                probe.check_identity()
                probe.x('mouseup', held_button, cleanup=True)
                held_button = None
            except Exception as error:
                report.setdefault('cleanup_errors', []).append(str(error))
        probe.release()
        report['unreleased_button'] = held_button is not None
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status=report['status'], error=report.get('error'),
                          screen_point=report['screen_point'],
                          inputs=len(report['inputs']))))
    return 0 if report['status'] == 'completed' else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--point', type=float, nargs=2, required=True,
                   help='content-area pixel in the saved 2560x1440 game frame')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--seconds', type=float, default=20.)
    p.add_argument('--steps', type=int, default=8)
    p.add_argument('--button-hold-ms', type=int, default=120)
    a = p.parse_args(argv)
    if not 5 <= a.seconds <= 30:
        p.error('seconds must be 5..30')
    if not 1 <= a.steps <= 20:
        p.error('steps must be 1..20')
    if not 0 <= a.button_hold_ms <= 250:
        p.error('button-hold-ms must be 0..250')
    return run(a)


if __name__ == '__main__':
    raise SystemExit(main())
