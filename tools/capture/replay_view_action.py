#!/home/lengbai/Downloads/wws/.venv/bin/python
"""One explicit replay-view action; default dry-run, identity/focus guarded.

Click coordinates are native capture pixels within the 2560x1440 content area.
No automatic menu recognition or live combat action is inferred. SIGINT/TERM
and the total time bound release only this command's acquired button/key.
"""
import argparse
import json
import math
from pathlib import Path
import re
import signal
import time

from probe_aim_input import Probe, ProbeError, game_identity, pointer_state
import resolution_map


class WindowActionProbe(Probe):
    """Explicit desktop-game capture, never a fallback from a missing host source."""
    def snapshot(self, name):
        from wows_window_capture import WindowCapture
        capture = WindowCapture()
        if capture.identity != self.identity or capture.window != self.window:
            raise ProbeError('action and window capture identities differ')
        path = Path(self.report['out']) / f'{name}.png'
        meta = capture.grab(path, wait_update=True)
        self.check_identity()
        self.report['snapshots'].append(dict(meta, name=name, path=str(path)))
        return path


def window_point(x, y, width, height):
    """Capture pixel -> window pixel through the shared resolution map."""
    if not all(math.isfinite(v) for v in (x, y, width, height)):
        raise ValueError('invalid capture point or window geometry')
    # The game is a 16:9 client; this one measures 1924x1058 (1.819) in windowed
    # mode, so keep a measured band rather than a strict ratio, and never map a
    # window that is not plausibly the game at all.
    if not 1.70 <= width / height <= 1.90:
        raise ValueError('unsupported game window aspect ratio')
    try:
        return resolution_map.capture_to_window((x, y), resolution_map.REFERENCE_FRAME,
                                                (width, height))
    except resolution_map.MappingError as exc:
        raise ValueError(str(exc)) from exc


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(action=args.action, execute=args.execute, seconds=args.seconds,
                  inputs=[], focus_checks=[], snapshots=[], out=str(args.out.resolve()),
                  capture_point=args.point, status='dry-run')
    report['key_hold_ms'] = getattr(args, 'key_hold_ms', 0)
    button_hold_ms = getattr(args, 'button_hold_ms', 0)
    if not 0 <= button_hold_ms <= 250:
        raise ValueError('button-hold-ms must be 0..250')
    report['button_hold_ms'] = button_hold_ms
    capture_source = getattr(args, 'capture_source', 'host')
    report['capture_source'] = capture_source
    (args.out/'plan.json').write_text(json.dumps(report, indent=2)+'\n')
    if not args.execute:
        print(json.dumps(report, indent=2)); return 0
    probe = (WindowActionProbe if capture_source == 'window' else Probe)(args.seconds, report)
    held_button = None
    old_pointer = None
    handlers = {}
    timer = (0., 0.)
    def stop(signum, frame):
        raise ProbeError(f'interrupted by signal {signum}')
    try:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM):
            handlers[sig] = signal.signal(sig, stop)
        timer = signal.setitimer(signal.ITIMER_REAL, max(.001, probe.action_deadline-time.monotonic()))
        probe.identity = game_identity()
        report['identity'] = probe.identity
        candidates = probe.x('search', '--onlyvisible', '--name', '.').splitlines()
        windows = []
        for window in candidates:
            try:
                probe.verify_window(window); windows.append(window)
            except ProbeError:
                continue
        if len(windows) != 1:
            raise ProbeError('expected exactly one verified game window')
        probe.window = windows[0]
        if getattr(args, 'activate', False):
            # The game only receives input while focused. Activation is an
            # explicit, recorded side effect, never an implicit convenience.
            before = probe.x('getwindowfocus')
            probe.x('windowactivate', '--sync', probe.window)
            report['activation'] = dict(window=probe.window,
                                        previous_focus=before,
                                        focus_changed=before != probe.window,
                                        transport='xdotool windowactivate')
        probe.focus()
        probe.snapshot('before')
        if args.action == 'click':
            old_pointer = pointer_state(probe.identity['display'])
            if old_pointer['mask'] & (256 | 512 | 1024 | 2048 | 4096):
                raise ProbeError('pointer button already held; refusing ownership')
            geometry = dict(line.split('=', 1) for line in probe.x('getwindowgeometry', '--shell', probe.window).splitlines())
            point = window_point(*args.point, int(geometry['WIDTH']), int(geometry['HEIGHT']))
            report['window_point'] = list(point)
            probe.focus()
            probe.x('mousemove', '--window', probe.window, *point)
            probe.focus()
            held_button = 1
            probe.x('mousedown', 1)
            if button_hold_ms:
                time.sleep(button_hold_ms / 1000)
                probe.focus()
            probe.x('mouseup', 1)
            held_button = None
            report['inputs'].append(dict(button=1, pulse=True, hold_ms=button_hold_ms,
                                         window_point=list(point)))
        elif args.action in ('zoom-in', 'zoom-out'):
            probe.focus()
            state = pointer_state(probe.identity['display'])
            if state['mask'] & (256 | 512 | 1024 | 2048 | 4096):
                raise ProbeError('pointer button already held; refusing ownership')
            button = 4 if args.action == 'zoom-in' else 5
            held_button = button
            probe.x('mousedown', button)
            probe.x('mouseup', button)
            held_button = None
            report['inputs'].append(dict(wheel_button=button, ticks=1))
        elif args.action == 'minimap-max':
            for _ in range(6):
                for key in ('KP_Add', 'plus'):
                    probe.key(key)
        elif args.action == 'key':
            probe.key(args.key_name)
        elif args.action in ('relative-click', 'relative-move'):
            # In-battle UI: the game tracks its own cursor from relative motion,
            # so an absolute warp click lands wherever that cursor already was.
            probe.focus()
            probe.x('mousemove_relative', '--', args.relative[0], args.relative[1])
            report['inputs'].append(dict(relative=list(args.relative),
                                         button=None if args.action == 'relative-move' else 1))
            if args.action == 'relative-click':
                probe.focus()
                held_button = 1
                probe.x('mousedown', 1)
                if button_hold_ms:
                    time.sleep(button_hold_ms / 1000)
                    probe.focus()
                probe.x('mouseup', 1)
                held_button = None
                report['inputs'].append(dict(button=1, pulse=True, hold_ms=button_hold_ms,
                                             relative=list(args.relative)))
        else:
            probe.key({'pause':'End', 'normal-speed':'Home', 'camera-link':'c',
                       'scope':'Shift_L', 'main-guns':'1', 'rewind':'comma',
                       'forward':'period', 'minimap-enlarge':'KP_Add',
                       'observation-zoom-in':'Prior', 'observation-zoom-out':'Next'}[args.action])
        probe.snapshot('after')
        report['status'] = 'completed'
    except (Exception, KeyboardInterrupt) as error:
        report.update(status='failed', error=str(error))
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
        if old_pointer is not None:
            try:
                probe.focus()
                probe.x('mousemove', old_pointer['x'], old_pointer['y'], cleanup=True)
            except Exception as error:
                report.setdefault('cleanup_errors', []).append(str(error))
        report['unreleased_button'] = held_button is not None
        report['elapsed_s'] = time.monotonic()-probe.started
        if held_button or report.get('unreleased_own_keys') or report.get('cleanup_errors'):
            report['status'] = 'failed'
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        signal.setitimer(signal.ITIMER_REAL, *timer)
        (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return int(report['status'] != 'completed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('click','relative-click','relative-move','pause','normal-speed','camera-link','scope','main-guns','zoom-in','zoom-out','rewind','forward','minimap-enlarge','minimap-max','observation-zoom-in','observation-zoom-out','key'), required=True)
    parser.add_argument('--point', type=float, nargs=2)
    parser.add_argument('--relative', type=int, nargs=2, help='relative mouse dx dy for --action relative-click')
    parser.add_argument('--key-name', help='X keysym for --action key (Escape, Return, x, 1, w ...)')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--activate', action='store_true',
                        help='activate the verified game window before input; recorded, never implicit')
    parser.add_argument('--capture-source', choices=('host', 'window'), default='host',
                        help='explicit identity-bound window PNGs for desktop games; no render-age claim')
    parser.add_argument('--seconds', type=float, default=15.)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--key-hold-ms', type=int, default=0)
    parser.add_argument('--button-hold-ms', type=int, default=0,
                        help='explicit menu-click hold duration, 0..250 ms')
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 5 <= args.seconds <= 30:
        parser.error('seconds must be 5..30')
    if not 0 <= args.key_hold_ms <= 250:
        parser.error('key-hold-ms must be 0..250')
    if not 0 <= args.button_hold_ms <= 250:
        parser.error('button-hold-ms must be 0..250')
    if (args.action == 'key') != (args.key_name is not None):
        parser.error('key-name is required only for --action key')
    if args.key_name is not None and not re.fullmatch(r'[A-Za-z0-9_+.-]{1,24}', args.key_name):
        parser.error('key-name must be a plain X keysym name')
    if (args.action in ('relative-click', 'relative-move')) != (args.relative is not None):
        parser.error('relative is required only for --action relative-click/relative-move')
    if args.relative is not None and max(abs(v) for v in args.relative) > 400:
        parser.error('relative must have each axis bounded to 400')
    if (args.action == 'relative-move' and args.relative == [0, 0]):
        parser.error('relative-move needs a nonzero delta')
    if (args.action == 'click') != (args.point is not None):
        parser.error('point is required only for click')
    if args.point:
        window_point(*args.point, 1920, 1080)
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
