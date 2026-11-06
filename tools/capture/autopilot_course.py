#!/usr/bin/env python3
"""Set a WoWS autopilot course by clicking an ocean point on the tactical map.

The game's own help states it: press M for the tactical map, left-click to set a
course, press M again to leave. In battle the game tracks its own cursor and
ignores absolute warps (measured), so the click is delivered as relative mouse
motion from the pointer's current position, split into bounded chunks.

Default is a dry-run. Execute requires one verified game process and window,
re-verifies identity and focus before every action, and releases every key it
holds. The caller must supply a point it has actually looked at: the tactical
map pans and zooms with interaction, so a remembered pixel is not reliable.
"""
import argparse
import json
import math
from pathlib import Path
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'tools/capture'))

from probe_aim_input import Probe, ProbeError, game_identity, pointer_state
import resolution_map

MAX_CHUNK = 400


def window_point(x, y, width, height):
    """Capture-space point (native 2560x1600 frame) to window coordinates."""
    if not all(math.isfinite(v) for v in (x, y, width, height)):
        raise ValueError('invalid capture point or window geometry')
    try:
        return resolution_map.capture_to_window((x, y), resolution_map.REFERENCE_FRAME,
                                                (width, height))
    except resolution_map.MappingError as exc:
        raise ValueError(str(exc)) from exc


def chunk_deltas(dx, dy, limit=MAX_CHUNK):
    """Split one relative move into chunks no larger than limit per axis."""
    chunks = []
    while dx or dy:
        step_x = max(-limit, min(limit, dx))
        step_y = max(-limit, min(limit, dy))
        chunks.append([step_x, step_y])
        dx -= step_x
        dy -= step_y
    return chunks


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(action='set_autopilot_course', execute=args.execute, point=list(args.point),
                  open_map=not args.no_open_map, close_map=not args.keep_map_open,
                  out=str(args.out.resolve()), inputs=[], snapshots=[], focus_checks=[],
                  status='dry-run',
                  semantics='sets the game autopilot course; no claim about route viability')
    (args.out/'plan.json').write_text(json.dumps(report, indent=2)+'\n')
    if not args.execute:
        print(json.dumps(report, indent=2))
        return 0
    probe = Probe(args.seconds, report)
    handlers, timer, map_open = {}, (0., 0.), False
    try:
        def stop(signum, frame):
            raise ProbeError(f'interrupted by signal {signum}')
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM):
            handlers[sig] = signal.signal(sig, stop)
        timer = signal.setitimer(signal.ITIMER_REAL, max(.001, probe.action_deadline-time.monotonic()))
        probe.identity = game_identity()
        report['identity'] = probe.identity
        windows = []
        for window in probe.x('search', '--onlyvisible', '--name', '.').splitlines():
            try:
                probe.verify_window(window); windows.append(window)
            except ProbeError:
                continue
        if len(windows) != 1:
            raise ProbeError('expected exactly one verified game window')
        probe.window = windows[0]
        geometry = dict(line.split('=', 1) for line in
                        probe.x('getwindowgeometry', '--shell', probe.window).splitlines())
        width, height = int(geometry['WIDTH']), int(geometry['HEIGHT'])
        report['window_geometry'] = [width, height]
        if not args.no_open_map:
            probe.key(args.map_key)
            report['inputs'].append(dict(key=args.map_key, reason='open_tactical_map'))
            map_open = True
            time.sleep(args.settle)
        target = window_point(args.point[0], args.point[1], width, height)
        report['target_window_point'] = list(target)
        pointer = pointer_state(probe.identity['display'])
        report['pointer_before'] = [pointer['x'], pointer['y']]
        chunks = chunk_deltas(target[0]-pointer['x'], target[1]-pointer['y'])
        for index, chunk in enumerate(chunks):
            probe.move(chunk[0], chunk[1])
            report['inputs'][-1].update(step=index, reason='relative_chunk')
            time.sleep(args.step_settle)
        pointer = pointer_state(probe.identity['display'])
        report['pointer_at_click'] = [pointer['x'], pointer['y']]
        report['click_offset_from_target'] = [pointer['x']-target[0], pointer['y']-target[1]]
        held = None
        try:
            probe.focus()
            held = 1
            probe.x('mousedown', 1)
            if args.button_hold_ms:
                time.sleep(args.button_hold_ms/1000)
                probe.focus()
            probe.x('mouseup', 1)
            held = None
            report['inputs'].append(dict(button=1, pulse=True, hold_ms=args.button_hold_ms,
                                         reason='set_course'))
        finally:
            if held:
                probe.check_identity()
                probe.x('mouseup', held, cleanup=True)
        if args.settle:
            time.sleep(args.settle)
        report['after_click'] = str(probe.snapshot('after_click'))
        if not args.keep_map_open:
            probe.key(args.map_key)
            report['inputs'].append(dict(key=args.map_key, reason='close_tactical_map'))
            map_open = False
            time.sleep(args.settle)
        report['after_close'] = str(probe.snapshot('after_close'))
        report['status'] = 'completed'
    except (Exception, KeyboardInterrupt) as error:
        report.update(status='failed', error=str(error))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if map_open and not args.keep_map_open:
            try:
                probe.key(args.map_key)
                report['inputs'].append(dict(key=args.map_key, reason='close_map_cleanup'))
            except Exception as error:
                report.setdefault('cleanup_errors', []).append(f'map exit: {error}')
        probe.release()
        report['unreleased_button'] = False
        report['unreleased_own_keys'] = report.get('unreleased_own_keys', [])
        report['elapsed_s'] = time.monotonic()-probe.started
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        signal.setitimer(signal.ITIMER_REAL, *timer)
        (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return int(report['status'] != 'completed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--point', type=float, nargs=2, required=True,
                        help='ocean point on the tactical map, in native capture pixels (x y)')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--no-open-map', action='store_true',
                        help='the tactical map is already open')
    parser.add_argument('--keep-map-open', action='store_true')
    parser.add_argument('--map-key', default='m')
    parser.add_argument('--settle', type=float, default=.8)
    parser.add_argument('--step-settle', type=float, default=.15)
    parser.add_argument('--button-hold-ms', type=int, default=120)
    parser.add_argument('--seconds', type=float, default=30.)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not 5 <= args.seconds <= 60:
        parser.error('seconds must be 5..60')
    for value in args.point:
        if not math.isfinite(value):
            parser.error('point must be finite')
    if not (0 <= args.point[0] < 2560 and 80 <= args.point[1] < 1520):
        parser.error('point must lie inside the 2560x1440 content area')
    if not 0 <= args.button_hold_ms <= 250:
        parser.error('button-hold-ms must be 0..250')
    window_point(args.point[0], args.point[1], 1920, 1080)
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
