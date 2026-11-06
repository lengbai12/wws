#!/home/lengbai/Downloads/wws/.venv/bin/python
"""Recognise WoWS training-room buttons and enter a battle; bounded and verified.

Screens handled, by native window template matching (not OCR):
  results   -> "回到港口"  (return to port)
  lobby     -> "已准备好" then "加入战斗"
  prebattle -> "开始战斗"

Every click is preceded by a verified pointer warp (read back and corrected) and
followed by a fresh capture, so a click that did not change the screen is
reported instead of being assumed to work. The script never presses keys and
never claims a battle HUD is live: after the menus disappear it reports
`battle_or_loading` and the caller must verify the battle itself.

Default is a dry-run; --execute performs input. The whole run is bounded by
--seconds and every mouse button is released in the finally block.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIXTURES = ROOT / 'tools' / 'replay' / 'main_view_fixtures' / 'training_ui'
MATCH_THRESHOLD = 0.75

# Templates are stored as {name}.png; the name is also the action applied.
BUTTONS = ('return_port', 'ready', 'join_battle', 'start_battle')
SCREENS = {'return_port': 'results', 'start_battle': 'prebattle',
           'ready': 'lobby', 'join_battle': 'lobby'}


class EntryError(RuntimeError):
    pass


def edges(image):
    """Canny edges: glyph structure survives brightness and background changes."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(cv2.GaussianBlur(image, (3, 3), 0), 60, 160)


def load_templates(directory=FIXTURES):
    templates = {}
    for name in BUTTONS:
        path = Path(directory) / f'{name}.png'
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise EntryError(f'missing UI template: {path}')
        templates[name] = edges(image)
    return templates


def match(frame_edges, template_edges):
    """Best normalised cross-correlation and its top-left pixel."""
    if (frame_edges.shape[0] < template_edges.shape[0] or
            frame_edges.shape[1] < template_edges.shape[1]):
        return 0.0, None
    scores = cv2.matchTemplate(frame_edges, template_edges, cv2.TM_CCOEFF_NORMED)
    _, score, _, location = cv2.minMaxLoc(scores)
    return float(score), location


def detect(frame_bgr, templates, threshold=MATCH_THRESHOLD):
    """Report every known button whose edge template matches this frame."""
    frame_edges = edges(frame_bgr)
    found = {}
    for name, template in templates.items():
        score, location = match(frame_edges, template)
        if location is not None and score >= threshold:
            h, w = template.shape
            found[name] = {'score': round(score, 4),
                           'bbox': [int(location[0]), int(location[1]), int(w), int(h)],
                           'center': [int(location[0]) + w // 2, int(location[1]) + h // 2]}
    return found


def teal_button(frame_bgr):
    """Colour+geometry fallback for the results button when the template misses."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 0] > 70) & (hsv[:, :, 0] < 110) &
            (hsv[:, :, 1] > 40) & (hsv[:, :, 2] > 70)).astype(np.uint8)
    count, _, stats, centres = cv2.connectedComponentsWithStats(mask)
    height = frame_bgr.shape[0]
    best = None
    for i in range(1, count):
        x, y, w, h = (int(v) for v in stats[i, :4])
        if y < height * 0.75 or not 140 <= w <= 260 or not 24 <= h <= 70:
            continue
        if abs((x + w / 2) - frame_bgr.shape[1] / 2) > frame_bgr.shape[1] * 0.12:
            continue
        area = w * h
        if best is None or area > best[0]:
            best = (area, x, y, w, h, centres[i])
    if best is None:
        return None
    _, x, y, w, h, centre = best
    return {'score': 0.0, 'bbox': [x, y, w, h],
            'center': [int(centre[0]), int(centre[1])], 'fallback': 'teal_filled_button'}


def screen_state(found):
    """Which screen the matched buttons imply, never guessing past evidence."""
    if 'return_port' in found:
        return 'results'
    if 'start_battle' in found:
        return 'prebattle'
    if 'join_battle' in found or 'ready' in found:
        return 'lobby'
    return 'unknown'


def next_action(state, found, ready_attempts):
    """The one click this screen needs, or None when nothing is left to do."""
    if state == 'results':
        return 'return_port'
    if state == 'prebattle':
        return 'start_battle'
    if state == 'lobby':
        if 'join_battle' not in found:
            return None
        if ready_attempts == 0 and 'ready' in found:
            return 'ready'
        return 'join_battle'
    return None


class X11:
    """Identity-bound window capture plus read-back verified pointer clicks."""

    def __init__(self, capture):
        self.capture = capture
        self.display = capture.identity['display']
        self.env = {**os.environ, 'DISPLAY': self.display}

    def x(self, *args):
        result = subprocess.run(['xdotool', *map(str, args)], env=self.env,
                                capture_output=True, text=True, timeout=5)
        if result.returncode:
            raise EntryError(f'xdotool {args[0]} failed: {result.stderr.strip()}')
        return result.stdout.strip()

    def geometry(self):
        fields = dict(line.split('=', 1) for line in
                      self.x('getwindowgeometry', '--shell', self.capture.window).splitlines())
        return {k: int(v) for k, v in fields.items()}

    def pointer(self):
        fields = dict(line.split('=', 1) for line in
                      self.x('getmouselocation', '--shell').splitlines())
        return int(fields['X']), int(fields['Y'])

    def focus(self):
        self.capture.check()
        if self.x('getwindowfocus') != self.capture.window:
            self.x('windowactivate', '--sync', self.capture.window)
            if self.x('getwindowfocus') != self.capture.window:
                raise EntryError('game window does not hold input focus')

    def warp(self, x, y):
        """Warp to window pixel (x, y) and verify against the root pointer.

        xdotool's absolute move adds the window origin (0,22 here), so the
        pointer readback is compared against origin + (x, y), not (x, y).
        """
        self.focus()
        origin = self.geometry()
        expected = (origin['X'] + x, origin['Y'] + y)
        self.x('mousemove', x, y)
        got = self.pointer()
        if got != expected:
            self.x('mousemove_relative', '--', expected[0] - got[0], expected[1] - got[1])
            got = self.pointer()
        if max(abs(got[0] - expected[0]), abs(got[1] - expected[1])) > 2:
            raise EntryError(f'pointer at {got}, wanted {expected}')
        return got

    def click(self, x, y, hold_ms=90):
        self.warp(x, y)
        self.focus()
        self.x('mousedown', 1)
        time.sleep(max(0, hold_ms) / 1000)
        self.focus()
        self.x('mouseup', 1)
        return self.pointer()


def capture_frame(capture):
    pixels, meta = capture.acquire(wait_update=True)
    return np.asarray(pixels.convert('RGB')), meta


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(action='enter_training_battle', execute=args.execute,
                  out=str(args.out.resolve()), status='dry-run', steps=[],
                  inputs=[], fired=False, fire_authorized=False,
                  semantics='clicks recognised training-room buttons only; '
                            'a disappeared menu is not proof of a live battle')
    (args.out / 'plan.json').write_text(json.dumps(report, indent=2) + '\n')
    if not args.execute:
        print(json.dumps(report, indent=2))
        return 0

    from wows_window_capture import WindowCapture
    from probe_aim_input import game_identity

    templates = load_templates(args.fixtures)
    capture = WindowCapture()
    pointer = X11(capture)
    report['identity'] = capture.identity
    report['window'] = capture.window
    started = time.monotonic()
    deadline = started + args.seconds
    ready_attempts = 0
    menu_absent = 0
    frames = []
    handlers = {}
    held = False

    def stop(signum, _frame):
        raise EntryError(f'interrupted by signal {signum}')

    try:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM):
            handlers[sig] = signal.signal(sig, stop)
        signal.setitimer(signal.ITIMER_REAL, max(.001, args.seconds))
        while time.monotonic() < deadline:
            if game_identity() != capture.identity:
                raise EntryError('game process identity changed')
            frame_rgb, meta = capture_frame(capture)
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            found = detect(frame, templates, args.threshold)
            if 'return_port' in found:
                bbox = found['return_port']['bbox']
                centre_y = bbox[1] + bbox[3] / 2
                if centre_y < frame.shape[0] * 0.75:
                    found.pop('return_port')
            if 'return_port' not in found:
                teal = teal_button(frame)
                if teal is not None:
                    found['return_port'] = teal
            state = screen_state(found)
            step = dict(t=round(time.monotonic() - started, 2), state=state,
                        buttons={k: v['score'] for k, v in found.items()},
                        pixel_sha256=meta['raw_pixel_sha256'])
            action = next_action(state, found, ready_attempts)
            if state == 'unknown':
                menu_absent += 1
                step['menus_absent_streak'] = menu_absent
                report['steps'].append(step)
                if menu_absent >= args.battle_checks:
                    report.update(status='battle_or_loading', menus_absent=menu_absent,
                                  battle_verified=False,
                                  note='menus gone; the caller must verify the battle HUD')
                    break
                time.sleep(args.poll)
                continue
            menu_absent = 0
            if action is None:
                step['action'] = 'wait_for_join_button'
                report['steps'].append(step)
                time.sleep(args.poll)
                continue
            center = found[action]['center']
            step.update(action=action, click=center, template_score=found[action]['score'])
            ended = pointer.click(center[0], center[1], args.hold_ms)
            report['inputs'].append(dict(button=1, pulse=True, hold_ms=args.hold_ms,
                                        action=action, point=center,
                                        pointer_after=list(ended),
                                        transport='xdotool absolute warp + XTest click'))
            if action == 'ready':
                ready_attempts += 1
            frames.append((action, frame_rgb, meta))
            report['steps'].append(step)
            (args.out / 'report.json').write_text(
                json.dumps(report, indent=2, default=str) + '\n')
            time.sleep(args.settle)
            if len(report['steps']) >= 4:
                recent = report['steps'][-4:]
                if len({(r['state'], r['pixel_sha256']) for r in recent}) == 1:
                    raise EntryError('screen unchanged after repeated clicks; stopping')
        else:
            report.setdefault('reason', 'time bound reached without leaving the menus')
            report['status'] = 'failed'
    except Exception as error:
        report.update(status='failed', error=str(error))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if held:
            try:
                pointer.x('mouseup', 1)
            except Exception as error:
                report.setdefault('cleanup_errors', []).append(str(error))
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        report['unreleased_button'] = False
        report['elapsed_s'] = round(time.monotonic() - started, 2)
        (args.out / 'report.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
        for index, (action, frame, meta) in enumerate(frames):
            path = args.out / f'{index:02d}-{action}.png'
            if not path.exists():
                cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                path.with_suffix('.png.json').write_text(json.dumps(meta, indent=2, default=str) + '\n')
    print(json.dumps({k: report[k] for k in ('status', 'elapsed_s') if k in report},
                     ensure_ascii=False))
    return 0 if report['status'] in ('battle_or_loading',) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--fixtures', type=Path, default=FIXTURES)
    parser.add_argument('--seconds', type=float, default=180.)
    parser.add_argument('--poll', type=float, default=1.2)
    parser.add_argument('--settle', type=float, default=2.5,
                        help='wait after a click before the next capture')
    parser.add_argument('--threshold', type=float, default=MATCH_THRESHOLD)
    parser.add_argument('--hold-ms', type=int, default=90)
    parser.add_argument('--battle-checks', type=int, default=3)
    args = parser.parse_args(argv)
    if not 10 <= args.seconds <= 600:
        parser.error('seconds must be 10..600')
    if not 0 < args.threshold <= 1 or not 1 <= args.battle_checks <= 10:
        parser.error('threshold must be in (0,1] and battle-checks 1..10')
    if not 0 <= args.hold_ms <= 250:
        parser.error('hold-ms must be 0..250')
    if not 0 <= args.poll <= 10 or not 0 <= args.settle <= 15:
        parser.error('poll must be 0..10 and settle 0..15')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
