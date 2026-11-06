#!/usr/bin/env python
"""Measure EI vs XTest relative-motion camera response and the turn limit.

The +-60 degree turning ceiling came from XTest motion: the X pointer is clamped
at the window/screen edge and every further relative move is discarded, so the
camera stops even though the script keeps sending deltas. gamescope's libei
socket (``$XDG_RUNTIME_DIR/gamescope-0-ei``) delivers relative motion before any
cursor clamp, so the same unit count should keep turning.

This tool drives one transport at a time, snapshots the scene before and after
each move, and converts the observed screen translation into pixels per unit
with ``main_view_perception.estimate_camera_shift`` (Lucas-Kanade flow, HUD
excluded). It then repeats the move until ``--repeat`` and records the first
attempt whose response collapses, which is the empirical turning limit.

Safety: the default is a dry-run that only prints the plan. ``--execute``
requires exactly one live WorldOfWarships process on ``--display``, checks its
identity and the pointer before every move, sends *only* relative motion (never
a key or a click), and records the final pointer state so a held button cannot
be mistaken for a clean release.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_OUT = ROOT / 'state' / 'ei-gain' / 'measure.json'
CAPTURE_HOST = HERE / 'capture_host.py'
EI_INPUT = HERE / 'ei_input'
SYSTEM_PYTHON = Path('/usr/bin/python3')
BUTTON_MASK = 0x1F00  # X11 Button1Mask..Button5Mask; keys live above this


class MeasureError(RuntimeError):
    pass


def px_per_unit(units: float, px: float) -> float | None:
    """Observed camera translation per injected unit (magnitude, no sign)."""
    if not units:
        return None
    return abs(float(px)) / abs(float(units))


def rotation_degrees(px: float, screen_w: int, fov_deg: float) -> float:
    """Screen translation in pixels -> approximate camera yaw, in degrees.

    This is a pinhole approximation with the vertical FOV applied horizontally;
    it is reported as an estimate, not as a measured world angle.
    """
    return abs(float(px)) / max(float(screen_w), 1.0) * float(fov_deg)


def ei_socket_path(environ=None) -> Path:
    environ = os.environ if environ is None else environ
    runtime = environ.get('XDG_RUNTIME_DIR') or f'/run/user/{os.getuid()}'
    return Path(runtime) / 'gamescope-0-ei'


def move_command(transport: str, units: float, dy: float = 0.0) -> list[str]:
    """Build the relative motion command for one transport; never a click/key."""
    if transport == 'ei':
        return [str(EI_INPUT), '--dx', f'{units:g}', '--dy', f'{dy:g}']
    if transport == 'xtest':
        return ['xdotool', 'mousemove_relative', '--', f'{units:g}', f'{dy:g}']
    raise MeasureError(f'unknown transport {transport!r}')


def _default_state_check(display: str, transport: str) -> dict:
    """Verify the live game and pointer without touching them."""
    sys.path.insert(0, str(HERE))
    from probe_aim_input import game_identity, pointer_state  # noqa: WPS433
    identity = game_identity()
    if identity['display'] != display:
        raise MeasureError(
            f"game runs on {identity['display']}, not requested {display}")
    if transport == 'ei':
        socket = ei_socket_path()
        if not socket.exists():
            raise MeasureError(f'EI socket missing: {socket}')
    return {'identity': identity, 'pointer': pointer_state(display)}


def _default_capture(path: Path) -> np.ndarray:
    """One live PNG snapshot through the project's capture host."""
    result = subprocess.run(
        [str(SYSTEM_PYTHON), str(CAPTURE_HOST), 'snapshot', str(path)],
        check=True, capture_output=True, text=True, timeout=60.0)
    evidence = json.loads(result.stdout or '{}')
    if evidence.get('status') != 'completed' or not path.is_file():
        raise MeasureError(f'snapshot failed: {evidence or result.stderr.strip()}')
    from PIL import Image
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB'))


def _default_shift(before: np.ndarray, after: np.ndarray) -> tuple[float, float, float]:
    """Camera translation between two native frames, in pixels."""
    sys.path.insert(0, str(ROOT / 'tools/replay'))
    from main_view_perception import estimate_camera_shift, FRAME_W, FRAME_H
    if before.shape[:2] != (FRAME_H, FRAME_W) or after.shape[:2] != (FRAME_H, FRAME_W):
        raise MeasureError(
            f'snapshot must be {FRAME_W}x{FRAME_H} RGB, got '
            f'{before.shape[1]}x{before.shape[0]} and {after.shape[1]}x{after.shape[0]}')
    return estimate_camera_shift(before, after)


def _default_run(argv: list[str], display: str) -> None:
    env = dict(os.environ)
    env['DISPLAY'] = display
    subprocess.run(argv, check=True, capture_output=True, text=True,
                   timeout=15.0, env=env)


class GainSession:
    """One transport measurement run; every side effect is injectable."""

    def __init__(self, *, transport: str = 'ei', units: float = 400.0,
                 repeat: int = 60, settle: float = 0.35, display: str = ':2',
                 execute: bool = False, out: Path | None = None,
                 fov_deg: float = 68.2, min_px: float = 2.0,
                 fail_fraction: float = 0.25, screen_w: int = 2560,
                 capture=None, shift=None, run=None, sleep=None,
                 state_check=None, clock=time.monotonic):
        if transport not in ('ei', 'xtest'):
            raise MeasureError(f'unknown transport {transport!r}')
        if units <= 0:
            raise MeasureError('units must be positive')
        if repeat < 1:
            raise MeasureError('repeat must be at least 1')
        self.transport = transport
        self.units = float(units)
        self.repeat = int(repeat)
        self.settle = float(settle)
        self.display = display
        self.execute = bool(execute)
        self.out = Path(out) if out is not None else DEFAULT_OUT
        self.fov_deg = float(fov_deg)
        self.min_px = float(min_px)
        self.fail_fraction = float(fail_fraction)
        self.screen_w = int(screen_w)
        self.capture = capture or _default_capture
        self.shift = shift or _default_shift
        self.run_command = run or (lambda argv: _default_run(argv, self.display))
        self.sleep = sleep or time.sleep
        self.state_check = state_check or (
            lambda: _default_state_check(self.display, self.transport))
        self.clock = clock
        self.commands: list[list[str]] = []
        self.input_injected = False
        self.frames: list[Path] = []
        self.last_state: dict | None = None

    # -- plan -------------------------------------------------------------
    def plan(self) -> list[dict]:
        return [{'attempt': index, 'transport': self.transport,
                 'move': move_command(self.transport, self.units),
                 'snapshot': f'attempt-{index:03d}-{{before,after}}.png',
                 'settle_s': self.settle}
                for index in range(1, self.repeat + 1)]

    # -- one attempt ------------------------------------------------------
    def attempt(self, index: int, directory: Path | None) -> dict:
        before_path = (directory or Path('.')) / f'attempt-{index:03d}-before.png'
        after_path = (directory or Path('.')) / f'attempt-{index:03d}-after.png'
        before = self.capture(before_path)
        command = move_command(self.transport, self.units)
        self.commands.append(command)
        self.run_command(command)
        self.input_injected = True
        self.sleep(self.settle)
        after = self.capture(after_path)
        dx, dy, score = self.shift(before, after)
        per_unit = px_per_unit(self.units, dx)
        return {'attempt': index, 'units': self.units,
                'command': command, 'shift_px': [round(dx, 2), round(dy, 2)],
                'confidence': round(float(score), 3),
                'px_per_unit': None if per_unit is None else round(per_unit, 4),
                'rotation_deg': round(rotation_degrees(dx, self.screen_w,
                                                       self.fov_deg), 3)}

    # -- whole run --------------------------------------------------------
    def run(self) -> dict:
        report = {
            'tool': 'measure_ei_gain',
            'semantics': 'screen-space camera flow; degrees use the configured '
                         'vertical FOV as a pinhole approximation',
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'transport': self.transport,
            'display': self.display,
            'units': self.units,
            'repeat': self.repeat,
            'settle_s': self.settle,
            'fov_deg': self.fov_deg,
            'execute': self.execute,
            'input_injected': False,
            'commands': [],
            'plan': self.plan(),
            'attempts': [],
        }
        if not self.execute:
            report['state'] = 'dry_run'
            self._write(report)
            return report

        directory = Path(tempfile.mkdtemp(prefix='ei-gain-'))
        report['frames_dir'] = str(directory)
        records: list[dict] = []
        reference: float | None = None
        degraded_from: int | None = None
        aborted: str | None = None
        try:
            state = self.state_check()
            self.last_state = state
            report['initial_state'] = state
            for index in range(1, self.repeat + 1):
                self.last_state = self.state_check()
                record = self.attempt(index, directory)
                px = abs(record['shift_px'][0])
                tracked = record['confidence'] > 0
                if reference is None and tracked and px > 0:
                    reference = px
                floor = max(self.min_px,
                            self.fail_fraction * (reference or 0.0))
                record['degraded'] = (not tracked) or px < floor
                record['px_floor'] = round(floor, 3)
                record['state'] = ('untracked' if not tracked else
                                   ('degraded' if record['degraded'] else 'tracked'))
                records.append(record)
                if record['degraded'] and degraded_from is None:
                    degraded_from = index
        except Exception as error:  # noqa: BLE001 - report, then release
            aborted = f'{type(error).__name__}: {error}'
        finally:
            try:
                self.last_state = self.state_check()
            except Exception as error:  # noqa: BLE001
                self.last_state = {'error': f'{type(error).__name__}: {error}'}
            shutil.rmtree(directory, ignore_errors=True)

        report.update(self._summarize(records, reference, degraded_from))
        report['input_injected'] = bool(self.input_injected)
        report['commands'] = [list(command) for command in self.commands]
        report['final_state'] = self.last_state
        report['released'] = self._released(self.last_state)
        report['state'] = 'aborted' if aborted else 'completed'
        if aborted:
            report['aborted_reason'] = aborted
        self._write(report)
        return report

    def _summarize(self, records, reference, degraded_from) -> dict:
        shifts = [record['shift_px'][0] for record in records]
        travelled = sum(abs(shift) for shift in shifts)
        net = sum(shifts)
        return {
            'attempt_count': len(records),
            'attempts': records,
            'reference_px': None if reference is None else round(reference, 2),
            'degraded_from_attempt': degraded_from,
            'healthy_attempts_before_failure': (
                len(records) if degraded_from is None else degraded_from - 1),
            'cumulative_units': round(self.units * len(records), 3),
            'travelled_px': round(travelled, 2),
            'net_px': round(net, 2),
            'cumulative_rotation_deg': round(
                rotation_degrees(travelled, self.screen_w, self.fov_deg), 3),
            'net_rotation_deg': round(
                net / max(self.screen_w, 1) * self.fov_deg, 3),
            'median_px_per_unit': (
                None if not records else round(float(np.median(
                    [r['px_per_unit'] for r in records if r['px_per_unit'] is not None]
                    or [0.0])), 4)),
        }

    @staticmethod
    def _released(state) -> dict:
        if not state:
            return {'released': None, 'reason': 'state_unavailable'}
        if 'error' in state:
            return {'released': False, 'reason': state['error']}
        pointer = state.get('pointer') or {}
        buttons = int(pointer.get('mask', 0)) & BUTTON_MASK
        return {'released': buttons == 0,
                'buttons_held': buttons,
                'pointer': pointer,
                'note': 'this tool never presses a button or key; the sampled '
                        'button mask proves no press was left behind'}

    def _write(self, report) -> None:
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--display', default=':2')
    parser.add_argument('--transport', choices=('ei', 'xtest'), default='ei')
    parser.add_argument('--units', type=float, default=400.0)
    parser.add_argument('--repeat', type=int, default=60)
    parser.add_argument('--settle', type=float, default=0.35)
    parser.add_argument('--fov-deg', type=float, default=68.2)
    parser.add_argument('--min-px', type=float, default=2.0,
                        help='absolute response floor counted as degraded')
    parser.add_argument('--fail-fraction', type=float, default=0.25,
                        help='fraction of the reference response counted as degraded')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--execute', action='store_true',
                        help='inject relative motion (default: plan only)')
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    session = GainSession(transport=args.transport, units=args.units,
                          repeat=args.repeat, settle=args.settle,
                          display=args.display, execute=args.execute,
                          out=args.out, fov_deg=args.fov_deg,
                          min_px=args.min_px,
                          fail_fraction=args.fail_fraction)
    try:
        report = session.run()
    except MeasureError as error:
        print(json.dumps({'tool': 'measure_ei_gain', 'state': 'refused',
                          'input_injected': False, 'reason': str(error)},
                         indent=2), file=sys.stderr)
        return 2
    print(json.dumps({key: report[key] for key in (
        'state', 'transport', 'units', 'repeat', 'input_injected',
        'attempt_count', 'degraded_from_attempt', 'cumulative_units',
        'travelled_px', 'cumulative_rotation_deg', 'median_px_per_unit') if key in report},
        indent=2, ensure_ascii=False))
    if report['state'] == 'dry_run':
        print(f"dry-run: {len(report['plan'])} planned moves; "
              f"add --execute to inject", file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
