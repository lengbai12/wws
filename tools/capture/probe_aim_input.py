#!/home/lengbai/Downloads/wws/.venv/bin/python
"""Bounded opt-in aim transport probe; never fires or claims calibration.

Relative motion goes over XTest by default; uinput and gamescope libei ('ei')
are explicit alternatives because a parked X pointer silently eats XTest moves.
Default dry-run is offline and records a plan only. Execute requires an already
running game and capture host. Opposite relative input is attempted as rollback;
it does not guarantee the original orientation. Replay camera ownership can
override inputs: camera-link (C) is an explicit experiment, not calibration.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
# Built from tools/capture/ei_input.c; sends libei relative motion to the
# gamescope session. Overridden by tests, never by the environment.
EI_INPUT = ROOT / 'tools/capture/ei_input'


class ProbeError(RuntimeError):
    pass


def process_identity(pid, proc=Path('/proc')):
    entry = proc / str(pid)
    if entry.stat().st_uid != os.getuid() or (entry / 'comm').read_text().strip() != 'WorldOfWarships':
        raise ProbeError('original game process is no longer present')
    environ = dict(part.split(b'=', 1) for part in (entry / 'environ').read_bytes().split(b'\0') if b'=' in part)
    display = environ.get(b'DISPLAY', b'').decode()
    if not display:
        raise ProbeError('game process has no DISPLAY')
    start = (entry / 'stat').read_text().rsplit(')', 1)[1].split()[19]
    return {'pid': int(pid), 'display': display, 'start_ticks': start}


def game_identity(proc=Path('/proc')):
    matches = []
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid == os.getuid() and (entry / 'comm').read_text().strip() == 'WorldOfWarships':
                matches.append(entry)
        except OSError:
            continue
    if len(matches) != 1:
        raise ProbeError(f'expected one real WorldOfWarships process, found {len(matches)}')
    return process_identity(int(matches[0].name), proc)


def ancestor_pids(pid, proc=Path('/proc')):
    result = set()
    while pid > 1 and pid not in result:
        result.add(pid)
        fields = (proc / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        pid = int(fields[1])
    return result


def pointer_state(display):
    """Read the X11 root-pointer position and button/key mask."""
    lib = ctypes.CDLL(ctypes.util.find_library('X11'))
    lib.XOpenDisplay.argtypes = [ctypes.c_char_p]
    lib.XOpenDisplay.restype = ctypes.c_void_p
    lib.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
    lib.XDefaultRootWindow.restype = ctypes.c_ulong
    lib.XQueryPointer.argtypes = [ctypes.c_void_p, ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_uint)]
    lib.XCloseDisplay.argtypes = [ctypes.c_void_p]
    connection = lib.XOpenDisplay(display.encode())
    if not connection:
        raise ProbeError('cannot query original display pointer')
    try:
        root, child = ctypes.c_ulong(), ctypes.c_ulong()
        rx, ry = ctypes.c_int(), ctypes.c_int()
        wx, wy, mask = ctypes.c_int(), ctypes.c_int(), ctypes.c_uint()
        if not lib.XQueryPointer(connection, lib.XDefaultRootWindow(connection),
                ctypes.byref(root), ctypes.byref(child), ctypes.byref(rx), ctypes.byref(ry),
                ctypes.byref(wx), ctypes.byref(wy), ctypes.byref(mask)):
            raise ProbeError('pointer is on another screen')
        return {'x': rx.value, 'y': ry.value, 'mask': mask.value}
    finally:
        lib.XCloseDisplay(connection)


def ei_socket():
    """gamescope's libei socket; WW_EI_SOCKET overrides the session default.

    Mirrors default_socket() in ei_input.c so the existence check here names the
    same endpoint the binary would connect to.
    """
    override = os.environ.get('WW_EI_SOCKET')
    if override:
        return Path(override)
    runtime = os.environ.get('XDG_RUNTIME_DIR') or f'/run/user/{os.getuid()}'
    return Path(runtime) / 'gamescope-0-ei'


def scene_motion(before, after):
    """Measure masked local scene response; never a firing calibration."""
    from analyze_aim_probe import scene_motion as masked_scene_motion
    return masked_scene_motion(before, after)


class Probe:
    def __init__(self, seconds, report, relative_transport=None):
        self.started = time.monotonic()
        self.deadline = self.started + seconds
        # Cleanup is part of the requested total bound, never an extension.
        self.action_deadline = self.deadline - min(2.0, seconds / 4)
        self.report = report
        self.identity = None
        self.window = None
        self.held = {}
        self.source = None
        # XTest motion reaches X11 clients (port UI) but a DirectX game that has
        # latched a kernel input device ignores it for camera rotation.  The
        # uinput transport writes to a real virtual mouse instead, so the game
        # receives ordinary relative motion.  Default stays XTest so every
        # existing acceptance result keeps its meaning.
        # 'ei' sends libei relative motion to gamescope, which forwards it as
        # wl_relative_pointer before any cursor clamping, so repeated turns are
        # not eaten at the window/screen edge.  An explicit argument wins over
        # the environment; either way the default remains XTest.
        self.relative_transport = (relative_transport
                                   or os.environ.get('WW_RELATIVE_TRANSPORT', 'xtest'))
        # The capture source cannot change while the game runs, yet the old code
        # re-ran `capture_host preflight` (a Python start plus a full PipeWire
        # registry parse) before *every* frame: measured ~1 s per grab, which was
        # the bulk of the time-to-fire. Cache it for a short TTL; an error or a
        # changed source still refreshes it immediately.
        self.preflight_ttl = float(os.environ.get('WW_PREFLIGHT_TTL', '5.0'))
        self._preflight = None
        self._preflight_at = 0.0
        # A capture costs ~0.6 s, and a step needs two of them (its own frame
        # and the pre-fire confirmation). The confirmation frame is already
        # the picture *after* the last move, so the next step may reuse it as
        # input: the convergence counter still reads a distinct confirmation
        # each time, because that grab happens after the next move.
        self._frame_cache = None
        self._inputs_at = 0.0
        self.frame_reuses = 0

    def command(self, argv, cleanup=False, identity=None):
        if argv and str(argv[0]) == 'xdotool':
            # Any injected input invalidates the cached frame.
            self._inputs_at = time.monotonic()
        remaining = (self.deadline if cleanup else self.action_deadline) - time.monotonic()
        if remaining <= 0:
            raise ProbeError('probe time bound expired')
        env = os.environ.copy()
        target = self.identity if identity is None else identity
        if target:
            env['DISPLAY'] = target['display']
        result = subprocess.run(argv, env=env, text=True, capture_output=True,
                                timeout=min(2 if cleanup else 5, remaining))
        if result.returncode:
            raise ProbeError(f'{argv[0]} failed: {result.stderr.strip()}')
        return result.stdout.strip()
    def key_is_down(self, key):
        import ctypes
        import ctypes.util
        lib = ctypes.CDLL(ctypes.util.find_library('X11'))
        lib.XOpenDisplay.argtypes = [ctypes.c_char_p]
        lib.XOpenDisplay.restype = ctypes.c_void_p
        lib.XStringToKeysym.argtypes = [ctypes.c_char_p]
        lib.XStringToKeysym.restype = ctypes.c_ulong
        lib.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        lib.XKeysymToKeycode.restype = ctypes.c_ubyte
        lib.XQueryKeymap.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.XCloseDisplay.argtypes = [ctypes.c_void_p]
        display = lib.XOpenDisplay(self.identity['display'].encode())
        if not display:
            raise ProbeError('cannot inspect existing held keys')
        try:
            code = lib.XKeysymToKeycode(display, lib.XStringToKeysym(key.encode()))
            if not code:
                raise ProbeError('unknown keycode')
            state = ctypes.create_string_buffer(32)
            lib.XQueryKeymap(display, state)
            return bool(state.raw[code // 8] & (1 << (code % 8)))
        finally:
            lib.XCloseDisplay(display)


    def x(self, *args, cleanup=False, identity=None):
        return self.command(['xdotool', *map(str, args)], cleanup=cleanup, identity=identity)

    def check_identity(self):
        if game_identity() != self.identity:
            raise ProbeError('game process identity changed')

    def verify_window(self, window):
        title = self.x('getwindowname', window)
        try:
            pid = int(self.x('getwindowpid', window))
        except (ValueError, ProbeError):
            pid = None
        if pid != self.identity['pid']:
            raise ProbeError('window PID does not match original game process')
        return {'window': window, 'pid': pid, 'title': title}

    def focus(self):
        self.check_identity()
        evidence = self.verify_window(self.window)
        if self.x('getwindowfocus') != self.window:
            raise ProbeError('game window does not have input focus')
        self.report['focus_checks'].append(evidence)

    def preflight(self, force=False):
        """Cached capture-source verification (TTL, refreshed on demand)."""
        now = time.monotonic()
        if (not force and self._preflight is not None
                and now - self._preflight_at <= self.preflight_ttl):
            return self._preflight
        result = json.loads(self.command(['/usr/bin/python3', str(HERE/'capture_host.py'), 'preflight']))
        if not result.get('target_resolved') or result.get('game_pids') != [self.identity['pid']]:
            raise ProbeError('capture host did not verify unique live game/source')
        self._preflight = result
        self._preflight_at = now
        return result

    def _host_call(self, command, path):
        """Ask the running host over its control socket instead of spawning one.

        Measured 2026-10-20: a per-frame subprocess costs ~0.56 s, almost all of
        it interpreter start-up; the socket call returns the same evidence in a
        fraction of that. Any failure falls back to the subprocess path so a
        host that is missing or older never blocks aiming.
        """
        if os.environ.get('WW_HOST_SOCKET', '1') == '0':
            return json.loads(self.command(['/usr/bin/python3',
                                            str(HERE / 'capture_host.py'), command, str(path)]))
        try:
            from capture_host import request as host_request, DEFAULT_STATE_DIR
            socket_path = Path(os.environ.get('WW_HOST_SOCKET_PATH',
                                             DEFAULT_STATE_DIR / 'control.sock'))
            return host_request(socket_path, command, {'out': str(path)}, timeout=10.0)
        except Exception:  # noqa: BLE001 - fall back to the documented CLI path
            return json.loads(self.command(['/usr/bin/python3',
                                            str(HERE / 'capture_host.py'), command, str(path)]))

    def grab(self, name, allow_reuse=False):
        """One raw frame from the host's persistent pipeline (about 4x faster)."""
        self.check_identity()
        cached = self._frame_cache
        ttl = float(os.environ.get('WW_FRAME_REUSE_S', '0'))
        if (allow_reuse and cached is not None and ttl > 0
                and time.monotonic() - cached[1] <= ttl
                and self._inputs_at <= cached[1] and Path(cached[0]).is_file()):
            self.frame_reuses += 1
            return Path(cached[0])
        # Off by default: existing callers (and their tests) must keep the
        # one-capture-per-read semantics. The fast loop opts in explicitly.
        ttl = float(os.environ.get('WW_FRAME_REUSE_S', '0'))
        cached = self._frame_cache
        if (cached is not None and ttl > 0
                and time.monotonic() - cached[1] <= ttl
                and self._inputs_at <= cached[1] and Path(cached[0]).is_file()):
            self.frame_reuses += 1
            return cached[0]
        preflight = self.preflight()
        source = preflight['node']
        if int(source['process_id']) not in ancestor_pids(self.identity['pid']):
            # Steam-launched sessions run the game under a compositor that is not
            # its ancestor, yet the host pipeline still yields this game's frames.
            # Accepting that is an explicit, recorded override, never a default.
            if os.environ.get('WW_ALLOW_FOREIGN_CAPTURE') != '1':
                raise ProbeError('capture compositor is not an ancestor of this game')
            self.report['capture_ancestry_override'] = source
        if self.source is not None and source != self.source:
            raise ProbeError('capture source changed')
        self.source = source
        path = Path(self.report['out']) / f'{name}.ppm'
        result = self._host_call('grab', path)
        if result.get('status') != 'completed' or str(result.get('process_id')) != str(source['process_id']) or result.get('node') != source['id'] or not path.is_file():
            raise ProbeError('grab source/output mismatch')
        self._frame_cache = (str(path), time.monotonic())
        return path

    def snapshot(self, name, allow_reuse=False):
        """Allow_reuse lets an *aiming* read take the newest frame again.

        A pre-fire confirmation always grabs fresh, so two consecutive
        confirmations can never collapse onto one picture.
        """
        self.check_identity()
        cached = self._frame_cache
        ttl = float(os.environ.get('WW_FRAME_REUSE_S', '0'))
        if (allow_reuse and cached is not None and ttl > 0
                and time.monotonic() - cached[1] <= ttl
                and self._inputs_at <= cached[1] and Path(cached[0]).is_file()):
            self.frame_reuses += 1
            self.report.setdefault('frame_reuses', 0)
            self.report['frame_reuses'] = self.frame_reuses
            return Path(cached[0])
        preflight = self.preflight()
        source = preflight['node']
        if int(source['process_id']) not in ancestor_pids(self.identity['pid']):
            # Steam-launched sessions run the game under a compositor that is not
            # its ancestor, yet the host pipeline still yields this game's frames.
            # Accepting that is an explicit, recorded override, never a default.
            if os.environ.get('WW_ALLOW_FOREIGN_CAPTURE') != '1':
                raise ProbeError('capture compositor is not an ancestor of this game')
            self.report['capture_ancestry_override'] = source
        if self.source is not None and source != self.source:
            raise ProbeError('capture source changed')
        self.source = source
        path = Path(self.report['out']) / f'{name}.png'
        started = time.monotonic() - self.started
        evidence = self._host_call('snapshot', path)
        evidence.update({'name': name, 'request_elapsed_s': started,
                         'completed_elapsed_s': time.monotonic() - self.started})
        if evidence.get('status') != 'completed' or str(evidence.get('process_id')) != str(source['process_id']) or evidence.get('node') != source['id'] or not path.is_file():
            raise ProbeError('snapshot source/output mismatch')
        self.report['snapshots'].append(evidence)
        self._frame_cache = (str(path), time.monotonic())
        return path

    def ei_move(self, dx, dy, click=False):
        """Send one relative motion through the gamescope libei socket.

        Fails loudly instead of falling back to XTest: the reason this channel
        exists is that XTest motion is clamped at the window/screen edge, so a
        silent fallback would look like a delivered move that never rotated.
        """
        binary = EI_INPUT
        socket = ei_socket()
        if not binary.is_file():
            raise ProbeError(f'EI transport binary missing {binary}')
        if not socket.exists():
            raise ProbeError(f'EI transport socket missing {socket}')
        remaining = self.action_deadline - time.monotonic()
        if remaining <= 0:
            raise ProbeError('probe time bound expired')
        argv = [str(binary), '--dx', str(dx), '--dy', str(dy)]
        if click:
            argv.append('--click')
        argv.append(str(socket))
        try:
            result = subprocess.run(argv, capture_output=True, text=True,
                                    timeout=max(1.0, min(5.0, remaining)))
        except subprocess.TimeoutExpired as error:
            raise ProbeError(f'ei_input timed out after {error.timeout:.2f}s') from error
        except OSError as error:
            raise ProbeError(f'EI transport cannot run {binary}: {error}') from error
        if result.returncode:
            detail = (result.stderr or result.stdout).strip() or 'no output'
            raise ProbeError(f'ei_input failed ({result.returncode}) on {socket}: {detail}')
        return result.stdout.strip()

    def move(self, dx, dy, click=False):
        uinput = self.relative_transport == 'uinput'
        ei = self.relative_transport == 'ei'
        if click and not ei:
            # Only the EI channel can press a button in the same frame as the
            # motion; XTest/uinput callers keep their existing semantics.
            raise ProbeError('mouse button click requires the ei relative transport')
        self.focus()
        # Sample the seat pointer for every transport.  Under uinput the sample
        # still shows whether the request actually moved anything: a pointer
        # parked on a screen edge reports a zero delta while the game silently
        # drops the move, which is what made the servo look like it was shaking.
        # Under EI the sample is diagnostics only -- ei_input.c sends libei
        # deltas that gamescope applies before cursor clamping, so the pointer
        # may stay pinned while the game received the whole motion.
        observation = None
        if ei:
            try:
                before = pointer_state(self.identity['display'])
            except ProbeError as error:
                before = None
                observation = {'before_error': str(error)}
        else:
            before = pointer_state(self.identity['display'])
        event = {
            'relative': [dx, dy],
            'click': bool(click),
            'elapsed_s': time.monotonic() - self.started,
            'transport': ('uinput_virtual_mouse' if uinput else
                          'ei_libei_gamescope' if ei else
                          'XTestFakeRelativeMotionEvent'),
            'pointer_before': before,
            'pointer_after': None,
            'pointer_delta': None,
            'pointer_mask_unchanged': None,
            'pointer_observation': observation,
            'transport_status': 'attempted',
            'interpretation': (
                'kernel uinput device; the compositor moves the seat pointer and the '
                'game receives ordinary relative motion; X pointer samples do not '
                'describe this transport' if uinput else
                'gamescope libei relative motion; deltas are compositor-level and are '
                'delivered before cursor clamping, so X pointer samples do not describe '
                'this transport and a clipped pointer is not a clipped event' if ei else
                'X root-pointer samples; clipping, warps and pointer lock may hide '
                'motion; not game receipt or calibration'),
        }
        self.report['inputs'].append(event)
        self.report['movement_delivery_attempted'] = True
        try:
            if uinput:
                binary = ROOT / 'state/cn360-tools/uinput_input'
                if not binary.is_file():
                    raise ProbeError(f'uinput transport missing {binary}')
                remaining = self.action_deadline - time.monotonic()
                if remaining <= 0:
                    raise ProbeError('probe time bound expired')
                result = subprocess.run([str(binary), '--dx', str(dx), '--dy', str(dy)],
                                        capture_output=True, text=True,
                                        timeout=max(1.0, min(5.0, remaining)))
                if result.returncode:
                    raise ProbeError(f'uinput move failed: {result.stderr.strip()}')
                event['transport_status'] = 'command_completed'
                after = pointer_state(self.identity['display']) if before else None
                if after is not None and before is not None:
                    delta = [after['x'] - before['x'], after['y'] - before['y']]
                    event['pointer_after'] = after
                    event['pointer_delta'] = delta
                    event['pointer_mask_unchanged'] = after['mask'] == before['mask']
                    event['pointer_clipped'] = bool(
                        (abs(dx) > 3 and delta[0] == 0)
                        or (abs(dy) > 3 and delta[1] == 0))
            elif ei:
                reply = self.ei_move(dx, dy, click=click)
                event['transport_status'] = 'command_completed'
                observation = dict(observation or {})
                if before is not None:
                    observation.setdefault('before', before)
                try:
                    after = pointer_state(self.identity['display'])
                except ProbeError as error:
                    observation['after_error'] = str(error)
                else:
                    observation['after'] = after
                    if before is not None:
                        observation['delta'] = [after['x'] - before['x'],
                                                after['y'] - before['y']]
                        observation['mask_unchanged'] = after['mask'] == before['mask']
                # No pointer_clipped here on purpose: the delivered delta lives
                # above the seat pointer, which may be clamped at a screen edge
                # while the game still received the full relative motion.
                observation['note'] = ('diagnostic only; X pointer samples cannot '
                                       'describe the EI channel')
                event['pointer_observation'] = observation
                if reply:
                    try:
                        event['transport_reply'] = json.loads(reply)
                    except ValueError:
                        event['transport_reply'] = reply
            else:
                self.x('mousemove_relative', '--', dx, dy)
                event['transport_status'] = 'command_completed'
                after = pointer_state(self.identity['display'])
                event.update(pointer_after=after,
                             pointer_delta=[after['x'] - before['x'], after['y'] - before['y']],
                             pointer_mask_unchanged=after['mask'] == before['mask'])
        except Exception as error:
            event['error'] = f'{type(error).__name__}: {error}'
            raise

    def key(self, key):
        self.focus()
        if self.key_is_down(key):
            raise ProbeError('requested key is already held; refusing ownership')
        self.focus()
        # Pin ownership before sending: timeout may follow a delivered keydown.
        self.held[key] = {'identity': dict(self.identity), 'window': self.window}
        self.x('keydown', key)
        hold_ms = self.report.get('key_hold_ms', 0)
        if hold_ms:
            remaining = self.action_deadline - time.monotonic()
            if remaining <= hold_ms / 1000:
                raise ProbeError('key pulse exceeds action deadline')
            time.sleep(hold_ms / 1000)
        self.focus()
        self.x('keyup', key)
        del self.held[key]
        self.report['inputs'].append({'key': key, 'elapsed_s': time.monotonic() - self.started})

    def release(self):
        for key, owner in tuple(self.held.items()):
            try:
                identity = owner['identity']
                if process_identity(identity['pid']) != identity:
                    raise ProbeError('owned key original game identity changed')
                # XTest resets server key state even after focus loss. It cannot
                # target a window: only our acquired key is released, on its
                # original display; the new focus can also observe this keyup.
                self.x('keyup', key, cleanup=True, identity=identity)
                self.report.setdefault('own_key_releases', []).append({
                    'key': key, **owner, 'transport': 'original-display XTest keyup',
                    'window_targeted': False})
                del self.held[key]
            except Exception as error:
                self.report.setdefault('cleanup_errors', []).append(str(error))
        self.report['unreleased_own_keys'] = sorted(self.held)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--dry-run', action='store_true')
    p.add_argument('--seconds', type=float, default=20)
    p.add_argument('--dx', type=int, default=40)
    p.add_argument('--dy', type=int, default=0)
    p.add_argument('--out', type=Path, default=ROOT/'state'/f'aim-probe-{time.time_ns()}')
    p.add_argument('--key-action', choices=('lock', 'scope', 'camera-link'))
    p.add_argument('--key-phase', choices=('before', 'after'), default='after',
                   help='apply optional key before movement or after rollback (default)')
    p.add_argument('--baseline', action='store_true',
                   help='capture one extra no-input frame before the movement pair')
    p.add_argument('--key-hold-ms', type=int, default=0,
                   help='bounded optional key pulse duration, 0..250ms')
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 30:
        p.error('--seconds must be finite, positive and <=30')
    if not 0 <= args.key_hold_ms <= 250:
        p.error('--key-hold-ms must be 0..250')
    if max(abs(args.dx), abs(args.dy)) > 200 or (args.dx == args.dy == 0):
        p.error('relative vector must be nonzero and each axis bounded to 200')
    if args.key_phase != 'after' and not args.key_action:
        p.error('--key-phase before requires --key-action')
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    report = {'out': str(args.out), 'execute': args.execute, 'identity_verified': False,
              'calibration_valid': False, 'world_calibration_valid': False,
              'focus_checks': [], 'snapshots': [], 'inputs': [],
              'plan': [{'snapshot': 'before'}, {'relative': [args.dx, args.dy]},
                       {'snapshot': 'after'}, {'relative': [-args.dx, -args.dy]}, {'snapshot': 'rollback'}],
              'limits': {'seconds': args.seconds, 'no_mouse_buttons': True},
              'key_hold_ms': args.key_hold_ms,
              'interpretation': 'Input transport/scene motion only; no world or aim calibration.'}
    key = {'lock': 'x', 'scope': 'Shift_L', 'camera-link': 'c'}.get(args.key_action)
    key_plan = [{'key': key}, {'snapshot': 'key-action'}] if key else []
    if args.baseline:
        report['plan'].insert(0, {'snapshot': 'baseline'})
    if args.key_phase == 'before' and key:
        report['plan'] = [{'snapshot': 'initial'}, *key_plan, *report['plan']]
    else:
        report['plan'] += key_plan
    report['limits']['cleanup_within_seconds'] = True
    report['key_release_policy'] = ('Only acquired keys, original identity/display; XTest keyup '
                                    'resets server state and may be seen by a newly focused window.')
    probe = Probe(args.seconds, report)
    previous = {}
    previous_timer = (0.0, 0.0)
    def cancel(signum, frame):
        raise ProbeError(f'cancelled by signal {signum}')
    try:
        if not args.execute:
            report['status'] = 'dry-run'
            report['reason'] = 'offline plan only; process, focus and capture source not verified'
        else:
            for sig in (signal.SIGINT, signal.SIGTERM):
                previous[sig] = signal.signal(sig, cancel)
            previous[signal.SIGALRM] = signal.signal(signal.SIGALRM, cancel)
            previous_timer = signal.setitimer(signal.ITIMER_REAL, max(.001, probe.action_deadline - time.monotonic()))
            # Fail missing analysis dependencies before sending any live input.
            import cv2
            probe.identity = game_identity()
            report['identity'] = probe.identity
            candidates = probe.x('search', '--onlyvisible', '--name', '.').splitlines()
            verified = []
            for window in candidates:
                try:
                    probe.verify_window(window)
                    verified.append(window)
                except ProbeError:
                    continue
            if len(verified) != 1:
                raise ProbeError(f'expected one verified game window, found {len(verified)}')
            probe.window = verified[0]
            probe.focus()
            report['identity_verified'] = True
            if key and args.key_phase == 'before':
                probe.snapshot('initial')
                probe.key(key)
                probe.snapshot('key-action')
            baseline = probe.snapshot('baseline') if args.baseline else None
            before = probe.snapshot('before')
            report['movement_delivery_attempted'] = False
            try:
                probe.move(args.dx, args.dy)
                after = probe.snapshot('after')
            finally:
                if report['movement_delivery_attempted']:
                    try:
                        probe.move(-args.dx, -args.dy)
                        report['rollback_input_sent'] = True
                    except Exception as error:
                        report['rollback_input_sent'] = False
                        report['rollback_error'] = str(error)
                        raise ProbeError('relative rollback failed') from error
            rollback = probe.snapshot('rollback')
            if key and args.key_phase == 'after':
                probe.key(key)
                probe.snapshot('key-action')
            # End key ownership before analysis; cleanup retains its reserved time.
            probe.release()
            def rgb(path):
                image = cv2.imread(str(path))
                return None if image is None else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            report['scene_motion'] = scene_motion(rgb(before), rgb(after))
            report['rollback_scene_motion'] = scene_motion(rgb(after), rgb(rollback))
            if baseline:
                report['baseline_scene_motion'] = scene_motion(rgb(baseline), rgb(before))
            report['interpretation'] += (' Scene flow can include replay animation/camera motion; '
                                         'compare the no-input baseline and static landmarks visually.')
            report['source'] = probe.source
            report['status'] = 'completed'
    except (Exception, KeyboardInterrupt) as error:
        report['status'] = 'failed'
        report['error'] = str(error)
    finally:
        if signal.SIGALRM in previous:
            signal.setitimer(signal.ITIMER_REAL, 0)
        probe.release()
        report['elapsed_s'] = time.monotonic() - probe.started
        if report.get('unreleased_own_keys'):
            report['status'] = 'failed'
            report.setdefault('error', 'failed to release owned keys')
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if signal.SIGALRM in previous:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return 1 if report['status'] == 'failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
