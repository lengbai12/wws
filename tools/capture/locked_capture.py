"""Pinned gamescope capture for the locked-target controller.

The host's segment clock bounds pipeline delivery age, not game rendering age.
No desktop capture, focus change, or input is performed by this adapter.

Each frame includes two CLOCK_MONOTONIC brackets around CLOCK_REALTIME reads
and the Linux boot ID. ``captured_wall_ns`` is an estimate of pipeline capture
time, never game render time or a replay-clock anchor. Its bounds include the
sampling brackets and an explicit 1000 ppm clock-rate assumption. Missing clock
evidence or a detected discontinuity leaves the wall conversion unknown; frame
freshness still uses only the original monotonic timestamp and 750 ms limit.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import uuid

from PIL import Image

from capture_host import DEFAULT_STATE_DIR, request
from probe_aim_input import ProbeError, ancestor_pids, game_identity

CAPTURE_SEMANTICS = 'gst_segment_running_time;upstream_render_latency_unknown'
MAX_AGE_NS = 750_000_000
CLOCK_RATE_ALLOWANCE_PPM = 1000
CAPTURE_TIME_SEMANTICS = 'pipeline_capture_timestamp_not_game_render_time'
BOOT_ID_PATH = Path('/proc/sys/kernel/random/boot_id')


def sample_clock_correlation():
    """Record a real wall-clock observation bracketed by monotonic reads."""
    try:
        boot_id = str(uuid.UUID(BOOT_ID_PATH.read_text().strip()))
    except (OSError, ValueError):
        boot_id = None
    before = time.monotonic_ns()
    wall = time.time_ns()
    after = time.monotonic_ns()
    return dict(boot_id=boot_id, monotonic_before_ns=before, wall_ns=wall,
                monotonic_after_ns=after, bracket_uncertainty_ns=after - before)


def _valid_clock_sample(sample):
    if not isinstance(sample, dict):
        return False
    try:
        if str(uuid.UUID(sample.get('boot_id'))) != sample['boot_id']:
            return False
    except (ValueError, TypeError, AttributeError):
        return False
    fields = ('monotonic_before_ns', 'wall_ns', 'monotonic_after_ns',
              'bracket_uncertainty_ns')
    if any(type(sample.get(key)) is not int for key in fields):
        return False
    return (0 < sample['monotonic_before_ns'] <= sample['monotonic_after_ns'] and
            sample['wall_ns'] > 0 and sample['bracket_uncertainty_ns'] ==
            sample['monotonic_after_ns'] - sample['monotonic_before_ns'])


def _clock_rate_allowance(elapsed_ns):
    return (elapsed_ns * CLOCK_RATE_ALLOWANCE_PPM + 999_999) // 1_000_000


def capture_wall_time(stamp, before, after, previous=None):
    """Convert pipeline monotonic time only with recorded, consistent clocks.

    Bounds assume no undetected jump between samples and rate difference within
    CLOCK_RATE_ALLOWANCE_PPM. They do not bound upstream rendering latency.
    """
    evidence = dict(status='unknown', monotonic_clock='CLOCK_MONOTONIC',
                    wall_clock='CLOCK_REALTIME', before=before, after=after,
                    previous=previous, clock_rate_allowance_ppm=CLOCK_RATE_ALLOWANCE_PPM,
                    semantics=CAPTURE_TIME_SEMANTICS,
                    assumptions='no undetected clock jump between samples; bounded clock rate')
    result = dict(captured_wall_ns=None, captured_wall_bounds_ns=None,
                  captured_wall_uncertainty_ns=None, clock_correlation=evidence)
    samples = [before, after] if previous is None else [previous, before, after]
    if any(not _valid_clock_sample(sample) for sample in samples):
        evidence['reason'] = 'missing or invalid clock correlation evidence (including boot ID)'
        return result
    if type(stamp) is not int or stamp <= 0:
        evidence['reason'] = 'missing or invalid capture monotonic timestamp'
        return result
    if len({sample['boot_id'] for sample in samples}) != 1:
        evidence['reason'] = 'boot ID changed between clock samples'
        return result
    for left, right in zip(samples, samples[1:]):
        if left['monotonic_after_ns'] > right['monotonic_before_ns']:
            evidence['reason'] = 'clock sample monotonic brackets overlap or regress'
            return result
        allowance = _clock_rate_allowance(right['monotonic_after_ns'] - left['monotonic_before_ns'])
        left_low = left['wall_ns'] - left['monotonic_after_ns']
        left_high = left['wall_ns'] - left['monotonic_before_ns']
        right_low = right['wall_ns'] - right['monotonic_after_ns']
        right_high = right['wall_ns'] - right['monotonic_before_ns']
        if right_low > left_high + allowance or left_low > right_high + allowance:
            evidence['reason'] = 'wall/monotonic clock jump exceeds brackets and rate allowance'
            return result
    if stamp > after['monotonic_after_ns']:
        evidence['reason'] = 'capture timestamp follows final clock sample'
        return result
    # Enclose both measured offsets, expanded for the distance to capture time.
    # The capture PTS can precede the grab request, so this is not render timing.
    bounds = []
    for sample in (before, after):
        allowance = _clock_rate_allowance(max(
            abs(stamp - sample['monotonic_before_ns']),
            abs(stamp - sample['monotonic_after_ns'])))
        bounds.append((stamp + sample['wall_ns'] - sample['monotonic_after_ns'] - allowance,
                       stamp + sample['wall_ns'] - sample['monotonic_before_ns'] + allowance))
    low, high = min(bound[0] for bound in bounds), max(bound[1] for bound in bounds)
    wall = (low + high) // 2
    evidence['status'] = 'correlated'
    result.update(captured_wall_ns=wall, captured_wall_bounds_ns=[low, high],
                  captured_wall_uncertainty_ns=max(wall - low, high - wall))
    return result


def require_fresh(meta, previous_ns=None):
    stamp = meta.get('captured_monotonic_ns')
    if (type(stamp) is not int or stamp <= 0 or
            meta.get('capture_age_semantics') != CAPTURE_SEMANTICS):
        raise ProbeError('capture timestamp or pipeline clock semantics unverified')
    age = time.monotonic_ns() - stamp
    if not 0 <= age <= MAX_AGE_NS:
        raise ProbeError('capture frame future or expired')
    if previous_ns is not None and stamp <= previous_ns:
        raise ProbeError('capture timestamp did not advance')
    return stamp


class LockedCapture:
    def __init__(self):
        self.identity = game_identity()
        self.socket = DEFAULT_STATE_DIR / 'control.sock'
        self.node = None
        self.previous_ns = None
        self._temporary = None
        self._previous_clock_sample = None
        env = dict(os.environ, DISPLAY=self.identity['display'])
        windows = subprocess.run(
            ['xdotool', 'search', '--onlyvisible', '--pid', str(self.identity['pid'])],
            env=env, capture_output=True, text=True, timeout=2, check=True).stdout.split()
        verified = []
        for window in windows:
            geometry = subprocess.run(['xdotool', 'getwindowgeometry', '--shell', window],
                                      env=env, capture_output=True, text=True,
                                      timeout=2, check=True).stdout
            values = dict(line.split('=', 1) for line in geometry.splitlines() if '=' in line)
            if int(values.get('WIDTH', 0)) >= 640 and int(values.get('HEIGHT', 0)) >= 480:
                verified.append(window)
        if len(verified) != 1:
            raise ProbeError(f'expected one game surface, found {len(verified)}')
        self.window = verified[0]
        self.check()
        self._temporary = tempfile.TemporaryDirectory(prefix='locked-capture-', dir=DEFAULT_STATE_DIR)

    def check(self):
        if game_identity() != self.identity:
            raise ProbeError('capture game generation changed')
        preflight = request(self.socket, 'preflight', timeout=2)
        if (preflight.get('status') != 'completed' or not preflight.get('target_resolved') or
                preflight.get('game_pids') != [self.identity['pid']]):
            raise ProbeError('unique live capture game/source unconfirmed')
        node = preflight.get('node') or {}
        if (any(node.get(key) is None for key in ('id', 'serial', 'process_id')) or
                int(node['process_id']) not in ancestor_pids(self.identity['pid'])):
            raise ProbeError('capture compositor is not a verified game ancestor')
        if self.node is not None and node != self.node:
            raise ProbeError('capture source generation changed')
        self.node = dict(node)
        return preflight

    def acquire(self, wait_update=True):
        self.check()
        clock_before = sample_clock_correlation()
        path = Path(self._temporary.name) / 'current.ppm'
        result = request(self.socket, 'grab', {'out': str(path)}, timeout=2)
        if (result.get('status') != 'completed' or result.get('node') != self.node['id'] or
                str(result.get('process_id')) != str(self.node['process_id'])):
            raise ProbeError('grab response source mismatch')
        sidecar = json.loads(Path(str(path)+'.json').read_text())
        if sidecar.get('clock_basis') != 'gst_segment_running_time':
            raise ProbeError('grab did not establish pipeline time')
        with Image.open(path) as image:
            if image.mode != 'RGB' or image.size != (2560, 1600):
                raise ProbeError('requires native 2560x1600 RGB capture; no rescaling')
            pixels = image.copy()
        clock_after = sample_clock_correlation()
        meta = dict(sidecar, source=dict(self.node), identity=dict(self.identity),
                    window=self.window, capture_age_semantics=CAPTURE_SEMANTICS,
                    source_render_age_verified=False,
                    capture_time_semantics=CAPTURE_TIME_SEMANTICS,
                    replay_clock_correlation=dict(status='unknown', reason='no explicit replay/capture clock anchor'),
                    raw_pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest())
        self.check()
        stamp = require_fresh(meta, self.previous_ns)
        meta.update(capture_wall_time(stamp, clock_before, clock_after,
                                      getattr(self, '_previous_clock_sample', None)))
        self._previous_clock_sample = clock_after
        self.previous_ns = stamp
        return pixels, meta

    def save(self, path, image, meta):
        path = Path(path)
        with path.open('xb') as handle:
            image.save(handle, format='PNG')
        saved = dict(meta, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        with Path(str(path)+'.json').open('x') as handle:
            json.dump(saved, handle, indent=2)

    def close(self):
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None
