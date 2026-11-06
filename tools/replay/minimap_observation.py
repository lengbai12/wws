#!/usr/bin/env python3
"""Monotonic frame freshness and conservative continuous minimap observations."""
from dataclasses import dataclass
import math
import time

from enemy_tracks import EnemyTracker
from enemy_tracks import _finite_number


@dataclass(frozen=True)
class FrameStamp:
    epoch: int
    seq: int
    captured_ns: int
    age_bound_ns: int = 0

    def __post_init__(self):
        for name in ('epoch', 'seq', 'captured_ns', 'age_bound_ns'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(name + ' must be a nonnegative integer')

    def age_ns(self, now_ns):
        return now_ns - self.captured_ns + self.age_bound_ns

    def fresh(self, now_ns, max_age_ns):
        return now_ns >= self.captured_ns and self.age_ns(now_ns) <= max_age_ns


class MinimapObservations:
    """No input injection. Every caller must re-check snapshot at decision time.

    capture monotonic time is separate from replay/container time. Epoch increases
    when the capture source changes; no state or IDs carry over to a new source.
    Invalid/stale/out-of-order input revokes usable observations immediately.
    Validity ends at valid_until_ns (exclusive). Decision clocks cannot regress,
    including across source epochs. A freshness gap starts new track evidence.
    """
    def __init__(self, max_age_s=0.75, tracker=None):
        if not math.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError('max_age_s must be finite and positive')
        self.max_age_ns = int(max_age_s * 1e9)
        self.tracker = tracker if tracker is not None else EnemyTracker()
        self.epoch = None
        self.seq = -1
        self.captured_ns = -1
        self.stamp = None
        self.own = None
        self.tracks = []
        self.reason = 'no_frame'
        self.now_ns = -1

    def revoke(self, reason='source_lost'):
        self.tracker.reset()
        self.own = None
        self.tracks = []
        self.stamp = None
        self.reason = reason

    def _clock(self, now_ns):
        if isinstance(now_ns, bool) or not isinstance(now_ns, int) or now_ns < 0:
            self.revoke('invalid_clock')
            raise ValueError('now_ns must be a nonnegative monotonic integer')
        if now_ns < self.now_ns:
            self.revoke('clock_regressed')
            return False
        self.now_ns = now_ns
        return True

    def _fresh(self, stamp, now_ns):
        return (stamp.fresh(now_ns, self.max_age_ns) and
                now_ns < stamp.captured_ns + self.max_age_ns - stamp.age_bound_ns)

    def update(self, detections, stamp, now_ns=None):
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        if not isinstance(stamp, FrameStamp):
            self.revoke('invalid_stamp')
            raise ValueError('stamp must be FrameStamp')
        if not self._clock(now_ns):
            return self._snapshot(now_ns)
        if self.epoch is not None and stamp.epoch < self.epoch:
            self.revoke('old_epoch')
            return self.snapshot(now_ns)
        if stamp.epoch != self.epoch:
            self.revoke('source_changed')
            self.epoch, self.seq, self.captured_ns = stamp.epoch, -1, -1
        if stamp.seq <= self.seq or stamp.captured_ns <= self.captured_ns:
            self.revoke('out_of_order')
            return self.snapshot(now_ns)
        self.seq, self.captured_ns = stamp.seq, stamp.captured_ns
        if not self._fresh(stamp, now_ns):
            self.revoke('future_frame' if now_ns < stamp.captured_ns else 'stale_frame')
            return self._snapshot(now_ns)
        if self.stamp is not None and not self._fresh(self.stamp, now_ns):
            self.revoke('observation_expired')
        try:
            own = detections['own']
            if own is not None:
                for key in ('u', 'v'):
                    value = _finite_number(own[key], 'own.' + key)
                    if not 0 <= value <= 1:
                        raise ValueError('own coordinates must be finite and normalized')
            tracks = self.tracker.update(detections['enemies'], stamp.captured_ns / 1e9)
        except (KeyError, TypeError, ValueError):
            self.revoke('invalid_detection')
            raise
        self.own = dict(own) if own is not None else None
        self.tracks, self.stamp, self.reason = tracks, stamp, 'fresh'
        return self.snapshot(now_ns)

    def snapshot(self, now_ns=None):
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        if not self._clock(now_ns):
            return self._snapshot(now_ns)
        if self.stamp is not None and not self._fresh(self.stamp, now_ns):
            self.revoke('observation_expired')
        return self._snapshot(now_ns)

    def _snapshot(self, now_ns):
        fresh = self.stamp is not None
        return dict(epoch=self.epoch, seq=self.seq, fresh=fresh, reason=self.reason,
                    captured_ns=self.stamp.captured_ns if fresh else None,
                    age_ns=self.stamp.age_ns(now_ns) if fresh else None,
                    valid_until_ns=(self.stamp.captured_ns + self.max_age_ns
                                    - self.stamp.age_bound_ns) if fresh else None,
                    own=dict(self.own) if fresh and self.own is not None else None,
                    enemies=[dict(track, velocity_px_s=(dict(track['velocity_px_s'])
                             if track['velocity_px_s'] is not None else None))
                             for track in self.tracks] if fresh else [],
                    fire_authorized=False)


class ContinuousMinimapRecognizer:
    """Raw crop consumer; stamp MUST be captured before inference starts."""
    def __init__(self, package, device='cpu', max_age_s=0.75):
        from minimap_recognizer import MinimapRecognizer
        self.recognizer = MinimapRecognizer(package, device)
        self.observations = MinimapObservations(max_age_s)

    def observe(self, rgb, stamp):
        if not isinstance(stamp, FrameStamp):
            self.observations.revoke('invalid_stamp')
            raise ValueError('stamp must be FrameStamp')
        now_ns = time.monotonic_ns()
        if not self.observations._clock(now_ns):
            return self.observations._snapshot(now_ns)
        if not self.observations._fresh(stamp, now_ns):
            return self.observations.update(dict(own=None, enemies=[]), stamp, now_ns)
        try:
            detections = self.recognizer.detect(rgb)
        except Exception:
            self.observations.revoke('perception_error')
            raise
        return self.observations.update(detections, stamp, time.monotonic_ns())

    def snapshot(self):
        return self.observations.snapshot()

    def revoke(self, reason='source_lost'):
        self.observations.revoke(reason)
