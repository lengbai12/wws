#!/usr/bin/env python3
"""Position-driven reference waypoint intents. This module never injects input."""
import argparse
import json
import math
from pathlib import Path
import time

from build_navigation_reference import PATTERN, build_reference


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def nonnegative_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


class NavigationIntent:
    """Progress is bounded arc-length projection, never replay-clock selection.

    Waypoints express a request to a separately confirmed autopilot mechanism.
    No observed arrow angle is interpreted as steering yaw. Revokes carry no
    movement, and consumers must independently check expiry before execution.
    Focus/source recovery requires a frame captured after the revocation. A
    higher source epoch retires route progress even if that frame is stale.
    """
    def __init__(self, reference=None, lookahead_px=15.0, max_forward_px=30.0,
                 corridor_px=25.0, backtrack_px=8.0, ttl_ns=500_000_000):
        self.reference = build_reference() if reference is None else reference
        for value in (lookahead_px, max_forward_px, corridor_px, backtrack_px):
            if not finite_number(value) or value <= 0:
                raise ValueError('navigation distances must be finite and positive')
        if not nonnegative_int(ttl_ns) or ttl_ns == 0:
            raise ValueError('ttl_ns must be a positive integer')
        self.lookahead, self.forward = lookahead_px, max_forward_px
        self.corridor, self.backtrack, self.ttl_ns = corridor_px, backtrack_px, ttl_ns
        self._validate_reference()
        self.progress = 0.0
        self.started = False
        self.epoch = None
        self.seq = -1
        self.captured_ns = -1
        self.now_ns = -1
        self.resume_after_ns = -1

    def _validate_reference(self):
        r = self.reference
        if not isinstance(r, dict):
            raise ValueError('reference must be an object')
        if (r.get('schema_version') != 1 or r.get('map') != 's10_USS_CL'
                or r.get('ship') != 'PGSB210-Schlieffen' or r.get('pattern') != PATTERN
                or r.get('session') != '236374'
                or r.get('coordinate') != dict(space='minimap_crop', units='pixels',
                    rect=[1836, 796, 720, 723], frame_shape_rgb=[1600, 2560, 3])):
            raise ValueError('unsupported reference')
        for name in ('reference_sha256', 'projection_sha256'):
            value = r.get(name)
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('invalid reference hash')
        points = r.get('waypoints', [])
        if len(points) < 2:
            raise ValueError('reference needs segments')
        previous = None
        for p in points:
            if not all(finite_number(p.get(k)) for k in ('x', 'y', 's_px')):
                raise ValueError('invalid waypoint')
            if not (0 <= p['x'] < 720 and 0 <= p['y'] < 723):
                raise ValueError('waypoint outside panel')
            if previous is None:
                if p['s_px'] != 0:
                    raise ValueError('nonzero route origin')
            else:
                length = math.hypot(p['x'] - previous['x'], p['y'] - previous['y'])
                if length <= 0 or not math.isclose(p['s_px'] - previous['s_px'], length, abs_tol=1e-6):
                    raise ValueError('invalid route arc length')
            previous = p
        if not finite_number(r.get('length_px')) or not math.isclose(r['length_px'], points[-1]['s_px'], abs_tol=1e-6):
            raise ValueError('invalid reference length')

    def point_at(self, distance):
        points = self.reference['waypoints']
        distance = min(max(distance, 0.0), self.reference['length_px'])
        for a, b in zip(points, points[1:]):
            if b['s_px'] >= distance:
                fraction = (distance - a['s_px']) / (b['s_px'] - a['s_px'])
                return dict(x=a['x'] + fraction * (b['x'] - a['x']),
                            y=a['y'] + fraction * (b['y'] - a['y']))
        return {k: points[-1][k] for k in ('x', 'y')}

    def project(self, x, y):
        lower = max(0.0, self.progress - max(self.forward, self.backtrack * 2))
        upper = min(self.reference['length_px'], self.progress + self.forward)
        best = None
        points = self.reference['waypoints']
        for a, b in zip(points, points[1:]):
            if b['s_px'] < lower or a['s_px'] > upper:
                continue
            dx, dy = b['x'] - a['x'], b['y'] - a['y']
            length = b['s_px'] - a['s_px']
            fraction = ((x - a['x']) * dx + (y - a['y']) * dy) / (length * length)
            fraction = max(0.0, (lower - a['s_px']) / length,
                           min(1.0, (upper - a['s_px']) / length, fraction))
            px, py = a['x'] + fraction * dx, a['y'] + fraction * dy
            candidate = (math.hypot(x - px, y - py), a['s_px'] + fraction * length)
            if best is None or candidate < best:
                best = candidate
        return best

    def decide(self, snapshot, *, map_id, ship, pattern, now_ns=None,
               focused=True, source_available=True, mechanism='unknown'):
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        result = dict(kind='navigation', action='revoke', movement_authorized=False,
                      mode='explicit_observed_reference', own_pixel=None,
                      spawn_pattern_recognized=False,
                      reason='invalid_snapshot', waypoint=None, valid_until_ns=now_ns,
                      progress=dict(s_px=self.progress, fraction=self.progress / self.reference['length_px'],
                                    position_projected=False), deviation_px=None,
                      reference=dict(map=self.reference['map'], ship=self.reference['ship'],
                                     pattern=self.reference['pattern'],
                                     reference_sha256=self.reference['reference_sha256'],
                                     projection_sha256=self.reference['projection_sha256']),
                      heading=dict(yaw_rad=None, reason='uncalibrated_arrow_angle'),
                      input_authorized=False, execution_requires='separate tactical UI calibration and confirmation',
                      assurance=self.reference['assurance'])
        def revoke(reason):
            result['reason'] = reason
            return result
        if not nonnegative_int(now_ns):
            result['valid_until_ns'] = None
            return revoke('invalid_clock')
        if now_ns < self.now_ns:
            return revoke('decision_clock_regressed')
        self.now_ns = now_ns
        if not focused:
            self.resume_after_ns = now_ns
            self.progress, self.started = 0.0, False
            result['progress'] = dict(s_px=0.0, fraction=0.0, position_projected=False)
            return revoke('focus_lost')
        if not source_available:
            self.resume_after_ns = now_ns
            self.progress, self.started = 0.0, False
            result['progress'] = dict(s_px=0.0, fraction=0.0, position_projected=False)
            return revoke('source_lost')
        if (map_id, ship, pattern) != ('s10_USS_CL', 'PGSB210-Schlieffen', PATTERN):
            return revoke('unknown_map_ship_or_pattern')
        if mechanism != 'autopilot_waypoint':
            return revoke('unknown_navigation_mechanism')
        if not isinstance(snapshot, dict):
            return revoke('invalid_snapshot')
        epoch, seq = snapshot.get('epoch'), snapshot.get('seq')
        captured, expiry = snapshot.get('captured_ns'), snapshot.get('valid_until_ns')
        # Source identity remains meaningful even when its observation lease has
        # expired. Retire the previous source before checking frame usability.
        source_changed = False
        if nonnegative_int(epoch) and self.epoch is not None and epoch != self.epoch:
            if epoch < self.epoch:
                return revoke('old_source_epoch')
            self.epoch, self.seq, self.captured_ns = epoch, -1, -1
            self.progress, self.started = 0.0, False
            result['progress'] = dict(s_px=0.0, fraction=0.0, position_projected=False)
            source_changed = True
        if snapshot.get('fresh') is not True:
            return revoke('snapshot_not_fresh')
        if not all(nonnegative_int(v) for v in (epoch, seq, captured, expiry)):
            return revoke('invalid_snapshot')
        if captured > now_ns or expiry <= now_ns or expiry < captured:
            return revoke('snapshot_expired_or_future')
        if captured <= self.resume_after_ns:
            return revoke('snapshot_before_recovery')
        if source_changed:
            self.seq, self.captured_ns = seq, captured
            own = snapshot.get('own')
            if own is None:
                return revoke('own_missing')
            if not isinstance(own, dict) or not all(finite_number(own.get(k)) for k in ('pixel_x', 'pixel_y')):
                return revoke('invalid_own')
            if not (0 <= own['pixel_x'] < 720 and 0 <= own['pixel_y'] < 723):
                return revoke('invalid_own')
            return revoke('source_changed')
        if seq <= self.seq or captured <= self.captured_ns:
            return revoke('out_of_order')
        self.epoch, self.seq, self.captured_ns = epoch, seq, captured
        own = snapshot.get('own')
        if own is None:
            return revoke('own_missing')
        if not isinstance(own, dict) or not all(finite_number(own.get(k)) for k in ('pixel_x', 'pixel_y')):
            return revoke('invalid_own')
        x, y = own['pixel_x'], own['pixel_y']
        if not (0 <= x < 720 and 0 <= y < 723):
            return revoke('invalid_own')
        result['own_pixel'] = dict(x=x, y=y, coordinate='minimap_crop', units='pixels')
        deviation, projected = self.project(x, y)
        result['deviation_px'] = deviation
        if deviation > self.corridor:
            return revoke('off_reference_corridor')
        if self.started and projected < self.progress - self.backtrack:
            return revoke('backtracking')
        self.progress = max(self.progress, projected)
        self.started = True
        result['progress'] = dict(s_px=self.progress, fraction=self.progress / self.reference['length_px'],
                                  position_projected=True)
        endpoint = self.point_at(self.reference['length_px'])
        if self.reference['length_px'] - self.progress <= 6 and math.hypot(x - endpoint['x'], y - endpoint['y']) <= 6:
            return revoke('reference_complete')
        target_s = min(self.progress + self.lookahead, self.reference['length_px'])
        result.update(action='autopilot_waypoint', movement_authorized=True, reason='reference_waypoint',
                      valid_until_ns=min(expiry, now_ns + self.ttl_ns),
                      waypoint=dict(self.point_at(target_s), s_px=target_s,
                                    coordinate='minimap_crop', units='pixels'))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--map', required=True)
    parser.add_argument('--ship', required=True)
    parser.add_argument('--pattern', required=True)
    parser.add_argument('--mechanism', default='unknown')
    parser.add_argument('--offline-captured-clock', action='store_true', help='Saved-data simulation only')
    parser.add_argument('--focus-lost', action='store_true')
    parser.add_argument('--source-lost', action='store_true')
    args = parser.parse_args()
    consumer = NavigationIntent(json.loads(args.reference.read_text()) if args.reference else None)
    with args.observations.open() as handle:
        for line in handle:
            row = json.loads(line)
            snapshot = row.get('observation', row)
            now = snapshot.get('captured_ns') if args.offline_captured_clock else time.monotonic_ns()
            intent = consumer.decide(snapshot, map_id=args.map, ship=args.ship, pattern=args.pattern,
                                     now_ns=now, focused=not args.focus_lost,
                                     source_available=not args.source_lost, mechanism=args.mechanism)
            print(json.dumps(dict(sample=row.get('sample'), session=row.get('session'), intent=intent), allow_nan=False))


if __name__ == '__main__':
    main()
