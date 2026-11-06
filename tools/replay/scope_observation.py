"""Healthbar -> bounded scope observation plans, with no input or fire authority.

Shift is a suggested binding, not a key-hold contract. A visible scope ruler does
not establish zoom calibration. Track IDs must be reacquired after view changes.
"""
import math
import time


class ScopeObservationPlanner:
    def __init__(self, ttl_ns=150_000_000, transition_ns=2_000_000_000):
        for value in (ttl_ns, transition_ns):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError('deadlines must be positive integer nanoseconds')
        self.ttl_ns, self.transition_ns = ttl_ns, transition_ns
        self.epoch = None
        self.seq = self.captured = self.now = -1
        self.mode = None
        self.track = None
        self.track_expiry = 0
        self.requested_at = None

    def decide(self, observation, now_ns=None, focused=False, source_available=True):
        now = time.monotonic_ns() if now_ns is None else now_ns
        valid_clock = isinstance(now, int) and not isinstance(now, bool) and now >= 0
        out = dict(kind='scope_observation', action='revoke', state='unavailable',
                   reason='invalid_observation', valid_until_ns=now if valid_clock else None,
                   healthbar_pixel=None, detection_index=None, screen_track_id=None,
                   suggested_key=None, key_semantics='unverified', input_authorized=False,
                   fire_authorized=False, release_buttons=True)

        def stop(reason):
            self.track, self.track_expiry, self.mode = None, 0, None
            # A lost acknowledgement never permits another blind Shift request.
            out.update(reason=reason, state='unavailable')
            return out

        if not valid_clock:
            return stop('invalid_clock')
        if now < self.now:
            return stop('clock_regressed')
        self.now = now
        if not source_available:
            return stop('source_lost')
        if not focused:
            return stop('focus_lost')
        if not isinstance(observation, dict):
            return stop('invalid_observation')
        o = observation
        epoch, seq, captured, expiry = [o.get(k) for k in
                                       ('epoch', 'seq', 'captured_ns', 'valid_until_ns')]
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 0
               for v in (epoch, seq, captured, expiry)):
            return stop('invalid_stamp')
        if self.epoch is not None and epoch < self.epoch:
            return stop('old_epoch')
        if epoch != self.epoch:
            self.epoch, self.seq, self.captured = epoch, -1, -1
            self.mode, self.track, self.track_expiry, self.requested_at = None, None, 0, None
        if seq <= self.seq or captured <= self.captured:
            return stop('out_of_order')
        consecutive = seq == self.seq + 1
        self.seq, self.captured = seq, captured
        if o.get('fresh') is not True or captured > now or expiry <= now:
            return stop('expired_or_future_observation')
        hud = o.get('hud') or {}
        if (hud.get('battle', {}).get('active') is not True or
                hud.get('weapon', {}).get('kind') != 'gun' or
                hud.get('weapon', {}).get('status') != 'observed'):
            return stop('battle_or_gun_unconfirmed')
        detections = (o.get('view') or {}).get('detections', [])
        candidates = []
        for index, row in enumerate(detections):
            box = row.get('nameplate_bbox')
            if (row.get('team') != 'enemy' or row.get('evidence') != 'nameplate_healthbar'
                    or row.get('track_state') == 'ambiguous' or
                    not isinstance(box, (tuple, list)) or len(box) != 4 or
                    any(not isinstance(v, (int, float)) or isinstance(v, bool) or
                        not math.isfinite(v) for v in box)):
                continue
            x, y, w, h = box
            if w > 0 and h > 0 and 0 <= x < x+w <= 2560 and 80 <= y < y+h <= 1520:
                candidates.append(index)
        lock = o.get('lock') or {}
        locked = lock.get('detection_index')
        if (lock.get('confirmed') is True and lock.get('target_association') == 'unique_healthbar'
                and isinstance(locked, int) and not isinstance(locked, bool) and locked in candidates):
            selected = locked
        elif len(candidates) == 1:
            selected = candidates[0]
        else:
            return stop('healthbar_ambiguous' if candidates else 'no_enemy_healthbar')
        row = detections[selected]
        x, y, w, h = row['nameplate_bbox']
        ruler = hud.get('aiming_view') or {}
        camera = o.get('camera') or {}
        mode = ('scope' if ruler.get('status') == 'observed' and ruler.get('mode') == 'scope_ruler'
                else 'external' if camera.get('status') == 'observed' and camera.get('mode') == 'external'
                else 'unknown')
        previous_mode, previous_track = self.mode, self.track
        self.mode = mode
        self.track = None
        out.update(action='inspect_healthbar', state='view_unknown', reason='view_mode_unconfirmed',
                   valid_until_ns=min(expiry, now+self.ttl_ns), detection_index=selected,
                   healthbar_pixel=[x+w/2, y+h/2], screen_track_id=row.get('track_id'))
        if mode == 'scope':
            self.requested_at = None
            track = row.get('track_id')
            if (previous_mode == 'scope' and consecutive and track is not None and
                    track == previous_track and row.get('track_state') == 'tracked' and
                    now < self.track_expiry):
                out.update(action='confirm_hull', state='scope_observed',
                           reason='scope_healthbar_reacquired; hull_and_fire_solution_unconfirmed')
            else:
                out.update(state='reacquiring', reason='scope_requires_new_healthbar_continuity')
            self.track, self.track_expiry = track, expiry
        elif self.requested_at is not None:
            out.update(action='hold', state='transition_failed' if now-self.requested_at >= self.transition_ns
                       else 'awaiting_scope', reason='scope_transition_timeout' if
                       now-self.requested_at >= self.transition_ns else 'awaiting_scope_evidence')
        elif mode == 'external':
            # Only a confirmed external view may propose one Shift transition.
            # Pixel coordinates remain an inspection cue, never a firing anchor.
            self.requested_at = now
            out.update(action='request_scope', state='awaiting_scope', suggested_key='Shift_L',
                       reason='confirmed_external_view; input_binding_and_alignment_require_verification')
        return out
