#!/usr/bin/env python3
"""Native RGB -> expiring UI, hazard and screen-motion evidence; no input.

A lock associates a ring with a detected healthbar only. Screen motion and
silhouette candidates never manufacture a confirmed hull, identity or calibration.
Capture stamps use monotonic time; container/replay time is not accepted here.
"""
import copy
import math
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from aircraft_perception import detect as detect_aircraft, prepare as prepare_aircraft
from fire_hud import FireHudReader
from main_view_perception import FRAME_H, FRAME_W, load_native_templates, perceive_frame
from minimap_observation import FrameStamp
from main_view_tracks import MainViewTracker
from target_confirmation import confirm_target
from torpedo_perception import detect as detect_torpedoes


class ContinuousMainViewRecognizer:
    def __init__(self, max_age_s=.75):
        if not math.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError('max_age_s must be finite and positive')
        self.max_age_ns = int(max_age_s * 1e9)
        self.reader = FireHudReader()
        self.templates = load_native_templates()
        prepare_aircraft()
        self.tracker = MainViewTracker()
        self.epoch = None
        self.seq = self.captured_ns = -1
        self.previous = None
        self.previous_view = None
        self.previous_stamp = None
        self.stamp = None
        self.result = None
        self.reason = 'no_frame'
        self.now_ns = -1


    def revoke(self, reason='source_lost'):
        self.tracker.reset()
        self.previous = self.previous_view = self.previous_stamp = self.stamp = self.result = None
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

    def observe(self, rgb, stamp):
        if not isinstance(stamp, FrameStamp):
            self.revoke('invalid_stamp')
            raise ValueError('stamp must be FrameStamp')
        now = time.monotonic_ns()
        if not self._clock(now):
            return self._snapshot(now)
        if self.epoch is not None and stamp.epoch < self.epoch:
            self.revoke('old_epoch')
            return self._snapshot(now)
        if stamp.epoch != self.epoch:
            self.revoke('source_changed')
            self.epoch, self.seq, self.captured_ns = stamp.epoch, -1, -1
        if stamp.seq <= self.seq or stamp.captured_ns <= self.captured_ns:
            self.revoke('out_of_order')
            return self._snapshot(now)
        self.seq, self.captured_ns = stamp.seq, stamp.captured_ns
        if not self._fresh(stamp, now):
            self.revoke('future_frame' if now < stamp.captured_ns else 'stale_frame')
            return self._snapshot(now)
        try:
            if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.shape != (FRAME_H, FRAME_W, 3):
                raise ValueError('frame must be native 2560x1600 uint8 RGB')
            dt = None
            if self.previous_stamp is not None and self._fresh(self.previous_stamp, now):
                dt = (stamp.captured_ns - self.previous_stamp.captured_ns) / 1e9
            else:
                self.previous = self.previous_view = None
                self.tracker.reset()
            # Both current-frame hazards overlap HUD work. Join the bounded
            # workers on every path, including HUD rejection and exceptions.
            with ThreadPoolExecutor(max_workers=2) as worker:
                aircraft_job = worker.submit(detect_aircraft, rgb)
                torpedo_job = worker.submit(detect_torpedoes, rgb)
                hud = self.reader.detect(rgb)
                view = (perceive_frame(rgb, self.previous, self.templates, dt_s=dt,
                                       previous_view=self.previous_view, aircraft_job=aircraft_job,
                                       torpedo_job=torpedo_job)
                        if hud['battle']['active'] is True else None)
                # A rejected HUD must not hide either detector's failure.
                if view is None:
                    aircraft_job.result()
                    torpedo_job.result()
            if view is None:
                completed = time.monotonic_ns()
                if not self._clock(completed):
                    return self._snapshot(completed)
                self.revoke('battle_hud_unconfirmed')
                return self._snapshot(completed)
            lock = confirm_target(rgb, view['detections'])
            previous = rgb.copy()
            self.tracker.update(view, stamp.captured_ns)
            self.tracker.bind_lock(lock, view)
            completed = time.monotonic_ns()
            if not self._clock(completed):
                return self._snapshot(completed)
            if not self._fresh(stamp, completed):
                self.revoke('processing_expired')
                return self._snapshot(completed)
            self.previous = previous
            self.previous_view = view
            self.previous_stamp = self.stamp = stamp
            target = self.tracker.ui_target(lock, view)
            if target is not None:
                target.update(epoch=stamp.epoch, captured_ns=stamp.captured_ns,
                    valid_until_ns=stamp.captured_ns+self.max_age_ns-stamp.age_bound_ns)
            self.result = dict(view=view, lock=lock, hud=hud, target=target,
                camera=dict(stable=None, zoom_confirmed=None, mode='unknown',
                            view_evidence=hud.get('aiming_view')),
                mouse_response=dict(status='unknown'), aim_settled=False,
                fire_authorized=False)
            self.reason = 'fresh'
            return self._snapshot(completed)
        except Exception:
            self.revoke('perception_error')
            raise

    def _snapshot(self, now_ns):
        fresh = self.stamp is not None
        out = dict(epoch=self.epoch, seq=self.seq, fresh=fresh, reason=self.reason,
            captured_ns=self.stamp.captured_ns if fresh else None,
            age_ns=self.stamp.age_ns(now_ns) if fresh else None,
            valid_until_ns=(self.stamp.captured_ns+self.max_age_ns-self.stamp.age_bound_ns)
                if fresh else None,
            view=None, lock=dict(confirmed=False, state='unknown', target_association='unknown'),
            hud=None, target=None, camera=dict(stable=None, zoom_confirmed=None, mode='unknown'),
            mouse_response=dict(status='unknown'), aim_settled=False, fire_authorized=False)
        if fresh:
            out.update(copy.deepcopy(self.result))
        return out

    def snapshot(self, now_ns=None):
        now = time.monotonic_ns() if now_ns is None else now_ns
        if not self._clock(now):
            return self._snapshot(now)
        if self.stamp is not None and not self._fresh(self.stamp, now):
            self.revoke('observation_expired')
        return self._snapshot(now)
