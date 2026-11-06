#!/usr/bin/env python3
"""Fire at a minimap contact when no nameplate is ever drawn.

A submarine at periscope depth, or a target beyond the cyclone's detection
range, appears on the minimap and nowhere else: the main-view detector has no
red health bar to find, so the plate-driven chain can never fire. This path uses
the two measurements that *do* exist -- the minimap contact's bearing and range
-- and the reticle's own ballistic range readout.

It never invents evidence: every step records what was measured, and the shot is
labelled ``shot_basis = "minimap_contact"`` in the report.
"""
import math
import time

import numpy as np

RANGE_KM_PER_UNIT = 0.0167   # measured 2026-10-23: 60 units ~ 1 km (third person)
BEARING_PX_PER_DEG = 37.5    # 2560 px / 68.2 deg FOV


def contact_from_frame(frame, contacts_fn):
    """(km, relative_bearing_deg, own_pixel) or None."""
    seen, _ = contacts_fn(frame)
    if not seen or not seen.get('nearest'):
        return None
    own = seen['own_pixel']
    near = seen['nearest']
    pixel = near['pixel']
    km = math.hypot(pixel[0] - own[0], pixel[1] - own[1]) * 0.0501
    return km, near.get('relative_bearing_deg'), own


def yaw_units(relative_deg, gain_px_per_unit=1.5):
    if relative_deg is None:
        return 0.0
    return relative_deg * BEARING_PX_PER_DEG / gain_px_per_unit


def elevation_units(aim_km, target_km, km_per_unit=RANGE_KM_PER_UNIT, limit=120.0):
    """Vertical units that move the reticle's range from aim_km to target_km.

    Measured 2026-10-23: negative dy looks further; 60 units moved the readout
    24.4 -> 18.2 km, so one unit is ~0.10 km upward ... the sign and scale are
    both taken from that measurement, and the caller re-reads after each move.
    """
    if aim_km is None or target_km is None:
        return 0.0
    units = (aim_km - target_km) / km_per_unit
    return max(-limit, min(limit, units))


def shot_allowed(aim_km, target_km, bearing_deg, km_tolerance=0.4, bearing_tolerance=2.0):
    """Every number must exist and be inside its tolerance; otherwise no shot."""
    if aim_km is None or target_km is None or bearing_deg is None:
        return False, 'reading_missing'
    if abs(aim_km - target_km) > km_tolerance:
        return False, f'range_gap_{abs(aim_km - target_km):.2f}km'
    if abs(bearing_deg) > bearing_tolerance:
        return False, f'bearing_gap_{abs(bearing_deg):.1f}deg'
    return True, 'ok'
