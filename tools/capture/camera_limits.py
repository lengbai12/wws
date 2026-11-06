#!/usr/bin/env python3
"""One place for the camera-motion limits every tool must obey.

Measured 2026-10-21:

* a vertical action of 30-160 units threw the view at the sky and then slammed
  it onto the sea (the "视角突然抬升/下降" symptom);
* a sweep that walked further than the screen width parked the pointer on the
  edge, where every further move in that direction was silently clipped and the
  controller re-read the same frame forever (the "抖动" symptom).

Both symptoms are prevented by bounding the motion, not by reacting to it, so
the limits live here and every tool imports them.
"""

MAX_VERTICAL_STEP_UNITS = 6

# The 6-unit cap comes from a 2026-10-21 incident where a 30-160 unit vertical
# action threw the view at the sky. It is a *default*, not a law: on the CN
# client a third-person pitch of 60 units was measured to move the aim from
# 24.4 km to 18.2 km without any cut, and a 475 px vertical error cannot be
# closed six units at a time inside a 15 s budget. Tools that have measured a
# larger safe step in their current view call set_vertical_cap().
_HARD_VERTICAL_CAP = MAX_VERTICAL_STEP_UNITS


def set_vertical_cap(units):
    """Raise or restore the per-action vertical ceiling (None restores 6)."""
    global _HARD_VERTICAL_CAP
    if units is None:
        _HARD_VERTICAL_CAP = MAX_VERTICAL_STEP_UNITS
        return _HARD_VERTICAL_CAP
    value = abs(int(units))
    if value < 1 or value > 400:
        raise ValueError('vertical cap must be 1..400 units')
    _HARD_VERTICAL_CAP = value
    return _HARD_VERTICAL_CAP
# How far a sweep may walk from where it started before it must turn back.  The
# session screen is 1920 px wide and the measured gain is ~1.3 px per unit.
SCAN_BUDGET_UNITS = 700
MAX_HORIZONTAL_STEP_UNITS = 400


def clamp_vertical(dy, limit=MAX_VERTICAL_STEP_UNITS):
    """Clamp one vertical mouse action; never returns more than the hard cap."""
    limit = min(abs(int(limit)), _HARD_VERTICAL_CAP)
    if dy is None:
        return 0
    return max(-limit, min(limit, int(dy)))


def clamp_horizontal(dx, limit=MAX_HORIZONTAL_STEP_UNITS):
    limit = min(abs(int(limit)), MAX_HORIZONTAL_STEP_UNITS)
    if dx is None:
        return 0
    return max(-limit, min(limit, int(dx)))
