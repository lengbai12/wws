"""Shell-fall lead for a moving target, using the client's own flight time.

The minimap ring is the impact point, so aiming is: put the ring where the
target *will be* when the shells arrive. That is a fixed point, because the
flight time itself depends on the distance to the predicted point:

    t = flight_time(|target + v*t - own|)
    aim = target + v*t

The client already solves the ballistics and shows the flight time ("13s")
beside the reticle, so callers pass that reading in and do not need a shell
speed table. Nothing here is a learned model: it is the closed-form geometry
that the model would have had to rediscover from replays.
"""
from __future__ import annotations

import math


def flight_seconds(distance_m, speed_m_s=800.0):
    """Nominal time of flight, used only when the client's own reading is absent."""
    if distance_m <= 0 or speed_m_s <= 0:
        return 0.0
    return distance_m / speed_m_s


def lead_offset(velocity, seconds):
    """Offset added to the observed target position, per axis."""
    return (velocity[0] * seconds, velocity[1] * seconds)


def solve_aim(target, velocity, flight_time, iterations=6, tolerance=1e-6):
    """Fixed point of `aim = target + velocity * flight_time(aim)`.

    `flight_time` is a callable taking the aim distance and returning seconds.
    Returns `(aim, seconds, residual)`. The iteration is bounded and the
    residual is reported so a caller can refuse a solution that did not settle.
    """
    aim = (float(target[0]), float(target[1]))
    seconds = 0.0
    for _ in range(max(1, iterations)):
        seconds = float(flight_time(aim))
        nxt = (target[0] + velocity[0] * seconds, target[1] + velocity[1] * seconds)
        residual = math.hypot(nxt[0] - aim[0], nxt[1] - aim[1])
        aim = nxt
        if residual <= tolerance:
            break
    return aim, seconds, residual


def map_velocity(previous, current, dt_s):
    """Screen/minimap velocity between two observations, in px per second."""
    if dt_s <= 0:
        return (0.0, 0.0)
    return ((current[0] - previous[0]) / dt_s, (current[1] - previous[1]) / dt_s)


def smooth_velocity(previous, current, blend=0.4):
    """Exponential smoothing with a sign-flip reset.

    Measured on live frames: frame-to-frame differences swing between -58 and
    +90 px/s, and multiplying that by a six-second flight time produced a
    400 px lead that made convergence impossible. A flipped sign means the
    estimate is noise, not a turn, so it restarts instead of blending.
    """
    if previous is None:
        return current
    if (previous[0] * current[0] < 0) or (previous[1] * current[1] < 0):
        return current
    return (previous[0] + blend * (current[0] - previous[0]),
            previous[1] + blend * (current[1] - previous[1]))


def clamp_lead(offset, limit):
    """Bound a lead vector so one bad velocity sample cannot dominate the aim."""
    norm = math.hypot(offset[0], offset[1])
    if norm <= limit or norm == 0:
        return offset
    scale = limit / norm
    return (offset[0] * scale, offset[1] * scale)
