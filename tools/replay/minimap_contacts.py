#!/usr/bin/env python3
"""Turn minimap detections into a bearing and a range for the aim pipeline.

The minimap is north-up, so the panel's +u axis is east and +v is south. The
own arrow's angle is the hull heading in the same frame. Ranges need the map's
own scale, so metres per pixel is an explicit argument, never a constant that
would silently be wrong on another map.
"""
import math


def bearing_deg(own_pixel, contact_pixel):
    """Compass bearing from the own pixel to a contact pixel, 0 = north."""
    east = contact_pixel[0] - own_pixel[0]
    south = contact_pixel[1] - own_pixel[1]
    return (math.degrees(math.atan2(east, -south)) + 360.0) % 360.0


def range_m(own_pixel, contact_pixel, metres_per_pixel):
    if not metres_per_pixel or metres_per_pixel <= 0:
        raise ValueError('metres_per_pixel must be positive')
    return math.hypot(contact_pixel[0] - own_pixel[0],
                      contact_pixel[1] - own_pixel[1]) * metres_per_pixel


def relative_bearing_deg(own_heading_deg, contact_bearing_deg):
    """Contact bearing relative to the hull: 0 ahead, +starboard, -port."""
    if own_heading_deg is None:
        return None
    return (contact_bearing_deg - own_heading_deg + 540.0) % 360.0 - 180.0


def nearest_contact(own_pixel, contacts, metres_per_pixel):
    """The closest contact by range, with its bearing and relative bearing."""
    best = None
    for contact in contacts:
        pixel = contact.get('pixel') or [contact.get('pixel_x'), contact.get('pixel_y')]
        if any(v is None for v in pixel):
            continue
        distance = range_m(own_pixel, pixel, metres_per_pixel)
        row = dict(contact=contact, pixel=list(pixel), bearing_deg=bearing_deg(own_pixel, pixel),
                   range_m=distance)
        if best is None or distance < best['range_m']:
            best = row
    return best


def lead_seconds(distance_m, shell_speed_mps, latency_s=0.0):
    """Nominal flight time for a lead, never a claim about the real ballistics.

    The speed is an average supplied by the caller; the game's own flight time
    is what a validated version must use.
    """
    if not shell_speed_mps or shell_speed_mps <= 0:
        raise ValueError('shell_speed_mps must be positive')
    if distance_m is None or distance_m < 0:
        raise ValueError('distance_m must be non-negative')
    return distance_m / shell_speed_mps + max(0.0, latency_s)


__all__ = ['bearing_deg', 'range_m', 'relative_bearing_deg', 'nearest_contact',
           'lead_seconds']
