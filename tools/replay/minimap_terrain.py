#!/usr/bin/env python3
"""Does land sit between us and the target? Read it off the minimap.

The minimap draws islands as tan polygons over dark water (measured on the
s06_Atoll panel: land mean RGB 116/102/76, water 0/32/32), so a straight segment
between the two ship marks can be tested for land crossings. That is the only
line-of-fire evidence available in a live frame: the 3-D view cannot tell us
whether a ridge is hiding the target's hull or standing in the shell's path.

A single land sample is not enough: measured 2026-10-23, a capture-point icon
(a ~35 px orange blob) on the line made the gate refuse two legitimate shots on an
open-ocean map. Land therefore has to be part of a blob of real size -- islands
measured >1000 px against an icon's ~35 px -- not one icon-sized mark. The length
the line spends inside the blob is reported as evidence, but it is not the
discriminator: a 200 m islet is a short crossing and still stops a shell.
"""
import numpy as np

# Land rule measured on a real 720x723 panel (s06_Atoll, 2026-10-23).
LAND_RULE = dict(red_min=90, green_min=75, blue_max=120, red_over_blue=25)


def land_mask(panel_rgb, rule=None):
    """Boolean mask of the minimap's land pixels for one 720x723 panel."""
    cfg = dict(LAND_RULE)
    cfg.update(rule or {})
    image = np.asarray(panel_rgb).astype(np.int16)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError('panel must be HxWx3 RGB')
    red, green, blue = image[..., 0], image[..., 1], image[..., 2]
    return ((red > cfg['red_min']) & (green > cfg['green_min'])
            & (blue < cfg['blue_max'])
            & (red - blue > cfg['red_over_blue']) & (green > blue))


def segment_points(start, end, samples=128):
    """Evenly spaced points along start->end, inclusive of both ends."""
    if samples < 2:
        raise ValueError('samples must be >= 2')
    x0, y0 = float(start[0]), float(start[1])
    x1, y1 = float(end[0]), float(end[1])
    ts = np.linspace(0.0, 1.0, samples)
    return np.stack([x0 + (x1 - x0) * ts, y0 + (y1 - y0) * ts], axis=1)


def longest_run_px(hits, step_tolerance=2.0):
    """Longest stretch of line that stays inside land, in pixels.

    Counting samples would make the threshold depend on how finely the segment is
    sampled: a 4 px icon covered 10 samples at 128 samples across a short line.
    The span of the run is a length the caller can reason about.
    """
    if not hits:
        return 0.0
    best = 0.0
    start = previous = hits[0]
    for current in hits[1:]:
        gap = max(abs(current[0] - previous[0]), abs(current[1] - previous[1]))
        if gap > step_tolerance:
            span = abs(previous[0] - start[0]) + abs(previous[1] - start[1])
            best = max(best, float(span))
            start = current
        previous = current
    span = abs(previous[0] - start[0]) + abs(previous[1] - start[1])
    return max(best, float(span))


def blob_area(mask, seed, limit=400000):
    """Area of the land blob containing ``seed`` (4-neighbour flood fill)."""
    height, width = mask.shape[:2]
    x0, y0 = int(round(seed[0])), int(round(seed[1]))
    if not (0 <= x0 < width and 0 <= y0 < height):
        return 0
    if not mask[y0, x0]:
        # The reported hit comes from the dilated test, so it can sit one pixel
        # beside the coast. Look a little around it before giving up.
        found = None
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                nx, ny = x0 + dx, y0 + dy
                if 0 <= nx < width and 0 <= ny < height and mask[ny, nx]:
                    found = (nx, ny)
                    break
            if found:
                break
        if found is None:
            return 0
        x0, y0 = found
    seen = np.zeros_like(mask, dtype=bool)
    stack = [(x0, y0)]
    seen[y0, x0] = True
    area = 0
    while stack and area < limit:
        x, y = stack.pop()
        area += 1
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < width and 0 <= ny < height and mask[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                stack.append((nx, ny))
    return area


def segment_clear(mask, start, end, samples=128, margin=1, min_run_px=0.0):
    """Is the straight line between the two marks free of land?

    ``margin`` dilates the land test by N pixels (a coast that the shell would
    graze is treated as blocking).
    """
    height, width = mask.shape[:2]
    hits = []
    for x, y in segment_points(start, end, samples):
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < width and 0 <= yi < height):
            continue
        y0, y1 = max(0, yi - margin), min(height, yi + margin + 1)
        x0, x1 = max(0, xi - margin), min(width, xi + margin + 1)
        if mask[y0:y1, x0:x1].any():
            hits.append([xi, yi])
    run = longest_run_px(hits)
    # Zero means "any hit blocks", which still requires a hit: 0 >= 0 would have
    # blocked an empty line.
    blocked = bool(hits) and run >= float(min_run_px)
    row = dict(clear=not blocked, land_samples=len(hits), samples=int(samples),
               first_hit=hits[0] if hits else None,
               longest_run_px=round(run, 1), min_run_px=float(min_run_px))
    if hits and not blocked:
        row['ignored'] = 'isolated_marks_not_a_coast'
    return row


def shot_terrain(panel_rgb, own_xy, enemy_xy, samples=128, margin=1,
                 min_run_px=0.0, min_blob=120):
    """Convenience: mask + segment test, with the numbers a report should keep."""
    if own_xy is None or enemy_xy is None:
        return dict(clear=None, reason='no_contact', land_samples=None)
    mask = land_mask(panel_rgb)
    row = segment_clear(mask, own_xy, enemy_xy, samples=samples, margin=margin,
                        min_run_px=min_run_px)
    row['land_pixels'] = int(mask.sum())
    if not row['clear']:
        area = blob_area(mask, row['first_hit'])
        row['blob_px'] = int(area)
        if area < min_blob:
            # A capture-point icon or a ship mark, not a coast: an icon measured
            # ~35 px and refused two shots on open water.
            row['clear'] = True
            row['ignored'] = 'blob_too_small_to_be_land'
            row['min_blob'] = int(min_blob)
    row['reason'] = 'clear' if row['clear'] else 'land_on_the_line'
    return row
