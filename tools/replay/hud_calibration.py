"""HUD calibration records and pixel measurement for captured WoWS frames.

Calibration is fixed per (resolution, ui_scale) and is written ONCE, then reused
for every frame of every recording. An element record stores the pixel rectangle
AND the mapping from pixels to the value that the element displays.

A rectangle is never trusted without a check: measure a frame whose expected
value is known from replay truth, then compare. A rectangle that cannot be
verified is reported as a problem, not silently used.

Frames are supplied as raw RGB24 bytes with their width and height. This module
does not open video files and has no third-party dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path

SCHEMA_VERSION = 1

# Element names used by the pipeline. Only own_hp_bar is required to produce
# labels today; the rest are placeholders that a later stage fills in.
ELEMENTS = (
    "own_hp_bar",
    "own_hp_text",
    "minimap",
    "score_bar",
    "timer",
    "consumable_bar",
)

FILL_DIRECTIONS = ("left_to_right", "right_to_left", "bottom_to_top", "top_to_bottom")

# A pixel whose every channel is below this is treated as letterbox or empty bar.
DARK_LEVEL = 24

# A column counts as part of the fill when this share of its pixels matches the
# reference colour. Rounded bar ends make a stricter value brittle.
FILL_COLUMN_RATIO = 0.6

# Coarse colour bucket width. Two pixels are "the same colour" when each channel
# falls in the same bucket.
BUCKET = 32

SATURATION_MIN = 40

# The reference colour of a readable bar must be a real colour. A flat dark
# region is an absent bar, not a full one.
MIN_REFERENCE_SATURATION = 30


def blank_calibration(resolution, content_rect=None):
    """Return an empty calibration record for a frame size."""
    return {
        "schema_version": SCHEMA_VERSION,
        "resolution": [int(resolution[0]), int(resolution[1])],
        "content_rect": list(content_rect) if content_rect else None,
        "ui_scale": None,
        "elements": {},
    }


def _is_int_pair(value):
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(v, int) and v > 0 for v in value)
    )


def _is_rect(value):
    return (
        isinstance(value, (list, tuple))
        and len(value) == 4
        and all(isinstance(v, int) for v in value)
        and value[2] > 0
        and value[3] > 0
    )


def validate_calibration(cal):
    """Return a list of problems. An empty list means the record is usable."""
    problems = []
    if not isinstance(cal, dict):
        return ["calibration is not an object"]
    if cal.get("schema_version") != SCHEMA_VERSION:
        problems.append("schema_version must be %d" % SCHEMA_VERSION)
    if not _is_int_pair(cal.get("resolution")):
        problems.append("resolution must be [width, height] of positive ints")
    content = cal.get("content_rect")
    if content is not None:
        if not _is_rect(content):
            problems.append("content_rect must be [x, y, w, h] or null")
        elif _is_int_pair(cal.get("resolution")):
            w, h = cal["resolution"]
            x, y, cw, ch = content
            if x < 0 or y < 0 or x + cw > w or y + ch > h:
                problems.append("content_rect falls outside resolution")
    elements = cal.get("elements")
    if not isinstance(elements, dict):
        problems.append("elements must be an object")
        return problems
    for name, element in elements.items():
        if name not in ELEMENTS:
            problems.append("unknown element %r" % name)
        if not isinstance(element, dict):
            problems.append("element %r is not an object" % name)
            continue
        if not _is_rect(element.get("rect")):
            problems.append("element %r has no usable rect" % name)
        elif _is_int_pair(cal.get("resolution")):
            w, h = cal["resolution"]
            x, y, ew, eh = element["rect"]
            if x < 0 or y < 0 or x + ew > w or y + eh > h:
                problems.append("element %r rect falls outside resolution" % name)
        direction = element.get("fill_direction")
        if direction is not None and direction not in FILL_DIRECTIONS:
            problems.append("element %r has an unknown fill_direction" % name)
    return problems


def load_calibration(path):
    """Load a calibration record and refuse one that does not validate."""
    cal = json.loads(Path(path).read_text(encoding="utf-8"))
    problems = validate_calibration(cal)
    if problems:
        raise ValueError("invalid calibration %s: %s" % (path, "; ".join(problems)))
    return cal


def save_calibration(path, cal):
    """Write a calibration record after validating it."""
    problems = validate_calibration(cal)
    if problems:
        raise ValueError("refusing to write invalid calibration: %s" % "; ".join(problems))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cal, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return path


def element_rect(cal, name):
    """Return [x, y, w, h] for an element, or None when it is not calibrated."""
    element = (cal.get("elements") or {}).get(name)
    if not isinstance(element, dict):
        return None
    rect = element.get("rect")
    return list(rect) if _is_rect(rect) else None


def frame_size(rgb, width, height):
    """Complain when the buffer does not match the declared frame size."""
    expected = width * height * 3
    if len(rgb) != expected:
        raise ValueError("frame buffer is %d bytes, expected %d for %dx%d RGB24"
                         % (len(rgb), expected, width, height))
    return width, height


def pixel(rgb, width, x, y):
    """Return the (r, g, b) tuple at a pixel."""
    i = (y * width + x) * 3
    return rgb[i], rgb[i + 1], rgb[i + 2]


def _bucket(rgb, width, x, y):
    r, g, b = pixel(rgb, width, x, y)
    return (r // BUCKET, g // BUCKET, b // BUCKET)


def _is_saturated(rgb, width, x, y):
    r, g, b = pixel(rgb, width, x, y)
    return max(r, g, b) - min(r, g, b) >= SATURATION_MIN


def _is_dark(rgb, width, x, y, level=DARK_LEVEL):
    r, g, b = pixel(rgb, width, x, y)
    return r < level and g < level and b < level


def letterbox_rect(rgb, width, height, level=DARK_LEVEL):
    """Return the [x, y, w, h] box that holds non-dark content.

    A gamescope capture can be larger than the game's render surface, so the
    delivered frame may carry solid bars. Labels and calibration rectangles are
    stated in frame pixels, and this box tells a later stage where the game
    actually is.
    """
    frame_size(rgb, width, height)
    top = None
    for y in range(height):
        if not all(_is_dark(rgb, width, x, y, level) for x in range(0, width, 8)):
            top = y
            break
    if top is None:
        return [0, 0, width, height]
    bottom = height - 1
    while bottom > top and all(_is_dark(rgb, width, x, bottom, level) for x in range(0, width, 8)):
        bottom -= 1
    left = 0
    while left < width and all(_is_dark(rgb, width, left, y, level) for y in range(top, bottom + 1, 8)):
        left += 1
    right = width - 1
    while right > left and all(_is_dark(rgb, width, right, y, level) for y in range(top, bottom + 1, 8)):
        right -= 1
    return [left, top, right - left + 1, bottom - top + 1]


def candidate_bars(rgb, width, height, region, min_length=40, min_rows=3):
    """Find saturated horizontal bars inside a region.

    Returns candidate rectangles sorted by area, largest first. Each candidate is
    ``{"rect": [x, y, w, h], "colour": [r, g, b], "rows": n}``. The caller picks
    one and verifies it; nothing here decides which bar is which HUD element.
    """
    frame_size(rgb, width, height)
    rx, ry, rw, rh = region
    runs = []
    for y in range(max(0, ry), min(height, ry + rh)):
        x = max(0, rx)
        end = min(width, rx + rw)
        while x < end:
            if not _is_saturated(rgb, width, x, y):
                x += 1
                continue
            key = _bucket(rgb, width, x, y)
            start = x
            while (x < end and _is_saturated(rgb, width, x, y)
                   and _bucket(rgb, width, x, y) == key):
                x += 1
            if x - start >= min_length:
                runs.append((y, start, x - 1, key))
    merged = []
    for y, start, stop, key in runs:
        placed = False
        for group in merged:
            if group["key"] != key or y != group["last_y"] + 1:
                continue
            overlap = min(group["stop"], stop) - max(group["start"], start) + 1
            if overlap <= 0 or overlap < 0.8 * min(group["stop"] - group["start"] + 1,
                                                   stop - start + 1):
                continue
            group["last_y"] = y
            group["start"] = min(group["start"], start)
            group["stop"] = max(group["stop"], stop)
            group["rows"] += 1
            placed = True
            break
        if not placed:
            merged.append({"key": key, "start": start, "stop": stop,
                           "first_y": y, "last_y": y, "rows": 1})
    out = []
    for group in merged:
        if group["rows"] < min_rows:
            continue
        rect = [group["start"], group["first_y"],
                group["stop"] - group["start"] + 1, group["rows"]]
        out.append({
            "rect": rect,
            "colour": [channel * BUCKET + BUCKET // 2 for channel in group["key"]],
            "rows": group["rows"],
        })
    out.sort(key=lambda item: item["rect"][2] * item["rect"][3], reverse=True)
    return out


def _reference_colour(rgb, width, rect, direction, share=0.1):
    """Return the most saturated pixel colour at the filled end of a bar.

    The filled end is guaranteed to hold bar pixels, and the most saturated one
    is the bar colour even when the bar is drawn with a bevel or gradient.
    """
    x, y, w, h = rect
    if direction in ("left_to_right", "right_to_left"):
        span = max(1, int(w * share))
        xs = range(x, x + span) if direction == "left_to_right" else range(x + w - span, x + w)
        ys = range(y, y + h)
    else:
        span = max(1, int(h * share))
        ys = range(y, y + span) if direction == "top_to_bottom" else range(y + h - span, y + h)
        xs = range(x, x + w)
    best = None
    best_saturation = -1
    for row in ys:
        for column in xs:
            r, g, b = pixel(rgb, width, column, row)
            saturation = max(r, g, b) - min(r, g, b)
            if saturation > best_saturation:
                best_saturation = saturation
                best = (r, g, b)
    return best


def _matches(colour, reference, tolerance):
    return all(abs(colour[i] - reference[i]) <= tolerance for i in range(3))


def measure_fill_fraction(rgb, width, height, rect,
                          direction="left_to_right", tolerance=48, min_pixels=1,
                          max_gap=2):
    """Measure how much of a bar is filled, as a fraction in [0, 1].

    A column belongs to the bar when it holds at least ``min_pixels`` pixels close
    to the bar colour, so a bevel, a gradient or a rounded end does not break the
    measurement. The fraction is the filled extent measured from the filled end,
    allowing ``max_gap`` empty columns inside the bar for rounded ends and ticks.

    The return value is ``None`` when the reference colour cannot be sampled,
    because an unreadable bar must not become a silent zero.
    """
    frame_size(rgb, width, height)
    x, y, w, h = rect
    if w <= 0 or h <= 0:
        raise ValueError("rect must have a positive size")
    reference = _reference_colour(rgb, width, rect, direction)
    if reference is None:
        return None
    # An empty or absent bar is a flat dark region. Without this guard its own
    # colour becomes the reference and the whole rectangle measures as full.
    if max(reference) - min(reference) < MIN_REFERENCE_SATURATION:
        return None
    if direction in ("left_to_right", "right_to_left"):
        columns = range(x, x + w) if direction == "left_to_right" else range(x + w - 1, x - 1, -1)
        filled = 0
        gap = 0
        for column in columns:
            count = sum(1 for row in range(y, y + h)
                        if _matches(pixel(rgb, width, column, row), reference, tolerance))
            if count >= min_pixels:
                filled += 1 + gap
                gap = 0
            else:
                gap += 1
                if gap > max_gap:
                    break
        return filled / w
    rows = range(y, y + h) if direction == "top_to_bottom" else range(y + h - 1, y - 1, -1)
    filled = 0
    gap = 0
    for row in rows:
        count = sum(1 for column in range(x, x + w)
                    if _matches(pixel(rgb, width, column, row), reference, tolerance))
        if count >= min_pixels:
            filled += 1 + gap
            gap = 0
        else:
            gap += 1
            if gap > max_gap:
                break
    return filled / h


def cropping_argv(video, at_s, rect, scale=None):
    """Return the ffmpeg argv that yields one raw RGB24 frame or rect crop.

    Kept here so the CLI and the tests describe the frame the same way.
    """
    filters = []
    if rect:
        filters.append("crop=%d:%d:%d:%d" % (rect[2], rect[3], rect[0], rect[1]))
    if scale:
        filters.append("scale=%d:%d" % (scale[0], scale[1]))
    argv = ["ffmpeg", "-v", "error"]
    if at_s is not None:
        argv += ["-ss", "%.6f" % at_s]
    argv += ["-i", str(video), "-frames:v", "1"]
    if filters:
        argv += ["-vf", ",".join(filters)]
    argv += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    return argv


def probe_size(video, runner):
    """Return (width, height) from ffprobe output, using the given runner."""
    text = runner(["ffprobe", "-v", "error", "-select_streams", "v:0",
                   "-show_entries", "stream=width,height",
                   "-of", "csv=p=0", str(video)])
    parts = text.strip().splitlines()[0].split(",")
    return int(parts[0]), int(parts[1])
