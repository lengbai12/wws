#!/usr/bin/env python3
"""Click an ocean point towards an enemy whose ship class was read off the tactical map.

Non-battle only: this tool never turns the rudder, moves the camera or fires. It
reads a captured tactical-map frame, finds the white own-ship marker and the red
enemy ship icons, classifies each icon as CV / BB / CA / DD / SS, and returns the
window point that sets an autopilot waypoint on the own->enemy line, short of
the enemy. `--execute` sends exactly one left click through
`click_game_point.click`, the clicker that is already verified on this client.

`--execute` is guarded because a left click in the 3D view is a salvo: the tool
only clicks when the frame proves an open tactical map (white own-ship marker
plus a classified enemy icon), the chosen point sits on flat sea, and the frame
is at most `--max-frame-age` seconds old. It never presses M, never moves the
camera and never fires. The dry run (the default) prints the point it would
click and writes the same report.

Icon conventions (researched 2026-10-23)
----------------------------------------
The minimap / tactical map draws one top-down ship silhouette per contact,
tinted by team, and the silhouette encodes the class. Wargaming's own wiki
documents the colours and the class code used in chat symbols:
https://wiki.worldofwarships.com/Ship:Controls
  * "a = ally (green, pointed right)", "e = enemy (red, pointed left)",
    "s = neutral (white)" and "0 = CV, 1 = BB, 2 = CA/CL, 3 = DD,
    4 = Auxiliary ship, 5 = SS".
  * "M: open/close tactical map (in battle) ... Click on tactical map: enable
    autopilot and set first waypoint" -- the click this tool places.
https://wiki.worldofwarships.com/Ship:Detection
  * allied ships and team-spotted enemies are drawn on the minimap; contacts
    outside view range are drawn "as outlined icons", which is why the game
    ships *_invisible / *_last_visible variants of every icon.
The same convention (own ship white arrow, allies green, enemies red) is
described for the Blitz client at https://wowsb.fandom.com/wiki/Controls
(the page itself refused to load here; only the search snippet was read).

The silhouettes are not a guess: this client installs them as SVG assets in
`state/toalba/gamedata/data/gui/fla/minimap/ship_icons/minimap_<class>.svg`, and
`ICON_POLYGONS` below transcribes the glyph paths from those files. All five
icons share one 9x16 dark hull (`#333333`), bow at the top, and differ only in
the white glyph drawn on it:
  * cv  9x16 -- flat deck: a thin chevron across the bow plus two long blocks
                side by side aft (blocks run fore-aft, split across the beam);
  * bb  9x16 -- three thick bands across the beam (bow wedge, midships band,
                stern wedge), i.e. the three turret groups;
  * ca  9x16 -- two thick bands with a gap around midships (bow wedge, stern
                wedge), wider than the BB bands;
  * dd  7x13 -- a single solid triangle, no bands at all;
  * ss  9x16 -- the DD triangle plus a separate thin bar across the stern
                (the submarine's casing / wake bar).
Because the glyph is drawn over a *dark* hull and the sea is dark blue, the
classifier only uses chromatic, bright pixels: that keeps the grey selection
ring, the white own-ship arrow and the grey "last visible" outlines out.

How a patch is classified
-------------------------
1. `icon_signal` turns the patch into a 0..1 glyph field: brightness above the
   patch's own background, weighted by colour saturation, so only the tinted
   class glyph survives.
2. `_moment_frame` measures the glyph's centre, size and length/width ratio
   from weighted second moments -- rotation and pixel-size independent. Patches
   whose ratio is outside 1.55..3.6 (a red disc 1.0, a HUD digit 1.5, a
   one-pixel line 4.9) are rejected before any matching.
3. `_resample` samples that field into a 25x25 canonical frame, and the frame is
   correlated with the five template glyphs at 10 degree steps
   (`_templates`, rasterised from `ICON_POLYGONS` by `_raster`). Class is the
   argmax; accepted only when the correlation is >= 0.72 and leads the runner-up
   by >= 0.04.

Honest coverage
---------------
* Real frames: the only enemy contact available in
  `state/cn360-flow-20261021/{map2,map4}/frame-000.png` and
  `course6/after_click.png` is a battleship. BB is therefore the only class
  verified on real client pixels -- three tactical-map crops plus one
  maximap-panel crop, all `bb` with score 0.93, and stable to a +/-1 px shift of
  the crop centre.
* CV, CA, DD and SS are verified only against synthetic icons rasterised from
  the game's own SVG glyph paths, at 1.0x..2.6x of the design grid and every 15
  degrees: 700 of 720 cases classified correctly, 20 returned `unknown`, 0 were
  given the wrong class. The 20 are dd/ss below ~1.2 px per design unit, where
  the submarine's 1 px stern gap closes and the two glyphs are the same picture.
  dd/ss is the weakest pair overall (margin 0.04..0.08); bb/cv/ca margins are
  much wider.
* The submarine icon is verified as the *surfaced* glyph only; the
  `minimap_submarine_invisible.svg` outline variant is not handled.
* Route direction is measured on real post-click frames: the three saved
  desktop captures `course2` and `course3` measure as 60.1 and 136.1 degrees
  off the bearing (they really are drawn elsewhere), and
  `standoff-session/course-toward-enemy` measures 1.5 degrees (aligned).
  Frames without a route (`lm3`, `lm2`) return None/unverifiable; the route in
  `goal-static-shot-20261017/course` is too faint to trace, so it also returns
  unverifiable rather than a guess. The capture -> measure -> correct loop is
  covered by synthetic frames only; no live `--execute` run with verification
  was performed while writing this.

Post-click verification
-----------------------
A waypoint click is only half the fact: the route the game *draws* must point
at the enemy. After the click this tool captures a fresh frame (the resident
`capture_host.py snapshot`, no new capture process) and measures the drawn
route: a dashed neutral line from the own-ship arrow to a small hollow waypoint
diamond. The dashes are traced from the arrow outward; the diamond ends the
chain. `measure_course` returns the bearing of own->diamond and the angle to
own->enemy; `course_direction` returns that angle alone. Both return None when
no chain (no dashed route) can be traced.

With `--verify-iterations` (default 1) the tool measures once after the click.
An angle above 15 degrees rotates the waypoint click around the own marker by
the signed error and sends exactly one correction, then stops; there are never
more than two clicks (one waypoint plus one correction). The correction itself
is guarded exactly like the first click: the fresh frame must show an open
tactical map with a classified enemy, the corrected point must be flat sea, and
the frame must be fresh. When the direction cannot be measured the report says
`verification.state = unverifiable` and no blind second click is sent. The
report also records the first expected bearing (`expected_direction_deg`) and,
per measurement, the drawn and expected bearings.

Geometry: on this client the tactical map is ~0.042 km/px at the 1924x1058
window size -- measured from the client's own captions on the same frames
(14.8 km / 356.8 px, 12.5 km / 292.7 px). The maximap panel is a different,
already-calibrated scale (0.0501 km/px in `tools/capture/aim_servo.py`) and must
not be reused here. The default map rectangle was measured from the grid lines
of `state/cn360-flow-20261021/map4/frame-000.png`
(11 horizontal and 11 vertical lines, 74..926 / 546..1364 in a 1924x1058
window).
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

# Measured on the 1924x1058 game window (state/cn360-flow-20261021/map4).
WINDOW_SIZE = (1924, 1058)
DEFAULT_MAP_RECT = (546, 74, 1364, 926)
# The tactical map at this window size: own->enemy pixel distances against the
# client's own captions on the same frames (14.8 km / 356.8 px, 12.5 km /
# 292.7 px) give 0.0415 and 0.0427 km/px. The maximap panel is a different,
# already-calibrated scale (0.0501 km/px in tools/capture/aim_servo.py) and does
# not apply here.
KM_PER_PX = .042
MIN_GAP_PX = 18          # a waypoint this close to the icon would select the ship
PATCH_HALF = 13          # half size of the icon patch handed to classify_icon
CLUSTER_GAP_PX = 10      # parts of one icon are at most this far apart

PATCH_SIZE = 25          # side of the canonical frame the glyph is sampled into
TARGET_LENGTH = 20.      # the glyph's own length inside that frame, in px
ANGLE_STEP = 10          # matcher heading resolution, degrees
MIN_SCORE = .72          # normalised correlation needed for a class
MIN_MARGIN = .04         # lead the winner must have over the runner-up
MIN_ASPECT, MAX_ASPECT = 1.55, 3.6   # length/width of a ship glyph

# Post-click verification (measured on state/cn360-flow-20261021/course2 and
# course3, 2560x1600 desktop captures): the autopilot route is a dashed neutral
# line that starts at the own-ship arrow and ends at a small hollow diamond.
COURSE_TOLERANCE_DEG = 15.    # route direction may differ from the bearing this much
MAX_CLICKS = 2                # one waypoint click plus at most one correction
MAX_VERIFY_ITERATIONS = 3     # capture/measure passes; clicks stay capped
NEUTRAL_MIN_CHANNEL = 120     # the dashed line is white, the sea is not
NEUTRAL_MAX_SPREAD = 45       # ... and neutral (no team tint)
DASH_MIN_AREA, DASH_MAX_AREA = 2, 80
DASH_MAX_EXTENT = 9           # a dash is a dot, not a glyph
MARKER_MIN_AREA, MARKER_MAX_AREA = 40, 200
MARKER_MIN_EXTENT, MARKER_MAX_EXTENT = 9, 24   # the waypoint diamond
MARKER_MIN_SIDE = 4
CHAIN_FIRST_STEP = 55.        # the arrow itself hides the first dashes
CHAIN_STEP = 40.              # dashes are 7-12 px apart but some are dim
CHAIN_MAX_TURN_DEG = 55.      # the route bends gently near the ship
CHAIN_MARKER_RADIUS = 22.     # the diamond sits at the end of the dashes
CHAIN_MIN_DASHES = 3
COURSE_MIN_NET_PX = 40.       # shorter chains are not a course
OWN_HINT_RADIUS = 60.         # how far the arrow may move between two frames

# Transcribed from state/toalba/gamedata/data/gui/fla/minimap/ship_icons/*.svg.
# Glyph polygons only: the shared #333333 hull is excluded on purpose.
ICON_POLYGONS = {
    'dd': [[(1.25, 15), (7.75, 15), (4.5, 3)]],
    'ss': [[(2.0625, 12), (6.9375, 12), (4.5, 3)],
           [(1.25, 13), (7.20833, 13), (7.75, 15), (1.79167, 15)]],
    'ca': [[(8, 6), (4.5, 2), (1, 6), (1, 7), (8, 10)],
           [(1, 15), (8, 15), (8, 12), (1, 9)]],
    'bb': [[(1, 6), (4.5, 2), (8, 6), (8, 9), (1, 6)],
           [(1, 7.456), (1, 8.544), (8, 11.544), (8, 10.456)],
           [(1, 15), (1, 10), (8, 13), (8, 15)]],
    'cv': [[(8, 6), (4.5, 2), (1, 6), (1, 7), (8, 7)],
           [(8, 8), (5, 8), (5, 15), (8, 15)],
           [(4, 8), (1, 8), (1, 15), (4, 15)]],
}
CLASSES = ('cv', 'bb', 'ca', 'dd', 'ss')


# --------------------------------------------------------------------------- png

def png_size(path):
    """Width and height from the IHDR chunk, without decoding any pixels."""
    data = Path(path).read_bytes()
    if data[:8] != b'\x89PNG\r\n\x1a\n' or data[12:16] != b'IHDR':
        raise SystemExit(f'{path}: not a PNG file')
    width, height = struct.unpack('>II', data[16:24])
    return width, height


def read_png(path):
    """uint8 HWC RGB, via Pillow when it is importable and ffmpeg otherwise.

    The accepted dry-run interpreter has numpy but no Pillow, so a hard Pillow
    requirement would make the tool unusable exactly where it is documented.
    """
    try:
        from PIL import Image
    except ImportError:
        result = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-f',
                                 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                                capture_output=True, check=False)
        if result.returncode != 0:
            raise SystemExit(f'{path}: decode failed ({result.stderr.decode().strip()[:200]}) '
                             'and Pillow is not installed')
        width, height = png_size(path)
        return np.frombuffer(result.stdout, np.uint8).reshape(height, width, 3)
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB'), dtype=np.uint8)


# ------------------------------------------------------------------- components

def label_components(mask):
    """4-connected components of a bool mask, as an int32 label image and count.

    Horizontal runs plus a union-find, so a full-map mask costs one pass over
    rows instead of one Python step per pixel.
    """
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    padded = np.zeros((height, width + 2), np.int8)
    padded[:, 1:-1] = mask
    edges = np.diff(padded, axis=1)
    starts_row, starts_x = np.nonzero(edges == 1)
    ends_row, ends_x = np.nonzero(edges == -1)
    runs = list(zip(starts_row.tolist(), starts_x.tolist(), ends_x.tolist()))
    if not runs:
        return np.zeros(mask.shape, np.int32), 0
    parent = list(range(len(runs)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    previous = []
    index = 0
    while index < len(runs):
        row = runs[index][0]
        current = []
        while index < len(runs) and runs[index][0] == row:
            current.append(index)
            index += 1
        for old in previous:
            old_start, old_end = runs[old][1], runs[old][2]
            for new in current:
                if old_start < runs[new][2] and runs[new][1] < old_end:
                    root_old, root_new = find(old), find(new)
                    if root_old != root_new:
                        parent[root_new] = root_old
        previous = current
    labels = np.zeros(mask.shape, np.int32)
    names = {}
    for position, (row, start, end) in enumerate(runs):
        root = find(position)
        name = names.setdefault(root, len(names) + 1)
        labels[row, start:end] = name
    return labels, len(names)


# ---------------------------------------------------------------------- features

def icon_signal(patch):
    """Glyph brightness in 0..1, weighted by colour saturation.

    Team colour carries the class glyph (green ally / red enemy); the dark hull,
    the grey selection ring, the white own-ship arrow and the sea are all either
    neutral or dim, so they contribute nothing to this field.
    """
    colour = patch.astype(np.float64)
    lum = colour @ np.array([.299, .587, .114])
    chroma = colour.max(axis=2) - colour.min(axis=2)
    border_lum = np.concatenate([lum[0], lum[-1], lum[:, 0], lum[:, -1]])
    border_chroma = np.concatenate([chroma[0], chroma[-1], chroma[:, 0], chroma[:, -1]])
    background = float(np.median(border_lum))
    peak = float(np.percentile(lum, 99.5))
    chroma_background = float(np.median(border_chroma))
    bright = np.clip((lum - background) / max(peak - background, 1e-6), 0., 1.)
    weighted = np.clip((chroma - chroma_background - 8.) / 50., 0., 1.)
    return bright * weighted


def _sorted_parts(labels, count):
    """Pixel coordinates of each label, ordered by label, in one pass."""
    ys, xs = np.nonzero(labels)
    values = labels[ys, xs]
    order = np.argsort(values, kind='stable')
    ys, xs, values = ys[order], xs[order], values[order]
    starts = np.searchsorted(values, np.arange(1, count + 1), 'left')
    ends = np.searchsorted(values, np.arange(1, count + 1), 'right')
    return ys, xs, starts, ends


def _moment_frame(signal):
    """Centre, sampling factor and elongated-ness of one glyph, or None.

    Everything comes from weighted second moments, so the numbers do not depend
    on the icon's rotation or its pixel size: the same 15 px client icon and
    60 px reference render land in the same canonical frame.
    """
    total = float(signal.sum())
    if total < 6.:
        return None
    ys, xs = np.mgrid[0:signal.shape[0], 0:signal.shape[1]]
    centre_x = float((signal * xs).sum() / total)
    centre_y = float((signal * ys).sum() / total)
    mxx = float((signal * (xs - centre_x) ** 2).sum() / total)
    myy = float((signal * (ys - centre_y) ** 2).sum() / total)
    mxy = float((signal * (xs - centre_x) * (ys - centre_y)).sum() / total)
    length = math.sqrt(12. * (mxx + myy))
    if length < 4.:
        return None
    values = np.maximum(np.linalg.eigvalsh(np.array([[mxx, mxy], [mxy, myy]])), 1e-9)
    return dict(centre=(centre_x, centre_y), factor=length / TARGET_LENGTH,
                aspect=float(math.sqrt(values[1] / values[0])),
                pixels=int((signal > .45).sum()))


def _resample(signal, centre, theta, factor, size=PATCH_SIZE):
    """Sample `signal` into a canonical `size` x `size` frame, rotated by theta."""
    ys, xs = np.mgrid[0:size, 0:size]
    u = xs - size / 2 + .5
    v = ys - size / 2 + .5
    cos, sin = math.cos(theta), math.sin(theta)
    sx = np.clip(centre[0] + (u * cos - v * sin) * factor, 0., signal.shape[1] - 1.001)
    sy = np.clip(centre[1] + (u * sin + v * cos) * factor, 0., signal.shape[0] - 1.001)
    x0 = np.floor(sx).astype(int)
    y0 = np.floor(sy).astype(int)
    fx, fy = sx - x0, sy - y0
    x1 = np.minimum(x0 + 1, signal.shape[1] - 1)
    y1 = np.minimum(y0 + 1, signal.shape[0] - 1)
    return (signal[y0, x0] * (1 - fx) * (1 - fy) + signal[y0, x1] * fx * (1 - fy)
            + signal[y1, x0] * (1 - fx) * fy + signal[y1, x1] * fx * fy)


def _normalised(image):
    """Zero-mean unit-length vector, or None for a flat image."""
    vector = image.ravel() - image.mean()
    norm = float(np.linalg.norm(vector))
    return None if norm < 1e-6 else vector / norm


def _raster(polygons, scale, size=None):
    """Scanline-fill the transcribed glyph polygons, bow up, centred in the canvas."""
    size = size or int(round(22 * scale))
    ys, xs = np.mgrid[0:size, 0:size]
    x = (xs + .5 - size / 2) / scale + 4.5
    y = (ys + .5 - size / 2) / scale + 8.
    mask = np.zeros((size, size), bool)
    for polygon in polygons:
        points = np.asarray(polygon, np.float64)
        inside = np.zeros((size, size), bool)
        for index in range(len(points)):
            x1, y1 = points[index]
            x2, y2 = points[(index + 1) % len(points)]
            crosses = (y1 > y) != (y2 > y)
            with np.errstate(divide='ignore', invalid='ignore'):
                crossing = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            inside ^= crosses & (x < crossing)
        mask |= inside
    return mask


@functools.lru_cache(maxsize=1)
def _templates():
    """(class, canonical unit vector) for each class, built from the SVG glyphs.

    A template goes through exactly the same signal and canonicalisation as a
    patch, so a client icon and a rasterised one are compared in one frame.
    """
    templates = []
    for label in CLASSES:
        glyph = _raster(ICON_POLYGONS[label], 2.)
        image = np.zeros((*glyph.shape, 3), np.uint8)
        image[:] = (6, 40, 60)                      # the tactical-map sea colour
        image[glyph] = (225, 55, 55)
        signal = icon_signal(image)
        frame = _moment_frame(signal)
        vector = _normalised(_resample(signal, frame['centre'], 0., frame['factor']))
        templates.append((label, vector))
    return tuple(templates)


def classify_icon(patch_rgb):
    """Classify one minimap ship icon patch -> (class, score).

    `patch_rgb` is a small uint8 RGB array (any size from 8x8 up) holding one
    contact: glyph plus some background. The glyph is reduced to a normalised
    signal, centred and scaled by its own second moments, then matched against
    the five silhouettes at 10 degree steps. Returns ``('cv'|'bb'|'ca'|'dd'|
    'ss', score)``, or ``('unknown', score of the best candidate)`` when the
    best match is weak, its lead over the runner-up is small, or the patch is
    not ship-shaped (a disc, a thin line, a HUD digit). Pure: no I/O, no state.
    """
    patch = np.asarray(patch_rgb)
    if patch.dtype != np.uint8 or patch.ndim != 3 or patch.shape[2] != 3:
        raise ValueError('expected a uint8 HxWx3 RGB patch')
    if min(patch.shape[:2]) < 8:
        raise ValueError('icon patch must be at least 8x8 px')
    signal = icon_signal(patch)
    frame = _moment_frame(signal)
    if frame is None:
        return 'unknown', 0.
    # Ship glyphs are 1.6-3.6 times longer than wide; a red disc (1.0), a HUD
    # digit (1.5) and a one-pixel line (4.9) fall outside that band.
    if frame['pixels'] < 14 or not MIN_ASPECT <= frame['aspect'] <= MAX_ASPECT:
        return 'unknown', 0.
    best = {}
    for degrees in range(0, 360, ANGLE_STEP):
        vector = _normalised(_resample(signal, frame['centre'],
                                       math.radians(degrees), frame['factor']))
        if vector is None:
            continue
        for label, template in _templates():
            score = float(template @ vector)
            if label not in best or score > best[label][0]:
                best[label] = (score, degrees)
    if not best:
        return 'unknown', 0.
    ranked = sorted(best.items(), key=lambda entry: -entry[1][0])
    score = ranked[0][1][0]
    margin = score - ranked[1][1][0]
    if score < MIN_SCORE or margin < MIN_MARGIN:
        return 'unknown', max(0., score)
    return ranked[0][0], score


# -------------------------------------------------------------------- detection

def _components_with_boxes(mask, minimum):
    labels, count = label_components(mask)
    boxes = []
    if not count:
        return boxes
    areas = np.bincount(labels.ravel())[1:]
    ys, xs, starts, ends = _sorted_parts(labels, count)
    for index in range(1, count + 1):
        area = int(areas[index - 1])
        if area < minimum:
            continue
        piece_x = xs[starts[index - 1]:ends[index - 1]]
        piece_y = ys[starts[index - 1]:ends[index - 1]]
        boxes.append(dict(area=area, x0=int(piece_x.min()), y0=int(piece_y.min()),
                          x1=int(piece_x.max()), y1=int(piece_y.max()),
                          cx=float(piece_x.mean()), cy=float(piece_y.mean())))
    return boxes


def _cluster(boxes, gap):
    clusters = []
    for box in sorted(boxes, key=lambda entry: entry['area'], reverse=True):
        for cluster in clusters:
            if (box['x0'] - gap <= cluster['x1'] and cluster['x0'] - gap <= box['x1']
                    and box['y0'] - gap <= cluster['y1'] and cluster['y0'] - gap <= box['y1']):
                cluster['x0'] = min(cluster['x0'], box['x0'])
                cluster['y0'] = min(cluster['y0'], box['y0'])
                cluster['x1'] = max(cluster['x1'], box['x1'])
                cluster['y1'] = max(cluster['y1'], box['y1'])
                cluster['area'] += box['area']
                cluster['parts'] += 1
                cluster['sizes'].append(box['area'])
                cluster['cx'] = (cluster['cx'] * (cluster['parts'] - 1) + box['cx']) / cluster['parts']
                cluster['cy'] = (cluster['cy'] * (cluster['parts'] - 1) + box['cy']) / cluster['parts']
                break
        else:
            clusters.append(dict(box, parts=1, sizes=[box['area']]))
    return clusters


def _own_clusters(rgb, map_rect):
    """Neutral, bright, compact clusters that could be the own-ship arrow."""
    x0, y0, x1, y1 = map_rect
    region = rgb[y0:y1, x0:x1]
    spread = region.max(axis=2).astype(np.int16) - region.min(axis=2).astype(np.int16)
    candidate = (region.min(axis=2) >= 165) & (spread <= 45)
    clusters = []
    for cluster in _cluster(_components_with_boxes(candidate, 20), 6):
        width = cluster['x1'] - cluster['x0'] + 1
        height = cluster['y1'] - cluster['y0'] + 1
        if not (12 <= width <= 60 and 12 <= height <= 60 and 120 <= cluster['area'] <= 900):
            continue
        blades = cluster['sizes']
        if len(blades) >= 3:
            # The arrow is one stiff shape (measured 1-2 parts on every saved
            # frame). Three or more parts is a capture-point rosette (three
            # equal blades) or a run of HUD glyphs merged by the 6 px cluster
            # gap (measured 4 parts on lm2's banner), never the own marker.
            continue
        clusters.append(cluster)
    return clusters, x0, y0


def _cluster_record(cluster, x0, y0):
    return dict(pixel=(int(round(cluster['cx'])) + x0, int(round(cluster['cy'])) + y0),
                bbox=[cluster['x0'] + x0, cluster['y0'] + y0,
                      cluster['x1'] + x0 + 1, cluster['y1'] + y0 + 1],
                area=cluster['area'], parts=cluster['parts'])


def _fill(cluster):
    width = cluster['x1'] - cluster['x0'] + 1
    height = cluster['y1'] - cluster['y0'] + 1
    return cluster['area'] / float(max(1, width * height))


def find_own_marker(rgb, map_rect):
    """The white own-ship marker: one neutral, bright cluster of 120-900 px.

    The marker's heading changes its bounding box (measured 17x36, 20x15 and
    30x24 on the three saved frames), so size and shape alone cannot prove it.
    What can: it is neutral (the arrow is white, the sea and the enemy tint are
    not) and it is a single compact cluster. The capture-point rosette is also
    neutral and bright but is drawn as three equal blades, which is excluded.
    """
    clusters, x0, y0 = _own_clusters(rgb, map_rect)
    if not clusters:
        return None, 'expected one own-ship marker, found 0'
    if len(clusters) > 1:
        # A white HUD banner sitting over the map is also neutral and bright,
        # so it shows up as a second candidate (measured 2026-10-23 on
        # state/cn360-flow-20261021/lm2). The own arrow is a filled triangle,
        # so rank by fill and keep the filled one.
        clusters.sort(key=_fill, reverse=True)
        if _fill(clusters[0]) < 0.15:
            return None, (f'expected one own-ship marker, found {len(clusters)} '
                          'and none is filled')
    return _cluster_record(clusters[0], x0, y0), None


def _pick_own(rgb, map_rect, hint=None):
    """The own marker, preferring the candidate nearest `hint`.

    `hint` is the own position measured on an earlier frame of the same map.
    Between two captures the arrow moves only a few pixels, while a HUD banner
    can appear anywhere, so the nearest candidate is the trustworthy one.
    """
    if hint is None:
        return find_own_marker(rgb, map_rect)
    clusters, x0, y0 = _own_clusters(rgb, map_rect)
    if not clusters:
        return None, 'expected one own-ship marker, found 0'
    nearest = min(clusters, key=lambda cluster: math.hypot(
        cluster['cx'] + x0 - hint[0], cluster['cy'] + y0 - hint[1]))
    distance = math.hypot(nearest['cx'] + x0 - hint[0], nearest['cy'] + y0 - hint[1])
    if distance > OWN_HINT_RADIUS:
        return None, (f'own marker is {distance:.1f} px from the expected point '
                      f'(limit {OWN_HINT_RADIUS})')
    return _cluster_record(nearest, x0, y0), None


def find_enemy_icons(rgb, map_rect, patch_half=PATCH_HALF):
    """Every red contact on the map, classified. Parts of one icon are merged."""
    x0, y0, x1, y1 = map_rect
    region = rgb[y0:y1, x0:x1].astype(np.int16)
    red = ((region[:, :, 0] > 110) & (region[:, :, 0] - region[:, :, 1] > 40)
           & (region[:, :, 0] - region[:, :, 2] > 40))
    found = []
    for cluster in _cluster(_components_with_boxes(red, 3), CLUSTER_GAP_PX):
        width = cluster['x1'] - cluster['x0'] + 1
        height = cluster['y1'] - cluster['y0'] + 1
        if not (5 <= width <= 40 and 5 <= height <= 40):
            continue
        cx = int(round(cluster['cx'])) + x0
        cy = int(round(cluster['cy'])) + y0
        px0, py0 = max(0, cx - patch_half), max(0, cy - patch_half)
        px1, py1 = min(rgb.shape[1], cx + patch_half), min(rgb.shape[0], cy + patch_half)
        label, score = classify_icon(rgb[py0:py1, px0:px1])
        found.append(dict(pixel=(cx, cy), bbox=[cluster['x0'] + x0, cluster['y0'] + y0,
                                                cluster['x1'] + x0 + 1, cluster['y1'] + y0 + 1],
                          parts=cluster['parts'], area=cluster['area'],
                          ship_class=label, score=round(float(score), 3)))
    return sorted(found, key=lambda entry: entry['score'], reverse=True)


def sea_flat(rgb, point, half=7, max_std=4.):
    """True when the click lands on empty sea rather than on a marker or icon.

    The tactical map's sea is a smooth gradient (measured std <= 0.6 over 15x15
    on the saved frames), while a nameplate, a selection ring or a coast line
    raises the local standard deviation above 5.
    """
    x, y = int(round(point[0])), int(round(point[1]))
    patch = rgb[max(0, y - half):y + half, max(0, x - half):x + half]
    if patch.size == 0:
        return False
    return bool(patch.astype(np.float64).std(axis=(0, 1)).max() <= max_std)


# ------------------------------------------------------------------ course point

def course_point(own_px, enemy_px, map_rect, fraction=.95, min_gap_px=MIN_GAP_PX,
                 is_sea=None):
    """Window point on the own->enemy line that sets an autopilot waypoint.

    `fraction` is how far along the own->enemy line the point sits; `min_gap_px`
    additionally keeps it away from the enemy icon so the click does not select
    the ship instead of steering. `is_sea`, when given, receives candidate
    points and pulls the result further back along the line until one is
    accepted, so the waypoint never lands on a nameplate or a marker. The result
    is clamped inside `map_rect`.
    """
    if not 0. < fraction <= 1.:
        raise ValueError('fraction must be in (0, 1]')
    if min_gap_px < 0:
        raise ValueError('min_gap_px must not be negative')
    x0, y0, x1, y1 = map_rect
    own = (float(own_px[0]), float(own_px[1]))
    enemy = (float(enemy_px[0]), float(enemy_px[1]))
    for name, point in (('own', own), ('enemy', enemy)):
        if not (x0 <= point[0] <= x1 and y0 <= point[1] <= y1):
            raise ValueError(f'{name} position {point} is outside map rect {map_rect}')
    dx, dy = enemy[0] - own[0], enemy[1] - own[1]
    distance = math.hypot(dx, dy)
    if distance < 1e-6:
        raise ValueError('own and enemy positions coincide')
    gap = min(max((1. - fraction) * distance, min_gap_px), .5 * distance)
    travel = (distance - gap) / distance
    while True:
        x = own[0] + dx * travel
        y = own[1] + dy * travel
        if is_sea is None or travel <= .2 or is_sea((x, y)):
            break
        travel -= 2. / distance
    margin = max(2., .02 * min(x1 - x0, y1 - y0))
    return (int(round(min(max(x, x0 + margin), x1 - margin))),
            int(round(min(max(y, y0 + margin), y1 - margin))))


# ------------------------------------------------------------- course direction

def _route_candidates(rgb, map_rect):
    """Dash dots and the hollow waypoint diamond of an autopilot route.

    The route is drawn in white, independent of team tint, so it separates from
    the blue sea by being neutral (`max - min` small) and bright. Dashes are a
    few pixels across; the waypoint diamond is roughly 10-24 px across and
    40-200 px of outline. Text glyphs are excluded by the diamond's size band
    and by the trace, not by shape alone.
    """
    x0, y0, x1, y1 = map_rect
    region = rgb[y0:y1, x0:x1].astype(np.int16)
    spread = region.max(axis=2) - region.min(axis=2)
    mask = ((region.min(axis=2) >= NEUTRAL_MIN_CHANNEL)
            & (spread <= NEUTRAL_MAX_SPREAD))
    dashes, markers = [], []
    for box in _components_with_boxes(mask, DASH_MIN_AREA):
        width = box['x1'] - box['x0'] + 1
        height = box['y1'] - box['y0'] + 1
        centroid = (box['cx'] + x0, box['cy'] + y0)
        if DASH_MIN_AREA <= box['area'] <= DASH_MAX_AREA and max(width, height) <= DASH_MAX_EXTENT:
            dashes.append(centroid)
        elif (MARKER_MIN_AREA <= box['area'] <= MARKER_MAX_AREA
              and MARKER_MIN_EXTENT <= max(width, height) <= MARKER_MAX_EXTENT
              and min(width, height) >= MARKER_MIN_SIDE):
            markers.append(dict(pixel=centroid, area=box['area'],
                                size=(width, height)))
    return dashes, markers


def _trace_route(own, dashes, markers):
    """Follow the dash chain from the own arrow to the waypoint diamond.

    Starts with a long first step because the arrow body hides the first
    dashes, then follows the nearest unused dash inside a shallow cone. The
    route bends near the ship, so the cone is wide; the diamond ends it.
    Returns ``(path, waypoint or None)``.
    """
    origin = (float(own[0]), float(own[1]))
    points = [tuple(point) for point in dashes if math.dist(point, origin) >= 12.]
    marker_points = [tuple(entry['pixel']) for entry in markers]
    used = set()
    path = []
    current = np.asarray(origin, dtype=np.float64)
    heading = None
    limit = CHAIN_FIRST_STEP
    while True:
        best = None
        for index, point in enumerate(points):
            if index in used:
                continue
            vector = np.asarray(point, dtype=np.float64) - current
            distance = float(np.linalg.norm(vector))
            if distance < 3. or distance > limit:
                continue
            if heading is not None:
                cosine = float(vector @ heading) / max(
                    distance * float(np.linalg.norm(heading)), 1e-9)
                if cosine < math.cos(math.radians(CHAIN_MAX_TURN_DEG)):
                    continue
            if best is None or distance < best[0]:
                best = (distance, index, vector)
        if best is None:
            return path, None
        _, index, vector = best
        used.add(index)
        path.append(points[index])
        heading = vector
        current = np.asarray(points[index], dtype=np.float64)
        limit = CHAIN_STEP
        near = [point for point in marker_points
                if math.dist(point, points[index]) <= CHAIN_MARKER_RADIUS]
        if near:
            return path, min(near, key=lambda point: math.dist(point, points[index]))


def measure_course(rgb, map_rect, own_pixel=None, enemy_pixel=None, own_hint=None):
    """Measure the drawn autopilot course and compare it with own->enemy.

    ``rgb`` is a frame captured after the waypoint click. ``own_pixel`` and
    ``enemy_pixel`` may be supplied in frame coordinates (the caller already
    knows them from the click frame); otherwise they are detected. ``own_hint``
    is the expected own position, used to reject an unrelated white HUD banner
    that would otherwise win the own-marker vote.

    Returns a dict with ``angle_deg`` (smallest angular difference, 0..180),
    ``course_deg`` (drawing, screen coordinates, 0 = +x, growing clockwise),
    ``expected_deg`` (own->enemy), the ``waypoint`` and the dash count, or
    ``None`` when no route can be measured. Pure: no I/O.
    """
    if own_pixel is None:
        own, _ = _pick_own(rgb, map_rect, own_hint)
    else:
        own = dict(pixel=(int(round(own_pixel[0])), int(round(own_pixel[1]))))
    if own is None:
        return None
    if enemy_pixel is None:
        accepted = [entry for entry in find_enemy_icons(rgb, map_rect)
                    if entry['ship_class'] != 'unknown']
        if not accepted:
            return None
        enemy = min(accepted, key=lambda entry: math.hypot(
            entry['pixel'][0] - own['pixel'][0],
            entry['pixel'][1] - own['pixel'][1]))['pixel']
    else:
        enemy = (float(enemy_pixel[0]), float(enemy_pixel[1]))
    origin = (float(own['pixel'][0]), float(own['pixel'][1]))
    if math.dist(origin, enemy) < 1e-6:
        return None
    dashes, markers = _route_candidates(rgb, map_rect)
    path, waypoint = _trace_route(origin, dashes, markers)
    if waypoint is None or len(path) < CHAIN_MIN_DASHES:
        return None
    if math.dist(waypoint, origin) < COURSE_MIN_NET_PX:
        return None
    course = math.degrees(math.atan2(waypoint[1] - origin[1],
                                     waypoint[0] - origin[0])) % 360.
    expected = math.degrees(math.atan2(enemy[1] - origin[1],
                                       enemy[0] - origin[0])) % 360.
    angle = abs((course - expected + 180.) % 360. - 180.)
    return dict(angle_deg=round(angle, 3), course_deg=round(course, 3),
                expected_deg=round(expected, 3),
                own=origin, waypoint=(round(waypoint[0], 3), round(waypoint[1], 3)),
                dashes=len(path))


def course_direction(rgb, map_rect, own_pixel=None, enemy_pixel=None, own_hint=None):
    """Angle in degrees between the drawn course and own->enemy, or None.

    The route is read from a *post-click* frame: a dashed white line from the
    own-ship arrow to a small hollow waypoint diamond. Returns None when that
    line cannot be measured, so a caller can refuse to correct blindly.
    """
    measured = measure_course(rgb, map_rect, own_pixel=own_pixel,
                              enemy_pixel=enemy_pixel, own_hint=own_hint)
    return None if measured is None else measured['angle_deg']


def angle_within(angle_deg, tolerance=COURSE_TOLERANCE_DEG):
    """True only for a measured angle inside the tolerance."""
    return angle_deg is not None and angle_deg <= tolerance


def rotate_about(own, point, degrees):
    """Rotate `point` around `own` in screen coordinates (y grows downwards)."""
    radians = math.radians(degrees)
    cosine, sine = math.cos(radians), math.sin(radians)
    dx, dy = point[0] - own[0], point[1] - own[1]
    return (own[0] + dx * cosine - dy * sine,
            own[1] + dx * sine + dy * cosine)


def corrected_click_point(measurement, click_frame, map_rect):
    """The click that would put the route on the own->enemy bearing.

    Rotates the original waypoint around the measured own-ship position by the
    signed error (expected - drawn), then clamps it inside the map.
    """
    delta = (measurement['expected_deg'] - measurement['course_deg'] + 180.) % 360. - 180.
    point = rotate_about(measurement['own'], click_frame, delta)
    x0, y0, x1, y1 = map_rect
    margin = max(2., .02 * min(x1 - x0, y1 - y0))
    return (int(round(min(max(point[0], x0 + margin), x1 - margin))),
            int(round(min(max(point[1], y0 + margin), y1 - margin))))


def _pull_back_to_sea(own, point, map_rect, rgb):
    """Walk from `point` back towards own until the point sits on flat sea."""
    x0, y0, x1, y1 = map_rect
    margin = max(2., .02 * min(x1 - x0, y1 - y0))
    for fraction in np.linspace(1., .2, 41):
        candidate = (own[0] + (point[0] - own[0]) * fraction,
                     own[1] + (point[1] - own[1]) * fraction)
        if not (x0 + margin < candidate[0] < x1 - margin
                and y0 + margin < candidate[1] < y1 - margin):
            continue
        if sea_flat(rgb, candidate):
            return (int(round(candidate[0])), int(round(candidate[1])))
    return None


def verify_and_correct(plan, capture, click, iterations=1,
                       tolerance=COURSE_TOLERANCE_DEG, max_age_s=None):
    """Close the loop on the drawn course, sending at most one correction.

    `plan` carries the original click in frame coordinates (`click_frame`).
    `capture()` returns a fresh frame plus what the caller measured about it:
    ``rgb``, ``map_rect``, ``own_hint``, ``enemy_hint``, ``map_verified`` (own
    marker and a classified enemy icon are present), ``age_s`` and ``path``.
    `click(frame_point)` sends one waypoint click and returns a dict with
    ``reached``. A click is only sent when the fresh frame proves an open map,
    is fresh enough, and the corrected point is on flat sea -- the same guards
    as the first click are never relaxed.

    Returns the report block ``{state, angle_deg, corrected, clicks, ...}``.
    ``state`` is one of: skipped, unverifiable (no route measurable -> no
    blind second click), refused (fresh frame failed a guard), corrected (one
    correction sent), ok (final angle inside tolerance), failed (still wrong
    after the one allowed correction).
    """
    iterations = max(0, min(int(iterations), MAX_VERIFY_ITERATIONS))
    verification = dict(state='skipped', angle_deg=None, corrected=False, clicks=1,
                        tolerance_deg=tolerance, iterations=iterations,
                        measurements=[], reason=None)
    for _ in range(iterations):
        frame = capture() or {}
        rgb = frame.get('rgb')
        if rgb is None:
            verification.update(state='unverifiable',
                                reason='verification frame could not be captured')
            break
        map_rect = tuple(frame['map_rect'])
        measurement = measure_course(rgb, map_rect, enemy_pixel=frame.get('enemy_hint'),
                                     own_hint=frame.get('own_hint'))
        if measurement is None:
            verification.update(state='unverifiable',
                                reason='route direction could not be measured in the new frame')
            break
        verification['angle_deg'] = measurement['angle_deg']
        verification['measurements'].append(dict(
            angle_deg=measurement['angle_deg'], course_deg=measurement['course_deg'],
            expected_deg=measurement['expected_deg'],
            waypoint=list(measurement['waypoint']), dashes=measurement['dashes'],
            frame=frame.get('path')))
        if angle_within(measurement['angle_deg'], tolerance):
            verification.update(state='ok', reason=None)
            break
        if verification['clicks'] >= MAX_CLICKS:
            verification.update(state='failed',
                                reason='angle is still above tolerance after the one allowed correction')
            break
        if max_age_s is not None and frame.get('age_s', 0.) > max_age_s:
            verification.update(state='refused',
                                reason='verification frame is too old to click')
            break
        if not frame.get('map_verified', True):
            verification.update(state='refused',
                                reason='verification frame does not prove an open tactical map')
            break
        click_frame = frame.get('click_hint', plan['click_frame'])
        point = corrected_click_point(measurement, click_frame, map_rect)
        if not sea_flat(rgb, point):
            point = _pull_back_to_sea(measurement['own'], point, map_rect, rgb)
        if point is None:
            verification.update(state='refused',
                                reason='corrected point is not on flat sea')
            break
        result = click(point) or {}
        verification['clicks'] += 1
        verification['corrected'] = True
        verification['corrected_point'] = list(point)
        verification['correction_click'] = result
        if not result.get('reached'):
            verification.update(state='failed',
                                reason='corrective click did not reach the point')
            break
        verification['state'] = 'corrected'
    return verification


# -------------------------------------------------------------------------- cli

def _window_for_size(display, size):
    """The single visible window of the given size, for the execute path."""
    out = subprocess.run(['xdotool', 'search', '--onlyvisible', '--name', '.'],
                         capture_output=True, text=True,
                         env={'DISPLAY': display, 'PATH': '/usr/bin:/bin'}).stdout
    matches = []
    for window in out.split():
        geometry = subprocess.run(['xdotool', 'getwindowgeometry', '--shell', window],
                                  capture_output=True, text=True,
                                  env={'DISPLAY': display, 'PATH': '/usr/bin:/bin'}).stdout
        fields = dict(line.split('=', 1) for line in geometry.split() if '=' in line)
        try:
            found = (int(fields['WIDTH']), int(fields['HEIGHT']))
        except (KeyError, ValueError):
            continue
        if found == tuple(size):
            matches.append(window)
    if len(matches) != 1:
        raise SystemExit(f'expected one visible window of size {tuple(size)}, '
                         f'found {len(matches)}; pass --window')
    return matches[0]


def frame_to_window(point, frame_rect, window_size):
    """Map a frame pixel to game-window content pixels.

    `frame_rect` is the part of the frame that holds the game content, so a
    window capture (frame_rect == the frame) and a scaled desktop capture both
    work. The click is always placed in window coordinates.
    """
    x0, y0, x1, y1 = frame_rect
    return (int(round(window_size[0] * (point[0] - x0) / (x1 - x0))),
            int(round(window_size[1] * (point[1] - y0) / (y1 - y0))))


def window_to_frame(point, frame_rect, window_size):
    x0, y0, x1, y1 = frame_rect
    return (x0 + point[0] * (x1 - x0) / window_size[0],
            y0 + point[1] * (y1 - y0) / window_size[1])


def content_rect(rgb):
    """The non-black part of a capture, for desktop captures with letterboxing.

    A window capture fills its frame (`frame_rect` == the frame), while the
    resident capture host snapshots the whole composited output: measured
    2026-10-23 on state/cn360-flow-20261021/course3/after_click.png the game
    content is 0,96..2560,1504 inside a 2560x1600 frame, exactly the black bars
    that `max(channel) >= 10` removes. Anything already full-frame stays full.
    """
    lit = rgb.max(axis=2) >= 10
    rows = np.flatnonzero(lit.any(axis=1))
    columns = np.flatnonzero(lit.any(axis=0))
    if rows.size == 0 or columns.size == 0:
        return (0, 0, rgb.shape[1], rgb.shape[0])
    rect = (int(columns[0]), int(rows[0]),
            int(columns[-1]) + 1, int(rows[-1]) + 1)
    if rect[2] - rect[0] < 64 or rect[3] - rect[1] < 64:
        return (0, 0, rgb.shape[1], rgb.shape[0])
    return rect


def _frame_hint(point, from_rect, to_rect, window_size):
    """Carry a frame point from the click frame into a newly captured frame."""
    return window_to_frame(frame_to_window(point, from_rect, window_size),
                           to_rect, window_size)


def _map_rect_in(map_rect, from_rect, to_rect, window_size):
    """Carry the tactical-map rectangle into a newly captured frame."""
    top_left = _frame_hint(map_rect[:2], from_rect, to_rect, window_size)
    bottom_right = _frame_hint(map_rect[2:], from_rect, to_rect, window_size)
    return (int(round(top_left[0])), int(round(top_left[1])),
            int(round(bottom_right[0])), int(round(bottom_right[1])))


def snapshot_capture(args, out, own_pixel, enemy_pixel, click_frame, frame_rect, map_rect):
    """A `capture()` for verify_and_correct: one fresh capture-host PNG.

    Returns a closure and a dict that remembers the last frame's geometry, so
    the corrective click can map its frame point back to window coordinates.
    Capture failures return None and the caller reports `unverifiable`.
    """
    latest = {'frame_rect': frame_rect, 'map_rect': map_rect, 'path': None}

    def capture():
        path = out / f'verify-{time.time_ns()}.png'
        started = time.time()
        try:
            result = subprocess.run(
                [sys.executable, str(HERE / 'capture_host.py'), 'snapshot', str(path)],
                capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode != 0 or not path.is_file() or path.stat().st_size == 0:
            return None
        try:
            rgb = read_png(path)
        except SystemExit:
            return None
        rect = content_rect(rgb)
        new_map = _map_rect_in(map_rect, frame_rect, rect, args.window_size)
        own_hint = _frame_hint(own_pixel, frame_rect, rect, args.window_size)
        enemy_hint = _frame_hint(enemy_pixel, frame_rect, rect, args.window_size)
        click_hint = _frame_hint(click_frame, frame_rect, rect, args.window_size)
        own_here, _ = _pick_own(rgb, new_map, own_hint)
        accepted = [entry for entry in find_enemy_icons(rgb, new_map)
                    if entry['ship_class'] != 'unknown']
        latest.update(frame_rect=rect, map_rect=new_map, path=str(path))
        return dict(rgb=rgb, map_rect=new_map, own_hint=own_hint,
                    enemy_hint=enemy_hint, click_hint=click_hint,
                    map_verified=bool(own_here and accepted),
                    age_s=time.time() - started, path=str(path))

    return capture, latest


def build_report(args, frame_size, frame_sha, frame_rect, map_rect, own, reason,
                 enemies, target, click_frame, click_point, evidence, execute_result,
                 expected_direction_deg=None, verification=None):
    accepted = [entry for entry in enemies if entry['ship_class'] != 'unknown']
    return dict(tool='minimap_enemy_click', execute=args.execute, status='failed',
                failure_reason=reason, frame=str(args.frame),
                frame_source=dict(path=str(args.frame), size=list(frame_size),
                                  sha256=frame_sha,
                                  age_s=round(time.time() - args.frame.stat().st_mtime, 3),
                                  content_rect=list(frame_rect)),
                window=dict(size=list(args.window_size), origin=list(args.origin),
                            display=args.display, id=args.window),
                map=dict(rect=list(map_rect),
                         source='argument' if args.map_rect else 'default/measured',
                         evidence=evidence),
                own=own,
                enemies=dict(candidates=len(enemies), accepted=len(accepted),
                             icons=enemies),
                chosen_target=target,
                click=dict(window=click_point, frame=click_frame),
                expected_direction_deg=expected_direction_deg,
                verification=verification or dict(state='skipped', angle_deg=None,
                                                  corrected=False, clicks=0),
                fraction=args.fraction, km_per_px=args.km_per_px,
                execute_result=execute_result,
                semantics='sets one autopilot waypoint on the tactical map; '
                          'no claim about route viability, and no battle input')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--frame', type=Path, required=True,
                        help='captured tactical-map frame')
    parser.add_argument('--map-rect', nargs=4, type=int, metavar=('X0', 'Y0', 'X1', 'Y1'),
                        help='tactical-map area inside the frame')
    parser.add_argument('--frame-rect', nargs=4, type=int, metavar=('X0', 'Y0', 'X1', 'Y1'),
                        help='game-content area inside the frame, for scaled captures')
    parser.add_argument('--out', type=Path, default=ROOT / 'state' / 'minimap-enemy-click',
                        help='directory for report.json')
    parser.add_argument('--execute', action='store_true',
                        help='send the click (default: dry run)')
    parser.add_argument('--fraction', type=float, default=.95)
    parser.add_argument('--km-per-px', type=float, default=KM_PER_PX,
                        help='tactical-map scale; zooming the map changes it')
    parser.add_argument('--display', default=':2')
    parser.add_argument('--window', help='X window id of the game; found by size if omitted')
    parser.add_argument('--origin', nargs=2, type=int, default=[0, 30],
                        help='root offset of the window content')
    parser.add_argument('--window-size', nargs=2, type=int, default=list(WINDOW_SIZE))
    parser.add_argument('--hold-ms', type=int, default=110)
    parser.add_argument('--max-frame-age', type=float, default=3.,
                        help='refuse --execute when the frame is older than this')
    parser.add_argument('--verify-iterations', type=int, default=1,
                        help='post-click captures used to measure the drawn course; '
                             'an angle above 15 degrees triggers one corrective click '
                             '(0 disables verification; at most 2 clicks in total)')
    args = parser.parse_args(argv)
    if args.verify_iterations < 0:
        parser.error('--verify-iterations must not be negative')

    rgb = read_png(args.frame)
    frame_size = (rgb.shape[1], rgb.shape[0])
    frame_rect = tuple(args.frame_rect) if args.frame_rect else (0, 0, *frame_size)
    if args.map_rect:
        map_rect = tuple(args.map_rect)
    else:
        top_left = window_to_frame(DEFAULT_MAP_RECT[:2], frame_rect, args.window_size)
        bottom_right = window_to_frame(DEFAULT_MAP_RECT[2:], frame_rect, args.window_size)
        map_rect = (int(round(top_left[0])), int(round(top_left[1])),
                    int(round(bottom_right[0])), int(round(bottom_right[1])))
    if not (0 <= map_rect[0] < map_rect[2] <= frame_size[0]
            and 0 <= map_rect[1] < map_rect[3] <= frame_size[1]):
        raise SystemExit(f'map rect {map_rect} does not fit frame {frame_size}')

    own, own_reason = find_own_marker(rgb, map_rect)
    enemies = find_enemy_icons(rgb, map_rect)
    accepted = [entry for entry in enemies if entry['ship_class'] != 'unknown']

    click_point, click_frame, failure = None, None, None
    target = None
    expected_direction = None
    if own is None:
        failure = f'own marker not found: {own_reason}'
    elif not accepted:
        failure = (f'no accepted enemy icon (candidates: {len(enemies)}); '
                   'refusing to guess a course')
    else:
        target = min(accepted, key=lambda entry: math.hypot(
            entry['pixel'][0] - own['pixel'][0], entry['pixel'][1] - own['pixel'][1]))
        target = dict(target, distance_px=round(math.hypot(
            target['pixel'][0] - own['pixel'][0],
            target['pixel'][1] - own['pixel'][1]), 1))
        target['distance_km'] = round(target['distance_px'] * args.km_per_px, 2)
        expected_direction = round(math.degrees(math.atan2(
            target['pixel'][1] - own['pixel'][1],
            target['pixel'][0] - own['pixel'][0])) % 360., 3)
        click_frame = course_point(own['pixel'], target['pixel'], map_rect, args.fraction,
                                   is_sea=lambda point: sea_flat(rgb, point))
        click_point = frame_to_window(click_frame, frame_rect, args.window_size)

    evidence = dict(own_marker=bool(own), accepted_icons=len(accepted),
                    click_on_flat_sea=bool(click_frame and sea_flat(rgb, click_frame)),
                    sources='white own-ship marker, red ship icon, flat sea at the click')
    map_verified = bool(own and accepted and evidence['click_on_flat_sea'])

    execute_result = None
    verification = dict(state='skipped', angle_deg=None, corrected=False, clicks=0,
                        tolerance_deg=COURSE_TOLERANCE_DEG,
                        iterations=args.verify_iterations, measurements=[], reason=None)
    status = 'dry-run' if click_point else 'failed'
    args.out.mkdir(parents=True, exist_ok=True)
    if args.execute and click_point:
        age = time.time() - args.frame.stat().st_mtime
        if not map_verified:
            failure = ('frame does not prove an open tactical map and an ocean click '
                       'point; refusing to click where it might be a battle input')
        elif age > args.max_frame_age:
            failure = (f'frame is {age:.1f}s old (limit {args.max_frame_age}s); '
                       'capture again before clicking')
        else:
            import click_game_point
            window = args.window or _window_for_size(args.display, args.window_size)
            reached, pointer = click_game_point.click(
                args.display, window, click_point, tuple(args.origin), args.hold_ms)
            execute_result = dict(window=window, reached=bool(reached),
                                  pointer=list(pointer), hold_ms=args.hold_ms)
            status = 'clicked' if reached else 'failed'
            verification['clicks'] = 1 if reached else 0
            if not reached:
                failure = 'pointer did not reach the target point; click not trustworthy'
            elif args.verify_iterations > 0:
                capture, latest = snapshot_capture(
                    args, args.out, own['pixel'], target['pixel'], click_frame,
                    frame_rect, map_rect)

                def click_frame_point(point):
                    window_point = frame_to_window(point, latest['frame_rect'],
                                                   args.window_size)
                    reached_again, pointer_again = click_game_point.click(
                        args.display, window, window_point, tuple(args.origin),
                        args.hold_ms)
                    return dict(window=list(window_point), reached=bool(reached_again),
                                pointer=list(pointer_again))

                verification = verify_and_correct(
                    dict(own=own['pixel'], enemy=target['pixel'],
                         click_frame=click_frame),
                    capture, click_frame_point, iterations=args.verify_iterations,
                    max_age_s=args.max_frame_age)

    frame_sha = hashlib.sha256(args.frame.read_bytes()).hexdigest()
    report = build_report(args, frame_size, frame_sha, frame_rect, map_rect, own, failure,
                          enemies, target, click_frame, click_point, evidence,
                          execute_result, expected_direction_deg=expected_direction,
                          verification=verification)
    report['status'] = status
    if failure:
        report['status'] = 'failed'
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')

    print(f'frame {args.frame} {frame_size} sha256 {frame_sha[:16]}')
    print(f'map rect {map_rect}')
    print(f'own marker {own["pixel"] if own else "not found"}')
    for entry in enemies:
        print(f'  enemy {entry["pixel"]} bbox={entry["bbox"]} parts={entry["parts"]} '
              f'class={entry["ship_class"]} score={entry["score"]}')
    if target:
        print(f'target {target["pixel"]} {target["ship_class"]} '
              f'{target["distance_km"]} km ({target["distance_px"]} px)')
        print(f'expected direction {expected_direction:.1f} deg '
              '(own marker -> target on the map)')
    if click_point and execute_result:
        print(f'clicked window {click_point} (frame {click_frame}) {execute_result}')
    elif click_point:
        verb = 'refused' if args.execute else 'dry run'
        print(f'would click window {click_point} (frame {click_frame}) [{verb}]')
    if verification['state'] != 'skipped' or verification.get('measurements'):
        angle = verification['angle_deg']
        angle_text = 'n/a' if angle is None else f'{angle:.1f} deg'
        print(f'course verification {verification["state"]} angle={angle_text} '
              f'corrected={verification["corrected"]} clicks={verification["clicks"]}')
        if verification.get('reason'):
            print(f'  verification note: {verification["reason"]}')
    if failure:
        print(f'failed: {failure}')
    print(f'report {args.out / "report.json"}')
    return 0 if report['status'] != 'failed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
