#!/usr/bin/env python3
"""Project reference-space HUD rectangles onto the frame that is live now.

Every HUD constant in this project was authored in one *reference* space: a
2560x1440 game surface inside a 2560x1600 captured canvas (80 px bars). That
space is a definition, not a pixel layout. The client on this machine renders
1924x1058 and the host stretches that surface into the canvas, so the same HUD
element lands elsewhere (measured on ``state/hud-now-20261019.png``: the
minimap panel is 2 px further right, 14 px higher and 1 % narrower than the
reference panel, and the three-mount turret bar row sits at y=1349 with a
144 px stride while the reference four-mount row is at y=1365 with 96 px).

The mapping itself is linear and per axis: ``x' = sx*x + ox`` and
``y' = sy*y + oy``. ``sx``/``sy`` are the game surface's width and height over
REFERENCE (2560x1440) -- independently, so a 16:9 surface stretched into a 16:10
canvas gets two scales -- and the offsets remove the reference letterbox and add
this frame's own content origin. ``projection_for`` builds that map from the
frame shape alone, ``content_bounds`` recovers the surface from the frame's
black bars, and ``from_ref``/``to_ref`` convert rectangles in both directions.
On the reference capture the map is the identity, which is why every constant
below can stay in reference canvas coordinates. A rectangle that leaves the
frame is clamped and recorded in ``report()['clamped']``; it is never dropped
silently.

On top of that geometric map, a projection can be *measured*, which is what the
live client needs:

* the minimap panel gives both horizontal edges (left and right) and its top;
* the bottom turret/ammo bar row gives its baseline, its first slot and its
  slot pitch.

Those measurements fit the same affine per axis. Two anchors, two axes: no
aspect-ratio assumption, no uniform-scale assumption. A missing anchor raises
``MappingError``, and a point or rectangle that leaves the frame raises too, so
a caller can never silently read the wrong pixels. The strictly-raised callers
are ``project_point``/``project_rect``; ``from_ref``/``clip_rect`` clamp
instead, and every clamp they make is recorded in ``report()['clamped']``, so
truncation is visible rather than silent.

The minimap panel is used for the horizontal axis because its left and right
edges are design-stable (``720`` px apart in reference space, 714 measured).
The bar row's pitch is *reported and cross-checked*, but it is a per-client
layout (4x96 reference, 3x144 measured here) and never overrides a directly
measured panel width; when the two disagree the projection says so in
``warnings``. The bar baseline anchors the vertical axis because every bottom
HUD readout hangs off that row.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

REFERENCE_FRAME = (2560, 1600)
REFERENCE_CONTENT = (2560, 1440)
REFERENCE_BAR_Y = 80
# The reference space itself, named the way callers quote it: the game surface,
# *without* the letterbox. Reference rectangles are written in this surface's
# canvas coordinates, so the 80 px top bar is subtracted before a reference row
# is scaled (see ``projection_for``); on the reference capture that cancels
# exactly and the projection is the identity.
REFERENCE = REFERENCE_CONTENT
REFERENCE_CANVAS = REFERENCE_FRAME
REFERENCE_CONTENT_ORIGIN = (0, REFERENCE_BAR_Y)

# Reference minimap panel, capture pixels: (x0, y0, x1, y1). Measured on
# state/mainview-numeric-bar-fixed-20261015/frame-000-394.000s.ppm, the native
# 2560x1600 reference capture: left border 1836, top 796, right 2554, bottom
# 1519 (the last content row).
REFERENCE_MINIMAP_PANEL = (1836, 796, 2554, 1519)
# Same element as a rectangle, for callers that already think in rects. The
# mask/ROI form keeps the historical 720x723 extent (1836..2556, 796..1519).
REFERENCE_MINIMAP = (1836, 796, 720, 723)
# Reference bottom turret-bar row: four mounts, 96 px stride, 60 px bodies.
REFERENCE_BAR_ROW = {"count": 4, "first": 1108, "pitch": 96, "width": 60,
                     "bar_y": 1365, "baseline": 1371}

# Detection limits. They describe geometry, not recognition thresholds.
PANEL_MIN_SIDE = 240
PANEL_EDGE_MARGIN = 48
PANEL_EDGE_CONTINUITY = 0.60
PANEL_EDGE_GRADIENT = 25.0
BAR_WIDTH_RANGE = (55, 280)
BAR_HEIGHT_RANGE = (3, 13)
BAR_ASPECT_MIN = 6.0
SCALE_RANGE = (0.5, 3.0)
# A measured projection is accurate to about a pixel per anchor, so a
# coordinate may land a few pixels outside the frame at its own edge. That is
# measurement slack and is clipped. Anything beyond it is a wrong mapping and
# raises, which is what keeps a missing anchor from silently reading the wrong
# pixels.
FRAME_TOLERANCE = 8.0
PROJECTION_ENV = "WWS_HUD_PROJECTION"


class MappingError(RuntimeError):
    """A projection cannot be measured, is ambiguous, or leaves the frame."""


@dataclass(frozen=True)
class Panel:
    """Measured minimap panel edges in frame coordinates (right/bottom exclusive)."""
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def as_dict(self) -> dict:
        return {"left": self.left, "top": self.top, "right": self.right,
                "bottom": self.bottom, "width": self.width, "height": self.height}


@dataclass(frozen=True)
class BarRow:
    """Measured bottom bar row: baseline, first slot and slot pitch in frame pixels."""
    baseline: int
    first: int
    pitch: float
    widths: tuple
    tops: tuple
    count: int

    def as_dict(self) -> dict:
        return {"baseline": self.baseline, "first": self.first, "pitch": self.pitch,
                "count": self.count, "widths": list(self.widths),
                "tops": list(self.tops)}


@dataclass(frozen=True)
class Projection:
    """One affine per axis, reference space -> frame space."""
    x_scale: float
    x_offset: float
    y_scale: float
    y_offset: float
    frame: tuple
    source: str = "measured_anchors"
    panel: Panel | None = None
    bar: BarRow | None = None
    initial: dict | None = None
    warnings: tuple = field(default_factory=tuple)
    anchors: dict | None = None
    # Every coordinate this projection had to clamp, appended by the mapping
    # calls themselves and published in ``report()``. A frozen dataclass still
    # lets the list be appended to; the tuple fields above stay immutable.
    clamps: list = field(default_factory=list)

    # -- primitives -----------------------------------------------------
    def project_x(self, x: float) -> float:
        return self.x_scale * float(x) + self.x_offset

    def project_y(self, y: float) -> float:
        return self.y_scale * float(y) + self.y_offset

    def _record_clamp(self, reference: Sequence[float], mapped: Sequence[float],
                      axis: str, reason: str = "frame_bounds") -> None:
        """Remember a clamped coordinate so it is reported, not silent."""
        if len(self.clamps) >= 32:
            return
        self.clamps.append({"reference": [round(float(v), 3) for v in reference],
                            "mapped": [round(float(v), 3) for v in mapped],
                            "axis": axis, "reason": reason})

    # -- reference-space conversion ------------------------------------
    def project_point(self, point) -> tuple[int, int]:
        """Reference capture point -> frame point."""
        if point is None or len(point) != 2:
            raise MappingError("point must be (x, y)")
        x, y = float(point[0]), float(point[1])
        _check_reference_point(x, y, point)
        wx, wy = self.project_x(x), self.project_y(y)
        _check_frame_point(wx, wy, self.frame, point)
        if not (0.0 <= wx <= self.frame[0] - 1 and 0.0 <= wy <= self.frame[1] - 1):
            self._record_clamp(point, (wx, wy), "both" if (
                not 0.0 <= wx <= self.frame[0] - 1) and not (
                0.0 <= wy <= self.frame[1] - 1) else (
                "x" if not 0.0 <= wx <= self.frame[0] - 1 else "y"))
        return _pixel(wx, self.frame[0]), _pixel(wy, self.frame[1])

    def from_ref(self, rect, *, clamp: bool = True) -> list[int]:
        """Reference rect -> frame rect; an out-of-frame rect is clamped and reported.

        ``clamp=False`` is the strict variant ``project_rect`` uses: a rect that
        leaves the frame by more than the measurement tolerance raises, so a
        reader never slices a window it could not place. ``clamp=True`` (the
        default here, and what a caller that has to keep running wants) keeps
        the part that is on screen and appends the clamp to
        ``report()['clamped']``.
        """
        if rect is None or len(rect) != 4:
            raise MappingError("rect must be [x, y, w, h]")
        x, y, w, h = (float(v) for v in rect)
        if w <= 0 or h <= 0:
            raise MappingError(f"rect {list(rect)} must have positive size")
        width, height = self.frame
        x0, y0 = self.project_x(x), self.project_y(y)
        x1, y1 = self.project_x(x + w), self.project_y(y + h)
        if not clamp:
            _check_frame_point(x0, y0, self.frame, rect)
            _check_frame_point(x1, y1, self.frame, rect)
        cx0, cy0 = min(max(0.0, x0), float(width)), min(max(0.0, y0), float(height))
        cx1, cy1 = min(max(0.0, x1), float(width)), min(max(0.0, y1), float(height))
        x_clamped = (cx0, cx1) != (x0, x1)
        y_clamped = (cy0, cy1) != (y0, y1)
        if clamp and (x_clamped or y_clamped):
            self._record_clamp(rect, (x0, y0, x1, y1),
                               "both" if x_clamped and y_clamped else
                               "x" if x_clamped else "y")
        left, top = _pixel(cx0, width), _pixel(cy0, height)
        return [left, top, max(1, _pixel(cx1, width) - left),
                max(1, _pixel(cy1, height) - top)]

    def to_ref(self, rect, *, clamp: bool = True) -> list[int]:
        """Frame rect -> reference rect; the inverse of ``from_ref``.

        The reference content region is the target: rows above the 80 px top
        bar and below the bottom bar are not HUD space. Rectangles that cross
        that boundary are clamped into it and recorded, so a caller converting
        a live detection back for comparison can see that it was truncated.
        """
        if rect is None or len(rect) != 4:
            raise MappingError("rect must be [x, y, w, h]")
        if self.x_scale == 0 or self.y_scale == 0:
            raise MappingError("zero-scale projection cannot be inverted")
        x, y, w, h = (float(v) for v in rect)
        if w <= 0 or h <= 0:
            raise MappingError(f"rect {list(rect)} must have positive size")
        rx0 = (x - self.x_offset) / self.x_scale
        ry0 = (y - self.y_offset) / self.y_scale
        rx1 = (x + w - self.x_offset) / self.x_scale
        ry1 = (y + h - self.y_offset) / self.y_scale
        lo_x, lo_y = REFERENCE_CONTENT_ORIGIN
        hi_x, hi_y = lo_x + REFERENCE[0], lo_y + REFERENCE[1]
        cx0, cy0 = min(max(lo_x, rx0), float(hi_x)), min(max(lo_y, ry0), float(hi_y))
        cx1, cy1 = min(max(lo_x, rx1), float(hi_x)), min(max(lo_y, ry1), float(hi_y))
        x_clamped = (cx0, cx1) != (rx0, rx1)
        y_clamped = (cy0, cy1) != (ry0, ry1)
        if clamp and (x_clamped or y_clamped):
            self._record_clamp(rect, (rx0, ry0, rx1, ry1),
                               "both" if x_clamped and y_clamped else
                               "x" if x_clamped else "y", reason="reference_bounds")
        left, top = int(round(cx0)), int(round(cy0))
        return [left, top, max(1, int(round(cx1)) - left),
                max(1, int(round(cy1)) - top)]

    def project_span_y(self, y0: float, y1: float) -> tuple[int, int]:
        """Project two reference rows to a frame row span (top, bottom).

        The bounds check probes the frame's horizontal centre, not a reference
        column: a reference column projected as if it were a frame coordinate
        would reject every frame narrower than 1836 px.
        """
        _check_reference_point(REFERENCE_MINIMAP_PANEL[0], float(y0), [0, y0])
        _check_reference_point(REFERENCE_MINIMAP_PANEL[0], float(y1), [0, y1])
        wy0, wy1 = self.project_y(y0), self.project_y(y1)
        centre_x = self.frame[0] / 2.0
        _check_frame_point(centre_x, wy0, self.frame, [0, y0])
        _check_frame_point(centre_x, wy1, self.frame, [0, y1])
        return (_pixel(wy0, self.frame[1]), _pixel(wy1, self.frame[1]))

    def project_rect(self, rect) -> list[int]:
        """Reference capture [x, y, w, h] -> frame [x, y, w, h] (strict).

        A rectangle that leaves the frame by more than the measurement
        tolerance raises instead of being truncated, because a reader would
        otherwise slice a window it could not place. ``from_ref`` is the same
        conversion with clamping and reporting.
        """
        if rect is None or len(rect) != 4:
            raise MappingError("rect must be [x, y, w, h]")
        x, y = float(rect[0]), float(rect[1])
        _check_reference_point(x, y, rect)
        return self.from_ref(rect, clamp=False)

    def project_rects(self, rects: Iterable[Sequence[float]]) -> list[list[int]]:
        return [self.project_rect(rect) for rect in rects]

    def clip_rect(self, rect) -> list[int]:
        """Project a reference rect and clip it to the frame.

        For masking only: unlike ``project_rect`` this cannot raise, because a
        mask rectangle has no meaning outside the frame and a mask must never
        abort a live loop. The projection itself is still the measured one; this
        only drops the part of the rectangle that falls outside the raster, and
        that clipping is recorded in ``report()['clamped']``.
        """
        if rect is None or len(rect) != 4:
            raise MappingError("rect must be [x, y, w, h]")
        x, y, w, h = (float(v) for v in rect)
        x0, y0 = self.project_x(x), self.project_y(y)
        x1, y1 = self.project_x(x + max(0.0, w)), self.project_y(y + max(0.0, h))
        left, right = sorted((x0, x1))
        top, bottom = sorted((y0, y1))
        clipped = [max(0, min(self.frame[0] - 1, int(round(left)))),
                   max(0, min(self.frame[1] - 1, int(round(top)))),
                   min(self.frame[0], int(round(right))),
                   min(self.frame[1], int(round(bottom)))]
        if clipped != [int(round(left)), int(round(top)),
                       int(round(right)), int(round(bottom))]:
            self._record_clamp(rect, (x0, y0, x1, y1), "both", reason="frame_bounds")
        left, top, right, bottom = clipped
        return [left, top, max(0, right - left), max(0, bottom - top)]

    # -- evidence -------------------------------------------------------
    def report(self) -> dict:
        out = {"source": self.source, "frame": list(self.frame),
               "reference": list(REFERENCE),
               "reference_frame": list(REFERENCE_FRAME),
               "scale": [round(self.x_scale, 6), round(self.y_scale, 6)],
               "offset": [round(self.x_offset, 3), round(self.y_offset, 3)],
               "uniform": bool(abs(self.x_scale - self.y_scale) <= 1e-3),
               "warnings": list(self.warnings),
               "clamped": list(self.clamps),
               "anchors": {"reference_minimap_panel": list(REFERENCE_MINIMAP_PANEL),
                           "reference_bar_row": dict(REFERENCE_BAR_ROW),
                           "minimap_panel": None if self.panel is None else self.panel.as_dict(),
                           "bar_row": None if self.bar is None else self.bar.as_dict()}}
        if self.initial is not None:
            out["initial"] = self.initial
        if self.anchors is not None:
            out["anchors_used"] = self.anchors
        if self.panel is not None:
            out["minimap_residual_px"] = round(
                self.project_y(REFERENCE_MINIMAP_PANEL[3]) - self.panel.bottom, 3)
        return out


def _as_frame(frame) -> np.ndarray:
    a = np.asarray(frame)
    if a.ndim != 3 or a.shape[2] < 3:
        raise MappingError("frame must be HxWx3")
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    if a.shape[0] < PANEL_MIN_SIDE or a.shape[1] < PANEL_MIN_SIDE:
        raise MappingError(f"frame {a.shape[:2]} is too small to hold a HUD")
    return a


def _check_reference_point(x: float, y: float, original) -> None:
    width, height = REFERENCE_FRAME
    bar = REFERENCE_BAR_Y
    if not (0 <= x < width):
        raise MappingError(f"point {list(original)} outside the reference frame width")
    if not (bar <= y < height - bar):
        raise MappingError(f"point {list(original)} lies in the reference bars, not content")


def _check_frame_point(wx: float, wy: float, frame, original) -> None:
    # A correctly measured projection can land a fraction of a pixel outside
    # the frame at its own edge; that is rasterisation, not a wrong mapping.
    if not (-FRAME_TOLERANCE <= wx <= frame[0] + FRAME_TOLERANCE and
            -FRAME_TOLERANCE <= wy <= frame[1] + FRAME_TOLERANCE):
        raise MappingError(f"point {list(original)} projects outside the frame {frame}")


def _pixel(value: float, limit: int) -> int:
    """Round a projected coordinate and keep it inside the rasterised frame."""
    return max(0, min(limit - 1, int(round(value))))


def _peak(score: np.ndarray, lo: int, hi: int) -> int:
    lo, hi = max(0, int(lo)), min(len(score), int(hi))
    if hi - lo < 1:
        raise MappingError("search range outside the score profile")
    return lo + int(np.argmax(score[lo:hi]))


def _continuous_edge(score: np.ndarray, continuity: np.ndarray, lo: int, hi: int,
                     minimum: float = PANEL_EDGE_CONTINUITY,
                     leftmost: bool = False) -> int:
    """Strongest edge in [lo, hi) that also runs the panel's whole height.

    Sea texture and map grid lines make single-row gradient spikes, so
    brightness alone picks the wrong border. A border line exists on (nearly)
    every row of the panel; requiring that continuity rejects the spikes.

    Inside the panel the grid lines are continuous too, so the left border is
    the *first* strong continuous edge (``leftmost``): every grid line sits to
    the right of it, while sea texture never reaches the continuity floor.
    """
    lo, hi = max(0, int(lo)), min(len(score), int(hi))
    if hi - lo < 1:
        raise MappingError("search range outside the edge profile")
    candidates = np.flatnonzero(continuity[lo:hi] >= minimum)
    if not len(candidates):
        raise MappingError(
            f"no continuous minimap border in columns {lo}..{hi} "
            f"(best continuity {float(continuity[lo:hi].max()):.2f})")
    strong = score[lo:hi][candidates]
    if leftmost:
        keep = candidates[strong >= 0.5 * float(strong.max())]
        return lo + int(keep[0])
    return lo + int(candidates[int(np.argmax(strong))])


def minimap_panel(frame) -> Panel:
    """Locate the minimap panel by its four border edges.

    Top and bottom come first from the row profile, because the panel is the
    widest horizontal edge structure in the right half of the frame. The
    column profile is then evaluated *inside* that vertical span, which is what
    makes the left and right borders continuous over the panel's full height
    and separates them from sea texture.
    """
    a = _as_frame(frame)
    height, width = a.shape[:2]
    gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    rows = gy[:, int(width * 0.72):int(width * 0.99)].mean(axis=1)
    top = _peak(rows, int(height * 0.30), int(height * 0.72))
    bottom = _peak(rows, top + 150, int(height * 0.975))
    if not PANEL_MIN_SIDE <= bottom - top <= int(height * 0.95):
        raise MappingError(f"minimap panel height {bottom - top} is implausible")
    band = gx[top + 20:bottom - 20, :]
    if band.shape[0] < 40:
        raise MappingError("minimap panel band is too thin to measure")
    columns = band.mean(axis=0)
    continuity = (band > PANEL_EDGE_GRADIENT).mean(axis=0)
    left = _continuous_edge(columns, continuity, int(width * 0.55), int(width * 0.88),
                            leftmost=True)
    right = _continuous_edge(columns, continuity, left + 200, width)
    panel = Panel(left, top, right, bottom)
    if panel.right < width - PANEL_EDGE_MARGIN:
        raise MappingError(
            f"minimap panel right edge {panel.right} is not against the frame edge {width}")
    if panel.width < PANEL_MIN_SIDE or panel.height < PANEL_MIN_SIDE:
        raise MappingError(f"minimap panel {panel.as_dict()} is too small to anchor")
    aspect = panel.width / panel.height
    if not 0.7 <= aspect <= 1.5:
        raise MappingError(f"minimap panel {panel.as_dict()} is not panel shaped")
    return panel


def bar_row(frame) -> BarRow:
    """Locate the bottom turret/ammo bar row: baseline, first slot, slot pitch.

    Bars are thin saturated horizontal bodies. Ticks and traverse indicators
    are shorter than ``BAR_WIDTH_RANGE`` and are rejected by width, not by
    colour, so ammo state and readiness colours do not matter.
    """
    a = _as_frame(frame)
    height, width = a.shape[:2]
    hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)
    saturated = ((hsv[..., 1].astype(np.int16) >= 90) &
                 (hsv[..., 2].astype(np.int16) >= 90)).astype(np.uint8)
    x0, x1 = int(width * 0.30), int(width * 0.74)
    y0, y1 = int(height * 0.70), int(height * 0.97)
    patch = saturated[y0:y1, x0:x1]
    found, _, stats, _ = cv2.connectedComponentsWithStats(patch, 8)
    bars = []
    for x, y, w, h, area in stats[1:]:
        w, h = int(w), int(h)
        if not (BAR_WIDTH_RANGE[0] <= w <= BAR_WIDTH_RANGE[1]):
            continue
        if not (BAR_HEIGHT_RANGE[0] <= h <= BAR_HEIGHT_RANGE[1]):
            continue
        if w / max(h, 1) < BAR_ASPECT_MIN or area < 2 * w:
            continue
        bars.append((int(x) + x0, int(y) + y0, w, h))
    if len(bars) < 2:
        raise MappingError(f"bottom bar row not found: {len(bars)} bars")
    groups: list = []
    for bar in sorted(bars, key=lambda b: b[1]):
        if groups and bar[1] - groups[-1][-1][1] <= 2:
            groups[-1].append(bar)
        else:
            groups.append([bar])
    row = max(groups, key=lambda group: (len(group), sum(b[2] for b in group)))
    if len(row) < 2:
        raise MappingError("bottom bar row not found: no shared baseline")
    row.sort(key=lambda b: b[0])
    xs = [b[0] for b in row]
    gaps = [b - a for a, b in zip(xs, xs[1:]) if b - a >= 8]
    if not gaps:
        raise MappingError("bottom bar row has no usable slot pitch")
    return BarRow(baseline=max(b[1] + b[3] for b in row), first=xs[0],
                  pitch=float(np.median(gaps)), widths=tuple(b[2] for b in row),
                  tops=tuple(b[1] for b in row), count=len(row))


def content_bounds(frame) -> tuple[int, int, int, int]:
    """Frame content rectangle (w, h, x0, y0) measured from the letterbox.

    The game surface is the live region; the canvas bars are pure black. No
    reference size is assumed: a frame with no bars reports the whole frame as
    its surface, and a frame with nothing but bars raises.
    """
    a = _as_frame(frame)
    gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
    live_rows = np.flatnonzero(gray.mean(axis=1) > 4.0)
    live_cols = np.flatnonzero(gray.mean(axis=0) > 4.0)
    if not len(live_rows) or not len(live_cols):
        raise MappingError("frame content is empty: the frame is a letterbox only")
    return (int(live_cols[-1] - live_cols[0] + 1), int(live_rows[-1] - live_rows[0] + 1),
            int(live_cols[0]), int(live_rows[0]))


def projection_for(frame_shape, content=None) -> Projection:
    """Declared linear map for a frame shape alone: reference space -> this frame.

    ``frame_shape`` is an array shape, ``(height, width)`` or
    ``(height, width, channels)``. The game surface is the whole frame unless
    ``content`` supplies its ``(x, y, w, h)``, or the shape is the reference
    capture canvas 2560x1600 -- whose content is the declared 2560x1440 surface
    between the two 80 px bars. Both axes scale independently
    (``sx = content_w / 2560``, ``sy = content_h / 1440``) and the offsets
    remove the reference letterbox and add this frame's content origin, so the
    reference capture maps to itself exactly: ``sx = sy = 1`` and no offset.

    No pixel is read here, which is why the result is labelled
    ``declared_shape`` and says so in ``warnings``. ``resolve`` measures the
    frame's own anchors when it can, and ``declared`` measures the letterbox;
    both report which they used, and a caller that must not fall back can pass
    ``clamp=False`` rectangles or use ``measure`` directly.
    """
    if frame_shape is None or len(frame_shape) < 2:
        raise MappingError("frame_shape must be (height, width[, channels])")
    height, width = int(frame_shape[0]), int(frame_shape[1])
    if height <= 0 or width <= 0:
        raise MappingError(f"frame shape {tuple(frame_shape)} is empty")
    if content is None:
        if (width, height) == REFERENCE_FRAME:
            content = (0, REFERENCE_BAR_Y, REFERENCE_CONTENT[0], REFERENCE_CONTENT[1])
        else:
            content = (0, 0, width, height)
    if len(content) != 4:
        raise MappingError("content must be [x, y, w, h]")
    cx, cy, cw, ch = (int(round(float(v))) for v in content)
    if cw <= 0 or ch <= 0:
        raise MappingError(f"content {list(content)} must have positive size")
    if cx < 0 or cy < 0 or cx + cw > width or cy + ch > height:
        raise MappingError(f"content {list(content)} does not fit the frame {width}x{height}")
    x_scale = cw / float(REFERENCE_CONTENT[0])
    y_scale = ch / float(REFERENCE_CONTENT[1])
    if not (SCALE_RANGE[0] <= x_scale <= SCALE_RANGE[1] and
            SCALE_RANGE[0] <= y_scale <= SCALE_RANGE[1]):
        raise MappingError(f"content {cw}x{ch} is not a plausible game surface")
    return Projection(x_scale=x_scale,
                      x_offset=float(cx) - x_scale * REFERENCE_CONTENT_ORIGIN[0],
                      y_scale=y_scale,
                      y_offset=float(cy) - y_scale * REFERENCE_CONTENT_ORIGIN[1],
                      frame=(width, height), source="declared_shape",
                      warnings=(f"declared_shape_projection:{cw}x{ch}+{cx}+{cy}",))


def declared(frame, initial: dict | None = None, reason: str = "") -> Projection:
    """Declared projection: reference content stretched onto this frame's content.

    This is geometry from the frame itself (its letterbox bounds), not a HUD
    measurement. It is the documented fallback when no HUD anchor is drawn, and
    it says so in ``source`` so no caller can mistake it for a measured fit.

    A reference-canvas frame whose content cannot be measured as a game surface
    (no letterbox, or only a pasted sprite region) is declared to *be* the
    reference space, and the warning says ``reference_canvas`` rather than
    pretending it was measured. Any other frame size whose content cannot be
    measured (an all-black frame, a frame narrower than the letterbox scan) is
    declared to be its own surface, with ``unmeasurable_content`` in the warning
    -- never silently rescaled to some invented rectangle.
    """
    a = _as_frame(frame)
    reference_canvas = (a.shape[1], a.shape[0]) == REFERENCE_FRAME
    note = ""
    try:
        cw, ch, ox, oy = content_bounds(a)
        whole_frame = (ox, oy, cw, ch) == (0, 0, a.shape[1], a.shape[0])
        plausible = (SCALE_RANGE[0] <= cw / REFERENCE_CONTENT[0] <= SCALE_RANGE[1] and
                     SCALE_RANGE[0] <= ch / REFERENCE_CONTENT[1] <= SCALE_RANGE[1])
        if reference_canvas and (whole_frame or not plausible):
            # No letterbox to measure, or only a fragment of art: a reference
            # canvas is declared to be the reference space rather than being
            # silently re-scaled to a 16:10 surface or to a pasted sprite.
            raise MappingError(
                f"content {cw}x{ch} at ({ox},{oy}) is not a measurable surface")
    except MappingError:
        if not reference_canvas:
            cw, ch, ox, oy = a.shape[1], a.shape[0], 0, 0
            note = "unmeasurable_content"
        else:
            cw, ch, ox, oy = (REFERENCE_CONTENT[0], REFERENCE_CONTENT[1], 0, REFERENCE_BAR_Y)
            note = "reference_canvas"
    x_scale = cw / REFERENCE_CONTENT[0]
    y_scale = ch / REFERENCE_CONTENT[1]
    if not (SCALE_RANGE[0] <= x_scale <= SCALE_RANGE[1] and
            SCALE_RANGE[0] <= y_scale <= SCALE_RANGE[1]):
        raise MappingError(f"content {cw}x{ch} is not a plausible game surface")
    detail = ":".join(part for part in (note, reason) if part) or "measured_content"
    return Projection(x_scale=x_scale, x_offset=float(ox),
                      y_scale=y_scale, y_offset=float(oy) - y_scale * REFERENCE_BAR_Y,
                      frame=(a.shape[1], a.shape[0]), source="declared_content",
                      initial=initial, warnings=(f"declared_content_projection:{detail}",))


def _fit_axis(anchors, tolerance: float, axis: str):
    """Weighted least-squares affine through measured anchors, dropping outliers.

    ``anchors`` are ``(reference_value, frame_value, label, weight)``. A single
    bad measurement (a minimap title row, a turret row whose mount count
    differs) would otherwise bias both the scale and the offset, so the worst
    anchor is dropped and the fit repeated, up to two times. The survivors keep
    the full span; at least two points and 100 px of reference span are
    required.
    """
    remaining = [tuple(a) for a in anchors]
    dropped = []
    while True:
        if len(remaining) < 2:
            raise MappingError(f"{axis} axis needs two measured anchors, got {len(remaining)}")
        ref = np.asarray([a[0] for a in remaining], dtype=np.float64)
        frame = np.asarray([a[1] for a in remaining], dtype=np.float64)
        weight = np.asarray([a[3] for a in remaining], dtype=np.float64)
        if ref.max() - ref.min() < 100:
            raise MappingError(f"{axis} anchors span only {ref.max() - ref.min():.0f} px")
        design = np.stack([ref, np.ones_like(ref)], axis=1) * weight[:, None]
        scale, offset = np.linalg.lstsq(design, frame * weight, rcond=None)[0]
        residuals = np.abs(scale * ref + offset - frame)
        # Low-weight letterbox checks never get dropped: they are the frame's
        # own geometry, so a disagreement is reported, not removed. Among the
        # measured HUD anchors the worst offender is dropped and the fit
        # repeated.
        droppable = np.flatnonzero(weight >= 0.5)
        if not len(droppable):
            return float(scale), float(offset), remaining, dropped
        worst = int(droppable[int(np.argmax(residuals[droppable]))])
        if residuals[worst] <= tolerance or len(dropped) >= 2:
            return float(scale), float(offset), remaining, dropped
        dropped.append((remaining.pop(worst)[2], round(float(residuals[worst]), 2)))


def fit(frame, panel: Panel, bar: BarRow | None = None,
        initial: dict | None = None) -> Projection:
    """Fit one affine per axis from every anchor the frame actually shows.

    The minimap panel carries both axes: its left and right borders for x, its
    top and bottom rows for y. The bottom bar row's baseline is a second
    vertical anchor and its pitch is reported, but the row is optional and
    droppable: on a scoped frame the thin saturated bars can belong to another
    HUD group (measured at y=1331 against a panel bottom of 1519), so a baseline
    that disagrees with the panel is dropped and named in ``warnings``.

    The frame's own letterbox bounds the game surface and is kept as a
    *low-weight* check: on this machine the client draws its HUD at nearly the
    reference size inside a 1408 px tall surface, so letting the letterbox
    dominate would stretch the HUD vertically and misplace the minimap by about
    15 px (measured). Horizontally the surface spans the whole frame, so its
    edges are real anchors there.
    """
    a = _as_frame(frame)
    cw, ch, cx, cy = content_bounds(a)
    ref_x0, ref_top, ref_x1, ref_bottom = REFERENCE_MINIMAP_PANEL
    ref_baseline = float(REFERENCE_BAR_ROW["baseline"])
    x_anchors = [(float(ref_x1), float(panel.right), "panel_right", 1.0),
                 # The surface spans the frame width, so its edges are real
                 # horizontal coordinate anchors. The panel's *left* border is
                 # not: the minimap has its own size setting, and a smaller
                 # panel (measured 458 px wide on state/current-view-20261017-q1
                 # .png) would otherwise scale the whole HUD by 0.64.
                 (0.0, float(cx), "content_left", 0.5),
                 (float(REFERENCE_CONTENT[0] - 1), float(cx + cw - 1), "content_right", 0.5)]
    y_anchors = [# The panel's top row can carry a title strip (measured: +40 px on
                 # state/tracking-mainview-236374.png), so it anchors at half
                 # weight and is the first anchor the robust pass drops; the
                 # bottom row and the bar baseline stay full weight because the
                 # bottom HUD is what gets read.
                 (float(ref_top), float(panel.top), "panel_top", 0.5),
                 (float(ref_bottom), float(panel.bottom), "panel_bottom", 1.0),
                 (float(REFERENCE_BAR_Y), float(cy), "content_top", 0.15),
                 (float(REFERENCE_FRAME[1] - REFERENCE_BAR_Y - 1), float(cy + ch - 1),
                  "content_bottom", 0.15)]
    if bar is not None:
        y_anchors.insert(1, (ref_baseline, float(bar.baseline), "bar_baseline", 1.0))
    # Measured HUD edges are accurate to about a pixel, so a 6 px disagreement
    # is a different layout rather than noise.
    tolerance = 6.0
    x_scale, x_offset, x_used, x_dropped = _fit_axis(x_anchors, tolerance, "x")
    y_scale, y_offset, y_used, y_dropped = _fit_axis(y_anchors, tolerance, "y")
    for axis, value in (("x", x_scale), ("y", y_scale)):
        if not SCALE_RANGE[0] <= value <= SCALE_RANGE[1]:
            raise MappingError(f"{axis} scale {value:.4f} is implausible for a game HUD")
    warnings = [f"dropped_{axis}_anchor:{name} (residual {residual} px)"
                for axis, dropped in (("x", x_dropped), ("y", y_dropped))
                for name, residual in dropped]
    if bar is not None:
        expected_pitch = REFERENCE_BAR_ROW["pitch"] * x_scale
        if abs(bar.pitch - expected_pitch) > 0.15 * expected_pitch:
            warnings.append(
                "bar_slot_pitch_differs_from_reference_layout: measured "
                f"{bar.pitch:.1f} px for {bar.count} slots, reference "
                f"{REFERENCE_BAR_ROW['pitch']} px for {REFERENCE_BAR_ROW['count']} slots "
                f"(x scale is {x_scale:.4f}); the row is a per-client layout, so the "
                "pitch is reported, not used to place reference rectangles")
    reference_width = ref_x1 - ref_x0
    if abs(panel.width - reference_width) > 0.03 * reference_width:
        warnings.append(
            f"minimap_size_differs_from_reference: measured {panel.width}x{panel.height} "
            f"against reference {reference_width}x{ref_bottom - ref_top}; the panel's top "
            "and left borders follow that size setting and are reported, not used for scale")
    if initial:
        guess = (initial.get("scale") or [None, None])[1]
        if guess and abs(guess - x_scale) > 0.05 * x_scale:
            warnings.append(
                f"initial_scale_disagrees: resolution_map scale {guess:.4f} vs "
                f"measured {x_scale:.4f}x/{y_scale:.4f}")
    return Projection(x_scale=x_scale, x_offset=x_offset, y_scale=y_scale,
                      y_offset=y_offset, frame=(a.shape[1], a.shape[0]),
                      source="measured_anchors", panel=panel, bar=bar,
                      initial=initial, warnings=tuple(warnings),
                      anchors={"x": [(n, r) for r, _, n, _ in x_used],
                               "y": [(n, r) for r, _, n, _ in y_used]})


def identity(frame) -> Projection:
    """Identity projection: the caller has already guaranteed reference space.

    It is only for callers that do not hold the frame (``hud_mask``), for tests
    that build reference-canvas frames on purpose, and for the explicit
    ``WWS_HUD_PROJECTION=reference`` run switch. ``source`` says so.
    """
    if isinstance(frame, (tuple, list)) and len(frame) >= 2:
        size = (int(frame[1]), int(frame[0]))
    else:
        a = _as_frame(frame)
        size = (a.shape[1], a.shape[0])
    return Projection(x_scale=1.0, x_offset=0.0, y_scale=1.0, y_offset=0.0,
                      frame=size, source="identity_reference",
                      warnings=("identity_reference_projection",))


def projection_disabled() -> bool:
    """True when the operator forced reference space for this run.

    ``WWS_HUD_PROJECTION=reference`` (also ``off``/``identity``/``0``) makes
    every ``resolve()`` return the identity, which is the pre-projection
    behaviour. It exists so a live run can keep going if a measurement ever
    misbehaves, without editing code.
    """
    value = os.environ.get(PROJECTION_ENV, "").strip().lower()
    return value in ("reference", "off", "identity", "0", "false", "no")


def measure(frame, initial: dict | None = None) -> Projection:
    """Measure both anchors in this frame and fit the per-axis affine.

    The minimap panel is required. The bottom bar row is an extra vertical
    anchor and is allowed to be missing (a scoped frame can hide or replace
    it); the projection then rests on the panel alone and records why.
    """
    a = _as_frame(frame)
    panel = minimap_panel(a)
    try:
        bar = bar_row(a)
    except MappingError as exc:
        projection = fit(a, panel, None, initial=initial)
        return replace(projection,
                       warnings=projection.warnings + (f"bar_row_unavailable:{exc}",))
    return fit(a, panel, bar, initial=initial)


def resolve(frame, initial: dict | None = None, *, allow_declared: bool = True) -> Projection:
    """Measured projection, or the recorded declared fallback when no HUD is drawn.

    ``WWS_HUD_PROJECTION=reference`` short-circuits to the identity and is
    reported in ``warnings``; nothing else about the call sites changes.
    """
    if projection_disabled():
        return replace(identity(frame), warnings=(f"{PROJECTION_ENV}=reference",))
    try:
        return measure(frame, initial=initial)
    except MappingError as exc:
        if not allow_declared:
            raise
        return declared(frame, initial=initial, reason=str(exc))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('frame', type=Path, help='captured PNG frame')
    parser.add_argument('--rect', nargs=4, type=float, action='append',
                        help='reference rect to project (repeatable)')
    parser.add_argument('--point', nargs=2, type=float, action='append',
                        help='reference point to project (repeatable)')
    parser.add_argument('--declared', action='store_true',
                        help='report the declared content projection instead of measuring')
    args = parser.parse_args(argv)
    image = cv2.imread(str(args.frame))
    if image is None:
        print(json.dumps({'status': 'error', 'reason': f'unreadable {args.frame}'}))
        return 1
    frame = np.ascontiguousarray(image[:, :, ::-1])
    try:
        projection = (declared(frame) if args.declared else measure(frame))
        out = {'status': 'ok', 'projection': projection.report()}
        if args.rect:
            out['rects'] = [{'reference': list(r), 'frame': projection.project_rect(r)}
                            for r in args.rect]
        if args.point:
            out['points'] = [{'reference': list(p), 'frame': list(projection.project_point(p))}
                             for p in args.point]
    except MappingError as exc:
        print(json.dumps({'status': 'error', 'reason': str(exc)}))
        return 1
    print(json.dumps(out, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
