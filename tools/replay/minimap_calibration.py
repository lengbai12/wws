"""Minimap calibration: own-ship marker detection and the world projection.

Two measured facts drive this module.

1. The own ship's marker on the minimap is a solid white triangle that rotates
   with the ship's heading. A rotating shape cannot be matched by a fixed
   template, so the marker is found as the largest near-white connected region
   inside the minimap rectangle, and its centroid is the observation.

2. The minimap is a fixed projection of the battle map, so world coordinates map
   to minimap pixels through one affine transform, fixed for a map and a camera
   setting. That transform is what turns a replay position into a pixel label.

Nothing here decides which minimap element is which, and nothing infers a marker
that is not visible: an absent observation returns None and is counted as absent,
never as a zero position.
"""

from __future__ import annotations

import numpy as np

# The marker is drawn in pure white. Range circles and text are lighter but not
# white, so a high floor separates them.
WHITE_MIN = 205

# A marker is a small solid shape. The bounds reject text glyphs and the large
# semi-transparent range rings, which carry far more pixels than a marker.
MARKER_AREA_MIN = 25
MARKER_AREA_MAX = 4000


def frame_array(rgb, width, height):
    """Return the frame as an (height, width, 3) uint8 array."""
    expected = width * height * 3
    if len(rgb) != expected:
        raise ValueError("frame buffer is %d bytes, expected %d for %dx%d RGB24"
                         % (len(rgb), expected, width, height))
    return np.frombuffer(rgb, dtype=np.uint8).reshape(height, width, 3)


def marker_mask(frame, rect, white_min=WHITE_MIN):
    """Return the near-white mask of a region, in that region's own coordinates."""
    x, y, w, h = rect
    crop = frame[y:y + h, x:x + w]
    return crop.min(axis=2) >= white_min


def largest_component(mask, min_area=MARKER_AREA_MIN, max_area=MARKER_AREA_MAX):
    """Return (area, centroid_yx) of the largest component, or None.

    Components are found by iterative flood fill over the boolean mask. A single
    minimap rectangle holds few thousand white pixels, so a dense label pass is
    not needed and no image library is required.
    """
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    rows, columns = np.nonzero(mask)
    best = None
    for start_row, start_column in zip(rows.tolist(), columns.tolist()):
        if visited[start_row, start_column]:
            continue
        stack = [(start_row, start_column)]
        visited[start_row, start_column] = True
        total_row = 0
        total_column = 0
        area = 0
        while stack:
            row, column = stack.pop()
            area += 1
            total_row += row
            total_column += column
            for next_row, next_column in ((row - 1, column), (row + 1, column),
                                          (row, column - 1), (row, column + 1)):
                if (0 <= next_row < height and 0 <= next_column < width
                        and mask[next_row, next_column]
                        and not visited[next_row, next_column]):
                    visited[next_row, next_column] = True
                    stack.append((next_row, next_column))
        if min_area <= area <= max_area and (best is None or area > best[0]):
            best = (area, (total_row / area, total_column / area))
    return best


def detect_marker(frame, rect, white_min=WHITE_MIN):
    """Return (x, y, area) of the own-ship marker in absolute frame pixels.

    Returns None when no component matches, which is the honest answer for a
    minimap that is hidden, covered, or not yet drawn.
    """
    found = largest_component(marker_mask(frame, rect, white_min))
    if found is None:
        return None
    area, (row, column) = found
    x, y = rect[0], rect[1]
    return (x + column, y + row, area)


def fit_affine(world, pixels):
    """Fit ``pixel = A @ (x, z) + t`` by least squares.

    ``world`` is a sequence of (x, z) and ``pixels`` a sequence of (px, py) of
    equal length, at least three. Returns (A, t, residuals) where residuals are
    the Euclidean pixel distances per sample, in input order.
    """
    world = np.asarray(world, dtype=float)
    pixels = np.asarray(pixels, dtype=float)
    if world.shape != pixels.shape or world.shape[1] != 2:
        raise ValueError("world and pixels must both be Nx2")
    if len(world) < 3:
        raise ValueError("an affine fit needs at least three samples, got %d" % len(world))
    design = np.column_stack([world, np.ones(len(world))])
    solution, _residuals, _rank, _singular = np.linalg.lstsq(design, pixels, rcond=None)
    matrix = solution[:2, :].T          # 2x2: pixel = A @ world + t
    translation = solution[2, :]        # (2,)
    predicted = design @ solution
    residuals = np.linalg.norm(predicted - pixels, axis=1)
    return matrix, translation, residuals


def apply_affine(matrix, translation, x, z):
    """Map one world point to minimap pixels."""
    point = np.asarray(matrix, dtype=float) @ np.array([x, z], dtype=float)
    return point + np.asarray(translation, dtype=float)


def ransac_affine(world, pixels, threshold_px=4.0, trials=2000, min_inliers=8, seed=0):
    """Fit the affine on the largest mutually consistent subset of the samples.

    Least squares answers a question nobody asked when the detector is unreliable:
    it returns the best average of a set that was never one relation. That is what
    happened on the snow-storm operation map, where the near-white marker rule
    locks onto snow as often as onto the ship — the fit came back with a 93 degree
    rotation and a 207 px mean residual, which reads like a broken model instead of
    what it is, a broken detection.

    An affine has a property that fixes this: three points determine it, and every
    other sample either agrees or does not. So sample minimal subsets (deterministic
    seeding, so a re-run reports the same thing), keep the subset with the most
    agreeing samples, then refit on those. Returns ``(matrix, translation, inliers,
    distances)`` over ALL samples, or None when no subset reaches ``min_inliers`` —
    which is the honest answer for a detector that found nothing consistent, and must
    never be smoothed over into a plausible-looking transform.
    """
    world = np.asarray(world, dtype=float)
    pixels = np.asarray(pixels, dtype=float)
    count = len(world)
    if count < 4:
        return None
    generator = np.random.default_rng(seed)
    best_inliers = None
    for _ in range(trials):
        pick = generator.choice(count, size=4, replace=False)
        try:
            matrix, translation, _ = fit_affine(world[pick], pixels[pick])
        except (np.linalg.LinAlgError, ValueError):
            continue
        # A degenerate transform (collinear picks) maps everything onto one line and
        # would collect a large "inlier" set, so it is rejected outright.
        if abs(float(np.linalg.det(np.asarray(matrix, dtype=float)))) < 1e-9:
            continue
        distance = np.linalg.norm(world @ matrix.T + translation - pixels, axis=1)
        inliers = np.nonzero(distance <= threshold_px)[0]
        if best_inliers is None or len(inliers) > len(best_inliers):
            best_inliers = inliers
    if best_inliers is None or len(best_inliers) < min_inliers:
        return None
    matrix, translation, _ = fit_affine(world[best_inliers], pixels[best_inliers])
    distance = np.linalg.norm(world @ matrix.T + translation - pixels, axis=1)
    return matrix, translation, best_inliers, distance


def inverse_affine(matrix, translation):
    """Return (matrix, translation) mapping minimap pixels back to world x, z."""
    inverse = np.linalg.inv(np.asarray(matrix, dtype=float))
    return inverse, -inverse @ np.asarray(translation, dtype=float)


def describe(matrix, translation, residuals):
    """Summarise a fit: scale, rotation and residual spread.

    The minimap is expected to be north-up, so the rotation should sit within a
    degree of a multiple of 90 degrees and both axis scales should agree. A fit
    that violates either is reported, not silently accepted.
    """
    matrix = np.asarray(matrix, dtype=float)
    first_axis = matrix[:, 0]
    second_axis = matrix[:, 1]
    scale_x = float(np.linalg.norm(first_axis))
    scale_z = float(np.linalg.norm(second_axis))
    angle_deg = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
    residuals = np.asarray(residuals, dtype=float)
    return {
        "scale_px_per_m_x": scale_x,
        "scale_px_per_m_z": scale_z,
        "scale_ratio": (scale_x / scale_z) if scale_z else None,
        "rotation_deg": angle_deg,
        "samples": int(len(residuals)),
        "residual_px_mean": float(residuals.mean()),
        "residual_px_max": float(residuals.max()),
        "residual_px_p95": float(np.percentile(residuals, 95)),
    }
