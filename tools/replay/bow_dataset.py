#!/usr/bin/env python3
"""Validate reviewed, auditable 256x256 bow annotations with truthful sources.

Positive rows have a binary hull mask and visible bow/stern points in crop
coordinates. ``orientation_bin`` is the OBSERVED CAMERA azimuth relative to the
ship, not the angle of the bow/stern line in the image: 0 bow-on, 4 starboard
(ship's right side), 8 stern-on, 12 port (ship's left side). Bin k is centered
at k*22.5 degrees and covers [center-11.25, center+11.25), wrapping at 360.
Missing orientation or ``"unknown"`` never supplies known-view supervision.
A 2D image rotation does not change or establish a 3D camera viewpoint.

Negative rows explicitly reject invisible, occluded, ambiguous, or absent ships:
mask is all zero, both points null, orientation unknown. They must not silently
become positive keypoint/orientation targets. All rows require reviewed=true,
annotation_source="manual" or "assistant_visual", reviewer, session, source_image, source_roi=[x,y,w,h],
source_sha256, image_sha256. Hashes identify decoded BGR pixels, not file bytes.
The saved crop must exactly equal the source ROI padded with black on its bottom
and right to a square, then INTER_AREA resized to 256. Source records distinguish
human and assistant visual review; validation checks consistency, not judgement.
Session-count and train/validation/test requirements belong to the trainer.
Use ``annotate_bow.py --help`` to create real labels offline; nothing here
generates bow labels from existing hull masks or infers viewpoint from 2D.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

SIZE = 256
ORIENTATION_BINS = 16
ORIENTATION_CONVENTION = (
    "camera-relative azimuth; 0=bow-on; 4=starboard (ship right); "
    "8=stern-on; 12=port (ship left); centers k*22.5deg; "
    "intervals [center-11.25,center+11.25) modulo 360"
)
PIXEL_HASH_CONVENTION = "sha256 decoded BGR uint8 row-major bytes"
POINT_MASK_TOLERANCE = 6.0
REJECTION_REASONS = ("invisible", "occluded", "ambiguous", "no_ship")
REQUIRED = (
    "image", "mask", "bow_point", "stern_point", "session", "label_status",
    "reviewed", "annotation_source", "reviewer", "source_image", "source_roi",
    "source_sha256", "image_sha256", "crop_transform",
)
CROP_CONVENTION = "source-roi; square-pad-black-bottom-right; resize-256-INTER_AREA"


def pixel_sha256(image: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()

def _point(value, name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be [x,y]")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value):
        raise ValueError(f"{name} coordinates must be finite numbers, not bool")
    point = [float(v) for v in value]
    if not all(math.isfinite(v) and 0 <= v < SIZE for v in point):
        raise ValueError(f"{name} outside finite 256x256 coordinates: {point}")
    return point


def source_crop(source: np.ndarray, roi) -> np.ndarray:
    """Apply the inference crop transform, preserving the ROI's top-left origin."""
    x, y, width, height = roi
    side = max(width, height)
    square = np.zeros((side, side, 3), dtype=np.uint8)
    square[:height, :width] = source[y:y + height, x:x + width]
    return cv2.resize(square, (SIZE, SIZE), interpolation=cv2.INTER_AREA)

def read_source(path):
    """Read saved native BGR/BGRA bytes; ignore alpha, never composite pixels."""
    source = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if (source is None or source.dtype != np.uint8 or source.ndim != 3
            or source.shape[2] not in (3, 4)):
        raise ValueError(f'source must be readable uint8 BGR/BGRA: {path}')
    return np.ascontiguousarray(source[:, :, :3])


def _text(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _read_image(root: Path, value, name: str) -> tuple[Path, np.ndarray]:
    path = root / _text(value, name)
    if not path.is_file():
        raise ValueError(f"{name} path does not exist: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{name} must be a readable uint8 three-channel image: {path}")
    return path, image


def _supported(point: list[float], mask: np.ndarray) -> bool:
    y, x = np.nonzero(mask)
    return bool(np.any((x - point[0]) ** 2 + (y - point[1]) ** 2 <= POINT_MASK_TOLERANCE ** 2))


def validate_row(row: dict, root: Path) -> dict:
    if not isinstance(row, dict):
        raise ValueError("annotation row must be an object")
    missing = [key for key in REQUIRED if key not in row]
    if missing:
        raise ValueError(f"missing fields: {','.join(missing)}")
    if row["reviewed"] is not True or row["annotation_source"] not in ("manual", "assistant_visual"):
        raise ValueError("reviewed must be true and annotation_source must be manual or assistant_visual")
    _text(row["reviewer"], "reviewer")
    _text(row["session"], "session")
    image_path, image = _read_image(root, row["image"], "image")
    if image.shape[:2] != (SIZE, SIZE):
        raise ValueError(f"image must be readable 256x256: {image_path}")
    mask_path = root / _text(row["mask"], "mask")
    if not mask_path.is_file():
        raise ValueError(f"mask path does not exist: {mask_path}")
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None or mask.dtype != np.uint8 or mask.shape != (SIZE, SIZE):
        raise ValueError(f"mask must be readable uint8 single-channel 256x256: {mask_path}")
    if not np.isin(mask, (0, 255)).all():
        raise ValueError("mask pixels must be binary 0 or 255")
    _, source = _read_image(root, row["source_image"], "source_image")
    roi = row["source_roi"]
    if (not isinstance(roi, (list, tuple)) or len(roi) != 4
            or any(isinstance(v, bool) or not isinstance(v, int) for v in roi)):
        raise ValueError("source_roi must be integer [x,y,width,height], not bool")
    x, y, width, height = roi
    if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > source.shape[1] or y + height > source.shape[0]:
        raise ValueError("source_roi outside source image")
    if row["source_sha256"] != pixel_sha256(source):
        raise ValueError("source_sha256 does not match source pixels")
    if row["image_sha256"] != pixel_sha256(image):
        raise ValueError("image_sha256 does not match crop pixels")
    if row["crop_transform"] != CROP_CONVENTION:
        raise ValueError("unsupported crop_transform")
    expected = source_crop(source, roi)
    if not np.array_equal(image, expected):
        raise ValueError("image pixels do not match source_image/source_roi crop")
    orientation = row.get("orientation_bin", "unknown")
    if not (orientation == "unknown" or (
            not isinstance(orientation, bool) and isinstance(orientation, int)
            and 0 <= orientation < ORIENTATION_BINS)):
        raise ValueError("orientation_bin must be integer 0..15 or 'unknown'")
    status = row["label_status"]
    if status == "negative":
        if row["bow_point"] is not None or row["stern_point"] is not None or np.any(mask):
            raise ValueError("negative requires null bow/stern points and all-zero mask")
        if orientation != "unknown":
            raise ValueError("negative orientation_bin must be unknown")
        if row.get("rejection_reason") not in REJECTION_REASONS:
            raise ValueError(f"negative rejection_reason must be one of {REJECTION_REASONS}")
        bow = stern = None
    elif status == "positive":
        if row.get("rejection_reason") is not None:
            raise ValueError("positive must not have rejection_reason")
        bow = _point(row["bow_point"], "bow_point")
        stern = _point(row["stern_point"], "stern_point")
        if bow == stern:
            raise ValueError("bow_point and stern_point must differ")
        if int(np.count_nonzero(mask)) < 32:
            raise ValueError("mask has fewer than 32 positive pixels")
        for name, point in (("bow_point", bow), ("stern_point", stern)):
            if not _supported(point, mask):
                raise ValueError(f"{name} unsupported by mask within {POINT_MASK_TOLERANCE:g}px")
    else:
        raise ValueError("label_status must be positive or negative")
    return dict(row, bow_point=bow, stern_point=stern, orientation_bin=orientation)


def load_annotations(path: Path) -> list[dict]:
    path = Path(path)
    rows = []
    crops = set()
    source_sessions = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"cannot read annotations {path}: {error}") from error
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = validate_row(json.loads(line), path.parent)
            identity = row["image_sha256"]
            if identity in crops:
                raise ValueError("duplicate crop pixel identity")
            source = row["source_sha256"]
            if source in source_sessions and source_sessions[source] != row["session"]:
                raise ValueError("same source frame pixels assigned to multiple sessions")
            crops.add(identity)
            source_sessions[source] = row["session"]
            rows.append(row)
        except (ValueError, TypeError, OSError) as error:
            raise ValueError(f"{path}:{number}: {error}") from error
    if not rows:
        raise ValueError(f"no annotations in {path}")
    return rows


def annotation_summary(rows: list[dict]) -> dict:
    counts = [0] * ORIENTATION_BINS
    unknown = negative = 0
    for row in rows:
        if row["label_status"] == "negative":
            negative += 1
        elif row["orientation_bin"] == "unknown":
            unknown += 1
        else:
            counts[row["orientation_bin"]] += 1
    return {
        "samples": len(rows), "positive_samples": len(rows) - negative,
        "negative_samples": negative, "unknown_orientation_positives": unknown,
        "annotation_source_counts": {source: sum(row["annotation_source"] == source for row in rows)
                                     for source in sorted({row["annotation_source"] for row in rows})},
        "sessions": sorted({row["session"] for row in rows}),
        "orientation_bins": [i for i, count in enumerate(counts) if count],
        "orientation_bin_counts": counts,
        "missing_orientation_bins": [i for i, count in enumerate(counts) if not count],
        "all_16_bins_observed": all(counts),
        "orientation_convention": ORIENTATION_CONVENTION,
    }


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path)
    args = parser.parse_args(argv)
    try:
        rows = load_annotations(args.annotations)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(dict(status="valid", **annotation_summary(rows)), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
