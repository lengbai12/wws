#!/usr/bin/env python3
"""Normalize one observed route with the exact stored world-to-frame affine."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from fsutil import sha256


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROUTE = ROOT / 'replays/parsed/routes/236374.json'
DEFAULT_PROJECTION = ROOT / 'replays/parsed/calibration/minimap-s10_USS_CL-720x723.json'
DEFAULT_POSITIONS = ROOT / 'replays/parsed/position-236374.jsonl'
PATTERN = 'reference_236374'


def build_reference(route_path=DEFAULT_ROUTE, projection_path=DEFAULT_PROJECTION,
                    positions_path=DEFAULT_POSITIONS):
    route = json.loads(Path(route_path).read_text())
    projection = json.loads(Path(projection_path).read_text())
    with Path(positions_path).open() as handle:
        identity = json.loads(next(handle))
    if (route.get('session') != '236374' or route.get('map') != 's10_USS_CL'
            or identity.get('map') != route['map']
            or identity.get('recording_vehicle') != 'PGSB210-Schlieffen'):
        raise ValueError('unsupported reference identity')
    rect = projection['minimap_rect']
    if rect != [1836, 796, 720, 723] or projection['resolution'] != [2560, 1600]:
        raise ValueError('unsupported projection geometry')
    affine = projection['world_to_pixel']
    matrix, translation = affine['matrix'], affine['translation']
    if len(matrix) != 2 or any(len(row) != 2 for row in matrix) or len(translation) != 2:
        raise ValueError('invalid affine shape')
    if not all(math.isfinite(v) for row in matrix for v in row) or not all(math.isfinite(v) for v in translation):
        raise ValueError('nonfinite affine')
    points = []
    distance = 0.0
    for index, row in enumerate(route['route']):
        x, z, clock = row['x'], row['z'], row['t']
        if not all(math.isfinite(v) for v in (x, z, clock)):
            raise ValueError('nonfinite source waypoint')
        px = matrix[0][0] * x + matrix[0][1] * z + translation[0] - rect[0]
        py = matrix[1][0] * x + matrix[1][1] * z + translation[1] - rect[1]
        if not (0 <= px < rect[2] and 0 <= py < rect[3]):
            raise ValueError('reference leaves calibrated panel')
        if points:
            step = math.hypot(px - points[-1]['x'], py - points[-1]['y'])
            if step < 1e-9:
                continue
            distance += step
        points.append(dict(x=px, y=py, s_px=distance, source_index=index,
                           source_clock_s=clock))
    if len(points) < 2 or not math.isfinite(distance):
        raise ValueError('reference has no finite path')
    return dict(schema_version=1, map=route['map'], ship='PGSB210-Schlieffen',
                pattern=PATTERN, session='236374', reference_sha256=sha256(route_path),
                projection_sha256=sha256(projection_path),
                identity_sha256=sha256(positions_path),
                source_coordinate=dict(axes=['x', 'z'], units='replay_world_units',
                                       clock='route-relative seconds; provenance only'),
                coordinate=dict(space='minimap_crop', units='pixels', rect=rect,
                                frame_shape_rgb=[1600, 2560, 3]),
                affine=affine, length_px=distance, waypoints=points,
                assurance='Observed reference only; no island/collision clearance guarantee')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--route', type=Path, default=DEFAULT_ROUTE)
    parser.add_argument('--projection', type=Path, default=DEFAULT_PROJECTION)
    parser.add_argument('--positions', type=Path, default=DEFAULT_POSITIONS)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    text = json.dumps(build_reference(args.route, args.projection, args.positions), allow_nan=False, indent=2) + '\n'
    if args.out:
        args.out.write_text(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
