#!/usr/bin/env python3
"""Report what the current perception stack can actually see, frame by frame.

Runs the existing consumers on saved native frames and prints one row per
frame: the minimap recogniser (own ship plus enemy markers), the battle HUD
gate, the red enemy nameplate detector, and the torpedo/aircraft hazard
detectors. Nothing here injects input; it is a capability audit, not a claim
that the pipeline can fight.
"""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'tools/replay'))
sys.path.insert(0, str(ROOT / 'tools/capture'))

import numpy as np
import cv2

MINIMAP_RECT = (1836, 796, 720, 723)


def minimap_crop(frame, rect=MINIMAP_RECT):
    x, y, width, height = rect
    return np.ascontiguousarray(frame[y:y + height, x:x + width])


def scan(path, package, use_minimap=True):
    image = cv2.imread(str(path))
    if image is None or image.shape != (1600, 2560, 3):
        return dict(frame=Path(path).name, error='unreadable or not native 2560x1600')
    rgb = np.ascontiguousarray(image[:, :, ::-1])
    row = dict(frame=Path(path).name)
    from fire_hud import FireHudReader
    hud = FireHudReader().detect(rgb)
    row['battle_hud'] = hud['battle']['active']
    row['weapon'] = hud['weapon'].get('kind')
    row['reload_ready'] = hud['reload'].get('ready')
    row['reticle'] = None if hud['reticle'].get('center') is None else 'observed'
    row['aiming_view'] = (hud.get('aiming_view') or {}).get('status')
    from main_view_perception import detect_nameplates, verify_ship_bodies
    plates = verify_ship_bodies(rgb, detect_nameplates(rgb))
    row['enemy_nameplates'] = len(plates)
    from torpedo_perception import detect as detect_torpedoes
    from aircraft_perception import detect as detect_aircraft
    row['torpedo_markers'] = len(detect_torpedoes(rgb) or [])
    row['aircraft_markers'] = len(detect_aircraft(rgb) or [])
    if use_minimap:
        from minimap_recognizer import MinimapRecognizer
        recogniser = MinimapRecognizer(str(package))
        seen = recogniser.detect(minimap_crop(rgb))
        own = seen.get('own')
        # The locator reports a normalised u/v, a pixel and the arrow angle, not
        # a "position" key; an earlier audit printed null because of that.
        row['minimap_own'] = None if not own else dict(
            pixel=[own.get('pixel_x'), own.get('pixel_y')],
            uv=[round(float(own.get('u', 0)), 4), round(float(own.get('v', 0)), 4)],
            heading_deg=own.get('angle_deg'),
            score=round(float(own.get('score', 0)), 3),
            margin=round(float(own.get('margin', 0)), 3))
        row['minimap_enemies'] = len(seen.get('enemies') or [])
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('frames', nargs='+', type=Path)
    parser.add_argument('--package', type=Path,
                        default=ROOT / 'replays/parsed/artifacts/enemy-marker-22s-v1/package.json')
    parser.add_argument('--no-minimap', action='store_true')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    rows = [scan(path, args.package, not args.no_minimap) for path in args.frames]
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
