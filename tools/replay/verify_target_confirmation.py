#!/usr/bin/env python3
"""Inspect target-lock evidence in an actual native RGB screenshot, without input.

Example:
  python tools/replay/verify_target_confirmation.py \
      --image state/aimprobe-live-scope/key-action.png --out state/lock-key-action

The JSON report and annotated PNG preserve native frame coordinates. Confirmation
means one white ring associated with one detected health bar; it does not confirm
ship identity, hull visibility, target motion, or permission to fire.
"""
from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from main_view_perception import FRAME_H, FRAME_W, detect_nameplates, verify_ship_bodies
from target_confirmation import confirm_target


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def annotate(rgb, detections, lock):
    out = rgb.copy()
    for index, detection in enumerate(detections):
        x, y, w, h = detection['nameplate_bbox']
        color = (80, 255, 80) if lock['detection_index'] == index else (255, 190, 60)
        cv2.rectangle(out, (x, y), (x+w-1, y+h-1), color, 2)
        cv2.putText(out, f'bar {index}', (x, y+h+18), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
        if detection.get('body_bbox'):
            bx, by, bw, bh = detection['body_bbox']
            cv2.rectangle(out, (bx, by), (bx+bw-1, by+bh-1), (255, 120, 120), 1)
    for index, ring in enumerate(lock['candidates']):
        x, y, w, h = ring['bbox']
        cv2.rectangle(out, (x, y), (x+w-1, y+h-1), (100, 220, 255), 1)
        compatible = ','.join(map(str, ring['compatible_detection_indices'])) or 'none'
        cv2.putText(out, f'ring {index} -> bars {compatible}', (x, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (100, 220, 255), 1)
    cv2.putText(out, f"UI lock: {lock['state']} | identity unknown | hull unverified",
                (560, 285), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 100), 2)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True, help='actual native 2560x1600 screenshot')
    parser.add_argument('--out', required=True, help='new output directory')
    args = parser.parse_args(argv)
    source = Path(args.image)
    source_bytes = source.read_bytes()
    with Image.open(BytesIO(source_bytes)) as image:
        if image.size != (FRAME_W, FRAME_H):
            parser.error(f'--image must be native {FRAME_W}x{FRAME_H}; resizing is not allowed')
        rgb = np.asarray(image.convert('RGB'))
    detections = verify_ship_bodies(rgb, detect_nameplates(rgb))
    lock = confirm_target(rgb, detections)
    implementation = {}
    for name in ('target_confirmation.py', 'main_view_perception.py', 'verify_target_confirmation.py'):
        path = Path(__file__).with_name(name)
        implementation[name] = _sha256(path.read_bytes())
    implementation_bytes = json.dumps(implementation, sort_keys=True, separators=(',', ':')).encode()
    report = {
        'source': {'path': str(source.resolve()), 'sha256': _sha256(source_bytes),
                   'rgb_sha256': _sha256(rgb.tobytes()), 'frame_shape': list(rgb.shape)},
        'implementation': {'sha256': _sha256(implementation_bytes), 'files': implementation,
                           'opencv_version': cv2.__version__, 'numpy_version': np.__version__},
        'coordinate_system': 'native_frame_xy_pixels', 'bbox_format': '[x,y,width,height]',
        'detections': detections, 'lock_confirmation': lock,
        'preview': 'annotated.png',
        'evidence': {'replay_truth_used': False, 'identity_input_used': False,
                     'game_input_sent': False, 'ring_bbox': 'fitted_circle_extent_not_hull'},
        'limitations': [
            'Unique means unique among detected health bars, not a guarantee of detector completeness.',
            'No ship-name OCR or replay identity association; visible UI text is not an identity input.',
            'White-circle support is a shape heuristic, not a calibrated probability or native-sprite proof.',
            'Ring evidence confirms UI acquisition only; silhouette candidates remain unverified.',
            'A still image supplies neither stable-camera velocity nor world/mouse calibration or firing permission.',
        ],
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    Image.fromarray(annotate(rgb, detections, lock)).save(out / 'annotated.png')
    (out / 'observations.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'out': str(out), 'source_sha256': report['source']['sha256'],
                      'implementation_sha256': report['implementation']['sha256'],
                      'detections': len(detections), 'lock_confirmation': lock}, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
