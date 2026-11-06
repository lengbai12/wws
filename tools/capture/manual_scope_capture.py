#!/home/lengbai/Downloads/wws/.venv/bin/python
"""Capture frames while the operator aims: scope + enemy blood bar required.

This tool sends NO input at all. It polls the live game, and only when the
picture is in binocular scope *and* the project's blood-bar coarse aim finds an
enemy nameplate does it save the native 2560x1600 frame plus a target crop.

Each record stores the measured minimap geometry (own ship, nearest enemy,
bearing of the own ship around that enemy, radius) so 5-degree coverage can be
audited afterwards; identical frames and shots that are too close in angle to
the previous one are skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
PY = str(ROOT / '.venv/bin/python')
CAPTURE_HOST = str(ROOT / 'tools/capture/capture_host.py')
PACKAGE = ROOT / 'replays/parsed/artifacts/enemy-marker-22s-v1/package.json'
MINIMAP = (1836, 796, 720, 723)
FRAME = (2560, 1600)


class CaptureError(RuntimeError):
    pass


def grab(path):
    result = subprocess.run([PY, CAPTURE_HOST, 'grab', str(path)],
                            capture_output=True, text=True, timeout=20)
    if result.returncode:
        raise CaptureError(f'host grab failed: {result.stderr.strip()}')
    image = np.asarray(Image.open(path).convert('RGB'))
    if image.shape[:2] != (FRAME[1], FRAME[0]):
        raise CaptureError(f'unexpected frame shape {image.shape}')
    return image


def scope_visual(frame):
    h, w = frame.shape[:2]
    top, bottom = (80, h - 80) if h == 1600 else (0, h)
    inner = bottom - top
    py, px = max(8, int(inner * 0.06)), max(8, int(w * 0.05))
    corners = [frame[top:top + py, 0:px], frame[top:top + py, w - px:],
               frame[bottom - py:bottom, 0:px], frame[bottom - py:bottom, w - px:]]
    centre = frame[top + int(inner * 0.45):top + int(inner * 0.55),
                   int(w * 0.45):int(w * 0.55)]
    corner_mean = float(np.mean([c.mean() for c in corners]))
    centre_mean = float(centre.mean())
    return dict(corner_mean=round(corner_mean, 1), centre_mean=round(centre_mean, 1),
                scope_like=bool(corner_mean < 45 and centre_mean > corner_mean * 1.4))


def coarse_target(frame, scope=True):
    """Any enemy blood bar (red nameplate) in this frame; hull checks are not required.

    The operator aims manually, so capture must trigger on the bar itself even
    when the hull verifier cannot see a body (bright water, partial occlusion).
    """
    import sys
    sys.path[:0] = [str(ROOT / 'tools/replay'), str(ROOT / 'tools/capture')]
    from main_view_perception import detect_nameplates
    plates = detect_nameplates(frame, scope_observed=scope)
    if not plates:
        return None
    best = min(plates, key=lambda p: abs(p['nameplate_bbox'][0] + p['nameplate_bbox'][2] / 2 - 1280))
    x, y, w, h = best['nameplate_bbox']
    return dict(bbox=[x, y, w, h], aim_px=[x + w / 2, y + h / 2],
                source='nameplate_healthbar', count=len(plates))


def geometry(frame, recognizer):
    x, y, w, h = MINIMAP
    out = recognizer.detect(frame[y:y + h, x:x + w])
    own = out.get('own')
    enemies = out.get('enemies') or []
    if not own:
        return None
    own_px = [own['u'] * w, own['v'] * h]
    if not enemies:
        return dict(own=own_px, enemy=None, bearing=None, radius=None)
    enemy = min(enemies, key=lambda e: (e['u'] - own['u']) ** 2 + (e['v'] - own['v']) ** 2)
    enemy_px = [enemy['u'] * w, enemy['v'] * h]
    dx, dy = own_px[0] - enemy_px[0], own_px[1] - enemy_px[1]
    return dict(own=own_px, enemy=enemy_px,
                bearing=(math.degrees(math.atan2(dx, -dy)) + 360.0) % 360.0,
                radius=math.hypot(dx, dy))


def angle_gap(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _tick(report, args, every=10.0):
    """Persist progress so a waiting operator can see the counters."""
    now = time.monotonic()
    if now - report.get('_last_write', 0.0) >= every:
        report['_last_write'] = now
        (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


def run(args):
    args.out = Path(args.out).resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    import sys
    sys.path[:0] = [str(ROOT / 'tools/replay'), str(ROOT / 'tools/capture')]
    from minimap_recognizer import MinimapRecognizer
    from bow_recognizer import crop_box

    recognizer = MinimapRecognizer(str(PACKAGE))
    report = dict(action='manual_scope_capture', out=str(args.out), status='running',
                  shots=[], skipped=dict(no_scope=0, no_target=0, duplicate=0, angle_gap=0),
                  input_sent=False, fired=False,
                  semantics='capture only; the operator aims and scopes manually')
    live = args.out / 'live.ppm'
    started = time.monotonic()
    deadline = started + args.seconds
    last_hash = None
    last_angle = None
    handlers = {}

    def stop(signum, _frame):
        raise CaptureError(f'interrupted by signal {signum}')

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, stop)
        while time.monotonic() < deadline and len(report['shots']) < args.shots:
            frame = grab(live)
            digest = hashlib.sha256(frame.tobytes()).hexdigest()
            visual = scope_visual(frame)
            if args.require_scope and not visual['scope_like']:
                report['skipped']['no_scope'] += 1
                _tick(report, args)
                time.sleep(args.poll)
                continue
            target = coarse_target(frame, scope=True)
            if target is None:
                report['skipped']['no_target'] += 1
                _tick(report, args)
                time.sleep(args.poll)
                continue
            info = geometry(frame, recognizer)
            if digest == last_hash:
                report['skipped']['duplicate'] += 1
                time.sleep(args.poll)
                continue
            if (info and info['bearing'] is not None and last_angle is not None
                    and angle_gap(info['bearing'], last_angle) < args.min_angle):
                report['skipped']['angle_gap'] += 1
                time.sleep(args.poll)
                continue
            index = len(report['shots']) + 1
            path = args.out / f'shot-{index:03d}.png'
            Image.fromarray(frame).save(path)
            x0, y0, x1, y1 = crop_box(target['bbox'], frame.shape)
            crop_path = None
            if min(x1 - x0, y1 - y0) >= 16:
                crop_path = args.out / f'shot-{index:03d}-target.png'
                Image.fromarray(frame[y0:y1, x0:x1]).save(crop_path)
            shot = dict(index=index, file=str(path.relative_to(ROOT)),
                        target_crop=None if crop_path is None else str(crop_path.relative_to(ROOT)),
                        target_bbox=target['bbox'], target_aim_px=target['aim_px'],
                        nameplate_count=target['count'],
                        scope_visual=visual, pixel_sha256=digest,
                        bearing=None if not info or info['bearing'] is None else round(info['bearing'], 2),
                        radius_px=None if not info or info['radius'] is None else round(info['radius'], 1),
                        own=None if not info else [round(v, 1) for v in info['own']],
                        enemy=None if not info or info['enemy'] is None else [round(v, 1) for v in info['enemy']],
                        t=round(time.monotonic() - started, 1))
            report['shots'].append(shot)
            last_hash, last_angle = digest, shot['bearing']
            (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps({'shot': index, 'bearing': shot['bearing'],
                              'radius_px': shot['radius_px']}), flush=True)
            time.sleep(args.poll)
        report['status'] = 'complete' if len(report['shots']) >= args.shots else 'partial'
    except Exception as error:
        report.update(status='failed', error=str(error))
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        report['elapsed_s'] = round(time.monotonic() - started, 1)
        report['angles_seen'] = sorted({s['bearing'] for s in report['shots'] if s['bearing'] is not None})
        (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        live.unlink(missing_ok=True)
    print(json.dumps({'status': report['status'], 'shots': len(report['shots']),
                      'skipped': report['skipped'], 'elapsed_s': report['elapsed_s']},
                     ensure_ascii=False))
    return 0 if report['shots'] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--shots', type=int, default=72)
    parser.add_argument('--seconds', type=float, default=1500.)
    parser.add_argument('--poll', type=float, default=0.6)
    parser.add_argument('--require-scope', action='store_true',
                        help='only capture verified binocular frames; default records the state')
    parser.add_argument('--min-angle', type=float, default=2.0,
                        help='skip shots closer than this bearing change (deg)')
    args = parser.parse_args(argv)
    if not 1 <= args.shots <= 500:
        parser.error('shots must be 1..500')
    if not 10 <= args.seconds <= 3600:
        parser.error('seconds must be 10..3600')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
