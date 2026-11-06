#!/usr/bin/env python3
"""Draw what the recognition chain believes, box by box, with its identity label.

The game runs inside gamescope, whose compositor keeps the game window above
every other client on its X server (measured 2026-10-23: an override-redirect,
shaped, ABOVE-hinted window stays below the game in the stacking order), so a
transparent overlay cannot be drawn *inside* the game frame. What this tool
produces instead is the same drawing on the captured frame: a live annotated
view plus a JSONL record, so identity claims and their accuracy can be inspected
directly rather than inferred from counts.

Nothing here decides anything. It reuses the perception modules the controller
uses and only annotates: a box labelled ENEMY means "the plate detector fired
with this confidence", not "this is an enemy".
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
import sys
for _extra in (ROOT / 'tools/replay', ROOT / 'tools/capture'):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))
MINIMAP_RECT = (1836, 796, 720, 723)
MINIMAP_KM_PER_PX = 0.0501  # measured against the printed 11.0 / 12.6 rings

COLOURS = {
    'enemy_plate': (255, 60, 60),
    'enemy_body': (190, 140, 60),   # the detector's search ROI, not an aim point
    'own_ship': (90, 255, 120),
    'contact': (255, 120, 255),
    'hazard': (60, 200, 255),
    'aim': (255, 255, 255),
}


def font(size=16):
    for name in ('DejaVuSans-Bold.ttf', 'DejaVuSans.ttf'):
        for base in ('/usr/share/fonts/TTF', '/usr/share/fonts/truetype/dejavu'):
            path = Path(base) / name
            if path.is_file():
                return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def grab(path):
    import subprocess
    subprocess.run(['/usr/bin/python3', str(ROOT / 'tools/capture/capture_host.py'),
                    'snapshot', str(path)], stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, timeout=30)
    image = Image.open(path).convert('RGB')
    return np.asarray(image)


def label(draw, xy, text, colour, fnt):
    x, y = xy
    box = draw.textbbox((0, 0), text, font=fnt)
    w, h = box[2] - box[0], box[3] - box[1]
    draw.rectangle([x, y - h - 6, x + w + 8, y + 2], fill=(18, 18, 18))
    draw.text((x + 4, y - h - 4), text, fill=colour, font=fnt)


def detect(frame, want_hazards=False):
    """The identity claims for one frame, as plain records."""
    from main_view_perception import detect_nameplates
    import aim_servo
    rows = []
    plates = detect_nameplates(frame)
    # The aim point is shown because "is the aim on the waterline" is exactly what
    # the operator judges from this window; without the marker the box they see is
    # only the detector's body ROI.
    best = max(plates, key=lambda p: float(p.get('confidence') or 0), default=None)
    if best is not None:
        try:
            point, meta = aim_servo.hull_aim_point(frame, best['nameplate_bbox'])
            rows.append(dict(kind='aim', label='AIM', confidence=1.0,
                             pixel=[float(point[0]), float(point[1])],
                             aim_source=meta.get('source'),
                             waterline_row=meta.get('waterline_row')))
        except (ValueError, OSError, KeyError):
            pass
    for plate in plates:
        rows.append(dict(kind='enemy_plate', bbox=list(plate['nameplate_bbox']),
                         label='ENEMY', confidence=round(float(plate.get('confidence') or 0), 3),
                         body_bbox=(list(plate['body_roi']) if plate.get('body_roi') else None),
                         body_visible=plate.get('body_visible')))
    if want_hazards:
        try:
            import torpedo_perception as TP
            for mark in TP.detect_torpedoes(frame) if hasattr(TP, 'detect_torpedoes') else []:
                box = mark.get('bbox') if isinstance(mark, dict) else None
                if box:
                    rows.append(dict(kind='hazard', bbox=list(box),
                                     label=str((mark or {}).get('type') or 'TORPEDO'),
                                     confidence=round(float((mark or {}).get('confidence') or 0), 3)))
        except Exception:
            pass
    return rows


def detect_minimap(frame):
    """Own ship and contacts from the trained minimap model, in panel pixels."""
    from minimap_recognizer import MinimapRecognizer
    x, y, w, h = MINIMAP_RECT
    crop = np.ascontiguousarray(frame[y:y + h, x:x + w])
    package = ROOT / 'replays/parsed/artifacts/enemy-marker-22s-v1/package.json'
    seen = MinimapRecognizer(str(package)).detect(crop)
    rows = []
    own = seen.get('own')
    if own:
        rows.append(dict(kind='own_ship', label='OWN',
                         confidence=round(float(own.get('score') or 0), 3),
                         pixel=[x + float(own['pixel_x']), y + float(own['pixel_y'])],
                         heading_deg=own.get('angle_deg')))
    own_pixel = None
    if own:
        own_pixel = [float(own['pixel_x']), float(own['pixel_y'])]
    for contact in seen.get('enemies') or []:
        u, v = contact.get('u'), contact.get('v')
        if u is None or v is None:
            continue
        pixel = [x + u * (w - 1), y + v * (h - 1)]
        row = dict(kind='contact', label=str(contact.get('species') or 'unknown').upper(),
                   confidence=round(float(contact.get('score') or 0), 3), pixel=pixel)
        if own_pixel is not None:
            # Range makes an identity claim checkable: "contact 13.5 km away" can be
            # compared with the ship actually visible at that range.
            row['km'] = round(math.hypot(u * (w - 1) - own_pixel[0],
                                         v * (h - 1) - own_pixel[1]) * MINIMAP_KM_PER_PX, 1)
        rows.append(row)
    return rows


EXPAND = 7


def annotate(frame, rows, fnt, counts, frame_index, battle=None):
    """One annotated frame. Claims outside a battle are drawn as raw detector fires."""
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    in_battle = battle is True
    for row in rows:
        colour = COLOURS.get(row['kind'], (255, 255, 255))
        if not in_battle:
            colour = (150, 150, 150)
        prefix = '' if in_battle else 'RAW '
        if 'bbox' in row and row['bbox']:
            x, y, w, h = [int(v) for v in row['bbox']]
            # A plate is 7 px tall; at viewer scale an exact outline is invisible,
            # so the annotation is padded rather than the detection changed.
            draw.rectangle([x - EXPAND, y - EXPAND, x + w + EXPAND, y + h + EXPAND],
                           outline=colour, width=5)
            label(draw, (x - EXPAND, y - EXPAND), f"{prefix}{row['label']} {row['confidence']:.2f}",
                  colour, fnt)
            body = row.get('body_bbox')
            if body:
                bx, by, bw, bh = [int(v) for v in body]
                # Thin and dim: this is the detector's search window, and drawing
                # it as strongly as the aim marker made it look like the aim.
                draw.rectangle([bx, by, bx + bw, by + bh],
                               outline=COLOURS['enemy_body'], width=1)
        elif 'pixel' in row and row.get('kind') == 'aim':
            px, py = row['pixel']
            draw.line([px - 26, py, px + 26, py], fill=colour, width=3)
            draw.line([px, py - 26, px, py + 26], fill=colour, width=3)
            draw.ellipse([px - 9, py - 9, px + 9, py + 9], outline=colour, width=3)
            text = f"AIM {row.get('aim_source') or ''}".strip()
            if row.get('waterline_row'):
                text += f" wl {row['waterline_row']}"
            label(draw, (px + 30, py + 8), text, colour, fnt)
        elif 'pixel' in row:
            px, py = row['pixel']
            r = 18
            draw.ellipse([px - r, py - r, px + r, py + r], outline=colour, width=4)
            text = f"{prefix}{row['label']} {row['confidence']:.2f}"
            if row.get('km') is not None:
                text += f' {row["km"]:.1f}km'
            label(draw, (px + r + 2, py - r), text, colour, fnt)
    header = ' | '.join(f'{k} {v}' for k, v in sorted(counts.items())) or 'no claims'
    if in_battle is False:
        header += '   [NOT IN BATTLE: raw detector output, not an identity claim]'
    label(draw, (24, 64), f'frame {frame_index}  {header}', (255, 255, 255), fnt)
    return image


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True, help='directory for latest.png and detections.jsonl')
    ap.add_argument('--seconds', type=float, default=60.0,
                    help='0 runs until killed')
    ap.add_argument('--interval', type=float, default=0.4)
    ap.add_argument('--record-every', type=int, default=0,
                    help='also keep every Nth annotated frame (0: only latest.png)')
    ap.add_argument('--minimap', dest='minimap', action='store_true', default=True)
    ap.add_argument('--no-minimap', dest='minimap', action='store_false')
    args = ap.parse_args()

    out = ROOT / args.out if not str(args.out).startswith('/') else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fnt = font(22)
    started = time.monotonic()
    index = 0
    counts = {}
    log = (out / 'detections.jsonl').open('a')
    try:
        # 0 means "until killed": the window is meant to stay up for a session.
        while args.seconds <= 0 or time.monotonic() - started < args.seconds:
            index += 1
            frame = grab(out / 'raw.png')
            battle = None
            try:
                import fire_hud
                battle = (fire_hud.FireHudReader().detect(frame).get('battle') or {}).get('active')
            except (ValueError, OSError, KeyError):
                battle = None
            rows = detect(frame)
            if args.minimap:
                try:
                    rows += detect_minimap(frame)
                except (ValueError, OSError, KeyError):
                    pass
            counts = {}
            for row in rows:
                counts[row['label']] = counts.get(row['label'], 0) + 1
            image = annotate(frame, rows, fnt, counts, index, battle=battle)
            # Write-then-rename: a reader polling latest.png must never load a
            # half-written file (observed as "image file is truncated").
            temporary = out / 'latest.tmp.png'
            image.save(temporary)
            os.replace(temporary, out / 'latest.png')
            # A compact feed for the layer-shell overlay: it must not decode a
            # 3 MB PNG five times a second.
            boxes_tmp = out / 'boxes.tmp.json'
            boxes_tmp.write_text(json.dumps(dict(frame=index, battle=battle,
                                                 captured_at=time.time(),
                                                 source_size=[frame.shape[1], frame.shape[0]],
                                                 claims=rows), default=str))
            os.replace(boxes_tmp, out / 'boxes.json')
            if args.record_every and index % args.record_every == 0:
                image.save(out / f'frame-{index:05d}.png')
            log.write(json.dumps(dict(frame=index, t_monotonic=round(time.monotonic() - started, 2),
                                      battle=battle, claims=rows), default=str) + '\n')
            log.flush()
            print(f"overlay-frame {index} battle={battle} claims={counts}", flush=True)
            time.sleep(max(0.0, args.interval))
    except KeyboardInterrupt:
        pass
    finally:
        log.close()
    print(json.dumps(dict(frames=index, last_counts=counts, out=str(out))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
