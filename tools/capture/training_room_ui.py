#!/usr/bin/env python3
"""Recognise the CN training-room menu chain and (only with --execute) click it.

The chain is: results --回到港口--> port --加入战斗--> lobby --已准备好->
lobby(ready) --加入战斗--> prebattle --开始战斗--> battle/loading.

Every button is found by edge-template matching in *window* pixels (the native
1924x1058 client frame), and every screen transition is decided by an explicit
pixel criterion that was measured on real frames (see
docs/training-room-restart.md).  A criterion that is not observable stops the
run; nothing is assumed to have been clicked successfully because a click was
sent.

Coordinates live in three spaces, which is the whole point of this module:

    content (2560x1440)  -- the project reference frame, where all documented
                           button points are written
    window  (WxH live)   -- the client area as reported by xdotool
    root    (logical px) -- what xdotool moves the pointer in

    window_x = round(content_x * W / 2560)
    window_y = round(content_y * H / 1440)
    root_x   = window_x + origin_x
    root_y   = window_y + origin_y

Read-only by default.  Input needs --execute and is one bounded click per
step: no keys, no held buttons, and a pointer read-back check before clicking.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import resolution_map

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'tools' / 'replay'))

import enter_training_battle as etb  # noqa: E402  (load_templates/detect/edges/teal)

CONTENT = (2560, 1440)
DISPLAY = ':2'

# Thresholds measured on real frames; margins are quoted in the doc.
READY_TEMPLATE_MIN = 0.45      # lobby 0.595..1.0 (banded); port 0.151, prebattle 0.0
START_TEMPLATE_MIN = 0.66      # prebattle 0.716..0.779; results false positive 0.538
JOIN_TEMPLATE_MIN = 0.70       # lobby 0.765..0.813; port 0.721
READY_GREEN_MIN = 200          # ready label turns bright green: 510..533 vs 0
PORT_PANEL_GREY = 0.15         # B 队面板区灰像素占比：大厅 0.007..0.021，港口 0.345

# 已准备好/未准备好 button row, content pixels: where the lobby is recognised.
READY_BAND = (692, 721, 1251, 953)
# 已准备好 label area inside that row (the label, not the frame).
READY_ROI = (875, 800, 1060, 862)
# 右侧房间面板（B 队）区域，content 像素：大厅为暗色半透明，港口是港口实景。
PORT_PANEL_ROI = (1200, 300, 1900, 900)

# Button points in content pixels.  "verified" means a click at this point has
# a recorded effect (see docs/training-room-restart.md for the evidence).
BUTTONS = {
    'return_port': dict(content=(1280, 1337), screen='results', verified='live frame 2026-10-20'),
    'join_port': dict(content=(1281, 91), screen='port', verified='state/join4-20261019'),
    'ready': dict(content=(968, 818), screen='lobby', verified='state/rooms13-ready-20261019'),
    'join_lobby': dict(content=(1281, 91), screen='lobby', verified='state/join4-20261019'),
    'start_battle': dict(content=(1298, 1396), screen='prebattle',
                         verified='detected bbox centre; layout shifts +-63 px'),
}


class UiError(RuntimeError):
    pass


# ---------------------------------------------------------------- coordinates
def read_geometry(display=DISPLAY, window=None):
    """Live client geometry: {'origin': [x, y], 'size': [w, h], 'window': id}."""
    env = {**os.environ, 'DISPLAY': display}
    if window is None:
        out = subprocess.run(['xdotool', 'search', '--onlyvisible', '--name', '战舰世界'],
                             env=env, capture_output=True, text=True, timeout=10)
        ids = [line.strip() for line in out.stdout.splitlines() if line.strip().isdigit()]
        if len(ids) != 1:
            raise UiError(f'expected one game window on {display}, found {ids}')
        window = ids[0]
    out = subprocess.run(['xdotool', 'getwindowgeometry', '--shell', window],
                         env=env, capture_output=True, text=True, timeout=10)
    if out.returncode:
        raise UiError(f'window geometry unavailable: {out.stderr.strip()}')
    fields = dict(line.split('=', 1) for line in out.stdout.splitlines())
    size = [int(fields['WIDTH']), int(fields['HEIGHT'])]
    origin = [int(fields['X']), int(fields['Y'])]
    if size[0] < 640 or size[1] < 480:
        raise UiError(f'window {window} is too small to be the game: {size}')
    return dict(window=window, origin=origin, size=size, display=display)


def content_to_window(point, geometry):
    try:
        return resolution_map.content_to_window(point, resolution_map.REFERENCE_FRAME,
                                                geometry['size'])
    except resolution_map.MappingError as exc:
        raise UiError(str(exc)) from exc


def window_to_root(point, geometry):
    return point[0] + geometry['origin'][0], point[1] + geometry['origin'][1]


def window_to_content(point, geometry):
    try:
        return resolution_map.window_to_content(point, resolution_map.REFERENCE_FRAME,
                                                geometry['size'])
    except resolution_map.MappingError as exc:
        raise UiError(str(exc)) from exc


def content_boxes(geometry):
    """Scales for the doc/printout: exact, per axis, never an aspect assumption."""
    return geometry['size'][0] / CONTENT[0], geometry['size'][1] / CONTENT[1]


# -------------------------------------------------------------------- capture
def capture(save=None):
    """One native client frame (window pixels) plus its evidence; read-only."""
    from wows_window_capture import WindowCapture
    shot = WindowCapture()
    pixels, meta = shot.acquire(wait_update=True)
    frame = np.asarray(pixels.convert('RGB'))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    image = dict(window=shot.window, identity=shot.identity, shape=list(frame.shape),
                 sha256=meta.get('raw_pixel_sha256'))
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save), frame)
        image['path'] = str(save)
    return frame, image


# -------------------------------------------------------------- pixel criteria
_TEMPLATES = None


def templates():
    global _TEMPLATES
    if _TEMPLATES is None:
        _TEMPLATES = etb.load_templates()
    return _TEMPLATES


def teal_bar(frame):
    """The teal '回到港口' bar: a wide, short teal blob low-centre of the screen.

    enter_training_battle.teal_button() was measured to pick the wrong blob on
    the 2026-10-20 results frame, so the geometry gate here is stricter.
    """
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask = ((hue > 78) & (hue < 100) & (sat > 70) & (val > 80)).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    best = None
    for index in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[index])
        if not (140 <= w <= 260 and 20 <= h <= 60 and area >= 2500):
            continue
        if y < height * 0.82 or abs((x + w / 2) - width / 2) > width * 0.15:
            continue
        if best is None or area > best['area']:
            best = dict(bbox=[x, y, w, h], area=area,
                        center=[x + w // 2, y + h // 2])
    return best


def banded_ready(frame, geometry):
    """Best ready-template score inside the lobby button row only.

    Region-restricted matching is what separates a lobby from the port: a
    full-frame match on the ready template scored 0.696 on a ready lobby and
    0.0 on the port, but the port's own join button dragged the full-frame
    score around by a few hundredths.  The band has no port UI in it.
    """
    x0, y0 = content_to_window((READY_BAND[0], READY_BAND[1]), geometry)
    x1, y1 = content_to_window((READY_BAND[2], READY_BAND[3]), geometry)
    crop = etb.edges(frame[y0:y1, x0:x1])
    template = etb.edges(cv2.imread(str(etb.FIXTURES / 'ready.png')))
    if crop.shape[0] < template.shape[0] or crop.shape[1] < template.shape[1]:
        return 0.0, None
    scores = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)
    _, best, _, location = cv2.minMaxLoc(scores)
    return round(float(best), 4), [x0 + int(location[0]), y0 + int(location[1])]


def panel_grey(frame, geometry):
    """Fraction of desaturated mid-bright pixels in the room-panel region.

    Used only to tell a lobby (dark translucent panels, 0.007..0.021) from the
    port (bright harbour behind the UI, 0.345) when the ready row is covered by
    a ship tooltip and the button template cannot be seen.
    """
    x0, y0 = content_to_window((PORT_PANEL_ROI[0], PORT_PANEL_ROI[1]), geometry)
    x1, y1 = content_to_window((PORT_PANEL_ROI[2], PORT_PANEL_ROI[3]), geometry)
    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        return 1.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    return round(float(((sat < 45) & (val > 85) & (val < 195)).mean()), 4)


def green_pixels(frame, geometry):
    """Count bright green pixels in the 已准备好 label area (content-space ROI)."""
    x0, y0, x1, y1 = READY_ROI
    wx0, wy0 = content_to_window((x0, y0), geometry)
    wx1, wy1 = content_to_window((x1, y1), geometry)
    roi = frame[wy0:wy1, wx0:wx1]
    if roi.size == 0:
        return 0, 0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask = (hue > 52) & (hue < 100) & (sat > 110) & (val > 120)
    return int(mask.sum()), int(mask.size)


def classify(frame, geometry):
    """Which menu screen this frame is, with the numbers that decided it."""
    found = etb.detect(frame, templates(), min(READY_TEMPLATE_MIN, JOIN_TEMPLATE_MIN))
    scores = {name: round(hit['score'], 4) for name, hit in found.items()}
    boxes = {name: hit['bbox'] for name, hit in found.items()}
    ready_score, ready_at = banded_ready(frame, geometry)
    green, green_total = green_pixels(frame, geometry)
    grey = panel_grey(frame, geometry)
    bar = teal_bar(frame)
    evidence = dict(template_scores=scores, template_boxes=boxes,
                    ready_band_score=ready_score, ready_band_at=ready_at,
                    ready_green_px=green, ready_green_box=green_total,
                    panel_grey=grey, teal_bar=bar)
    # start_battle before teal_bar: 开始战斗 is itself a teal bar low-centre, so
    # the template has to win first or prebattle would read as results.
    if scores.get('start_battle', 0) >= START_TEMPLATE_MIN:
        screen, why = 'prebattle', 'start_battle>=%.2f (%.4f)' % (
            START_TEMPLATE_MIN, scores['start_battle'])
    elif bar is not None:
        screen, why = 'results', 'teal_bar %s' % (bar['bbox'],)
    elif ready_score >= READY_TEMPLATE_MIN:
        ready = green >= READY_GREEN_MIN
        screen = 'lobby_ready' if ready else 'lobby_not_ready'
        why = 'ready_band>=%.2f (%.4f) and green%s%d (%d)' % (
            READY_TEMPLATE_MIN, ready_score, '>=' if ready else '<',
            READY_GREEN_MIN, green)
    elif scores.get('join_battle', 0) >= JOIN_TEMPLATE_MIN:
        if grey >= PORT_PANEL_GREY:
            screen, why = 'port', 'join_battle>=%.2f and panel_grey>=%.2f (%.4f)' % (
                JOIN_TEMPLATE_MIN, PORT_PANEL_GREY, grey)
        else:
            # Same top button as the port, but the lobby's dark panels are still
            # behind it: the ready row is covered (a ship card/tooltip), so the
            # ready state is unreadable and no click may be guessed.
            screen, why = 'lobby_occluded', (
                'join_battle>=%.2f but panel_grey %.4f < %.2f and ready row unseen' % (
                    JOIN_TEMPLATE_MIN, grey, PORT_PANEL_GREY))
    else:
        screen, why = 'unknown', 'no menu criterion matched'
    return dict(screen=screen, decided_by=why, evidence=evidence)


# ------------------------------------------------------------------ next step
def button_point(name, detection, geometry):
    """Window/root point for a button: live detection first, documented constant last."""
    entry = BUTTONS[name]
    if detection and 'bbox' in detection and name != 'ready':
        x, y, w, h = detection['bbox']
        window = [x + w // 2, y + h // 2]
        content = list(window_to_content(window, geometry))
        source = 'live_template'
    else:
        content = list(entry['content'])
        window = list(content_to_window(content, geometry))
        source = 'documented_content_point'
    return dict(button=name, source=source, content=content,
                window=window, root=list(window_to_root(window, geometry)))


CHAIN = [
    ('results', 'return_port', '回到港口', ['port', 'lobby_not_ready', 'lobby_ready'],
     'teal bar w 140..260, h 20..60, area>=2500, centre within 15% of screen width'),
    ('port', 'join_port', '加入战斗 (top)', ['lobby_not_ready', 'lobby_ready'],
     'join_battle template >= 0.70 (port 0.721, lobby 0.766..0.813)'),
    ('lobby_not_ready', 'ready', '已准备好', ['lobby_ready'],
     'ready-band template >= 0.45 (observed 0.595..1.0) and green label px < 200 (observed 0)'),
    ('lobby_ready', 'join_lobby', '加入战斗 (top)', ['prebattle'],
     'ready-band template >= 0.45 and green label px >= 200 (observed 510..533)'),
    ('prebattle', 'start_battle', '开始战斗', ['unknown'],
     'start_battle template >= 0.66 observed 0.716..0.779; after: 3 consecutive frames without any menu match'),
]


def chain_table(geometry):
    """Every menu step with its coordinates in all three spaces, for the dry run."""
    rows = []
    for number, (state, button, label, expect, criterion) in enumerate(CHAIN, 1):
        point = button_point(button, None, geometry)
        rows.append(dict(step=number, from_state=state, button=button, label=label,
                         content=point['content'], window=point['window'],
                         root=point['root'], expect=expect, criterion=criterion,
                         point_note=('detected at run time; this is the documented fallback'
                                     if button == 'start_battle' else 'fixed point')))
    return rows


def plan_for(state, classification, geometry):
    """The single click this screen needs, with its criteria, or None."""
    evidence = classification['evidence']
    if state == 'results':
        return dict(action='click', button='return_port',
                    point=button_point('return_port', evidence['teal_bar'], geometry),
                    expect=['port', 'lobby_not_ready', 'lobby_ready'])
    if state == 'port':
        return dict(action='click', button='join_port',
                    point=button_point('join_port', None, geometry),
                    expect=['lobby_not_ready', 'lobby_ready'])
    if state == 'lobby_not_ready':
        return dict(action='click', button='ready',
                    point=button_point('ready', None, geometry), expect=['lobby_ready'])
    if state == 'lobby_ready':
        return dict(action='click', button='join_lobby',
                    point=button_point('join_lobby', None, geometry), expect=['prebattle'])
    if state == 'prebattle':
        detected = None
        if 'start_battle' in evidence['template_boxes']:
            detected = dict(bbox=evidence['template_boxes']['start_battle'])
        return dict(action='click', button='start_battle',
                    point=button_point('start_battle', detected, geometry),
                    expect=['unknown'])
    return None


# ----------------------------------------------------------------- input path
def xdotool(geometry, *args):
    env = {**os.environ, 'DISPLAY': geometry['display']}
    out = subprocess.run(['xdotool', *map(str, args)], env=env,
                         capture_output=True, text=True, timeout=10)
    if out.returncode:
        raise UiError(f'xdotool {args[0]} failed: {out.stderr.strip()}')
    return out.stdout.strip()


def pointer(geometry):
    fields = dict(line.split('=', 1) for line in
                  xdotool(geometry, 'getmouselocation', '--shell').splitlines())
    return int(fields['X']), int(fields['Y'])


def focus_check(geometry):
    current = xdotool(geometry, 'getwindowfocus')
    if current != geometry['window']:
        raise UiError(f'game window does not hold focus (focus={current})')


def click_root(geometry, root_point, hold_ms=120, tolerance=6):
    """Move the pointer to a root point, verify it arrived, then one click."""
    focus_check(geometry)
    before = pointer(geometry)
    xdotool(geometry, 'mousemove', root_point[0], root_point[1])
    time.sleep(0.15)
    after = pointer(geometry)
    offset = math.hypot(after[0] - root_point[0], after[1] - root_point[1])
    if offset > tolerance:
        raise UiError(f'pointer at {after}, wanted {list(root_point)} (offset {offset:.1f} px)')
    focus_check(geometry)
    xdotool(geometry, 'mousedown', 1)
    time.sleep(max(0, hold_ms) / 1000)
    focus_check(geometry)
    xdotool(geometry, 'mouseup', 1)
    return dict(pointer_before=list(before), pointer_after=list(after),
                offset_px=round(offset, 2), hold_ms=hold_ms,
                transport='xdotool absolute move + XTest button 1')


# ------------------------------------------------------------------- the run
def wait_for(geometry, wanted, timeout, save_dir, poll=0.5, consecutive=1):
    """Re-capture until the screen is one of `wanted`; returns (state, shots).

    `consecutive` >1 requires that many frames in a row to agree, which is what
    'the menus are gone' needs: a single transition frame must not be mistaken
    for a live battle.
    """
    deadline = time.monotonic() + timeout
    shots = []
    streak = 0
    while True:
        frame, image = capture()
        found = classify(frame, geometry)
        shots.append(dict(t=round(time.monotonic() - (deadline - timeout), 2),
                          screen=found['screen'], sha256=image['sha256'],
                          scores=found['evidence']['template_scores']))
        if found['screen'] in wanted:
            streak += 1
            if streak >= consecutive:
                return found, shots
        else:
            streak = 0
        if time.monotonic() >= deadline:
            path = Path(save_dir) / 'failed.png'
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), frame)
            raise UiError('expected %s, still %s after %.1f s (saved %s)' % (
                '|'.join(wanted), found['screen'], timeout, path))
        time.sleep(poll)


def render_text(report):
    """Human-readable plan: coordinates in all three spaces plus the criteria."""
    geometry = report['window_geometry']
    scale = report['mapping']['scale']
    origin = geometry['origin']
    lines = [
        '== restart_training_room ==',
        'mode            %s' % ('EXECUTE (xdotool clicks)' if report['execute'] else
                                 'dry-run (no input; xdotool clicks are never invoked)'),
        'display/window  %s  %s' % (geometry['display'], geometry['window']),
        'window client   %dx%d at root origin (%d,%d)' % (
            geometry['size'][0], geometry['size'][1], origin[0], origin[1]),
        'content frame   %dx%d (project reference, no letterbox bars)' % tuple(report['mapping']['content']),
        'formula         window = round(content * [%d/%d, %d/%d]);  root = window + (%d,%d)'
        % (geometry['size'][0], report['mapping']['content'][0],
           geometry['size'][1], report['mapping']['content'][1], origin[0], origin[1]),
        'scale           sx=%.6f  sy=%.6f' % tuple(scale),
        '',
        'current screen  %s   (%s)' % (report['initial']['screen'],
                                       report['initial']['decided_by']),
    ]
    evidence = report['initial']['evidence']
    bar = evidence['teal_bar']
    lines.append('  evidence      ready_band=%.4f  panel_grey=%.4f  templates=%s  '
                 'green=%d/%d  teal_bar=%s' % (
        evidence['ready_band_score'], evidence['panel_grey'],
        json.dumps(evidence['template_scores'], sort_keys=True),
        evidence['ready_green_px'], evidence['ready_green_box'],
        'none' if bar is None else '%s centre %s' % (bar['bbox'], bar['center'])))
    if 'path' in report['frame']:
        lines.append('  frame         %s  sha256=%s' % (report['frame']['path'],
                                                       report['frame']['sha256'][:16]))
    lines += ['', 'remaining chain (content / window / root are the click points)',
              '%-4s %-20s %-13s %-12s %-12s %-12s %s' % (
                  'step', 'from screen', 'button', 'content', 'window', 'root', 'expected next')]
    for row in report['chain']:
        lines.append('%-4d %-20s %-13s %-12s %-12s %-12s %s' % (
            row['step'], row['from_state'], row['button'],
            '%d,%d' % tuple(row['content']), '%d,%d' % tuple(row['window']),
            '%d,%d' % tuple(row['root']), '|'.join(row['expect'])))
    lines += ['', 'criteria (measured on real frames; thresholds in plan.json)']
    for row in report['chain']:
        lines.append('  step %d %-13s %s' % (row['step'], row['button'], row['criterion']))
    lines += ['', 'every click is followed by a fresh capture; the run stops and saves '
              'failed.png when the expected screen does not appear.']
    plan = report.get('next')
    if plan is None:
        hint = ''
        if report['initial']['screen'] == 'lobby_occluded':
            hint = ('  (大厅房间面板可见，但"已准备好"按钮行被舰船卡片/提示框遮住：'
                    '先把指针移开再重试，脚本不会猜)')
        lines += ['', 'verdict         no action for this screen; the run would stop here.%s'
                  % hint]
    else:
        point = plan['point']
        lines += ['', 'verdict         next: click %s at content (%d,%d) -> window (%d,%d) '
                  '-> root (%d,%d), then expect %s' % (
                      point['button'], point['content'][0], point['content'][1],
                      point['window'][0], point['window'][1],
                      point['root'][0], point['root'][1], '|'.join(plan['expect']))]
    if not report['execute']:
        lines += ['no input sent.']
    print('\n'.join(lines))


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    geometry = read_geometry(args.display, args.window)
    sx, sy = content_boxes(geometry)
    report = dict(action='restart_training_room', execute=args.execute,
                  out=str(out.resolve()), display=geometry['display'],
                  window=geometry['window'], window_geometry=geometry,
                  mapping=dict(content=list(CONTENT), size=geometry['size'],
                               origin=geometry['origin'], scale=[round(sx, 6), round(sy, 6)],
                               content_to_window='round(content * size / content_size)',
                               window_to_root='window + origin',
                               bars='none: the client area itself is captured'),
                  thresholds=dict(ready_template=READY_TEMPLATE_MIN,
                                  start_template=START_TEMPLATE_MIN,
                                  join_template=JOIN_TEMPLATE_MIN,
                                  ready_green_px=READY_GREEN_MIN,
                                  ready_roi=list(READY_ROI)),
                  state='init', steps=[], inputs=[], fired=False,
                  chain=chain_table(geometry),
                  semantics='menu clicks only; a vanished menu is not proof of a battle')

    frame, image = capture(out / '00-current.png')
    current = classify(frame, geometry)
    report['initial'] = current
    report['frame'] = image
    report['state'] = current['screen']
    plan = plan_for(current['screen'], current, geometry)
    report['next'] = plan
    (out / 'plan.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')

    if not args.execute:
        report['status'] = 'dry-run'
        (out / 'plan.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
        if args.text:
            render_text(report)
        else:
            print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    try:
        state = current['screen']
        for index in range(args.max_steps):
            plan = plan_for(state, current, geometry)
            if plan is None:
                report['status'] = 'stopped_%s' % state
                break
            step = dict(index=index + 1, from_state=state, button=plan['button'],
                        point=plan['point'], expect=plan['expect'])
            sent = click_root(geometry, plan['point']['root'], args.hold_ms)
            report['inputs'].append(dict(button=1, pulse=True, action=plan['button'],
                                         root_point=plan['point']['root'], **sent))
            step['input'] = sent
            after, shots = wait_for(geometry, plan['expect'], args.step_timeout,
                                    out, args.poll,
                                    args.absent_checks if plan['expect'] == ['unknown'] else 1)
            step['to_state'] = after['screen']
            step['evidence'] = after['evidence']
            step['frames'] = shots
            path = out / f'{index + 1:02d}-{plan["button"]}.png'
            _, image = capture(path)
            step['frame'] = image
            report['steps'].append(step)
            report['state'] = after['screen']
            report['next'] = plan_for(after['screen'], after, geometry)
            (out / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
            print('step %d: %s -> %s  (%s at root %s)' % (
                index + 1, step['from_state'], after['screen'], plan['button'],
                plan['point']['root']))
            state, current = after['screen'], after
            if state == 'unknown':
                report['status'] = ('menus_absent_after_start'
                                    if plan['button'] == 'start_battle'
                                    else 'menus_absent_unexpectedly')
                break
        else:
            report['status'] = 'step_budget_exhausted'
    except Exception as error:                     # noqa: BLE001 - report, never swallow
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        report['unreleased_button'] = False
        (out / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: report[k] for k in ('status', 'state') if k in report},
                     ensure_ascii=False))
    return 0 if report.get('status') == 'menus_absent_after_start' else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--display', default=DISPLAY)
    parser.add_argument('--window', help='window id override (default: search by title)')
    parser.add_argument('--max-steps', type=int, default=5)
    parser.add_argument('--step-timeout', type=float, default=12.0)
    parser.add_argument('--poll', type=float, default=0.6)
    parser.add_argument('--hold-ms', type=int, default=120)
    parser.add_argument('--absent-checks', type=int, default=3,
                        help='consecutive frames without any menu before a battle is assumed')
    parser.add_argument('--text', action='store_true',
                        help='dry run: print the human-readable plan instead of JSON')
    args = parser.parse_args(argv)
    if not 1 <= args.max_steps <= 8:
        parser.error('max-steps must be 1..8')
    if not 2 <= args.step_timeout <= 60 or not 0.1 <= args.poll <= 5:
        parser.error('step-timeout must be 2..60 and poll 0.1..5')
    if not 0 <= args.hold_ms <= 250:
        parser.error('hold-ms must be 0..250')
    if not 1 <= args.absent_checks <= 10:
        parser.error('absent-checks must be 1..10')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
