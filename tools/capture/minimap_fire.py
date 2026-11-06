#!/usr/bin/env python3
"""Aim and fire using the minimap contact when no nameplate is ever drawn.

Sequence (the operator's own procedure, with the minimap as the sensor):
  1. read the nearest minimap contact: range and bearing relative to our course;
  2. yaw onto that bearing (chunked, with the M-recentre when the pointer runs
     out of travel);
  3. close the elevation until the reticle's ballistic range matches the contact
     range (measured: 60 units ~ 1 km, negative dy looks further);
  4. re-read both, and fire only when both are inside tolerance.

Dry-run by default. Every number that authorised the shot is written to the
report, and the shot is labelled ``shot_basis = "minimap_contact"``: this path
exists because a submarine or a cyclone-hidden target never draws a nameplate.
"""
import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

import aim_servo as A
import minimap_shot as MS

ROOT = Path(__file__).resolve().parents[2]

# The reticle is a fixed HUD element: every successful read in this project
# returned [1280.5, 805] (native band, green ring and scope ruler agree), so a
# failed read does not leave us without a centre.
RETICLE_FIXED = (1280.5, 805.0)


def snapshot(path):
    subprocess.run(['/usr/bin/python3', str(ROOT / 'tools/capture/capture_host.py'),
                    'snapshot', str(path)], capture_output=True, timeout=90)
    return np.asarray(Image.open(path).convert('RGB'))


def press_key(key):
    subprocess.run(['xdotool', 'key', key], env={'DISPLAY': ':2', 'PATH': '/usr/bin:/bin'})


def move(dx, dy=0.0, chunk=40.0, display=':2'):
    remaining = float(dx)
    first = True
    while abs(remaining) > 0.5:
        step = max(-chunk, min(chunk, remaining))
        cmd = ['xdotool', 'mousemove_relative', '--', str(int(step)),
               str(int(dy) if first else 0)]
        subprocess.run(cmd, env={'DISPLAY': display, 'PATH': '/usr/bin:/bin'})
        first = False
        remaining -= step
        time.sleep(0.012)
    time.sleep(0.2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--fire', action='store_true')
    parser.add_argument('--rounds', type=int, default=8)
    parser.add_argument('--bearing-gain', type=float, default=1.5)
    parser.add_argument('--km-tolerance', type=float, default=0.4)
    parser.add_argument('--bearing-tolerance', type=float, default=2.0)
    parser.add_argument('--recentre-key', default='m')
    parser.add_argument('--settle', type=float, default=0.35)
    parser.add_argument('--display', default=':2')
    args = parser.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    report = dict(tool='minimap_fire', inputs=[], rounds=[], shot_basis='minimap_contact',
                  execute=args.execute, fire_requested=args.fire)

    frame = snapshot(out / 'frame-000.png')
    reticle, active, weapon, source = A.reticle_centre(frame)
    report['reticle'] = reticle
    report['battle'] = active
    report['weapon'] = weapon
    contact = MS.contact_from_frame(frame, A.minimap_contacts_from_frame)
    report['contact'] = None if contact is None else dict(
        km=round(contact[0], 2), relative_bearing_deg=round(contact[1], 1))
    if contact is None or reticle is None or active is not True:
        report['status'] = 'refused'
        report['reason'] = 'missing contact, reticle or battle'
        (out / 'report.json').write_text(json.dumps(report, indent=2, default=str))
        return 1
    if not args.execute:
        report['status'] = 'dry_run'
        report['reason'] = 'execute not requested'
        (out / 'report.json').write_text(json.dumps(report, indent=2, default=str))
        return 0

    km, bearing, _own = contact
    units = MS.yaw_units(bearing, args.bearing_gain)
    report['rounds'].append(dict(kind='yaw', units=round(units, 1), bearing_deg=bearing))
    sign = 1 if units >= 0 else -1
    left = abs(units)
    while left > 1:
        step = min(900.0, left) * sign
        move(step, display=args.display)
        report['inputs'].append(dict(relative=[round(step, 1), 0], reason='yaw'))
        left -= abs(step)
        if left > 120 and args.recentre_key:
            press_key(args.recentre_key)
            time.sleep(0.9)
            press_key(args.recentre_key)
            time.sleep(1.0)
            report['inputs'].append(dict(key=args.recentre_key, reason='recentre'))
    time.sleep(args.settle)

    km_per_unit = 0.004
    seconds_per_km = 0.6
    seconds_per_unit = 0.01
    last_elevation = None
    last_aim_km = None
    last_flight = None
    for index in range(args.rounds):
        # The green reticle ring is only painted after mouse motion (measured
        # 2026-10-23: an idle frame reports no reticle at all), so every read is
        # preceded by a two-unit nudge that costs nothing and rotates nothing
        # measurable.
        move(2.0, 0.0, display=args.display)
        frame = snapshot(out / f'frame-{index + 1:03d}.png')
        reticle, active, weapon, source = A.reticle_centre(frame)
        if reticle is None:
            reticle = RETICLE_FIXED
            report.setdefault('reticle_fallbacks', 0)
            report['reticle_fallbacks'] += 1
        # hud_aim_km reads one probe point and returned None on six of eight
        # frames here; aim_distance_km fans over neighbouring offsets and takes
        # the majority, which is what the earlier closure used successfully.
        aim_km = A.hud_aim_km(frame, reticle)
        if aim_km is None:
            aim_km = A.aim_distance_km(frame, reticle)
        contact = MS.contact_from_frame(frame, A.minimap_contacts_from_frame)
        target_km = None if contact is None else contact[0]
        b = None if contact is None else contact[1]
        # The ballistic "N.N s" beside the reticle reads on frames where both km
        # readers fail (measured 2026-10-23: flight 4.5 s at conf 0.84 while
        # hud_aim_km and aim_distance_km both returned None). Flight time is
        # monotonic in range, so it can drive the elevation: convert the target
        # range with a seconds-per-km constant and refine it whenever a real km
        # reading appears.
        flight = None
        try:
            info = A.hud_flight_seconds(frame)
            flight = None if not info else info.get('value')
        except Exception:
            flight = None
        target_flight = None if target_km is None else target_km * seconds_per_km
        if aim_km is not None and flight:
            observed = flight / aim_km
            if 0.2 <= observed <= 2.0:
                seconds_per_km = 0.7 * seconds_per_km + 0.3 * observed
        report['flight'] = dict(value=flight, target=target_flight,
                                seconds_per_km=round(seconds_per_km, 4))
        report['rounds'].append(dict(kind='measure', aim_km=aim_km,
                                     target_km=None if target_km is None else round(target_km, 2),
                                     bearing_deg=None if b is None else round(b, 1)))
        if active is not True:
            report['status'] = 'refused'
            report['reason'] = 'battle or reticle lost'
            report['last_read'] = dict(reticle=reticle, battle=active, weapon=weapon,
                                       source=source, aim_km=aim_km,
                                       target_km=target_km)
            break
        if abs(b or 0) > args.bearing_tolerance and b is not None:
            units = MS.yaw_units(b, args.bearing_gain)
            move(units, display=args.display)
            report['inputs'].append(dict(relative=[round(units, 1), 0], reason='yaw_fix'))
            time.sleep(args.settle)
            continue
        # The vertical scale is not a constant: measured 2026-10-23, 120 units
        # moved the readout 0.1 -> 0.4 -> 1.2 km (about 0.004 km/unit), while an
        # earlier calibration in another pitch range gave 0.017. Learn it from
        # the last move instead of trusting either number.
        if last_elevation is not None and aim_km is not None and last_aim_km is not None:
            moved = last_elevation[1]
            if abs(moved) >= 20:
                observed = (aim_km - last_aim_km) / moved
                if 0.0005 <= abs(observed) <= 0.05:
                    km_per_unit = 0.5 * km_per_unit + 0.5 * observed
        if aim_km is None and flight is not None and target_flight is not None:
            # No km reading: steer on flight time alone. Its vertical scale is
            # learned the same way, from the last move.
            if last_flight is not None and last_elevation is not None:
                moved = last_elevation[1]
                if abs(moved) >= 20:
                    observed_s = (flight - last_flight) / moved
                    if 0.0005 <= abs(observed_s) <= 0.2:
                        seconds_per_unit = 0.5 * seconds_per_unit + 0.5 * observed_s
            last_flight = flight
            dy = (flight - target_flight) / seconds_per_unit
            dy = max(-240.0, min(240.0, dy))
            last_elevation = ('elevation', dy)
            last_aim_km = None
            report.setdefault('flight_seconds_per_unit', round(seconds_per_unit, 6))
            report['rounds'][-1]['flight'] = flight
            report['rounds'][-1]['target_flight'] = target_flight
            if abs(flight - target_flight) <= 0.6:
                allowed = True
                report['shot_check'] = dict(allowed=True, basis='flight_time',
                                            flight=flight, target_flight=target_flight,
                                            target_km=target_km, bearing_deg=b)
                if args.fire:
                    subprocess.run(['xdotool', 'mousedown', '1'],
                                   env={'DISPLAY': args.display, 'PATH': '/usr/bin:/bin'})
                    time.sleep(0.1)
                    subprocess.run(['xdotool', 'mouseup', '1'],
                                   env={'DISPLAY': args.display, 'PATH': '/usr/bin:/bin'})
                    report['fired'] = True
                    report['status'] = 'fired'
                else:
                    report['status'] = 'ready'
                break
            move(0.0, dy, display=args.display)
            report['inputs'].append(dict(relative=[0, round(dy, 1)], reason='elevation_flight'))
            time.sleep(args.settle)
            continue
        if aim_km is None:
            dy = -120.0      # the readout vanished: walk the aim up until it returns
        else:
            dy = (aim_km - target_km) / km_per_unit
            dy = max(-240.0, min(240.0, dy))
        last_elevation = ('elevation', dy)
        last_aim_km = aim_km
        report.setdefault('km_per_unit', round(km_per_unit, 6))
        if abs(dy) > 1.5:
            move(0.0, dy, display=args.display)
            report['inputs'].append(dict(relative=[0, round(dy, 1)], reason='elevation'))
            time.sleep(args.settle)
            continue
        allowed, why = MS.shot_allowed(aim_km, target_km, b, args.km_tolerance,
                                       args.bearing_tolerance)
        report['shot_check'] = dict(allowed=allowed, reason=why, aim_km=aim_km,
                                    target_km=target_km, bearing_deg=b)
        if not allowed:
            report['status'] = 'refused'
            report['reason'] = why
            break
        if args.fire:
            subprocess.run(['xdotool', 'mousedown', '1'],
                           env={'DISPLAY': args.display, 'PATH': '/usr/bin:/bin'})
            time.sleep(0.1)
            subprocess.run(['xdotool', 'mouseup', '1'],
                           env={'DISPLAY': args.display, 'PATH': '/usr/bin:/bin'})
            report['fired'] = True
        report['status'] = 'ready' if not args.fire else 'fired'
        break
    else:
        report['status'] = 'exhausted'

    (out / 'report.json').write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: report.get(k) for k in ('status', 'reason', 'fired', 'shot_check')},
                     indent=2, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
