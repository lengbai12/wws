#!/usr/bin/env python3
"""Training-room wide-view scan; all angles are commanded, never measured coverage.

Default dry-run is offline. Execute requires an operator-supplied signed angle
mapping bound to the exact current game identity (JSON via --angle-identity).
Integer relative input must represent each route endpoint exactly; incompatible
mappings are rejected rather than silently overshooting pitch or inventing angles.
"""
import argparse
import json
import math
from pathlib import Path
import signal
import time
from types import SimpleNamespace

import numpy as np

from prepare_locked_aim import (AcquisitionProbe, FireHudReader, ProbeError,
                                detect_nameplates, view_state)
from locked_capture import LockedCapture, require_fresh
import prepare_locked_aim as locked_aim

PITCHES = (-10.0, 0.0, 5.0)
ANGLE_ROUTE = [dict(pitch_deg=pitch, yaw_delta_deg=yaw)
               for pitch, yaw in zip(PITCHES, (360.0, -360.0, 360.0))]


class ScanProbe(AcquisitionProbe):
    def move(self, dx, dy):
        super().move(dx, dy)
        if self.relative_transport == 'xtest':
            delivered = self.report['inputs'][-1].get('pointer_delta')
            if delivered != [dx, dy]:
                raise ProbeError('pointer clipping or warp observed; refusing scan without recenter')


def validate(args):
    if type(args.steps) is not int or not 1 <= args.steps <= 4096:
        raise ProbeError('steps must be an integer in 1..4096')
    if not math.isfinite(args.seconds) or not 1 <= args.seconds <= 120:
        raise ProbeError('seconds must be finite in 1..120')
    if args.execute and args.inspect:
        raise ProbeError('execute and inspect are mutually exclusive')
    if args.execute and not args.training_room:
        raise ProbeError('--execute requires --training-room operator attestation')
    maximum = getattr(args, 'max_range_km', None)
    if args.execute and (maximum is None or not math.isfinite(maximum) or maximum <= 4):
        raise ProbeError('explicit main battery maximum range above 4 km required for coarse handoff')
    values = (args.yaw_deg_per_unit, args.pitch_deg_per_unit, args.initial_pitch_deg)
    supplied = any(v is not None for v in values) or args.angle_source is not None or args.angle_identity is not None
    if not supplied:
        if args.execute:
            raise ProbeError('execute requires explicit identity-bound angle mapping')
        return None
    if (any(v is None or not math.isfinite(v) for v in values) or
            values[0] == 0 or values[1] == 0 or not -10 <= values[2] <= 5 or
            not args.angle_source or not args.angle_source.strip() or
            not isinstance(args.angle_identity, dict) or not args.angle_identity):
        raise ProbeError('angle mapping requires finite nonzero signed gains, pitch in [-10,5], source and game identity')
    return dict(yaw_deg_per_unit=values[0], pitch_deg_per_unit=values[1],
                initial_pitch_deg=values[2], source=args.angle_source,
                identity=args.angle_identity)


def integer_units(angle, gain):
    value = angle / gain
    if not math.isfinite(value) or abs(value) > 1e12:
        raise ProbeError('angle mapping requires excessive relative units')
    units = round(value)
    if not math.isclose(units * gain, angle, rel_tol=0, abs_tol=1e-8):
        raise ProbeError('angle route endpoints not representable by integer mouse units')
    return units


def route_segments(mapping):
    """Small segment list; expand lazily so tiny gains cannot allocate a huge route."""
    segments = []
    pitch = mapping['initial_pitch_deg']
    for row, (target, yaw) in enumerate(zip(PITCHES, (360.0, -360.0, 360.0))):
        segments.append((row, 1, integer_units(target-pitch, mapping['pitch_deg_per_unit'])))
        segments.append((row, 0, integer_units(yaw, mapping['yaw_deg_per_unit'])))
        pitch = target
    return segments


def route_commands(segments):
    for row, axis, total in segments:
        limit = 150 if axis == 0 else 3
        while total:
            units = min(abs(total), limit) * (1 if total > 0 else -1)
            yield row, (units, 0) if axis == 0 else (0, units)
            total -= units


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(status='dry-run', inputs=[], focus_checks=[], frames=[],
                  commanded_angles=[], angle_route=ANGLE_ROUTE, coverage_verified=False,
                  actual_angles_observed=False, fired=False, handoff=None,
                  training_room_attested=bool(args.training_room),
                  training_room_evidence='operator_assertion_not_visual_proof',
                  frame_artifacts_complete=False)
    capture = probe = None
    handlers = {}
    previews = {}
    last_ns = None
    source_key = None
    pending_hash = None
    wait_started = None

    def stop(signum, _):
        raise ProbeError(f'bounded scan interrupted by signal {signum}')

    def write_report():
        (args.out/'report.json').write_text(json.dumps(report, indent=2))

    try:
        mapping = validate(args)
        report['angle_mapping'] = mapping
        segments = route_segments(mapping) if mapping else None
        if segments is not None:
            report['required_input_steps'] = sum((abs(n)+(149 if axis == 0 else 2)) //
                                                (150 if axis == 0 else 3)
                                                for _, axis, n in segments)
        if not args.execute and not args.inspect:
            return 0
        probe = ScanProbe(args.seconds, report)
        for sig in (signal.SIGALRM, signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, stop)
        signal.setitimer(signal.ITIMER_REAL, max(.001, probe.action_deadline-time.monotonic()))
        capture = LockedCapture()
        probe.capture = capture
        probe.identity, probe.window = capture.identity, capture.window
        probe.relative_transport = args.transport
        if args.execute and mapping['identity'] != capture.identity:
            raise ProbeError('angle mapping game identity does not match current game generation')
        reader = FireHudReader()
        commands = iter(route_commands(segments)) if args.execute else iter(())
        yaw = 0.0
        pitch = mapping['initial_pitch_deg'] if mapping else None
        action_count = 0
        report['status'] = 'running'
        # Bound observations independently: unchanged renders must not cause an
        # unbounded metadata/report accumulation, even with a very fast source.
        for observation in range(8192):
            if time.monotonic() >= probe.action_deadline:
                raise ProbeError('bounded scan deadline reached')
            pixels, meta = capture.acquire(wait_update=True)
            last_ns = require_fresh(meta, previous_ns=last_ns)
            key = json.dumps([meta.get(k) for k in ('source', 'identity', 'window')], sort_keys=True)
            if source_key is not None and key != source_key:
                raise ProbeError('pinned capture source changed')
            source_key = key
            if meta.get('identity') != capture.identity or meta.get('window') != capture.window:
                raise ProbeError('capture and input identity mismatch')
            frame = np.asarray(pixels)
            if frame.shape != (1600, 2560, 3) or frame.dtype != np.uint8:
                raise ProbeError('requires exact native 2560x1600 RGB')
            pixel_hash = meta.get('raw_pixel_sha256')
            if not isinstance(pixel_hash, str) or not pixel_hash:
                raise ProbeError('capture pixel hash missing')
            probe.evidence = meta
            hud = reader.detect(frame)
            state = view_state(hud, frame)
            candidates = (detect_nameplates(frame, scope_observed=False)
                          if hud.get('battle', {}).get('active') is True and state == 'wide' else [])
            row = dict(observation=observation, capture=meta, view=state,
                       battle_active=hud.get('battle', {}).get('active') is True,
                       candidates=candidates)
            report['frames'].append(row)
            if not previews:
                previews['first'] = (pixels, meta)
            previews['last'] = (pixels, meta)
            if not row['battle_active'] or state != 'wide':
                raise ProbeError('active battle and wide view required; scope is never toggled')
            if candidates:
                report['handoff'] = dict(candidates=candidates, capture=meta,
                                         preview='last.png', view=state,
                                         next_stage='coarse_aim', scan_stopped=True,
                                         target_identity_verified=False,
                                         range_eligibility_verified=False)
                row['action'] = 'stop_all_scan_for_coarse_aim'
                report['status'] = 'enemy_found'
                if args.execute:
                    # No route commands are consumed after detection, including
                    # on handoff failure. Preserve the first acquisition clock.
                    probe.release()
                    remaining = probe.action_deadline-time.monotonic()
                    if remaining < 5:
                        raise ProbeError('insufficient bounded time for coarse handoff; scan stopped')
                    aim_args = SimpleNamespace(out=args.out/'coarse-aim', execute=True,
                        inspect=False, fire=False, training_room=args.training_room,
                        max_range_km=args.max_range_km, steps=100, seconds=min(remaining,120),
                        transport=args.transport, scope_only=False)
                    code = locked_aim.run(aim_args, acquisition=dict(frame=frame,
                        candidates=candidates, capture=meta))
                    report['handoff'].update(exit_code=code, report='coarse-aim/report.json')
                    report['status'] = 'coarse_aim_completed' if code == 0 else 'coarse_aim_refused'
                break
            if args.inspect:
                report['status'] = 'inspected'
                break
            if pending_hash == pixel_hash:
                row['action'] = 'wait_for_changed_pixels'
                if time.monotonic()-wait_started >= 2:
                    raise ProbeError('render pixels unchanged after movement; refusing stacked input')
                time.sleep(min(.05, max(0, probe.action_deadline-time.monotonic())))
                continue
            pending_hash = None
            command = next(commands, None)
            if command is None:
                report['status'] = 'route_commanded'
                break
            if action_count >= args.steps:
                raise ProbeError('bounded scan step budget exhausted')
            route_row, (dx, dy) = command
            # AcquisitionProbe refuses observed pointer clipping. No recenter,
            # key, scope or fire commands are ever issued by this controller.
            yaw += dx * mapping['yaw_deg_per_unit']
            pitch += dy * mapping['pitch_deg_per_unit']
            report['commanded_angles'].append(dict(row=route_row, relative=[dx, dy],
                yaw_deg=yaw, pitch_deg=pitch, yaw_delta_deg=dx*mapping['yaw_deg_per_unit'],
                pitch_delta_deg=dy*mapping['pitch_deg_per_unit'], transport_status='attempted'))
            probe.move(dx, dy)
            report['commanded_angles'][-1]['transport_status'] = 'command_completed'
            action_count += 1
            row['action'] = 'relative_scan'
            pending_hash, wait_started = pixel_hash, time.monotonic()
        else:
            raise ProbeError('bounded scan observation budget exhausted')
    except Exception as exc:
        report.update(status='refused', reason=str(exc))
    finally:
        if handlers:
            signal.setitimer(signal.ITIMER_REAL, 0)
        if probe is not None:
            try:
                probe.release()
                if report.get('unreleased_own_keys') or report.get('unreleased_own_left_button'):
                    report.update(status='refused', reason='owned input release incomplete')
            except Exception as exc:
                report.update(status='refused', reason=f'input release failed: {exc}')
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        # Persist an inspectable report after release, before expensive encoding.
        write_report()
        try:
            for label, (pixels, meta) in previews.items():
                capture.save(args.out/f'{label}.png', pixels, meta)
            report['frame_artifacts_complete'] = True
        except Exception as exc:
            report['artifact_error'] = str(exc)
        finally:
            if capture is not None:
                try:
                    capture.close()
                except Exception as exc:
                    report['capture_close_error'] = str(exc)
            write_report()
    return 0 if report['status'] in ('dry-run', 'inspected', 'enemy_found',
                                    'coarse_aim_completed', 'route_commanded') else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--execute', action='store_true')
    modes.add_argument('--inspect', action='store_true', help='one read-only live frame')
    modes.add_argument('--dry-run', action='store_true', help='offline angular route (default)')
    parser.add_argument('--training-room', action='store_true')
    parser.add_argument('--seconds', type=float, default=120)
    parser.add_argument('--steps', type=int, default=4096)
    parser.add_argument('--transport', choices=('xtest', 'uinput'), default='xtest')
    parser.add_argument('--yaw-deg-per-unit', type=float)
    parser.add_argument('--pitch-deg-per-unit', type=float)
    parser.add_argument('--initial-pitch-deg', type=float)
    parser.add_argument('--angle-source', help='provenance of operator-supplied mapping')
    parser.add_argument('--max-range-km', type=float, help='actual main battery range for coarse handoff')
    parser.add_argument('--angle-identity', type=json.loads, help='exact current game identity JSON object')
    return run(parser.parse_args(argv))


if __name__ == '__main__':
    raise SystemExit(main())
