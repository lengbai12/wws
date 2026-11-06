#!/usr/bin/env python3
"""Evaluate saved observations; replay truth and clocks never enter decisions."""
import argparse
from bisect import bisect_right
from collections import Counter
import copy
import json
import math
from pathlib import Path

from build_navigation_reference import DEFAULT_POSITIONS, PATTERN, ROOT, build_reference, sha256
from navigation_intent import NavigationIntent

CONTEXT = dict(map_id='s10_USS_CL', ship='PGSB210-Schlieffen', pattern=PATTERN,
               mechanism='autopilot_waypoint')


def summary(values):
    if not values:
        return dict(count=0, mean=None, p95=None, maximum=None)
    ordered = sorted(values)
    return dict(count=len(values), mean=sum(values) / len(values),
                p95=ordered[math.ceil(.95 * len(ordered)) - 1], maximum=ordered[-1])


def verify(observations, out, positions=DEFAULT_POSITIONS):
    reference = build_reference(positions_path=positions)
    consumer = NavigationIntent(reference)
    out.mkdir(parents=True, exist_ok=False)
    counts, deviations, errors, progress_errors = Counter(), [], [], []
    first = None
    final_progress = 0.0
    # Independent, unsampled position stream for offline metrics only.
    truth = []
    with Path(positions).open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get('event') == 'position':
                truth.append((row['clock'], row['x'], row['z']))
    truth.sort()
    clocks = [p[0] for p in truth]
    affine = reference['affine']
    matrix, translation = affine['matrix'], affine['translation']
    points = reference['waypoints']
    with Path(observations).open() as source, (out / 'intents.jsonl').open('w') as destination:
        for line in source:
            row = json.loads(line)
            if row['session'] != '236374':
                continue
            snapshot = row['observation']
            # Real recorded decisions take precedence over acquisition time.
            # Older saved simulations have no decision_ns and are explicitly
            # evaluated at acquisition, never at replay/container time.
            if 'decision_ns' in row:
                now = row['decision_ns']
                clock_source = 'decision_ns'
            else:
                now = snapshot.get('captured_ns')
                clock_source = 'captured_ns_simulation'
                if not isinstance(now, int) or isinstance(now, bool) or now < 0:
                    now = row.get('synthetic_capture_ns')
                    clock_source = 'synthetic_capture_ns_simulation'
            if not isinstance(now, int) or isinstance(now, bool) or now < 0:
                raise ValueError('observation requires a valid decision_ns or explicit acquisition clock; replay clock is metrics-only')
            intent = consumer.decide(snapshot, now_ns=now, **CONTEXT)
            if first is None and snapshot.get('fresh') and snapshot.get('own'):
                first = copy.deepcopy(snapshot)
            counts[intent['reason']] += 1
            final_progress = intent['progress']['s_px']
            if intent['deviation_px'] is not None:
                deviations.append(intent['deviation_px'])
            # All use of replay_clock_s is below the decision boundary.
            clock = row['replay_clock_s']
            index = bisect_right(clocks, clock) - 1
            metrics = None
            if index >= 0 and snapshot.get('own'):
                _, x, z = truth[index]
                tx = matrix[0][0] * x + matrix[0][1] * z + translation[0] - 1836
                ty = matrix[1][0] * x + matrix[1][1] * z + translation[1] - 796
                own = snapshot['own']
                error = math.hypot(tx - own['pixel_x'], ty - own['pixel_y'])
                errors.append(error)
                # Clock-based reference progress exists solely as a reporting comparator.
                eligible = [p for p in points if p['source_clock_s'] <= clock]
                truth_progress = eligible[-1]['s_px'] if eligible else 0.0
                progress_error = final_progress - truth_progress
                progress_errors.append(abs(progress_error))
                metrics = dict(own_error_px=error, reference_clock_progress_px=truth_progress,
                               signed_progress_difference_px=progress_error)
            destination.write(json.dumps(dict(sample=row['sample'], session=row['session'],
                                              intent=intent, decision_clock_source=clock_source,
                                              offline_metrics=metrics), allow_nan=False) + '\n')
    if first is None:
        raise ValueError('no usable own observations for session 236374')
    probes = {}
    cases = dict(stale=dict(now_ns=first['valid_until_ns'] + 1),
                 focus_loss=dict(focused=False), source_loss=dict(source_available=False),
                 unknown_mechanism=dict(mechanism='unknown'), unknown_pattern=dict(pattern='unknown'),
                 unknown_map=dict(map_id='unknown'), unknown_ship=dict(ship='unknown'))
    for name, override in cases.items():
        kwargs = dict(CONTEXT, now_ns=first['captured_ns'])
        kwargs.update(override)
        intent = NavigationIntent(reference).decide(first, **kwargs)
        probes[name] = intent
        if intent['movement_authorized'] or intent['waypoint'] is not None or intent['action'] != 'revoke':
            raise AssertionError('failed revoke probe: ' + name)
    source_probe = NavigationIntent(reference)
    source_probe.decide(first, now_ns=first['captured_ns'], **CONTEXT)
    changed = copy.deepcopy(first)
    changed['epoch'] += 1
    changed['seq'] = 0
    changed['captured_ns'] += 1
    changed['valid_until_ns'] += 1
    changed_result = source_probe.decide(changed, now_ns=changed['captured_ns'], **CONTEXT)
    probes['source_change'] = changed_result
    if changed_result['movement_authorized'] or changed_result['waypoint'] is not None:
        raise AssertionError('failed revoke probe: source_change')
    recovery = copy.deepcopy(changed)
    recovery['seq'] = 1
    recovery['captured_ns'] += 1
    recovery['valid_until_ns'] += 1
    recovery_result = source_probe.decide(recovery, now_ns=recovery['captured_ns'], **CONTEXT)
    probes['source_recovery'] = recovery_result
    if not recovery_result['movement_authorized'] or recovery_result['waypoint'] is None:
        raise AssertionError('failed recovery probe: source_change')
    report = dict(schema_version=1, mode='offline_saved_observations_explicit_reference_pattern',
                  identity=dict(observations_sha256=sha256(observations), reference_sha256=reference['reference_sha256'],
                                projection_sha256=reference['projection_sha256'], truth_sha256=sha256(positions)),
                  counts=dict(counts), final_progress_px=final_progress,
                  final_progress_fraction=final_progress / reference['length_px'],
                  bounded_projection_deviation_px=summary(deviations),
                  independent_position_error_px=summary(errors),
                  absolute_clock_comparator_progress_difference_px=summary(progress_errors),
                  revoke_probes=probes,
                  limitations=['Explicit reference mode, not a recognized or validated spawn mechanism',
                               'No tactical UI click calibration; no input authorized',
                               'Arrow angle is not calibrated yaw; no steering intent',
                               'Same-session observation agreement is not closed-loop navigation success',
                               'Observed route gives no island or collision clearance guarantee'])
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observations', type=Path, default=ROOT / 'state/minimap-tracks-heldout-resume/observations.jsonl')
    parser.add_argument('--positions', type=Path, default=DEFAULT_POSITIONS)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.observations, args.out, args.positions), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
