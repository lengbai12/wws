#!/home/lengbai/Downloads/wws/.venv/bin/python
"""Analyze time-resolved native reticle ROI evidence; never authorize firing."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from analyze_aim_probe import RETICLE_LOCAL_POLICY, reticle_local_motion

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools/replay'))
from fire_hud import FireHudReader


def stable_tail(pairs, min_duration_ns=500_000_000, tolerance_px=.5):
    """Earliest measured stable tail; every subsequent interval must pass.

    No interpolation across missing/ambiguous pairs. This is ROI stability on
    sampled frames, not a bound on unobserved motion between frames.
    """
    start = None
    count = 0
    displacement = np.zeros(2)
    positions = [displacement.copy()]
    previous_end = None
    for pair in pairs:
        fit = pair['motion']
        stable = (fit.get('sufficient') is True and
                  0 < pair['to_ns']-pair['from_ns'] <= 250_000_000 and
                  (previous_end is None or pair['from_ns'] == previous_end) and
                  np.hypot(fit['screen_dx_px'], fit['screen_dy_px']) <= tolerance_px)
        previous_end = pair['to_ns']
        if stable:
            displacement += [fit['screen_dx_px'], fit['screen_dy_px']]
            stable = all(np.linalg.norm(displacement-p) <= tolerance_px for p in positions)
        if not stable:
            start, count = None, 0
            displacement = np.zeros(2)
            positions = [displacement.copy()]
        else:
            if start is None:
                start = pair['from_ns']
            positions.append(displacement.copy())
            count += 1
    if start is None or count < 3 or pairs[-1]['to_ns'] - start < min_duration_ns:
        return None
    return dict(start_ns=start, end_ns=pairs[-1]['to_ns'], intervals=count,
                tolerance_px=tolerance_px, minimum_duration_ns=min_duration_ns,
                maximum_sample_gap_ns=250_000_000)


def analyze(path):
    report = json.loads(path.read_text())
    rejected = []
    if report.get('status') != 'completed' or not report.get('execute'):
        rejected.append('probe_not_completed')
    if report.get('rollback_input_sent') is not True or report.get('unreleased_own_keys') != []:
        rejected.append('rollback_or_release_unconfirmed')
    if report.get('roi_xyxy') != RETICLE_LOCAL_POLICY['roi_xyxy']:
        raise ValueError('reticle ROI must match existing local-motion policy')
    if not report.get('identity') or not report.get('source') or not report.get('focus_checks'):
        rejected.append('missing_identity_source_or_focus')
    rows = report['samples']
    if any(b['received_ns'] <= a['received_ns'] or b['source_pts_ns'] <= a['source_pts_ns']
           for a, b in zip(rows, rows[1:])):
        rejected.append('nonmonotonic_frames')
    if any(row['queue_age_ns'] > 100_000_000 or row['queue_age_ns'] < 0 for row in rows):
        rejected.append('invalid_source_queue_age')
    if not (report.get('pulse_started_ns', 0) <= report.get('pulse_completed_ns', 0)
            < report.get('rollback_started_ns', 0) <= report.get('rollback_completed_ns', 0)):
        rejected.append('invalid_input_time_order')
    for phase, boundary in (('response', report.get('pulse_completed_ns', 0)),
                            ('rollback', report.get('rollback_completed_ns', 0))):
        # Reject queued samples predating delivery; do not assign negative settling.
        if any(row['received_ns']-row['queue_age_ns'] <= boundary
               for row in rows if row['phase'] == phase):
            rejected.append(phase+': frame_may_predate_input')
    reader = FireHudReader()
    phases = {}
    for phase in ('baseline', 'response', 'rollback'):
        samples = [row for row in rows if row['phase'] == phase]
        pairs, previous, previous_row = [], None, None
        hud = None
        for row in samples:
            image_path = Path(row['path'])
            if hashlib.sha256(image_path.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('sample identity changed: '+str(image_path))
            crop = np.load(image_path, allow_pickle=False)
            if crop.shape != (490, 560, 3) or crop.dtype != np.uint8:
                raise ValueError('invalid native RGB ROI')
            if row.get('full_frame'):
                from PIL import Image
                full = np.asarray(Image.open(row['full_frame']).convert('RGB'))
                hud = reader.detect(full)
                if not np.array_equal(full[560:1050, 1000:1560], crop):
                    raise ValueError('full frame does not match saved ROI')
            image = np.zeros((1600, 2560, 3), np.uint8)
            image[560:1050, 1000:1560] = crop
            if previous is not None:
                fit = reticle_local_motion(previous, image)
                pairs.append(dict(from_ns=previous_row['received_ns'], to_ns=row['received_ns'],
                                  interval_ms=(row['received_ns']-previous_row['received_ns'])/1e6,
                                  motion=fit))
            previous, previous_row = image, row
        if len(samples) < 4:
            rejected.append(phase+': insufficient_samples')
        tail = stable_tail(pairs) if pairs else None
        if tail is None:
            rejected.append(phase+': no_quality_accepted_stable_tail')
        if (hud or {}).get('aiming_view', {}).get('mode') != 'scope_ruler':
            rejected.append(phase+': scope_ruler_unconfirmed')
        phases[phase] = dict(samples=len(samples), pairs=pairs, stable_tail=tail,
                             phase_start_hud=hud)
    response_tail = phases['response']['stable_tail']
    return dict(schema_version=1, report=dict(path=str(path.resolve()),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
                mode='time_resolved_local_screen_diagnostic', phases=phases,
                status='diagnostic_only' if not rejected else 'rejected', rejections=rejected,
                observed_stable_tail_delay_ms=((response_tail['start_ns']-report['pulse_completed_ns'])/1e6
                                               if not rejected and response_tail else None),
                calibration_valid=False, world_calibration_valid=False,
                zoom_confirmed=None, target_anchor_association='unknown',
                mouse_response=None,
                limits=['sampled ROI translation, not pointwise aim or hull tracking',
                        'receive timestamps include source delivery delay; render age unknown',
                        'phase-start HUD evidence does not prove camera state on every frame',
                        'no bound on between-sample motion or firing precision'],
                policy=RETICLE_LOCAL_POLICY)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.report)
    with args.out.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(dict(out=str(args.out.resolve()), status=result['status'],
                         rejections=result['rejections'],
                         observed_stable_tail_delay_ms=result['observed_stable_tail_delay_ms']), indent=2))


if __name__ == '__main__':
    main()
