#!/usr/bin/env python3
"""Bounded real-pixel hazard recognition and leased continuity evidence; no input.

Commands (each --out must be new):
  .venv/bin/python tools/replay/verify_hazard_observation.py --case recognition --out state/hazard-recognition-current
  .venv/bin/python tools/replay/verify_hazard_observation.py --case scout --out state/hazard-scout-adjacent-current
  .venv/bin/python tools/replay/verify_hazard_observation.py --case danger --out state/hazard-danger-adjacent-current
  .venv/bin/python tools/replay/verify_hazard_observation.py --case torpedo --out state/hazard-torpedo-adjacent-current
  .venv/bin/python tools/replay/verify_hazard_observation.py --case scout --video recordings/236818.15.9/capture.mkv --times 146.383,146.583,146.783 --out state/hazard-scout-video-current

Saved aircraft adjacent images have sequence order and nominal fps=1 extraction,
not exact PTS. Their source timing stays null and cannot pass dense continuity.
Torpedo defaults to three bounded forward-decoded real frames at 410/.2/.4 s.
Actual monotonic acquisition precedes file/decode work; all work consumes the
unchanged 750ms lease. Standalone recognition diagnostics run after sampling and
expiry, so they cannot delay the integrated sampling loop. This is offline
consumer evidence, never live acquisition or calibrated/world movement.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

import aircraft_perception as aircraft
import torpedo_perception as torpedo
from fire_control import FireControl
from main_view_observation import ContinuousMainViewRecognizer
from minimap_observation import FrameStamp
from verify_main_view import parse_times
from verify_navigation_frames import NativeVideoSequence

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
EVIDENCE = ROOT / 'state/aircraft-consumer-20261015'


def fingerprint(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB'))


def standalone(rgb):
    started = time.monotonic_ns()
    markers = aircraft.detect(rgb) + torpedo.detect(rgb)
    return dict(markers=markers, elapsed_ms=(time.monotonic_ns()-started)/1e6,
                semantics='raw_current_frame_marker_recognizers; no lease, HUD gate or persistent IDs')


def recognition_cases():
    """Full saved RGB positives plus explicitly static crop-placement controls."""
    cases = []
    for name, filename, kind, expected in (
        ('scout-full', '236818.15.9-guided-146.383s.png', 'aircraft', 'scout'),
        ('danger-full', '236925.15.9-guided-1061.937s.png', 'aircraft', 'central_dangers'),
    ):
        cases.append(dict(name=name, path=EVIDENCE / filename, kind=kind,
                          expected=expected, placement=None))
    for name, filename, kind, expected in (
        ('torpedo-real-normal', 'torpedo_fixtures/normal-recording-236374-410s.png', 'torpedo', 'normal'),
        ('torpedo-alert-recording-rejection', 'torpedo_fixtures/alert-lookalike-236374-410s.png', 'torpedo', None),
        ('torpedo-alert-screenshot-rejection', 'torpedo_fixtures/alert-lookalike-aimprobe.png', 'torpedo', None),
        ('aircraft-ship-label-rejection', 'aircraft_fixtures/ship-label-native-context.png', 'aircraft', None),
        ('aircraft-building-label-rejection', 'aircraft_fixtures/aerodrome-native-context.png', 'aircraft', None),
        ('aircraft-torpedo-rejection', 'torpedo_fixtures/normal-recording-236374-410s.png', 'aircraft', None),
        ('aircraft-alert-rejection', 'torpedo_fixtures/alert-lookalike-236374-410s.png', 'aircraft', None),
    ):
        cases.append(dict(name=name, path=HERE / filename, kind=kind,
                          expected=expected, placement=[800, 480]))
    return cases


def matches_expected(markers, case):
    relevant = [marker for marker in markers if marker['class'] == case['kind']]
    expected = case['expected']
    if expected is None:
        return not relevant
    if expected == 'scout':
        return any(marker.get('aircraft_type') == 'scout' and marker.get('team') == 'ally'
                   and marker.get('marker_family') == 'consumables' for marker in relevant)
    if expected == 'central_dangers':
        return any(marker.get('marker_family') == expected and marker.get('team') == 'unknown'
                   and marker.get('aircraft_type') == 'unknown' for marker in relevant)
    return any(expected in marker.get('marker_type_candidates', []) for marker in relevant)


def continuity_report(records):
    """Hazard IDs only; source, acquisition, decision and expiry gates all count."""
    runs, dense_runs, longest, longest_dense = {}, {}, 0, 0
    pairs, frame_validity, states = [], [], Counter()
    previous = None
    for row in records:
        observation = row['observation']
        expiry = observation['valid_until_ns']
        fresh = (observation['fresh'] and expiry is not None
                 and row['captured_ns'] <= row['decision_ns'] < expiry)
        frame_validity.append(bool(fresh))
        linked = False
        dense = False
        if previous is not None:
            prior = previous['observation']
            source_dt = (row['video_time_s']-previous['video_time_s']
                         if row['video_time_s'] is not None and previous['video_time_s'] is not None else None)
            capture_dt = (row['captured_ns']-previous['captured_ns'])/1e9
            decision_dt = (row['decision_ns']-previous['decision_ns'])/1e9
            prior_expiry = prior['valid_until_ns']
            covered = prior_expiry is not None and row['decision_ns'] < prior_expiry
            linked = observation['epoch'] == prior['epoch'] and observation['seq'] == prior['seq']+1
            dense = (fresh and frame_validity[-2] and linked and covered
                     and source_dt is not None and 0 < source_dt <= .5+1e-6
                     and 0 < capture_dt <= .5 and 0 < decision_dt <= .5)
            pairs.append(dict(seq=observation['seq'], source_dt_s=source_dt,
                acquisition_dt_s=capture_dt, decision_dt_s=decision_dt,
                previous_lease_covers_decision=covered, dense=bool(dense)))
        current, current_dense = {}, {}
        view = observation.get('view')
        for marker in view.get('markers', []) if fresh and view is not None else []:
            states[marker.get('track_state', 'unknown')] += 1
            track_id = marker.get('track_id')
            if track_id is None:
                continue
            tracked = linked and marker.get('track_state') == 'tracked'
            current[track_id] = runs.get(track_id, 0)+1 if tracked else 1
            current_dense[track_id] = dense_runs.get(track_id, 0)+1 if tracked and dense else 1
        longest = max(longest, max(current.values(), default=0))
        longest_dense = max(longest_dense, max(current_dense.values(), default=0))
        runs, dense_runs, previous = current, current_dense, row
    return dict(fresh_frames=sum(frame_validity), marker_track_states=dict(states),
        longest_same_id_run_frames=longest, longest_dense_same_id_run_frames=longest_dense,
        cadence=pairs, entire_sequence_dense_and_fresh=(len(records) >= 2
            and all(frame_validity) and all(pair['dense'] for pair in pairs)),
        dense_hazard_continuity_demonstrated=longest_dense >= 2,
        semantics='consecutive HUD sprite evidence; no entity identity or world velocity')


def sample(inputs, video=None):
    consumer, policy = ContinuousMainViewRecognizer(), FireControl()
    source = NativeVideoSequence(video) if video else None
    records, frames = [], []
    deadline = time.monotonic()
    try:
        for seq, item in enumerate(inputs):
            remaining = deadline-time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            captured = time.monotonic_ns()
            if source is not None:
                rgb, actual_s, skipped = source.read_at(item['requested_video_time_s'])
            else:
                rgb, actual_s, skipped = load_rgb(item['path']), None, 0
            observed = consumer.observe(rgb, FrameStamp(0, seq, captured))
            decision = time.monotonic_ns()
            intent = policy.decide(observed, now_ns=decision, focused=True)
            if intent['fire_authorized']:
                raise AssertionError('hazard evidence authorized fire')
            records.append(dict(seq=seq, input=str(item.get('path', video)),
                requested_video_time_s=item.get('requested_video_time_s'), video_time_s=actual_s,
                skipped_video_frames=skipped, captured_ns=captured, decision_ns=decision,
                elapsed_ms=(decision-captured)/1e6, observation=observed, intent=intent))
            frames.append(rgb)
            if seq+1 < len(inputs):
                deadline += inputs[seq+1]['schedule_s']-item['schedule_s']
    finally:
        if source is not None:
            source.close()
    last_expiry = records[-1]['observation']['valid_until_ns']
    if last_expiry is not None:
        remaining = last_expiry+1-time.monotonic_ns()
        if remaining > 0:
            time.sleep(remaining/1e9)
    expired_at = time.monotonic_ns()
    expired = consumer.snapshot(expired_at)
    intent = policy.decide(expired, now_ns=expired_at, focused=True)
    if expired['fresh'] or expired['view'] is not None or intent['fire_authorized']:
        raise AssertionError('expiry retained hazard observations or fire authority')
    return records, frames, dict(now_ns=expired_at, observation=expired, intent=intent)


def identities():
    sources = ('verify_hazard_observation.py', 'aircraft_perception.py', 'torpedo_perception.py',
        'main_view_observation.py', 'main_view_perception.py', 'main_view_tracks.py',
        'fire_hud.py', 'fire_control.py', 'minimap_observation.py', 'target_confirmation.py',
        'verify_main_view.py', 'verify_navigation_frames.py')
    assets = sorted(aircraft.ASSET_ROOT.glob('plane/**/*.png'))
    assets += sorted(aircraft.ASSET_ROOT.glob('ship/*.png'))
    assets += sorted(aircraft.ASSET_ROOT.glob('mines/*.png'))
    assets += sorted((aircraft.ASSET_ROOT.parent / 'central_dangers/icons').glob('icon_danger_plane_*.png'))
    assets += sorted(torpedo.ASSET_ROOT.glob('*.png'))
    # Include all loaded positive and confuser assets, including state/torpedo symbols.
    for templates in aircraft._templates():
        assets += [aircraft.ASSET_ROOT / template.asset for template in templates]
    return dict(code={name:fingerprint(HERE / name) for name in sources},
                assets={str(path.relative_to(ROOT)):fingerprint(path) for path in sorted(set(assets))},
                aircraft_fixture_provenance_sha256=fingerprint(HERE / 'aircraft_fixtures/provenance.json'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('recognition', 'scout', 'danger', 'torpedo'), required=True)
    parser.add_argument('--video', type=Path)
    parser.add_argument('--times', nargs='+')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--opencv-threads', type=int, default=8,
                        help='bounded OpenCV pool alongside detector threads (default: 8)')
    args = parser.parse_args(argv)
    if not 1 <= args.opencv_threads <= 32:
        parser.error('--opencv-threads must be 1..32')
    cv2.setNumThreads(args.opencv_threads)
    if args.case == 'recognition' and (args.video or args.times):
        parser.error('recognition uses the bounded saved evidence cases')
    if args.times and not args.video:
        parser.error('--times requires --video')
    if args.video and not args.times:
        parser.error('--video requires --times')
    inputs, cases, video = [], None, args.video
    if args.case == 'recognition':
        cases = recognition_cases()
        # Crops are standalone controls, not complete integrated battle inputs.
        inputs = [dict(path=case['path'], schedule_s=index) for index, case in enumerate(cases[:2])]
    elif args.video or args.case == 'torpedo':
        video = args.video or ROOT / 'recordings/236374/capture.mkv'
        times = parse_times(args.times) if args.times else [410., 410.2, 410.4]
        if len(times) > 20 or any(b <= a for a, b in zip(times, times[1:])) or times[-1]-times[0] > 5:
            parser.error('require <=20 increasing video times spanning <=5 seconds')
        inputs = [dict(requested_video_time_s=value, schedule_s=value) for value in times]
    else:
        session = '236818.15.9' if args.case == 'scout' else '236925.15.9'
        inputs = [dict(path=EVIDENCE / f'{session}-adjacent-{index:02d}.png', schedule_s=index)
                  for index in range(1, 5)]
    args.out.mkdir(parents=True, exist_ok=False)
    records, frames, expiry = sample(inputs, video)
    for row, rgb in zip(records, frames):
        row['rgb_sha256'] = hashlib.sha256(rgb).hexdigest()
        if video is None:
            row['source_sha256'] = fingerprint(row['input'])
        row['standalone'] = standalone(rgb)
    checks = []
    if cases is not None:
        for index, case in enumerate(cases):
            rgb = frames[index] if index < 2 else load_rgb(case['path'])
            if case['placement'] is not None:
                crop = rgb
                rgb = np.full(aircraft.FRAME_SHAPE, 96, np.uint8)
                x, y = case['placement']
                h, w = crop.shape[:2]
                rgb[y:y+h, x:x+w] = crop
            raw = records[index]['standalone'] if index < 2 else standalone(rgb)
            checks.append(dict(name=case['name'], source=str(case['path']),
                source_sha256=fingerprint(case['path']), kind=case['kind'], expected=case['expected'],
                static_crop_placement_xy=case['placement'], standalone=raw,
                passed=matches_expected(raw['markers'], case),
                evidence_scope='untouched_full_frame' if case['placement'] is None else
                    'untouched_real_crop_on_neutral_canvas; static recognition control only'))
    identity = identities()
    if video is not None:
        stat = video.stat()
        identity['video'] = dict(path=str(video), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
            sha256=None, identity_scope='filesystem identity plus per-decoded-RGB SHA256; video not fully scanned')
    report = dict(case=args.case, mode='offline_real_saved_inputs_actual_monotonic_consumer',
        input_injected=False, identity=identity, records=records, recognition_checks=checks,
        opencv_threads=cv2.getNumThreads(),
        recognition_checks_passed=all(check['passed'] for check in checks) if checks else None,
        expiry_probe=expiry, continuity=continuity_report(records),
        provenance=dict(saved_adjacent_nominal_extraction_fps=1 if video is None and cases is None else None,
            source_timing='decoded PTS' if video else 'unknown; sequence order only',
            historical_evidence=str(EVIDENCE / 'recognition-adaptation-evidence.json'),
            historical_evidence_sha256=fingerprint(EVIDENCE / 'recognition-adaptation-evidence.json')),
        constraints=['750ms lease includes acquisition/decode and processing; no synthetic clock',
            'Dense requires source/acquisition/decision intervals <=500ms and previous lease coverage',
            'Only fresh integrated view.markers count for continuity; raw marker output is separate',
            'Same known subtype/team, camera score >=.35, residual <60px; isolated or glyph-separated two-way nearest match required',
            'Central dangers and unknown subtype cannot produce entity tracks or velocity',
            'Compensated screen-marker px/s is not physical/world velocity, wake or trajectory',
            'Crop placement is static control only; no shifted or composited input counts as movement',
            'No real-positive coverage claimed for torpedo families beyond normal',
            'Missing output means unknown; current recognizer runs do not inherit historical results',
            'No calibrated firing, visible aircraft bodies, plane counts or entity identity established'])
    (args.out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(out=str(args.out), recognition_checks_passed=report['recognition_checks_passed'],
        continuity=report['continuity']), indent=2))
    return 1 if checks and not report['recognition_checks_passed'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
