#!/home/lengbai/Downloads/wws/.venv/bin/python
"""Exercise native frame -> leased main-view observation -> FireControl, no input."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import cv2
from PIL import Image

from fire_control import FireControl
from main_view_observation import ContinuousMainViewRecognizer
from minimap_observation import FrameStamp
from scope_observation import ScopeObservationPlanner
from verify_main_view import parse_times
from verify_navigation_frames import NativeVideoSequence
from verify_target_confirmation import annotate
from fsutil import sha256



def continuity_report(records):
    """Separate same-ID evidence from uninterrupted, dense fresh decisions."""
    runs, dense_runs = {}, {}
    longest = longest_dense = 0
    gaps, cadence, decision_validity = [], [], []
    previous = None
    for row in records:
        observed = row['observation']
        fresh = (observed['fresh'] and observed['valid_until_ns'] is not None
                 and row['captured_ns'] <= row['decision_ns'] < observed['valid_until_ns'])
        decision_validity.append(bool(fresh))
        dense_pair = False
        if previous is not None:
            acquisition_dt = (row['captured_ns']-previous['captured_ns'])/1e9
            decision_dt = (row['decision_ns']-previous['decision_ns'])/1e9
            source_dt = (row['video_time_s']-previous['video_time_s']
                         if row['video_time_s'] is not None and
                         previous['video_time_s'] is not None else None)
            dense_pair = (0 < acquisition_dt <= .5 and 0 < decision_dt <= .5
                          and source_dt is not None and 0 < source_dt <= .5+1e-6
                          and observed['epoch'] == previous['observation']['epoch']
                          and observed['seq'] == previous['observation']['seq']+1)
            cadence.append(dict(seq=observed['seq'], source_dt_s=source_dt,
                requested_dt_s=row['requested_video_time_s']-previous['requested_video_time_s'],
                acquisition_dt_s=acquisition_dt, decision_dt_s=decision_dt,
                dense=dense_pair))
            expiry = previous['observation']['valid_until_ns']
            if expiry is None or row['decision_ns'] >= expiry:
                gaps.append(dict(seq=observed['seq'], previous_valid_until_ns=expiry,
                    next_decision_ns=row['decision_ns'],
                    uncovered_ns=row['decision_ns']-expiry if expiry is not None else None))
                dense_pair = False
        current, current_dense = {}, {}
        if fresh and observed.get('view') is not None:
            for detection in observed['view']['detections']:
                track_id = detection.get('track_id')
                if track_id is None:
                    continue
                linked = (detection.get('track_state') == 'tracked' and previous is not None
                          and observed['epoch'] == previous['observation']['epoch']
                          and observed['seq'] == previous['observation']['seq']+1)
                current[track_id] = runs.get(track_id, 0)+1 if linked else 1
                current_dense[track_id] = dense_runs.get(track_id, 0)+1 if linked and dense_pair else 1
            longest = max(longest, max(current.values(), default=0))
            longest_dense = max(longest_dense, max(current_dense.values(), default=0))
        runs, dense_runs = current, current_dense
        previous = row
    return dict(fresh_frames=sum(decision_validity),
        longest_same_id_run_frames=longest, longest_dense_same_id_run_frames=longest_dense,
        freshness_gaps=gaps, cadence=cadence,
        entire_sequence_dense_and_fresh=(len(records) >= 2 and not gaps
            and all(c['dense'] for c in cadence) and all(decision_validity)),
        semantics='screen evidence only; same ID does not establish hull identity or firing authority')



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--video', type=Path)
    source.add_argument('--image', type=Path)
    parser.add_argument('--times', nargs='+')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--opencv-threads', type=int, default=8,
                        help='bounded OpenCV pool alongside detector threads (default: 8)')
    args = parser.parse_args()
    if not 1 <= args.opencv_threads <= 32:
        parser.error('--opencv-threads must be 1..32')
    cv2.setNumThreads(args.opencv_threads)
    if bool(args.video) != bool(args.times):
        parser.error('--times is required only for video')
    times = parse_times(args.times) if args.video else [0.0]
    if len(times) > 20 or any(b <= a for a, b in zip(times, times[1:])):
        parser.error('require <=20 strictly increasing frame times')
    args.out.mkdir(parents=True, exist_ok=False)
    consumer, policy = ContinuousMainViewRecognizer(), FireControl()
    scope_policy = ScopeObservationPlanner()
    records, counts = [], Counter()
    previews = []
    source = NativeVideoSequence(args.video) if args.video else None
    deadline = time.monotonic()
    try:
        for seq, video_s in enumerate(times):
            remaining = deadline-time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            captured = time.monotonic_ns()
            if source is not None:
                rgb, actual_video_s, skipped = source.read_at(video_s)
            else:
                with Image.open(args.image) as image:
                    rgb = np.asarray(image.convert('RGB'))
                actual_video_s, skipped = None, 0
            observation = consumer.observe(rgb, FrameStamp(0, seq, captured))
            now = time.monotonic_ns()
            intent = policy.decide(observation, now_ns=now, focused=True)
            scope_intent = scope_policy.decide(observation, now_ns=now, focused=True)
            if intent['fire_authorized']:
                raise AssertionError('unconfirmed hull/camera authorized firing')
            if not observation['fresh'] and observation['view'] is not None:
                raise AssertionError('expired snapshot retained visual observations')
            records.append(dict(video_time_s=actual_video_s, requested_video_time_s=video_s,
                skipped_video_frames=skipped, captured_ns=captured, decision_ns=now,
                elapsed_ms=(now-captured)/1e6,
                observation=observation, intent=intent, scope_intent=scope_intent))
            counts[observation['reason']] += 1
            previews.append(rgb)
            records[-1]['consumer_completed_ns'] = time.monotonic_ns()
            if seq+1 < len(times):
                deadline += times[seq+1]-video_s
    finally:
        if source is not None:
            source.close()
    last = records[-1]['observation']
    expired_at = (last['valid_until_ns'] or 0)+1
    remaining_ns = expired_at-time.monotonic_ns()
    if remaining_ns > 0:
        time.sleep(remaining_ns/1e9)
    expired_at = time.monotonic_ns()
    expired = consumer.snapshot(expired_at)
    intent = policy.decide(expired, now_ns=expired_at, focused=True)
    scope_intent = scope_policy.decide(expired, now_ns=expired_at, focused=True)
    if expired['fresh'] or expired['view'] is not None or expired['lock']['confirmed'] or intent['fire_authorized']:
        raise AssertionError('expiry retained target or fire authority')
    # Bounded by the existing 20-frame limit (at most 246 MB of RGB). Evidence
    # encoding happens after sampling and the expiry probe, never between frames.
    for seq, rgb in enumerate(previews):
        record = records[seq]
        record['rgb_sha256'] = hashlib.sha256(rgb).hexdigest()
        observed = record['observation']
        preview = (annotate(rgb, observed['view']['detections'], observed['lock'])
                   if observed['fresh'] else rgb)
        Image.fromarray(preview).save(args.out / f'frame-{seq:03d}.png')
    report = dict(mode='offline_native_frames_actual_monotonic_processing_clock',
        source=str(args.video or args.image), input_injected=False, counts=dict(counts),
        opencv_threads=cv2.getNumThreads(),
        records=records, expiry_probe=dict(observation=expired, intent=intent, scope_intent=scope_intent),
        continuity=continuity_report(records),
        identity=dict(source_sha256=sha256(args.video or args.image),
            sources={name:sha256(Path(__file__).with_name(name)) for name in
                ('main_view_observation.py','main_view_tracks.py','main_view_perception.py','target_confirmation.py',
                 'fire_hud.py','fire_control.py','scope_observation.py','aircraft_perception.py','torpedo_perception.py',
                 'verify_main_view_observation.py', 'verify_navigation_frames.py')}),
        decoder='pyav_synchronous_forward' if args.video else 'saved_image',
        limits=['Frame decode and processing consume 750ms lease; file request is not live capture',
                'Video PTS is provenance only; actual monotonic acquisition drives tracking',
                'At most 20 RGB frames retained; hashes, annotation and PNG written after sampling and expiry',
                'UI lock is not hull/identity confirmation; target stays unknown',
                'Screen-motion evidence is not calibrated world motion or firing accuracy'])
    (args.out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(out=str(args.out), frames=len(records), counts=dict(counts),
        elapsed_ms=[r['elapsed_ms'] for r in records], fire_authorized=False,
        expiry_reason=expired['reason']), indent=2))


if __name__ == '__main__':
    main()
