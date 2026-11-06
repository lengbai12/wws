#!/home/lengbai/Downloads/wws/.venv/bin/python
"""Exercise raw video -> current RGB recognizer -> expiring navigation Intent.

Video timestamps are provenance only. One synchronous decoder is paced on a real
monotonic clock; acquisition, skipped-frame decode and inference consume the frame
lease. This never injects input or establishes closed-loop navigation success.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import math
import time

import av
import numpy as np
from PIL import Image, ImageDraw

from build_navigation_reference import build_reference
from minimap_observation import ContinuousMinimapRecognizer, FrameStamp
from navigation_intent import NavigationIntent, finite_number
from verify_navigation_intent import CONTEXT
from report_replay_meta import read_meta


def fingerprint(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def recording_identity(video, reference):
    """Check stored replay provenance before assigning offline route context.

    This checks the recording's declared binding, not scene recognition. It
    cannot prove that a mislabeled or edited video shows the declared replay.
    """
    binding = Path(video).parent / 'replay_path.txt'
    try:
        replay = Path(binding.read_text().strip())
        if not replay.is_absolute():
            raise ValueError('replay binding must be an absolute path')
        meta = read_meta(replay)
    except (OSError, ValueError) as error:
        raise ValueError(f'invalid recording replay binding: {binding}: {error}') from error
    actual = (replay.stem, (meta.get('mapName') or '').removeprefix('spaces/'),
              meta.get('playerVehicle'))
    expected = (reference['session'], reference['map'], reference['ship'])
    if actual != expected:
        raise ValueError(f'recording identity {actual!r} does not match reference {expected!r}')
    return dict(session=actual[0], map=actual[1], ship=actual[2],
                replay=str(replay), replay_sha256=fingerprint(replay),
                binding_sha256=fingerprint(binding),
                evidence='stored replay binding and replay header; not image-recognized identity')


class NativeVideoSequence:
    """Seek once and decode forward synchronously with frame PTS provenance.

    read_at must follow the offline acquisition stamp. Container timestamps
    never become capture timestamps; AV1 uses PyAV's software decoder.
    """
    def __init__(self, path):
        self.path = path
        self.container = None
        self.frames = None
        self.previous_video_s = None

    def read_at(self, target_s):
        # Backend import is process setup; seek/decode remain inside read_at
        # and therefore inside the caller's acquisition-age accounting.
        if self.container is None:
            self.container = av.open(str(self.path))
            stream = self.container.streams.video[0]
            if (stream.width, stream.height) != (2560, 1600):
                raise ValueError('video frame must be native 2560x1600')
            stream.codec_context.thread_count = 2
            self.container.seek(int(target_s / stream.time_base), stream=stream,
                                backward=True, any_frame=False)
            self.frames = self.container.decode(stream)
        skipped = 0
        for frame in self.frames:
            if frame.pts is None or frame.time_base is None:
                raise ValueError('video backend returned missing timestamps')
            video_s = float(frame.pts * frame.time_base)
            if (not math.isfinite(video_s) or video_s < 0 or
                    (self.previous_video_s is not None and
                     video_s <= self.previous_video_s)):
                raise ValueError('video backend returned missing/regressing timestamps')
            self.previous_video_s = video_s
            if video_s + 1e-6 >= target_s:
                rgb = frame.to_ndarray(format='rgb24')
                if rgb.shape != (1600, 2560, 3):
                    raise ValueError('video frame must be native 2560x1600 RGB')
                return rgb, video_s, skipped
            skipped += 1
        raise EOFError(f'video ended before requested time {target_s}')

    def close(self):
        if self.container is not None:
            self.container.close()


def continuity_report(records):
    """Count measured ID/waypoint runs across fresh same-source frames.

    Waypoint evidence needs explicit reference identity and finite projected
    progress; regressing progress cannot join a continuous navigation run.
    """
    states = Counter()
    confirmed_ids, usable_ids = set(), set()
    runs, longest = {}, 0
    gaps, cadence_failures, identity_failures = [], [], []
    navigation_run = longest_navigation_run = waypoint_frames = 0
    navigation_gaps = []
    decision_validity = []
    previous = None
    for row in records:
        observation = row['observation']
        observation_expiry = observation.get('valid_until_ns')
        fresh = (observation.get('fresh') is True
            and isinstance(observation_expiry, int) and not isinstance(observation_expiry, bool)
            and row['captured_ns'] <= row['decision_ns'] < observation_expiry)
        decision_validity.append(bool(fresh))
        dense_pair = False
        track_identity_pair = False
        if previous is not None:
            previous_observation = previous['observation']
            source_dt = row['video_time_s'] - previous['video_time_s']
            capture_dt = (row['captured_ns'] - previous['captured_ns']) / 1e9
            decision_dt = (row['decision_ns'] - previous['decision_ns']) / 1e9
            dense_pair = (0 < source_dt <= .5 + 1e-6 and
                          0 < capture_dt <= .5 and 0 < decision_dt <= .5)
            previous_epoch = previous_observation.get('epoch')
            current_epoch = observation.get('epoch')
            previous_seq = previous_observation.get('seq')
            current_seq = observation.get('seq')
            track_identity_pair = (isinstance(previous_epoch, int)
                and not isinstance(previous_epoch, bool)
                and isinstance(current_epoch, int) and not isinstance(current_epoch, bool)
                and current_epoch == previous_epoch
                and isinstance(previous_seq, int) and not isinstance(previous_seq, bool)
                and isinstance(current_seq, int) and not isinstance(current_seq, bool)
                and current_seq == previous_seq + 1)
            if not dense_pair:
                cadence_failures.append(dict(seq=row['seq'], source_dt_s=source_dt,
                    acquisition_dt_s=capture_dt, decision_dt_s=decision_dt))
            if not track_identity_pair:
                identity_failures.append(dict(seq=row['seq'], previous_epoch=previous_epoch,
                    epoch=current_epoch, previous_seq=previous_seq, observation_seq=current_seq))
            valid_until = previous_observation['valid_until_ns']
            if valid_until is None or row['decision_ns'] >= valid_until:
                gaps.append(dict(seq=row['seq'], previous_valid_until_ns=valid_until,
                    next_decision_ns=row['decision_ns'],
                    uncovered_ns=(row['decision_ns'] - valid_until
                                  if valid_until is not None else None)))
                dense_pair = False
        intent = row.get('intent', {})
        own = observation.get('own')
        waypoint = intent.get('waypoint')
        own_valid = (isinstance(own, dict)
            and all(finite_number(own.get(key)) for key in ('pixel_x', 'pixel_y'))
            and 0 <= own['pixel_x'] < 720 and 0 <= own['pixel_y'] < 723)
        waypoint_geometry_valid = (isinstance(waypoint, dict)
            and all(finite_number(waypoint.get(key)) for key in ('x', 'y'))
            and 0 <= waypoint['x'] < 720 and 0 <= waypoint['y'] < 723)
        expiry = intent.get('valid_until_ns')
        reference = intent.get('reference')
        reference_valid = (isinstance(reference, dict)
            and all(isinstance(reference.get(key), str) and reference[key]
                    for key in ('map', 'ship', 'pattern'))
            and all(isinstance(reference.get(key), str) and len(reference[key]) == 64
                    and all(c in '0123456789abcdef' for c in reference[key])
                    for key in ('reference_sha256', 'projection_sha256')))
        progress = intent.get('progress', {})
        progress_valid = (isinstance(progress, dict)
            and progress.get('position_projected') is True
            and finite_number(progress.get('s_px')) and progress['s_px'] >= 0
            and finite_number(progress.get('fraction')) and 0 <= progress['fraction'] <= 1)
        waypoint_valid = (fresh and own_valid
            and intent.get('action') == 'autopilot_waypoint'
            and intent.get('movement_authorized') is True
            and waypoint_geometry_valid
            and reference_valid and progress_valid
            and isinstance(expiry, int) and not isinstance(expiry, bool)
            and row['decision_ns'] < expiry <= observation['valid_until_ns'])
        navigation_pair = dense_pair
        if previous is not None and navigation_run:
            previous_intent = previous['intent']
            previous_expiry = previous_intent['valid_until_ns']
            if row['decision_ns'] >= previous_expiry:
                navigation_gaps.append(dict(seq=row['seq'],
                    previous_valid_until_ns=previous_expiry,
                    next_decision_ns=row['decision_ns'],
                    uncovered_ns=row['decision_ns']-previous_expiry))
                navigation_pair = False
            if (not track_identity_pair
                    or intent.get('reference') != previous_intent.get('reference')):
                navigation_pair = False
            if (progress_valid and previous_intent.get('progress')
                    and (progress['s_px'] < previous_intent['progress']['s_px']
                         or progress['fraction'] < previous_intent['progress']['fraction'])):
                navigation_pair = False
        if waypoint_valid:
            waypoint_frames += 1
            navigation_run = navigation_run + 1 if navigation_pair else 1
        else:
            navigation_run = 0
        longest_navigation_run = max(longest_navigation_run, navigation_run)
        current = {}
        for track in observation['enemies']:
            states[track['state']] += 1
            if track['state'] == 'confirmed':
                confirmed_ids.add(track['track_id'])
            if fresh and track['observed'] and track['usable']:
                usable_ids.add(track['track_id'])
                current[track['track_id']] = (runs.get(track['track_id'], 0) + 1
                                             if dense_pair and track_identity_pair else 1)
        runs = current
        longest = max(longest, max(runs.values(), default=0))
        previous = row
    demonstrated = longest >= 2
    return dict(fresh_frames=sum(decision_validity),
        track_states=dict(states), confirmed_track_ids=sorted(confirmed_ids),
        usable_track_ids=sorted(usable_ids),
        confirmed_frames=sum(valid and any(t['state'] == 'confirmed' for t in r['observation']['enemies'])
                             for r, valid in zip(records, decision_validity)),
        usable_frames=sum(valid and any(t['usable'] for t in r['observation']['enemies'])
                         for r, valid in zip(records, decision_validity)),
        freshness_gaps=gaps, cadence_failures=cadence_failures,
        identity_failures=identity_failures,
        longest_dense_usable_id_run_frames=longest,
        dense_track_continuity_demonstrated=demonstrated,
        entire_sequence_dense_and_fresh=(len(records) > 1 and not gaps and
            not cadence_failures and not identity_failures and all(decision_validity)),
        navigation=dict(valid_waypoint_frames=waypoint_frames,
            intent_expiry_gaps=navigation_gaps,
            longest_dense_waypoint_run_frames=longest_navigation_run,
            dense_waypoint_continuity_demonstrated=longest_navigation_run >= 2,
            entire_sequence_dense_with_valid_waypoints=(len(records) > 1 and
                longest_navigation_run == len(records)),
            input_execution_verified=False),
        verdict=('measured_same_usable_id_across_dense_adjacent_decisions' if demonstrated
                 else 'dense_usable_track_continuity_not_demonstrated'))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--start', type=float, default=50.0)
    parser.add_argument('--frames', type=int, default=12)
    parser.add_argument('--interval', type=float, default=.5)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if (not np.isfinite(args.start) or args.start < 0 or not 2 <= args.frames <= 120
            or not np.isfinite(args.interval) or not .1 <= args.interval <= 5):
        parser.error('require start >=0, 2..120 frames, interval .1..5 seconds')
    reference = build_reference()
    recording = recording_identity(args.video, reference)
    context = dict(CONTEXT, map_id=recording['map'], ship=recording['ship'])
    args.out.mkdir(parents=True, exist_ok=False)
    setup_started_ns = time.monotonic_ns()
    recognizer = ContinuousMinimapRecognizer(args.package)
    policy = NavigationIntent(reference)
    rect = reference['coordinate']['rect']
    x, y, width, height = rect
    # Resolve lazy inference kernels before accepting frames. This is not an
    # observation: never feed setup pixels into tracking or navigation state.
    warmup_started_ns = time.monotonic_ns()
    recognizer.recognizer.detect(np.zeros((height, width, 3), dtype=np.uint8))
    warmup_ns = time.monotonic_ns() - warmup_started_ns
    setup_ns = time.monotonic_ns() - setup_started_ns
    counts = Counter()
    records = []
    previews = []
    source = NativeVideoSequence(args.video)
    deadline = time.monotonic()
    try:
        for seq in range(args.frames):
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            requested_s = args.start + seq * args.interval
            # Actual offline acquisition boundary, before open/seek/decode. Never
            # replace this with video PTS or reset it after expensive work.
            captured = time.monotonic_ns()
            frame, video_s, skipped = source.read_at(requested_s)
            acquired = time.monotonic_ns()
            crop = frame[y:y+height, x:x+width]
            recognizer.observe(crop, FrameStamp(0, seq, captured))
            inferred = time.monotonic_ns()
            crop_hash = hashlib.sha256(crop.tobytes()).hexdigest()
            now = time.monotonic_ns()
            observation = recognizer.observations.snapshot(now)
            intent = policy.decide(observation, now_ns=now, **context)
            processed = time.monotonic_ns()
            if intent['movement_authorized'] and (not observation['fresh']
                    or intent['valid_until_ns'] > observation['valid_until_ns']):
                raise AssertionError('navigation exceeds observation lease')
            counts[intent['reason']] += 1
            row = dict(seq=seq, requested_video_time_s=requested_s, video_time_s=video_s,
                video_target_offset_s=video_s-requested_s, skipped_video_frames=skipped,
                captured_ns=captured, acquired_ns=acquired, decision_ns=now,
                processing_ns=processed-captured, acquisition_ns=acquired-captured,
                inference_ns=inferred-acquired, crop_sha256=crop_hash,
                observation=observation, intent=intent)
            records.append(row)
            if seq in (0, args.frames-1):
                # Retain only the two endpoint crops (about 3 MB total). Evidence
                # encoding must not delay the next acquisition or decision.
                previews.append((seq, crop.copy()))
            row['consumer_completed_ns'] = time.monotonic_ns()
            row['consumer_ns'] = row['consumer_completed_ns'] - captured
            deadline += args.interval
    finally:
        source.close()
    # Actually let the last acquisition lease expire; no synthetic future clock.
    expiry_deadline = records[-1]['captured_ns'] + recognizer.observations.max_age_ns
    while True:
        remaining_ns = expiry_deadline - time.monotonic_ns()
        if remaining_ns <= 0:
            break
        time.sleep(remaining_ns / 1e9)
    expiry_now = time.monotonic_ns()
    expired = recognizer.observations.snapshot(expiry_now)
    revoked = policy.decide(expired, now_ns=expiry_now, **context)
    if (expired['fresh'] or expired['own'] is not None or expired['enemies']
            or revoked['movement_authorized'] or revoked['waypoint']):
        raise AssertionError('expired observation retained navigation authority')
    for seq, crop in previews:
        observation, intent = records[seq]['observation'], records[seq]['intent']
        preview = Image.fromarray(crop)
        draw = ImageDraw.Draw(preview)
        own = observation['own']
        if own:
            px, py = own['pixel_x'], own['pixel_y']
            draw.ellipse((px-7, py-7, px+7, py+7), outline='yellow', width=2)
            waypoint = intent['waypoint']
            if waypoint:
                draw.line((px, py, waypoint['x'], waypoint['y']), fill='cyan', width=2)
        preview.save(args.out / f'preview-{seq:03d}.png')
    with (args.out / 'observations.jsonl').open('x') as handle:
        for record in records:
            handle.write(json.dumps(record, allow_nan=False) + '\n')
    report = dict(mode='paced_offline_video_actual_processing_clock', input_injected=False,
        decoder='pyav_synchronous_native_forward_decode',
        capture_stamp='actual_monotonic_before_offline_acquisition_not_original_capture',
        video_timestamp='decoded_frame_PTS_provenance_only',
        setup_ms=setup_ns / 1e6, warmup_ms=warmup_ns / 1e6,
        requested_interval_s=args.interval,
        frames=len(records), counts=dict(counts), expiry_probe=revoked,
        expiry_observation=expired, expiry_probe_now_ns=expiry_now,
        expiry_deadline_ns=expiry_deadline, continuity=continuity_report(records),
        skipped_video_frames=sum(r['skipped_video_frames'] for r in records),
        processing_ms=[r['processing_ns']/1e6 for r in records],
        consumer_ms=[r['consumer_ns']/1e6 for r in records],
        acquisition_ms=[r['acquisition_ns']/1e6 for r in records],
        inference_ms=[r['inference_ns']/1e6 for r in records],
        identity=dict(video=str(args.video), video_sha256=fingerprint(args.video),
            recording=recording,
            package_sha256=fingerprint(args.package),
            reference=dict(map=reference['map'], ship=reference['ship'],
                pattern=reference['pattern'], session=reference['session'],
                coordinate=reference['coordinate'],
                reference_sha256=reference['reference_sha256'],
                projection_sha256=reference['projection_sha256'],
                identity_sha256=reference['identity_sha256']),
            reference_sha256=reference['reference_sha256'],
            sources={name:fingerprint(Path(__file__).with_name(name)) for name in
                ('verify_navigation_frames.py', 'minimap_observation.py',
                 'navigation_intent.py', 'build_navigation_reference.py',
                 'enemy_tracks.py', 'minimap_recognizer.py', 'report_replay_meta.py')}),
        limits=['Offline reacquisition clock is not original live capture latency',
                'Model-only zero-input warmup runs before acquisition and never updates observation or policy state',
                'Initial open/seek, forward decode, RGB conversion, ROI and inference consume the 750ms lease',
                'Hashing/policy/crop-copy overhead remains in consumer timing; two endpoint PNGs are encoded after sampling and expiry',
                'Frame PTS is checked for increasing values but remains container provenance',
                'Source, acquisition and decision cadence must each be <=500ms for a dense usable ID run',
                'Local usable ID runs do not imply gapless coverage of the entire sequence or real ship identity',
                'Actual monotonic expiry probe clears own/enemies and revokes navigation',
                'Explicit map/ship/reference context, not image-recognized spawn mode',
                'Recording identity is declared by replay_path.txt plus replay header, not recognized from pixels',
                'No ship dynamics, navigation input or collision avoidance validated'])
    (args.out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
