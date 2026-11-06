#!/usr/bin/env python3
"""Exercise continuous observations on saved detections using original replay clocks.

Replay time drives a synthetic monotonic clock for offline testing only. No model
inference is rerun. No observed velocity is interpreted as world speed or heading.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np

from minimap_observation import FrameStamp, MinimapObservations
from fsutil import sha256



def small_member(path, key):
    with zipfile.ZipFile(path) as archive:
        member = archive.getinfo(key + '.npy')
        if member.file_size > 32 * 1024 * 1024:
            raise ValueError('timeline member unexpectedly large: ' + key)
        with archive.open(member) as handle:
            return np.lib.format.read_array(handle, allow_pickle=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    package = json.loads(args.package.read_text())
    sessions = small_member(package['dataset']['heatmaps'], 'session_ids')
    # Merge preserves per-session ordering. Clocks themselves were not merged;
    # recover them from source timelines, never invent 1s steps from frame index.
    local_indices = np.empty(len(sessions), dtype=np.int64)
    offsets = Counter()
    for i, session in enumerate(sessions):
        local_indices[i] = offsets[str(session)]
        offsets[str(session)] += 1
    clocks = {}
    args.out.mkdir(parents=True, exist_ok=False)
    consumer = MinimapObservations()
    current_session = None
    epoch = -1
    counts = Counter()
    clock_offset_ns = 0
    previous_now_ns = None
    previous_capture_ns = None
    states = Counter()
    elapsed = []
    seen = set()
    timeline_identity = {}
    with args.records.open() as records, (args.out / 'observations.jsonl').open('w') as output:
        for line in records:
            record = json.loads(line)
            session, sample = record['session'], record['sample']
            if not 0 <= sample < len(sessions) or str(sessions[sample]) != session:
                raise ValueError('detection sample/session differs from dataset identity')
            if session not in clocks:
                timeline = Path(package['dataset']['images']).parent / ('minimap-' + session + '.npz')
                clocks[session] = small_member(timeline, 'clocks')
                if len(clocks[session]) != offsets[session]:
                    raise ValueError('source timeline sample count differs from merged dataset')
                timeline_identity[session] = dict(path=str(timeline), clocks_sha256=hashlib.sha256(clocks[session].tobytes()).hexdigest())
            if session != current_session:
                if session in seen:
                    raise ValueError('interleaved sessions require separate stream consumers')
                current_session = session
                seen.add(session)
                epoch += 1
                seq = 0
            clock = float(clocks[session][local_indices[sample]])
            replay_ns = int(round(clock * 1e9))
            if seq == 0:
                # Capture time remains globally monotonic across replay sources.
                # The offset changes no within-session observation interval.
                clock_offset_ns = (0 if previous_now_ns is None else
                                   previous_now_ns + 1 - replay_ns)
                previous_capture_ns = None
            captured_ns = replay_ns + clock_offset_ns
            if previous_capture_ns is not None:
                counts['freshness_gap_frames'] += captured_ns - previous_capture_ns >= consumer.max_age_ns
            previous_capture_ns = captured_ns
            previous_now_ns = captured_ns
            start = time.perf_counter_ns()
            observation = consumer.update(record, FrameStamp(epoch, seq, captured_ns), captured_ns)
            elapsed.append((time.perf_counter_ns() - start) / 1e6)
            seq += 1
            counts['frames'] += 1
            counts['own_observed'] += observation['own'] is not None
            counts['detections'] += len(record['enemies'])
            for track in observation['enemies']:
                states[track['state']] += 1
                counts['usable_track_observations'] += track['usable']
                counts['velocity_observations'] += track['velocity_px_s'] is not None and track['usable']
            if record['own'] is None and observation['own'] is not None:
                raise AssertionError('missing own observation was carried forward')
            if any(t['usable'] and not t['observed'] for t in observation['enemies']):
                raise AssertionError('unobserved enemy marked usable')
            output.write(json.dumps(dict(sample=sample, session=session, replay_clock_s=clock,
                                         synthetic_capture_ns=captured_ns,
                                         observation=observation)) + '\n')
    if not elapsed:
        raise ValueError('records contain no observations')
    expired = consumer.snapshot(previous_now_ns + consumer.max_age_ns)
    if expired['fresh'] or expired['own'] is not None or expired['enemies']:
        raise AssertionError('observation expiry did not revoke outputs')
    report = dict(schema_version=1, mode='saved_detections_offline_clock_replay',
                  counts=dict(counts), states=dict(states), timelines=timeline_identity,
                  timing_ms=dict(mean=float(np.mean(elapsed)), p95=float(np.percentile(elapsed,95))),
                  identity=dict(records_sha256=sha256(args.records), package_sha256=sha256(args.package),
                                sources={p:sha256(Path(__file__).with_name(p)) for p in
                                         ['minimap_observation.py','enemy_tracks.py','verify_minimap_tracks.py']}),
                  limits='Freshness tested with synthetic monotonic time, preserving within-session replay dt. '
                         'Cross-session clock offsets are synthetic. Freshness gaps restart confirmation; '
                         'sparse replay observations do not establish continuous live coverage. Pixel-space '
                         'association only; no live latency, world heading, physical identity accuracy or firing validation.')
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
