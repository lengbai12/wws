#!/usr/bin/env python3
"""Bounded PipeWire -> continuous main-view evidence -> FireControl; no input.

Run with system Python for GStreamer. The one inference worker uses project venv.
Only the newest appsink frame is consumed. Source identity is pinned per run;
source loss ends the run and revokes observations, rather than silently switching.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FRAME_BYTES = 1600 * 2560 * 3


def delivery_age_ns(pts, running, refresh_ns):
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0
           for v in (pts, running, refresh_ns)) or pts > running:
        raise ValueError('invalid source PTS/running clock')
    return running - pts + refresh_ns


def read_line(process, timeout):
    deadline = time.monotonic() + max(0., timeout)
    # Keep partial bytes across timeouts: shutdown may need to drain an
    # in-flight frame before it receives its own correlated revoke response.
    data = getattr(process, '_response_buffer', None)
    if data is None:
        data = process._response_buffer = bytearray()
    while True:
        end = data.find(b'\n')
        if end > 4*1024*1024:
            raise ValueError('oversized inference response')
        if end >= 0:
            line = bytes(data[:end])
            del data[:end+1]
            return json.loads(line)
        if len(data) > 4*1024*1024:
            raise ValueError('oversized inference response')
        remaining = deadline-time.monotonic()
        if remaining <= 0 or not select.select([process.stdout], [], [], remaining)[0]:
            raise TimeoutError('inference worker response deadline')
        block = os.read(process.stdout.fileno(), 65536)
        if not block:
            raise RuntimeError('inference worker exited')
        data.extend(block)


def read_response(process, request_id, timeout):
    deadline = time.monotonic() + max(0., timeout)
    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError('inference worker correlated response deadline')
        response = read_line(process, deadline-time.monotonic())
        if response.get('request_id') == request_id:
            return response


def send(process, request, payload=b'', timeout=2.):
    deadline = time.monotonic()+max(0., timeout)
    for block in (json.dumps(request).encode()+b'\n', payload):
        pending = memoryview(block)
        while pending:
            remaining = deadline-time.monotonic()
            if remaining <= 0 or not select.select([], [process.stdin], [], remaining)[1]:
                raise TimeoutError('inference worker request deadline')
            try:
                count = os.write(process.stdin.fileno(), pending)
            except BlockingIOError:
                continue
            if not count:
                raise BrokenPipeError('inference pipe stopped')
            pending = pending[count:]

def receive_observation(response, now_ns, *, focused, source_available):
    """Apply the receiver's clock and control context before publishing evidence."""
    observation, intent = response['observation'], response['intent']
    response['receiver'] = dict(received_ns=now_ns, focused=focused,
        source_available=source_available, worker_fresh=observation['fresh'],
        worker_reason=observation['reason'])
    reason = None
    if not source_available:
        reason = 'source_lost'
    elif not focused:
        reason = 'focus_lost'
    elif observation['fresh']:
        captured, expiry = observation['captured_ns'], observation['valid_until_ns']
        if captured > now_ns or now_ns >= expiry:
            reason = 'receiver_observation_expired'
        else:
            observation['age_ns'] += now_ns-response['completed_ns']
    if reason is not None:
        observation.update(fresh=False, reason=reason, captured_ns=None,
            age_ns=None, valid_until_ns=None, view=None, hud=None, target=None,
            lock=dict(confirmed=False, state='unknown', target_association='unknown'),
            camera=dict(stable=None, zoom_confirmed=None, mode='unknown'),
            mouse_response=dict(status='unknown'), aim_settled=False,
            fire_authorized=False)
    if not observation['fresh']:
        reason = reason or observation['reason']
    elif intent['action'] != 'revoke' and now_ns >= intent['valid_until_ns']:
        reason = 'receiver_intent_expired'
    if reason is not None:
        intent.update(action='revoke', fire_authorized=False, aim_pixel=None,
            relative_mouse=None, target=None, valid_until_ns=now_ns,
            reason=reason, release_buttons=True)
    response['receiver']['fresh_at_receipt'] = observation['fresh']
    return response



def _track_evidence(observation):
    saved = observation.get('track_evidence')
    if saved is not None:
        return saved
    evidence = []
    view = observation.get('view') or {}
    for kind, entries in (('target', view.get('detections', [])),
                          ('marker', view.get('markers', []))):
        for entry in entries or []:
            track_id = entry.get('track_id')
            state = entry.get('track_state')
            if (isinstance(track_id, int) and not isinstance(track_id, bool)
                    and state in ('tentative', 'tracked')):
                evidence.append(dict(kind=kind, track_id=track_id, state=state))
            elif state == 'ambiguous':
                evidence.append(dict(kind=kind, track_id=None, state='ambiguous'))
    return evidence


def track_continuity(records):
    """Summarize consecutive fresh, receipt-covered UI track evidence.

    IDs are screen-evidence identities only.  Each edge requires the same
    receipt lease coverage and epoch/seq+1, with source and receipt deltas at
    most 500 ms.  Missing or ambiguous observations therefore break runs.
    """
    states = {'target': {}, 'marker': {}}
    longest = {'target': {}, 'marker': {}}
    prior_current = {}
    previous = None
    for row in records:
        observed = row['observation']
        received = row['receiver']['received_ns']
        expiry = observed.get('valid_until_ns')
        fresh = (observed.get('fresh') and expiry is not None
                 and observed.get('captured_ns') <= received < expiry)
        dense = False
        if previous is not None:
            prior = previous['observation']
            prior_received = previous['receiver']['received_ns']
            prior_expiry = prior.get('valid_until_ns')
            dense = bool(fresh and prior.get('fresh') and prior_expiry is not None
                and prior_received < prior_expiry and received < prior_expiry
                and observed.get('epoch') == prior.get('epoch')
                and observed.get('seq') == prior.get('seq') + 1
                and 0 < row['source_pts_ns'] - previous['source_pts_ns'] <= 500_000_000
                and 0 < received - prior_received <= 500_000_000)
        current_evidence = (row['track_evidence'] if 'track_evidence' in row
                            else _track_evidence(observed))
        current = {(item['kind'], item['track_id']): item.get('state') for item in
                   (current_evidence if fresh else [])
                   if item.get('track_id') is not None}
        for kind in ('target', 'marker'):
            active = {item_id for item_kind, item_id in current if item_kind == kind}
            for track_id in active:
                prior_state = prior_current.get((kind, track_id))
                current_state = current[(kind, track_id)]
                edge = (dense and current_state == 'tracked' and
                        prior_state in ('tentative', 'tracked'))
                run = states[kind].get(track_id, 0) + 1 if edge else 1
                states[kind][track_id] = run
                longest[kind][track_id] = max(longest[kind].get(track_id, 0), run)
            for track_id in set(states[kind]) - active:
                states[kind][track_id] = 0
        previous = row
        prior_current = current
    def pack(kind):
        return dict(track_count=len(longest[kind]),
                    longest_run_frames=max(longest[kind].values(), default=0),
                    tracks=[dict(track_id=track_id, longest_run_frames=length)
                            for track_id, length in sorted(longest[kind].items())])
    return dict(target=pack('target'), marker=pack('marker'),
                semantics='fresh receipt-covered screen evidence only; IDs are not hull or world identity')


def receipt_continuity(records):
    """Measure receiver coverage without extending any worker deadline."""
    pairs, fresh_count, run, longest = [], 0, 0, 0
    previous = None
    for row in records:
        observed = row['observation']
        received = row['receiver']['received_ns']
        expiry = observed['valid_until_ns']
        fresh = (observed['fresh'] and expiry is not None and
                 observed['captured_ns'] <= received < expiry)
        dense = False
        if previous is not None:
            prior = previous['observation']
            prior_receipt = previous['receiver']['received_ns']
            prior_expiry = prior['valid_until_ns']
            source_dt = row['source_pts_ns'] - previous['source_pts_ns']
            receipt_dt = received - prior_receipt
            covered = (prior['fresh'] and prior_expiry is not None and
                       prior_receipt < prior_expiry and received < prior_expiry)
            dense = bool(fresh and covered and
                observed['epoch'] == prior['epoch'] and
                observed['seq'] == prior['seq'] + 1 and
                0 < source_dt <= 500_000_000 and 0 < receipt_dt <= 500_000_000)
            pairs.append(dict(seq=observed['seq'], dense=dense,
                source_dt_ns=source_dt, receipt_dt_ns=receipt_dt,
                uncovered_ns=max(0, received-prior_expiry) if prior_expiry is not None else None,
                previous_lease_covers_receipt=bool(covered)))
        fresh_count += bool(fresh)
        run = run + 1 if dense else int(bool(fresh))
        longest = max(longest, run)
        previous = row
    return dict(fresh_frames=fresh_count, longest_dense_fresh_run_frames=longest,
        entire_sequence_dense_and_fresh=(len(records) >= 2 and
            fresh_count == len(records) and all(pair['dense'] for pair in pairs)),
        pairs=pairs,
        semantics='receiver availability only; no target identity or upstream render-delay proof')

def worker(opencv_threads=8):
    sys.path.insert(0, str(ROOT / 'tools/replay'))
    import numpy as np
    import cv2
    cv2.setNumThreads(opencv_threads)
    from main_view_observation import ContinuousMainViewRecognizer
    from minimap_observation import FrameStamp
    from fire_control import FireControl
    consumer, policy = ContinuousMainViewRecognizer(), FireControl()
    print(json.dumps({'ready': True}), flush=True)
    while line := sys.stdin.buffer.readline():
        request = json.loads(line)
        started_ns = time.monotonic_ns()
        payload_read_ns = observed_ns = started_ns
        if request.get('revoke'):
            consumer.revoke(request['revoke'])
            observation = consumer.snapshot()
        else:
            if request.get('bytes') != FRAME_BYTES:
                raise ValueError('native RGB payload length required')
            raw = sys.stdin.buffer.read(FRAME_BYTES)
            if len(raw) != FRAME_BYTES:
                raise EOFError('incomplete RGB frame')
            rgb = np.frombuffer(raw, np.uint8).reshape(1600, 2560, 3)
            payload_read_ns = time.monotonic_ns()
            observation = consumer.observe(rgb, FrameStamp(**request['stamp']))
            observed_ns = time.monotonic_ns()
        now = time.monotonic_ns()
        observation = consumer.snapshot(now)
        intent = policy.decide(observation, now_ns=now,
                               focused=request.get('focused', bool(request.get('revoke'))),
                               source_available=not bool(request.get('revoke')))
        print(json.dumps(dict(observation=observation, intent=intent,
                              request_id=request.get('request_id'),
                              completed_ns=now,
                              worker_timings_ms=dict(
                                  payload_read=(payload_read_ns-started_ns)/1e6,
                                  observe=(observed_ns-payload_read_ns)/1e6,
                                  snapshot_policy=(time.monotonic_ns()-observed_ns)/1e6)),
              allow_nan=False), flush=True)


def _game_preflight():
    from capture_host import game_pids
    from probe_aim_input import process_identity, ancestor_pids
    pids = game_pids()
    if len(pids) != 1:
        raise RuntimeError('expected unique game and verified capture source')
    game = process_identity(next(iter(pids)))
    return game, ancestor_pids(game['pid'])


def preflight():
    # These independent live reads overlap, but neither result survives this
    # call. Keep source selection and process checks in their shared helpers.
    from capture_host import resolve_target, STEAM_SESSION_DIR
    with ThreadPoolExecutor(max_workers=1) as checks:
        game_check = checks.submit(_game_preflight)
        node = resolve_target(None, STEAM_SESSION_DIR)
        game, ancestors = game_check.result()
    node = {key: node.get(key) for key in ('id', 'name', 'serial', 'process_id')}
    if int(node['process_id']) not in ancestors:
        raise RuntimeError('capture compositor is not a game ancestor')
    return game, node


def verify_source(game, node):
    """Re-resolve game identity, capture source and focus after inference.

    No context verdict may be reused from before the worker response. Source
    failure also prevents a focus query against an unrelated game generation.
    """
    current_game, current_node = preflight()
    available = (current_game, current_node) == (game, node)
    return current_game, current_node, available and focused(game)


def receive_verified_response(process, request_id, game, node, timeout=2.):
    response = read_response(process, request_id, timeout)
    responded = time.monotonic_ns()
    current_game, current_node, current_focus = verify_source(game, node)
    verified = time.monotonic_ns()
    available = (current_game, current_node) == (game, node)
    received = time.monotonic_ns()
    receive_observation(response, received,
                        focused=current_focus, source_available=available)
    response['receiver'].update(context_check_started_ns=responded,
                                context_check_completed_ns=verified)
    return response, responded, verified, received, available


def focused(game):
    env = dict(os.environ, DISPLAY=game['display'])
    try:
        # xdotool's command stack passes the focused window directly to
        # getwindowpid, preserving the same lookup without a second process.
        pid = subprocess.check_output(['xdotool', 'getwindowfocus', 'getwindowpid'],
                                      env=env, timeout=1).strip()
        return int(pid) == game['pid']
    except (subprocess.SubprocessError, ValueError):
        return False


def run(args):
    import gi
    gi.require_version('Gst', '1.0')
    gi.require_version('GstVideo', '1.0')
    from gi.repository import Gst, GstVideo
    Gst.init(None)
    game, node = preflight()
    args.out.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + args.seconds
    process = subprocess.Popen([str(ROOT/'.venv/bin/python'), '-B', str(Path(__file__).resolve()),
                                '--worker', '--opencv-threads', str(args.opencv_threads)],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
    os.set_blocking(process.stdin.fileno(), False)
    pipeline = None
    report = dict(mode='live_pipewire_native_rgb_intents_only', game=game, node=node,
                  input_injected=False, frames=0, reasons={}, status='running',
                  opencv_threads=args.opencv_threads,
                  verification_overlap='live source/focus checks after each worker response; no cached context verdict',
                  observer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  source_delivery_bound='PTS queue age plus caller refresh allowance; upstream render latency uncalibrated',
                  sources={p:hashlib.sha256((ROOT/'tools/replay'/p).read_bytes()).hexdigest()
                           for p in ('main_view_observation.py','main_view_tracks.py','main_view_perception.py',
                                     'fire_hud.py','fire_control.py','aircraft_perception.py','torpedo_perception.py',
                                     'target_confirmation.py','minimap_observation.py')})
    failure = None
    request_boundary = True
    previews = {}
    continuity_records = []
    previous_handlers = {}
    try:
        def interrupted(signum, frame):
            raise InterruptedError(f'observer interrupted by signal {signum}')
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, interrupted)
        if read_line(process, min(30., deadline-time.monotonic())) != {'ready': True}:
            raise RuntimeError('invalid inference readiness')
        pipeline = Gst.parse_launch(f'pipewiresrc target-object={node["serial"]} '
            'client-name=wws-main-view-observer do-timestamp=true ! videoconvert ! '
            'video/x-raw,format=RGB ! appsink name=sink max-buffers=1 drop=true sync=false')
        sink = pipeline.get_by_name('sink')
        pipeline.set_state(Gst.State.PLAYING)
        with (args.out/'observations.jsonl').open('x') as records:
            seq = 0
            while time.monotonic() < deadline and seq < args.frames:
                gate_started = time.monotonic_ns()
                current_game, current_node = preflight()
                if (current_game, current_node) != (game, node):
                    raise RuntimeError('source identity changed')
                acquisition_started = time.monotonic_ns()
                sample = sink.emit('try-pull-sample', int(min(.75, max(0., deadline-time.monotonic()))*Gst.SECOND))
                if sample is None:
                    raise RuntimeError('source frame timeout')
                arrival = time.monotonic_ns()
                buffer = sample.get_buffer()
                info = GstVideo.VideoInfo.new_from_caps(sample.get_caps())
                if (info.width, info.height) != (2560, 1600):
                    raise ValueError('source must be native 2560x1600 RGB')
                running = pipeline.get_clock().get_time()-pipeline.get_base_time()
                if buffer.pts == Gst.CLOCK_TIME_NONE:
                    raise ValueError('source PTS unavailable')
                age = delivery_age_ns(int(buffer.pts), int(running), args.refresh_bound_ns)
                raw = buffer.extract_dup(0, buffer.get_size())
                stride = info.stride[0]
                if stride < 2560*3 or len(raw) < stride*1600:
                    raise ValueError('invalid RGB stride or buffer extent')
                if stride != 2560*3:
                    raw = b''.join(raw[y*stride:y*stride+2560*3] for y in range(1600))
                else:
                    raw = raw[:FRAME_BYTES]
                if time.monotonic() >= deadline:
                    break
                packed = time.monotonic_ns()
                request = dict(bytes=FRAME_BYTES, focused=focused(game), request_id=seq,
                               stamp=dict(epoch=0, seq=seq, captured_ns=arrival, age_bound_ns=age))
                focus_checked = time.monotonic_ns()
                # --seconds bounds acquisition. Finish one already-started
                # exchange with its own deadline rather than truncating RGB.
                request_boundary = False
                send(process, request, raw, 2.)
                request_boundary = True
                sent = time.monotonic_ns()
                response, responded, verified, received, available = receive_verified_response(
                    process, seq, game, node)
                response.update(seq=seq, capture=request['stamp'], source_pts_ns=int(buffer.pts),
                                elapsed_ms=(response['completed_ns']-arrival)/1e6)
                response['receiver_timings_ms'] = dict(
                    source_before=(acquisition_started-gate_started)/1e6,
                    acquisition=(arrival-acquisition_started)/1e6,
                    frame_pack=(packed-arrival)/1e6,
                    focus_before=(focus_checked-packed)/1e6,
                    send=(sent-focus_checked)/1e6,
                    response_wait=(responded-sent)/1e6,
                    source_after=(verified-responded)/1e6,
                    focus_after=(received-verified)/1e6,
                    verification_total=(verified-responded)/1e6,
                    completed_to_receipt=(received-response['completed_ns'])/1e6)
                for side in ('worker', 'receiver'):
                    for stage, value in response[f'{side}_timings_ms'].items():
                        report.setdefault('stage_timings_ms', {}).setdefault(
                            f'{side}.{stage}', []).append(value)
                response['observation']['track_evidence'] = _track_evidence(response['observation'])
                records.write(json.dumps(response, allow_nan=False)+'\n'); records.flush()
                continuity_records.append(dict(source_pts_ns=response['source_pts_ns'],
                    receiver=dict(received_ns=received),
                    observation={key: response['observation'][key] for key in
                        ('fresh', 'captured_ns', 'valid_until_ns', 'epoch', 'seq')},
                    track_evidence=_track_evidence(response['observation'])))
                reason = response['observation']['reason']
                report['reasons'][reason] = report['reasons'].get(reason, 0)+1
                report['frames'] += 1
                if seq == 0 or (not response['observation']['fresh'] and 'rejected' not in previews):
                    # At most two native frames; evidence I/O follows sampling.
                    previews['first' if seq == 0 else 'rejected'] = (seq, raw)
                seq += 1
                if not available:
                    raise RuntimeError('source identity changed during inference')
            report['status'] = 'completed'
    except BaseException as error:
        failure = error
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        if pipeline is not None:
            pipeline.set_state(Gst.State.NULL)
        try:
            if not request_boundary:
                raise RuntimeError('incomplete RGB request; cannot safely frame revoke')
            send(process, {'revoke': 'observer_stopped', 'request_id': 'shutdown'}, timeout=.5)
            report['final_revoke'] = read_response(process, 'shutdown', 2.)
        except Exception as error:
            report['final_revoke_error'] = str(error)
        process.stdin.close()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
        # No input executor exists in this observer; termination leaves no held input.
        process.stdout.close()
        report['worker_returncode'] = process.returncode
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        report['track_continuity'] = track_continuity(continuity_records)
        report['continuity'] = receipt_continuity(continuity_records)
        report['stage_timings_ms'] = {
            stage: dict(min=min(values), max=max(values), mean=sum(values)/len(values))
            for stage, values in report.get('stage_timings_ms', {}).items()}
        for seq, raw in previews.values():
            (args.out/f'frame-{seq:03d}.ppm').write_bytes(b'P6\n2560 1600\n255\n'+raw)
        (args.out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2))
    return 1 if failure else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--seconds', type=float, default=15)
    parser.add_argument('--frames', type=int, default=20)
    parser.add_argument('--refresh-bound-ns', type=int, default=33_333_334)
    parser.add_argument('--opencv-threads', type=int, default=8,
                        help='inference worker OpenCV pool, 1..32 (default: 8)')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if not 1 <= args.opencv_threads <= 32:
        parser.error('--opencv-threads must be 1..32')
    if args.worker:
        worker(args.opencv_threads); return 0
    if (not args.out or not math.isfinite(args.seconds) or not 3 <= args.seconds <= 120
            or not 1 <= args.frames <= 500 or not 0 <= args.refresh_bound_ns <= 250_000_000):
        parser.error('require new --out, seconds 3..120, frames 1..500, refresh bound 0..250ms')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
