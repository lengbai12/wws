#!/usr/bin/env python3
"""Bounded PipeWire samples around one reversible mouse pulse; dry-run by default.

Records screen evidence only. PTS is pipeline time, not game render time; this
cannot establish world calibration, target identity, zoom or firing authority.
Run with system Python (GStreamer); analyze with analyze_aim_settling.py.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import time

from probe_aim_input import Probe, ProbeError
from observe_main_view import preflight, delivery_age_ns

ROOT = Path(__file__).resolve().parents[2]
ROI = (1000, 560, 1560, 1050)


def acquire(sink, pipeline, phase, duration, probe, records, out):
    import numpy as np
    from gi.repository import Gst, GstVideo
    end = min(time.monotonic() + duration, probe.action_deadline)
    next_sample = 0.
    while time.monotonic() < end:
        probe.focus()
        remaining = end-time.monotonic()
        if remaining <= 0:
            break
        sample = sink.emit('try-pull-sample', int(min(.25, remaining) * Gst.SECOND))
        if sample is None:
            continue
        arrival = time.monotonic_ns()
        if arrival / 1e9 < next_sample:
            continue
        buffer = sample.get_buffer()
        info = GstVideo.VideoInfo.new_from_caps(sample.get_caps())
        if (info.width, info.height) != (2560, 1600) or buffer.pts == Gst.CLOCK_TIME_NONE:
            raise ProbeError('native RGB source with PTS required')
        running = pipeline.get_clock().get_time() - pipeline.get_base_time()
        age = delivery_age_ns(int(buffer.pts), int(running), 0)
        if age > 100_000_000:
            raise ProbeError('source queue age exceeded 100ms')
        raw = buffer.extract_dup(0, buffer.get_size())
        stride = info.stride[0]
        if stride < 7680 or len(raw) < stride * 1600:
            raise ProbeError('invalid native RGB stride')
        rgb = np.frombuffer(raw, np.uint8).reshape(-1)[:stride*1600].reshape(1600, stride)[:, :7680].reshape(1600, 2560, 3)
        x0, y0, x1, y1 = ROI
        path = out / f'{len(records):03d}-{phase}.npy'
        np.save(path, rgb[y0:y1, x0:x1], allow_pickle=False)
        # Full frames allow independent HUD/scene inspection at each phase start.
        full = None
        if not records or records[-1]['phase'] != phase:
            full = out / f'{phase}.ppm'
            full.write_bytes(b'P6\n2560 1600\n255\n' + rgb.tobytes())
        records.append(dict(phase=phase, path=str(path.resolve()),
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            full_frame=str(full.resolve()) if full else None,
                            received_ns=arrival, source_pts_ns=int(buffer.pts),
                            pipeline_running_ns=int(running), queue_age_ns=age))
        next_sample = arrival / 1e9 + .1
    if sum(r['phase'] == phase for r in records) < 3:
        raise ProbeError(f'insufficient time samples in {phase}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--dx', type=int, default=4)
    parser.add_argument('--dy', type=int, default=0)
    parser.add_argument('--seconds', type=float, default=15)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.execute and args.dry_run:
        parser.error('choose execute or dry-run')
    if not math.isfinite(args.seconds) or not 10 <= args.seconds <= 30:
        parser.error('seconds must be 10..30')
    if not 0 < max(abs(args.dx), abs(args.dy)) <= 20:
        parser.error('nonzero pulse with each axis <=20 required')
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(schema_version=1, out=str(args.out), status='dry-run',
                  execute=args.execute, roi_xyxy=list(ROI), inputs=[], focus_checks=[],
                  samples=[], calibration_valid=False, world_calibration_valid=False,
                  limits=dict(seconds=args.seconds, no_mouse_buttons=True),
                  plan=[{'phase': 'baseline', 'duration_s': 1.},
                        {'relative': [args.dx, args.dy]},
                        {'phase': 'response', 'duration_s': 2.},
                        {'relative': [-args.dx, -args.dy]},
                        {'phase': 'rollback', 'duration_s': 1.5}],
                  time_semantics='monotonic receive and pipeline PTS; upstream render age unknown',
                  source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    probe = Probe(args.seconds, report)
    pipeline = None
    previous = {}
    timer = (0., 0.)
    moved = False
    try:
        if args.execute:
            import gi
            gi.require_version('Gst', '1.0')
            gi.require_version('GstVideo', '1.0')
            from gi.repository import Gst
            Gst.init(None)
            def stop(signum, frame):
                raise ProbeError(f'interrupted by signal {signum}')
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM):
                previous[sig] = signal.signal(sig, stop)
            timer = signal.setitimer(signal.ITIMER_REAL, max(.001, probe.action_deadline-time.monotonic()))
            probe.identity, node = preflight()
            report.update(identity=probe.identity, source=node)
            verified = []
            for window in probe.x('search', '--onlyvisible', '--name', '.').splitlines():
                try:
                    probe.verify_window(window)
                    verified.append(window)
                except ProbeError:
                    continue
            if len(verified) != 1:
                raise ProbeError('expected one verified game window')
            probe.window = verified[0]
            probe.focus()
            pipeline = Gst.parse_launch(f'pipewiresrc target-object={node["serial"]} '
                'client-name=wws-aim-settling do-timestamp=true ! videoconvert ! '
                'video/x-raw,format=RGB ! appsink name=sink max-buffers=1 drop=true sync=false')
            sink = pipeline.get_by_name('sink')
            pipeline.set_state(Gst.State.PLAYING)
            acquire(sink, pipeline, 'baseline', 1., probe, report['samples'], args.out)
            # Mark before delivery: a timed-out command may already have moved.
            moved = True
            report['pulse_started_ns'] = time.monotonic_ns()
            probe.move(args.dx, args.dy)
            report['pulse_completed_ns'] = time.monotonic_ns()
            acquire(sink, pipeline, 'response', 2., probe, report['samples'], args.out)
            report['rollback_started_ns'] = time.monotonic_ns()
            probe.move(-args.dx, -args.dy)
            moved = False
            report['rollback_completed_ns'] = time.monotonic_ns()
            report['rollback_input_sent'] = True
            acquire(sink, pipeline, 'rollback', 1.5, probe, report['samples'], args.out)
            if preflight() != (probe.identity, node):
                raise ProbeError('capture source identity changed')
            report['status'] = 'completed'
    except (Exception, KeyboardInterrupt) as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        if previous:
            signal.setitimer(signal.ITIMER_REAL, 0)
        if moved:
            try:
                probe.focus()
                probe.x('mousemove_relative', '--', -args.dx, -args.dy, cleanup=True)
                report['inputs'].append(dict(relative=[-args.dx, -args.dy],
                    elapsed_s=time.monotonic()-probe.started, cleanup=True))
                report['rollback_input_sent'] = True
            except Exception as error:
                report.update(rollback_input_sent=False, rollback_error=str(error))
        if pipeline is not None:
            pipeline.set_state(Gst.State.NULL)
        probe.release()
        report['elapsed_s'] = time.monotonic() - probe.started
        if report.get('unreleased_own_keys'):
            report['status'] = 'failed'
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if previous:
            signal.setitimer(signal.ITIMER_REAL, *timer)
        (args.out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:report.get(k) for k in ('out','status','elapsed_s','error','rollback_input_sent','unreleased_own_keys')}, indent=2))
    return int(report['status'] == 'failed')


if __name__ == '__main__':
    raise SystemExit(main())
