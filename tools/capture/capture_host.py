#!/usr/bin/env python3
"""Resident capture host with a command socket, modelled on fps-vision.

The host is started once and stays up. It owns the capture source (the
gamescope PipeWire node) and the GStreamer pipeline; recordings are driven by
commands over a Unix socket instead of by re-wrapping the game in a new capture
process per replay. The structure follows the reference project
``fps-vision-main`` (``src/session.rs``, ``src/experiments.rs``):

* A Unix socket at a fixed path, directory 0700 and socket 0600, with a
  same-user peer check (``SO_PEERCRED``). The service never falls back to
  another video source: an absent or ambiguous node is an error.
* One line of JSON per request, one line of JSON per reply.
* A long command answers ``{"status": "accepted"}`` first and the final result
  later, so a caller can tell "accepted" from "hung".
* The recorder is a singleton: a second ``record_start`` is refused.
* The output directory is created exclusively. An existing recording is never
  overwritten or deleted.

Commands (request ``{"command": ..., "args": {...}}``):

===============  ==========================================================
``status``       recorder state, frame count, elapsed time, target
``preflight``    resolve the capture source without touching the recorder
``record_start`` start a recording; ``out`` is required
``record_stop``  finalise the container, verify it, write ``session.json``
``snapshot``     write one PNG frame from the live source
``quit``         stop any recording and shut the host down
===============  ==========================================================

The recording artefacts keep the contract of ``gamescope_record.py`` so the
label pipeline keeps working unchanged: ``capture.mkv``, ``frames.jsonl`` (one
flushed line per frame with an arrival time), ``session.json`` holding
``wall_start`` and ``first_pts_s``, which define video time zero as
``wall_start + since_start - first_pts``.

Run the host with the system interpreter, which provides the GStreamer bindings:

    /usr/bin/python3 tools/capture/capture_host.py host
    /usr/bin/python3 tools/capture/capture_host.py record start recordings/<session> \\
        --wait-game --until-exit
    /usr/bin/python3 tools/capture/capture_host.py snapshot recordings/<session>/proof-live.png
    /usr/bin/python3 tools/capture/capture_host.py quit
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
# The target selection and the pipeline text live in one place. Importing them
# keeps a single definition of the "one node, verified by process identity" rule.
import gamescope_record as gsrec  # noqa: E402

WORKSPACE = HERE.parent.parent
DEFAULT_STATE_DIR = WORKSPACE / "state" / "capture-host"
STEAM_SESSION_DIR = WORKSPACE / "state" / "steam-session"
# The client writes its log next to its own profile. Which profile depends on the
# route: the direct launch used the workspace copy, a Steam launch uses the
# installed game directory. The watchdog picks the newest candidate that exists
# so the "replay finished" signal is found without a manual path.
PYLOG_CANDIDATES = (
    Path.home() / ".local/share/Steam/steamapps/common/World of Warships"
    / "profile" / "python.log",
    WORKSPACE / "profile" / "python.log",
)
GAME_COMM = "WorldOfWarships"
REQUEST_LIMIT = 64 * 1024
# A recording with no stop command still has to end. ``until_exit`` uses this cap
# so a replay whose client never exits cannot hold the pipeline forever.
MAX_SECONDS = 3600.0
# Commands that answer "accepted" before their result, because they can run for
# minutes (waiting for the client, finalising a container, verifying a video).
LONG_COMMANDS = ("record_start", "record_stop", "quit")


class HostError(RuntimeError):
    """A condition that makes a command meaningless if ignored."""


def iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def game_pids() -> set[int]:
    """Process ids whose command name is the game client.

    The command name is the readiness signal used by the older capture script,
    and it is the only one that identifies the client while it starts.
    """
    pids: set[int] = set()
    with os.scandir("/proc") as entries:
        for entry in entries:
            if not entry.name.isdigit():
                continue
            try:
                with open(entry.path + "/comm", encoding="utf-8") as stream:
                    comm = stream.read().strip()
            except OSError:
                continue
            if comm == GAME_COMM:
                pids.add(int(entry.name))
    return pids


LOG_READ_LIMIT = 64 * 1024 * 1024


def read_log_since(path: Path, offset: int) -> bytes:
    """Everything the log gained since ``offset``, up to ``LOG_READ_LIMIT``.

    Reading a fixed small amount would stop short of a signal that arrives later
    and would then never report it. For the battle gate that is not a missed
    message but a recording of the login screen instead of a refusal.
    """
    try:
        with path.open("rb") as stream:
            stream.seek(offset)
            return stream.read(LOG_READ_LIMIT)
    except OSError:
        return b""


def matching_line(data: bytes, needle: bytes) -> str:
    """The first line of ``data`` that holds ``needle``, or an empty string.

    The delta starts wherever the last write ended, so its first line is usually
    a fragment. Reporting that fragment would name the wrong line, or none.
    """
    for raw in data.splitlines():
        if needle in raw:
            return raw.decode("utf-8", "replace")
    return ""


def steam_session_state(state_dir: Path = STEAM_SESSION_DIR) -> dict:
    path = state_dir / "session.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def resolve_target(explicit_pid: int | None, state_dir: Path) -> dict:
    """The gamescope video source to capture, verified by process identity.

    An explicit process id wins. Otherwise the resident Steam session records
    the compositor it started, and that is used when it is still alive. With
    neither, discovery runs over the PipeWire dump and fails closed on an
    ambiguous result, exactly like the single-recording recorder.
    """
    if explicit_pid is None:
        recorded = steam_session_state(state_dir).get("gamescope_pid")
        if recorded and gsrec.process_alive(recorded, os.getuid()):
            explicit_pid = int(recorded)
    nodes = gsrec.video_source_nodes(gsrec.read_pipewire_dump())
    return gsrec.select_target(nodes, os.getuid(), gsrec.DEFAULT_NODE_NAME, explicit_pid)


def probe_first_pts(path: Path) -> float | None:
    """First frame presentation time without decoding the whole recording.

    ``first_pts_s`` is required by the label pipeline to place video time zero,
    but a full frame count costs minutes on a long recording. Reading only the
    head of the file gives this value at a fixed small cost.
    """
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "frame=pts_time", "-read_intervals", "%+#2",
         "-of", "json", str(path)],
        capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise HostError(f"ffprobe failed: {result.stderr.strip()}")
    frames = json.loads(result.stdout).get("frames", [])
    for frame in frames:
        if "pts_time" in frame:
            return float(frame["pts_time"])
    return None


def verify_recording(video: Path, sidecar: Path, mode: str) -> dict:
    """Container facts and the checks that were actually performed.

    ``full`` counts every frame and reads every presentation time. ``fast``
    reads the duration and the arrival-time series only, and states in the
    metadata that the frame count was not checked rather than leaving it to be
    assumed. Both modes read the first presentation time, which the label
    pipeline needs.
    """
    rows = gsrec.read_sidecar(sidecar)
    meta: dict = {"sidecar_frames": len(rows)}
    problems: list[str] = gsrec.check_sidecar([int(r["monotonic_ns"]) for r in rows])
    meta["first_pts_s"] = probe_first_pts(video)
    if mode == "full":
        probed = gsrec.probe(video)
        meta["frames"] = probed["frames"]
        meta["duration_s"] = probed["duration_s"]
        meta["last_pts_s"] = probed["last_pts_s"]
        meta["frame_count_checked"] = True
        meta["pts_checked"] = True
        problems += gsrec.check_frame_count(len(rows), probed["frames"])
        problems += gsrec.check_timestamps(probed["pts"])
    else:
        meta["duration_s"] = gsrec.probe_metadata(video)["duration_s"]
        meta["frame_count_checked"] = False
        meta["pts_checked"] = False
    meta["problems"] = problems
    return meta


class Recorder:
    """One recording of one target node into one output directory.

    Every method that touches the pipeline or GLib runs on the host main loop.
    The caller polls :meth:`finished` from another thread, so a long container
    finalisation never blocks the socket service.
    """

    def __init__(self, target: dict, out: Path, encoder: str, fps: int,
                 bitrate_kbps: int, rc_mode: str, cap_seconds: float):
        self.target = target
        self.out = out
        self.video = out / "capture.mkv"
        self.sidecar_path = out / "frames.jsonl"
        self.encoder = encoder
        self.fps = fps
        self.bitrate_kbps = bitrate_kbps
        self.rc_mode = rc_mode
        self.cap_seconds = cap_seconds
        self.state = "idle"
        self.failure: str | None = None
        self.frames_seen = 0
        self.started_wall: float | None = None
        self.started_mono_ns: int | None = None
        self.stopped_mono_ns: int | None = None
        self.elapsed_s: float | None = None
        self.stop_reason: str | None = None
        self.pipeline_text: str | None = None
        self._pipeline = None
        self._sidecar = None
        self._timeout_id = None
        self._final = threading.Event()

    def finished(self) -> bool:
        return self.state in ("stopped", "failed")

    def wait_finished(self, timeout: float) -> bool:
        """Wait for the container to be finalised, from another thread."""
        return self._final.wait(timeout)

    # --- main loop ---------------------------------------------------------

    def start(self) -> None:
        """Build the pipeline and start capturing. Main loop only."""
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import GLib, Gst

        self.pipeline_text = gsrec.build_pipeline(
            self.target["serial"], self.video, self.encoder, self.fps,
            self.bitrate_kbps, self.rc_mode)
        try:
            self._pipeline = Gst.parse_launch(self.pipeline_text)
        except Exception as error:
            self.state = "failed"
            self.failure = f"the pipeline text was rejected: {error}"
            self._final.set()
            raise HostError(self.failure) from error
        # The sidecar carries the arrival time of every frame, so it has to exist
        # before the first buffer arrives and it is the file that must survive an
        # interrupted run.
        self._sidecar = self.sidecar_path.open("w", encoding="utf-8")
        self.started_wall = time.time()
        self.started_mono_ns = time.monotonic_ns()
        stamp = self._pipeline.get_by_name("stamp")
        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::error", self._on_error)
        bus.connect("message::eos", self._on_eos)
        stamp.connect("handoff", self._on_handoff)
        self._pipeline.set_state(Gst.State.PLAYING)
        self._timeout_id = GLib.timeout_add(
            int(max(self.cap_seconds, 1.0) * 1000), self._on_cap)
        self.state = "recording"

    def _on_handoff(self, _identity, buffer, *_rest) -> None:
        """Arrival time of this frame, the value a label is aligned against."""
        now = time.monotonic_ns()
        pts = buffer.pts
        self._sidecar.write(json.dumps({
            "index": self.frames_seen,
            "monotonic_ns": now,
            "since_start_ns": now - self.started_mono_ns,
            "pipeline_pts_ns": pts if pts is not None and pts >= 0 else None,
            "bytes": buffer.get_size(),
        }) + "\n")
        # Flushed per frame: an interrupted recording must not lose the times.
        self._sidecar.flush()
        self.frames_seen += 1

    def _on_error(self, _bus, message) -> None:
        error, debug = message.parse_error()
        self.failure = f"{error.message}" + (
            f" ({debug.splitlines()[0]})" if debug else "")
        self._teardown()

    def _on_eos(self, _bus, _message) -> None:
        self._teardown()

    def _on_cap(self) -> bool:
        self.stop_reason = self.stop_reason or "cap-reached"
        self.request_stop()
        return False

    def _teardown(self) -> None:
        from gi.repository import Gst, GLib

        if self._pipeline is None:
            return
        pipeline, self._pipeline = self._pipeline, None
        pipeline.set_state(Gst.State.NULL)
        if self._timeout_id is not None:
            GLib.source_remove(self._timeout_id)
            self._timeout_id = None
        if self._sidecar is not None:
            self._sidecar.close()
            self._sidecar = None
        self.stopped_mono_ns = time.monotonic_ns()
        self.elapsed_s = round(
            (self.stopped_mono_ns - self.started_mono_ns) / 1e9, 3)
        self.state = "failed" if self.failure else "stopped"
        self._final.set()

    # --- called from the service thread -----------------------------------

    def request_stop(self) -> None:
        """Ask the pipeline to end. Main loop only; returns immediately."""
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        if self._pipeline is None:
            self._final.set()
            return
        self.state = "stopping"
        self._pipeline.send_event(Gst.Event.new_eos())

    def session_meta(self, verify: dict) -> dict:
        """The artefact contract. ``wall_start`` and ``first_pts_s`` define
        video time zero as ``wall_start + since_start - first_pts``."""
        meta = {
            "host": "capture_host.py",
            "host_pid": os.getpid(),
            "target": self.target,
            "video": str(self.video),
            "frames_jsonl": str(self.sidecar_path),
            "pipeline": self.pipeline_text,
            "encoder": self.encoder,
            "fps_cap": self.fps,
            "bitrate_kbps": self.bitrate_kbps,
            "rc_mode": self.rc_mode,
            "cap_seconds": self.cap_seconds,
            "wall_start": self.started_wall,
            "started_wall_iso": time.strftime(
                "%Y-%m-%dT%H:%M:%S%z", time.localtime(self.started_wall)),
            "monotonic_start_ns": self.started_mono_ns,
            "monotonic_stop_ns": self.stopped_mono_ns,
            "elapsed_s": self.elapsed_s,
            "stop_reason": self.stop_reason,
            "failure": self.failure,
            "video_bytes": self.video.stat().st_size if self.video.is_file() else 0,
        }
        meta.update(verify)
        return meta


class Grabber:
    """Newest frame from a persistent pipeline, written as a raw PPM.

    Starting a gst-launch pipeline per frame costs about 1.2 s, which dominates
    a live aim loop, so the host keeps one appsink alive and pulls from it.
    """

    def __init__(self, serial, node_id, process_id):
        self.serial = serial
        self.node_id = node_id
        self.process_id = process_id
        self.pipeline = None
        self.sink = None
        self.Gst = None
        self.GstVideo = None

    def ensure(self):
        if self.pipeline is not None:
            return
        import gi
        gi.require_version("Gst", "1.0")
        gi.require_version("GstVideo", "1.0")
        from gi.repository import Gst, GstVideo
        Gst.init(None)
        self.Gst, self.GstVideo = Gst, GstVideo
        self.pipeline = Gst.parse_launch(
            f"pipewiresrc target-object={self.serial} client-name=wws-grab "
            "do-timestamp=true ! videoconvert ! video/x-raw,format=RGB ! "
            "appsink name=sink max-buffers=1 drop=true sync=false")
        self.sink = self.pipeline.get_by_name("sink")
        self.pipeline.set_state(Gst.State.PLAYING)

    def pull(self, out, timeout_s=1.5):
        self.ensure()
        sample = self.sink.emit("try-pull-sample", int(timeout_s * self.Gst.SECOND))
        if sample is None:
            return None
        buffer = sample.get_buffer()
        info = self.GstVideo.VideoInfo.new_from_caps(sample.get_caps())
        if (info.width, info.height) != (2560, 1600):
            return None
        raw = buffer.extract_dup(0, buffer.get_size())
        stride = info.stride[0]
        if stride != 2560 * 3:
            raw = b"".join(raw[y * stride:y * stride + 2560 * 3] for y in range(1600))
        else:
            raw = raw[:2560 * 1600 * 3]
        out.write_bytes(b"P6\n2560 1600\n255\n" + raw)
        # PTS is segment time, not the host's monotonic clock. Compare running
        # times in the pipeline clock before mapping the age to host time.
        written_ns = time.monotonic_ns()
        pts_ns = int(buffer.pts) if buffer.pts != self.Gst.CLOCK_TIME_NONE else None
        age_ns = None
        clock = self.pipeline.get_clock()
        if pts_ns is not None and clock is not None:
            running_pts = sample.get_segment().to_running_time(self.Gst.Format.TIME, pts_ns)
            running_now = clock.get_time() - self.pipeline.get_base_time()
            if running_pts != self.Gst.CLOCK_TIME_NONE and 0 <= running_pts <= running_now:
                age_ns = int(running_now - running_pts)
        meta = {"pts_ns": pts_ns, "written_monotonic_ns": written_ns, "age_ns": age_ns,
                "captured_monotonic_ns": None if age_ns is None else written_ns - age_ns,
                "clock_basis": "gst_segment_running_time", "width": 2560, "height": 1600}
        Path(str(out) + ".json").write_text(json.dumps(meta))
        return len(raw)


class CaptureHost:
    """Owns the main loop, the singleton recorder and the command service."""

    def __init__(self, state_dir: Path, socket_path: Path,
                 gamescope_pid: int | None, verify_mode: str,
                 bitrate_mbps: float, encoder: str, fps: int):
        self.state_dir = state_dir
        self.socket_path = socket_path
        self.gamescope_pid = gamescope_pid
        self.verify_mode = verify_mode
        self.bitrate_mbps = bitrate_mbps
        self.encoder = encoder
        self.fps = fps
        self.lifecycle = threading.RLock()
        # Set while record_start waits for the game client. The wait is long, so
        # it must not hold the lifecycle lock: quit and record_stop stay
        # answerable, and they cancel the pending start through this event.
        self.pending_cancel = threading.Event()
        self.pending_out: Path | None = None
        self.recorder: Recorder | None = None
        self.finished_metrics: dict | None = None
        self.phase = "idle"
        self.detail = ""
        self.started_at: str | None = None
        self.shutdown = threading.Event()
        self._loop = None
        self._preflight_target: dict | None = None

    # --- main loop plumbing ------------------------------------------------

    def run_in_loop(self, fn, timeout: float = 300.0):
        """Run ``fn`` on the GLib main loop and return its value."""
        from gi.repository import GLib

        slot: dict = {}
        done = threading.Event()

        def invoke():
            try:
                slot["value"] = fn()
            except BaseException as error:  # reported to the caller verbatim
                slot["error"] = error
            finally:
                done.set()
            return False

        GLib.idle_add(invoke)
        if not done.wait(timeout):
            raise HostError("the host main loop did not run the command in time")
        if "error" in slot:
            raise slot["error"]
        return slot.get("value")

    def set_phase(self, phase: str, detail: str = "") -> None:
        self.phase = phase
        self.detail = detail
        self.write_host_state()

    def write_host_state(self) -> None:
        payload = {
            "pid": os.getpid(),
            "socket": str(self.socket_path),
            "started_at": self.started_at,
            "phase": self.phase,
            "detail": self.detail,
            "recorder": None if self.recorder is None else {
                "out": str(self.recorder.out),
                "state": self.recorder.state,
                "frames": self.recorder.frames_seen,
                "elapsed_s": self.recorder.elapsed_s,
                "stop_reason": self.recorder.stop_reason,
                "failure": self.recorder.failure,
                "target": self.recorder.target,
            },
        }
        path = self.state_dir / "host.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                       encoding="utf-8")
        tmp.replace(path)

    # --- commands ----------------------------------------------------------

    def status(self, args: dict | None = None) -> dict:
        recorder = self.recorder
        live = None
        if recorder is not None:
            elapsed = recorder.elapsed_s
            if elapsed is None and recorder.started_mono_ns:
                elapsed = round(
                    (time.monotonic_ns() - recorder.started_mono_ns) / 1e9, 3)
            live = {
                "out": str(recorder.out),
                "state": recorder.state,
                "frames": recorder.frames_seen,
                "elapsed_s": elapsed,
                "wall_start": recorder.started_wall,
                "stop_reason": recorder.stop_reason,
                "failure": recorder.failure,
                "target": recorder.target,
                "session": str(recorder.out / "session.json"),
            }
        return {
            "status": "completed",
            "host_pid": os.getpid(),
            "socket": str(self.socket_path),
            "phase": self.phase,
            "detail": self.detail,
            "recording": live,
            "finished": self.finished_metrics,
        }

    def preflight(self, args: dict) -> dict:
        """Resolve the capture source and report process state. No side effects."""
        report = {
            "status": "completed",
            "game_pids": sorted(game_pids()),
            "steam_session": steam_session_state(),
        }
        try:
            target = self._resolve_source(args)
        except HostError as error:
            report.update({"target_resolved": False, "error": str(error)})
            return report
        report.update({
            "target_resolved": True,
            "node": {"id": target.get("id"), "name": target.get("name"),
                     "serial": target.get("serial"),
                     "process_id": target.get("process_id")},
        })
        return report

    def _resolve_source(self, args: dict) -> dict:
        """The compositor node this host may capture, or an error.

        One definition of which process may be captured, shared by the pipeline,
        the pre-flight check and the snapshot. Resolving before the wait for the
        client matters: the node exists earlier, so a source that will never
        appear is an error now rather than a timeout in ten minutes. The
        selection error is re-raised as ``HostError`` so every caller sees one
        error type and no failure escapes the phase bookkeeping.
        """
        explicit = args.get("gamescope_pid")
        if explicit is None:
            explicit = self.gamescope_pid
        try:
            return resolve_target(int(explicit) if explicit is not None else None,
                                  STEAM_SESSION_DIR)
        except gsrec.CaptureError as error:
            raise HostError(str(error)) from error

    def record_start(self, args: dict) -> dict:
        if not args.get("out"):
            raise HostError("record start needs an output directory")
        out = Path(args["out"])
        if not out.is_absolute():
            out = WORKSPACE / out
        out = out.resolve()
        if str(out).startswith("/tmp"):
            raise HostError("refusing to put a recording under /tmp")
        cap_seconds = float(args.get("seconds") or MAX_SECONDS)
        until_exit = bool(args.get("until_exit"))
        if until_exit:
            cap_seconds = max(cap_seconds, MAX_SECONDS)
        replay = args.get("replay")

        with self.lifecycle:
            if self.pending_out is not None:
                raise HostError(
                    f"a recording is already starting for {self.pending_out}")
            if self.recorder is not None and not self.recorder.finished():
                raise HostError(
                    f"a recording is already active in {self.recorder.out}")
            present = gsrec.existing_outputs(out)
            if present:
                raise HostError(
                    f"{out} already holds {', '.join(present)} from an earlier "
                    "run; the host never overwrites a recording")
            out.mkdir(parents=True, exist_ok=True)
            self.pending_out = out
            self.pending_cancel.clear()

        try:
            baseline = game_pids()
            # Resolving the source now, before any wait, is what makes a wrong
            # route fail loudly: without a gamescope node there is nothing to
            # capture, and waiting for the client first would report that as a
            # timeout instead of as a missing source.
            self._resolve_source(args)
            matched = self._match_game(baseline, replay)
            if args.get("wait_battle"):
                # A client that cannot log in sits on the login screen forever:
                # the game process exists, but no replay ever starts. Gate on the
                # client's own statement that it started the replay, so a bad run
                # never produces a recording of a login screen.
                self.set_phase("waiting_for_battle", str(out))
                matched = self._await_replay_start(
                    out, args.get("pylog"),
                    float(args.get("battle_timeout_s") or 900),
                    matched)
                matched = matched or game_pids()
            if args.get("wait_game") and not matched:
                # Readiness gate: the compositor node exists during bootstrap,
                # so recording before the client appears would capture that
                # stage instead of the battle. The wait releases the lifecycle
                # lock, so quit and record_stop can cancel it.
                self.set_phase("waiting_for_game", str(out))
                matched = self._await_game(
                    baseline, replay, float(args.get("wait_timeout_s") or 600))
            self._start_pipeline(args, out, matched, replay, cap_seconds,
                                 until_exit)
        except HostError as error:
            # A cancelled start reported itself through its own phase already;
            # any other error means the recording that was armed did not happen.
            if not self.pending_cancel.is_set():
                self.set_phase("failed", str(error))
            raise
        finally:
            with self.lifecycle:
                self.pending_out = None
        result = self.status()
        result["action"] = "record_start"
        result["watchdog"] = bool(until_exit)
        return result

    def _await_game(self, baseline: set[int], replay: str | None,
                    timeout_s: float) -> set[int]:
        """Wait for the game client, unless the start is cancelled."""
        deadline = time.monotonic() + timeout_s
        self.pending_cancel.clear()
        while time.monotonic() < deadline:
            if self.pending_cancel.is_set():
                self.set_phase("idle", "the pending start was cancelled")
                raise HostError("the pending recording start was cancelled")
            fresh = game_pids() - baseline
            if fresh:
                matched = self._match_game(fresh, replay) or fresh
                return matched
            time.sleep(1.0)
        self.set_phase("idle", "no game process appeared")
        raise HostError(
            "no game client process appeared; the replay was not launched, or "
            "the launch went to another compositor")

    def _await_replay_start(self, out: Path, pylog: str | None,
                            timeout_s: float,
                            matched: set[int]) -> set[int]:
        """Wait until the client reports that it started the replay.

        The signal is a new ``[BattleReplay] play`` line in the client log. The
        log is watched from its current length, so a line from an earlier replay
        cannot satisfy this wait. A client stuck on login never writes one, and
        then this raises instead of recording the login screen.
        """
        path = self._newest_pylog(pylog)
        try:
            deadline = time.monotonic() + timeout_s
            # Every candidate is watched from its own current length, and the list is
            # re-read every second so a log that only APPEARS once the game starts is
            # picked up too. A log that shows up later starts at its own end, so a line
            # from an earlier replay still cannot satisfy the wait.
            offsets = {item: (item.stat().st_size if item.is_file() else 0)
                       for item in self._pylog_candidates(pylog)}
            (out / "battle_gate.txt").write_text(
                "watching %s from %s for [BattleReplay] play\n"
                % (", ".join(str(item) for item in offsets),
                   {str(item): value for item, value in offsets.items()}),
                encoding="utf-8")
            while time.monotonic() < deadline:
                if self.pending_cancel.is_set():
                    raise HostError("the pending recording start was cancelled")
                for item in self._pylog_candidates(pylog):
                    if item not in offsets:
                        offsets[item] = item.stat().st_size if item.is_file() else 0
                        continue
                    if not item.is_file():
                        continue
                    fresh = read_log_since(item, offsets[item])
                    if fresh:
                        offsets[item] += len(fresh)
                    if b"[BattleReplay] play" in fresh:
                        line = matching_line(fresh, b"[BattleReplay] play")
                        with (out / "battle_gate.txt").open(
                                "a", encoding="utf-8") as stream:
                            stream.write(f"replay started: {line}\nin {item}\n")
                        return matched
                time.sleep(1.0)
            raise HostError(
                f"the client never reported [BattleReplay] play within "
                f"{timeout_s:.0f} s; it is not playing the replay (login or "
                f"launcher), so nothing was recorded")
        finally:
            if not matched:
                self.set_phase("idle", "no game client process appeared")

    def _start_pipeline(self, args: dict, out: Path, matched: set[int],
                        replay: str | None, cap_seconds: float,
                        until_exit: bool) -> None:
        """Create the recorder and start capturing. Resolves the source here so
        a recording only ever attaches to a node that exists at this moment."""
        with self.lifecycle:
            if self.pending_cancel.is_set():
                raise HostError("the pending recording start was cancelled")
            if matched and replay:
                (out / "wows.pid").write_text(
                    f"{sorted(matched)[0]}\n", encoding="utf-8")
            if replay:
                (out / "replay_path.txt").write_text(
                    f"{Path(replay).resolve()}\n", encoding="utf-8")
            (out / "started_at.txt").write_text(f"{iso_now()}\n", encoding="utf-8")
            if self.finished_metrics is not None:
                (out / "host-previous.json").write_text(
                    json.dumps(self.finished_metrics, indent=2) + "\n",
                    encoding="utf-8")
            target = self._resolve_source(args)
            encoder = self.encoder
            if encoder == "auto":
                encoder = gsrec.choose_encoder(list(gsrec.DEFAULT_ENCODERS))
            recorder = Recorder(target, out, encoder, self.fps,
                                int(self.bitrate_mbps * 1000),
                                gsrec.DEFAULT_RC_MODE, cap_seconds)
            self.recorder = recorder
            self.finished_metrics = None
            try:
                self.run_in_loop(recorder.start)
            except BaseException as error:
                self.recorder = None
                raise HostError(
                    f"the capture pipeline did not start: {error}") from error
            self.set_phase("recording", str(out))
            (out / "recorder.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
            if until_exit:
                threading.Thread(
                    target=self._watch_until_exit,
                    args=(set(matched) or game_pids(), out, args.get("pylog")),
                    name="wws-replay-watchdog", daemon=True).start()

    def _match_game(self, candidates: set[int], replay: str | None) -> set[int]:
        """Candidates whose command line carries the replay, when one is named."""
        if not replay:
            return set(candidates)
        needle = str(Path(replay).resolve())
        matched = set()
        for pid in candidates:
            try:
                cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(
                    b"\0", b" ").decode("utf-8", "replace")
            except OSError:
                continue
            if needle in cmdline or Path(replay).name in cmdline:
                matched.add(pid)
        return matched

    def record_stop(self, args: dict) -> dict:
        if self.pending_out is not None and self.recorder is None:
            self.pending_cancel.set()
            return {"status": "completed", "action": "record_stop",
                    "cancelled_start": str(self.pending_out),
                    "metrics": None}
        with self.lifecycle:
            if self.finished_metrics is not None and self.recorder is None:
                return {"status": "completed", "action": "record_stop",
                        "metrics": self.finished_metrics}
            if self.recorder is None:
                raise HostError("no recording is active")
            reason = args.get("reason") or "requested"
            return self._finish(self.recorder, reason)

    def stop_and_finalize(self, reason: str) -> dict:
        """The watchdog path: stop the singleton recorder and finalise it."""
        with self.lifecycle:
            if self.recorder is None or self.finished_metrics is not None:
                return self.finished_metrics or {}
            return self._finish(self.recorder, reason)

    def _finish(self, recorder: Recorder, reason: str) -> dict:
        if recorder.stop_reason in (None, "cap-reached"):
            recorder.stop_reason = reason
        if not recorder.finished():
            self.run_in_loop(recorder.request_stop)
            if not recorder.wait_finished(120):
                self.run_in_loop(recorder._teardown)
                recorder.failure = recorder.failure or (
                    "the pipeline did not finalise within 120 s")
        self.set_phase("finalising", str(recorder.out))
        # The verification decodes the container and is run here, on the service
        # thread, so the main loop stays answerable while it runs.
        verify: dict
        problems: list[str] = []
        try:
            if recorder.video.is_file() and recorder.video.stat().st_size:
                verify = verify_recording(recorder.video, recorder.sidecar_path,
                                          self.verify_mode)
                problems = verify.get("problems", [])
            else:
                verify = {"frame_count_checked": False, "pts_checked": False,
                          "problems": ["no output was written to "
                                       f"{recorder.video}"], "sidecar_frames":
                          recorder.frames_seen, "first_pts_s": None}
                problems = verify["problems"]
        except HostError as error:
            verify = {"frame_count_checked": False, "pts_checked": False,
                      "problems": [str(error)], "sidecar_frames":
                      recorder.frames_seen, "first_pts_s": None}
            problems = verify["problems"]
        if not recorder.frames_seen:
            recorder.failure = recorder.failure or "the pipeline produced no frames"
        meta = recorder.session_meta(verify)
        meta["verify_mode"] = self.verify_mode
        meta["failed"] = bool(recorder.failure) or bool(problems)
        (recorder.out / "session.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
        (recorder.out / "stop_reason.txt").write_text(
            f"stop reason: {recorder.stop_reason}\n", encoding="utf-8")
        (recorder.out / "stopped_at.txt").write_text(f"{iso_now()}\n",
                                                    encoding="utf-8")
        self.finished_metrics = meta
        self.recorder = None
        self.set_phase("idle", str(recorder.out))
        return {"status": "completed", "action": "record_stop", "metrics": meta,
                "session": str(recorder.out / "session.json")}

    @staticmethod
    def snapshot_elements(out):
        """Encode one frame without compression when the caller asks for PPM.

        PNG encoding a 2560x1600 frame costs about a second, which dominates a
        live aim loop; a PPM is a header plus the raw pixels and reads back with
        the same loaders.
        """
        writer = (["filesink", f"location={out}"] if str(out).endswith(".ppm")
                  else ["pngenc", "!", "filesink", f"location={out}"])
        return writer

    def grab(self, args: dict) -> dict:
        """One raw PPM frame from a persistent pipeline (about 4x faster)."""
        if not args.get("out"):
            raise HostError("grab needs an output path")
        out = Path(args["out"])
        if not out.is_absolute():
            out = WORKSPACE / out
        out.parent.mkdir(parents=True, exist_ok=True)
        explicit = args.get("gamescope_pid", self.gamescope_pid)
        if self.recorder is not None:
            explicit = self.recorder.target.get("process_id") or explicit
        target = self._resolve_source({"gamescope_pid": explicit})
        key = (target.get("id"), target.get("process_id"))
        if getattr(self, "grabber", None) is None or self.grabber_serial != target.get("serial"):
            self.grabber = Grabber(target.get("serial"), target.get("id"),
                                   target.get("process_id"))
            self.grabber_serial = target.get("serial")
        if out.exists():
            out.unlink()
        Path(str(out) + ".json").unlink(missing_ok=True)
        size = self.grabber.pull(out)
        if size is None:
            raise HostError("the grab pipeline produced no frame")
        return {"status": "completed", "action": "grab", "path": str(out), "bytes": size,
                "node": target.get("id"), "process_id": target.get("process_id")}

    def snapshot(self, args: dict) -> dict:
        """One frame from the live source. Independent of any recording."""
        if not args.get("out"):
            raise HostError("snapshot needs an output path")
        out = Path(args["out"])
        if not out.is_absolute():
            out = WORKSPACE / out
        out.parent.mkdir(parents=True, exist_ok=True)
        explicit = args.get("gamescope_pid", self.gamescope_pid)
        if self.recorder is not None:
            explicit = self.recorder.target.get("process_id") or explicit
        target = self._resolve_source({"gamescope_pid": explicit})
        # gst-launch argument form: properties are separate arguments, and the
        # link marker is its own argument, exactly as the shell would pass them.
        elements = ["pipewiresrc", f"target-object={target['serial']}",
                    "client-name=wws-snapshot", "do-timestamp=true",
                    # A live source has no end: without a buffer limit the
                    # pipeline would run until the caller gives up on it.
                    "num-buffers=1", "!",
                    "videoconvert", "!", *self.snapshot_elements(out)]
        if out.exists():
            out.unlink()
        result = subprocess.run(["gst-launch-1.0", "-e", *elements],
                                capture_output=True, text=True, timeout=120)
        if result.returncode != 0 or not out.is_file():
            raise HostError(
                "the snapshot pipeline failed: "
                + (result.stderr.strip().splitlines() or ["no output"])[-1])
        return {"status": "completed", "action": "snapshot", "path": str(out),
                "bytes": out.stat().st_size, "node": target.get("id"),
                "process_id": target.get("process_id")}

    def quit(self, args: dict) -> dict:
        """Stop any recording, finalise it, and report before the host exits."""
        metrics = None
        cancelled = None
        if self.pending_out is not None:
            self.pending_cancel.set()
            cancelled = str(self.pending_out)
        with self.lifecycle:
            if self.recorder is not None and not self.recorder.finished():
                metrics = self._finish(self.recorder, "host-quit").get("metrics")
        self.shutdown.set()
        return {"status": "completed", "action": "quit", "metrics": metrics,
                "cancelled_start": cancelled}

    def _watch_until_exit(self, pids: set[int], out: Path, pylog: str | None) -> None:
        """Stop the recording when the replay ends or the client exits.

        With no known client process the watchdog adopts the first one that
        appears, so a recording started before the launch still has an exit
        condition. It never stops on a train of absent processes before any
        client was ever seen.
        """
        path = Path(pylog) if pylog else self._newest_pylog()
        (out / "watchdog.log").write_text(
            f"watchdog: watching {path} and the game process set\n",
            encoding="utf-8")
        try:
            offset = path.stat().st_size if path.is_file() else 0
        except OSError:
            offset = 0
        reason = ""
        while not self.shutdown.is_set():
            live = game_pids()
            if not pids:
                pids = live
                if not pids:
                    time.sleep(2.0)
                    continue
            # The signal is "no game process at all", not "the process seen at
            # start is gone": a Steam launch chain can hand over from a launcher
            # process to the client, and that handover is not an exit.
            if not live:
                reason = "game-exited"
                break
            if path.is_file():
                tail = read_log_since(path, offset)
                if b"onReplayFinished" in tail:
                    reason = "replay-finished"
                    break
            time.sleep(2.0)
        if not reason:
            return
        with (out / "watchdog.log").open("a", encoding="utf-8") as stream:
            stream.write(f"watchdog: {reason} at {iso_now()}\n")
        self.stop_and_finalize(reason)

    @staticmethod
    def _pylog_candidates(path: Path | None = None) -> list[Path]:
        """Every client log the gate may have to read.

        The two routes write DIFFERENT files: a direct launch writes the workspace's
        ``profile/python.log``, and Steam writes the one inside the game install. Which
        one exists can change while the gate is waiting, because ``record start`` is
        issued BEFORE the game is launched -- so the gate follows all of them instead of
        picking one up front. Choosing one up front is exactly how the 2026-10-14 07:29
        15.9 recording stood at ``waiting_for_battle`` for minutes while the `play` line
        sat in a file nobody was reading.
        """
        return [Path(path)] if path else list(PYLOG_CANDIDATES)

    @staticmethod
    def _newest_pylog(path: Path | None = None) -> Path:
        """The newest candidate, or the first one when none exists yet."""
        candidates = CaptureHost._pylog_candidates(path)
        existing = [item for item in candidates if item.is_file()]
        if not existing:
            return candidates[0]
        return max(existing, key=lambda item: item.stat().st_mtime)


def serve(host: CaptureHost, listener: socket.socket) -> None:
    """Accept control connections and run one thread per command."""
    listener.setblocking(False)
    clients: list[tuple[socket.socket, bytearray]] = []
    while not host.shutdown.is_set():
        try:
            stream, _ = listener.accept()
        except BlockingIOError:
            stream = None
        except OSError:
            if host.shutdown.is_set():
                break
            raise
        if stream is not None:
            if not same_user(stream):
                stream.close()
            else:
                stream.setblocking(False)
                clients.append((stream, bytearray()))
        for client in list(clients):
            try:
                handle_client(host, client, clients)
            except Exception as error:  # noqa: BLE001 - one bad client only
                sys.stderr.write(f"control client dropped: {error!r}\n")
                sys.stderr.flush()
                if client in clients:
                    clients.remove(client)
                try:
                    client[0].close()
                except OSError:
                    pass
        time.sleep(0.005)


def handle_client(host: CaptureHost, client, clients: list) -> None:
    stream, buffer = client
    try:
        chunk = stream.recv(4096)
    except BlockingIOError:
        return
    except OSError:
        chunk = b""
    if not chunk:
        stream.close()
        clients.remove(client)
        return
    buffer.extend(chunk)
    if len(buffer) > REQUEST_LIMIT:
        reply(stream, HostError("request exceeds 64 KiB"))
        stream.close()
        clients.remove(client)
        return
    if b"\n" not in buffer:
        return
    line, _rest = bytes(buffer).split(b"\n", 1)
    clients.remove(client)
    threading.Thread(target=run_request, args=(host, stream, line),
                     name="wws-capture-command", daemon=True).start()


def run_request(host: CaptureHost, stream: socket.socket, line: bytes) -> None:
    try:
        request = json.loads(line.decode("utf-8"))
        if not isinstance(request, dict) or "command" not in request:
            raise HostError('a request must be {"command": ..., "args": {...}}')
        # The wire form is snake_case, matching the reference project's request
        # enum. The spaced form is accepted too, because that is how the
        # commands are written in the documentation and on the command line.
        command = str(request["command"]).strip().replace(" ", "_") \
            .replace("-", "_")
        args = request.get("args") or {}
        if command in LONG_COMMANDS:
            # Two-phase acknowledgement: an accepted long command is neither a
            # result nor a failure, so it must not be mistakable for either.
            reply(stream, {"status": "accepted"})
        handler = getattr(host, command, None)
        if handler is None:
            raise HostError(f"unknown command {command!r}")
        result = handler(args)
        reply(stream, result)
        if command == "quit":
            # The reply must reach the caller before the process tears down, so
            # the loop stops shortly after this connection is closed.
            threading.Timer(0.5, host._loop.quit).start()
    except Exception as error:  # noqa: BLE001 - reported to the caller
        reply(stream, error)
    finally:
        try:
            stream.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        stream.close()


def same_user(stream: socket.socket) -> bool:
    """SO_PEERCRED check: the host serves the user that owns it."""
    try:
        credentials = stream.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
    except OSError:
        return False
    uid = int.from_bytes(credentials[4:8], sys.byteorder)
    return uid == os.geteuid()


def reply(stream: socket.socket, outcome) -> None:
    if isinstance(outcome, Exception):
        payload = {"ok": False, "result": None, "error": str(outcome)}
    else:
        payload = {"ok": True, "result": outcome, "error": None}
    data = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
    try:
        stream.setblocking(True)
        stream.settimeout(5.0)
        stream.sendall(data)
    except OSError:
        pass


def request(socket_path: Path, command: str, args: dict | None = None,
            timeout: float = 3600.0, accepted_only: bool = False) -> dict:
    """Send one command to a running host and return its result.

    A long command acknowledges with ``{"status": "accepted"}`` first; that
    acknowledgement is not a result, so the loop keeps reading until the real
    reply arrives. ``accepted_only`` returns the acknowledgement instead, which
    is what a batch script needs in order to chain its own next step.
    """
    if not socket_path.exists():
        raise HostError(f"no capture host is listening at {socket_path}")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stream:
        stream.settimeout(timeout)
        stream.connect(str(socket_path))
        stream.sendall((json.dumps({"command": command, "args": args or {}}) + "\n")
                       .encode("utf-8"))
        reader = stream.makefile("rb")
        while True:
            line = reader.readline()
            if not line:
                raise HostError(
                    "the host disconnected before completing the command")
            payload = json.loads(line.decode("utf-8"))
            if not payload.get("ok"):
                raise HostError(payload.get("error") or "the host reported no error")
            result = payload.get("result")
            if isinstance(result, dict) and result.get("status") == "accepted":
                if accepted_only:
                    return result
                continue
            return result


def run_host(args) -> int:
    """Start the resident host. Returns the process exit code."""
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import GLib, Gst

    Gst.init(None)
    state_dir = Path(args.state_dir).resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(state_dir, 0o700)
    socket_path = (Path(args.socket).resolve() if args.socket
                   else state_dir / "control.sock")
    if socket_path.exists():
        # Replacing the socket of a live host would leave the running one
        # unreachable while pretending to serve its callers.
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        probe.settimeout(2.0)
        try:
            probe.connect(str(socket_path))
        except OSError:
            # A socket file with nothing behind it is a leftover from a crash.
            socket_path.unlink()
        else:
            probe.close()
            raise HostError(f"a capture host is already listening at {socket_path}")
        probe.close()
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(socket_path))
    os.chmod(socket_path, 0o600)
    listener.listen(8)

    host = CaptureHost(state_dir, socket_path, args.gamescope_pid,
                       args.verify, args.bitrate_mbps, args.encoder, args.fps)
    host.started_at = iso_now()
    host._loop = GLib.MainLoop()
    host.write_host_state()
    print(f"capture host listening on {socket_path} (pid {os.getpid()})",
          flush=True)
    service = threading.Thread(target=serve, args=(host, listener),
                              name="wws-capture-service", daemon=True)
    service.start()
    try:
        host._loop.run()
    finally:
        host.shutdown.set()
        try:
            listener.close()
        except OSError:
            pass
        if socket_path.exists():
            socket_path.unlink()
        host.set_phase("stopped", "")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--socket", help="control socket path "
                        f"(default {DEFAULT_STATE_DIR / 'control.sock'})")
    sub = parser.add_subparsers(dest="group", required=True)

    host = sub.add_parser("host", help="run the resident host (blocks)")
    host.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    host.add_argument("--gamescope-pid", type=int,
                      help="require this compositor process id")
    host.add_argument("--verify", choices=("fast", "full"), default="fast",
                      help="fast reads duration and arrival times; full also "
                           "counts every frame (minutes on a long recording)")
    host.add_argument("--bitrate-mbps", type=float,
                      default=gsrec.DEFAULT_BITRATE_KBPS / 1000)
    host.add_argument("--encoder", default="auto")
    host.add_argument("--fps", type=int, default=gsrec.DEFAULT_FPS)

    record = sub.add_parser("record", help="recording commands")
    action = record.add_subparsers(dest="action", required=True)
    start = action.add_parser("start", help="start one recording")
    start.add_argument("out", help="output directory, never overwritten")
    start.add_argument("--seconds", type=float,
                       help=f"recording cap (default {MAX_SECONDS:.0f} s)")
    start.add_argument("--wait-game", dest="wait_game", action="store_true",
                       default=True,
                       help="wait for the game client before capturing "
                            "(default); the compositor node exists earlier and "
                            "would only record the bootstrap stage")
    start.add_argument("--no-wait-game", dest="wait_game", action="store_false")
    start.add_argument("--wait-timeout-s", type=float, default=600.0)
    start.add_argument("--wait-battle", action="store_true",
                       help="do not capture until the client reports "
                            "[BattleReplay] play, so a client stuck on login "
                            "never yields a recording")
    start.add_argument("--battle-timeout-s", type=float, default=900.0,
                       help="how long to wait for the replay to start")
    start.add_argument("--until-exit", dest="until_exit", action="store_true",
                       default=True,
                       help="stop when the replay finishes or the client exits "
                            "(default)")
    start.add_argument("--no-until-exit", dest="until_exit",
                       action="store_false")
    start.add_argument("--replay", help="expected replay path; used to identify "
                       "the game process and recorded in the session")
    start.add_argument("--pylog", help="client log watched for onReplayFinished")
    start.add_argument("--gamescope-pid", type=int)
    start.add_argument("--accepted", action="store_true",
                       help="return as soon as the host accepts the command, "
                            "instead of waiting for the recording to start")
    action.add_parser("stop", help="finalise, verify and write session.json")

    grab = sub.add_parser("grab", help="one raw PPM frame from a persistent pipeline")
    grab.add_argument("out")
    snapshot = sub.add_parser("snapshot", help="one PNG frame from the live source")
    snapshot.add_argument("out")
    sub.add_parser("status", help="recorder state")
    sub.add_parser("preflight", help="resolve the capture source, no side effects")
    sub.add_parser("quit", help="stop any recording and shut the host down")
    return parser


def socket_path(args) -> Path:
    if args.socket:
        return Path(args.socket).resolve()
    if args.group == "host":
        return Path(args.state_dir).resolve() / "control.sock"
    return (DEFAULT_STATE_DIR / "control.sock").resolve()


def main() -> int:
    args = build_parser().parse_args()
    if args.group == "host":
        return run_host(args)
    path = socket_path(args)
    if args.group == "record":
        if args.action == "start":
            payload = {"out": args.out, "seconds": args.seconds,
                       "wait_game": args.wait_game,
                       "wait_battle": args.wait_battle or None,
                       "battle_timeout_s": args.battle_timeout_s,
                       "wait_timeout_s": args.wait_timeout_s,
                       "until_exit": args.until_exit, "replay": args.replay,
                       "pylog": args.pylog,
                       "gamescope_pid": args.gamescope_pid}
            payload = {k: v for k, v in payload.items() if v is not None}
            print(json.dumps(request(path, "record_start", payload,
                                     accepted_only=args.accepted),
                             indent=2, ensure_ascii=False))
        else:
            print(json.dumps(request(path, "record_stop", {}),
                             indent=2, ensure_ascii=False))
        return 0
    if args.group == "grab":
        print(json.dumps(request(path, "grab", {"out": args.out}),
                         indent=2, ensure_ascii=False))
        return 0
    if args.group == "snapshot":
        print(json.dumps(request(path, "snapshot", {"out": args.out}),
                         indent=2, ensure_ascii=False))
        return 0
    print(json.dumps(request(path, args.group, {}), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except HostError as error:
        raise SystemExit(f"capture host error: {error}") from error
