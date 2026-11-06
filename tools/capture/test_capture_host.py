#!/usr/bin/env python3
"""Tests for capture_host.py that need no game, no Steam and no compositor.

Run with the system interpreter, which provides the GStreamer bindings:

    /usr/bin/python3 tools/capture/test_capture_host.py

The suite covers the pieces a live run cannot cheaply exercise on demand: the
request/reply framing, the guards that protect an existing recording, the
fail-closed source resolution, and the container checks used by the stop path.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import capture_host as host_module  # noqa: E402

WORKSPACE = HERE.parent.parent
STATE_ROOT = WORKSPACE / "state" / "tests"
HOST = str(HERE / "capture_host.py")
# A process id that cannot exist. Pinning the host to it keeps source resolution
# fail-closed no matter what this machine happens to be running: without it a
# live gamescope session would be picked up and the tests would depend on it.
ABSENT_PID = 2 ** 22 - 1


class HostProcess:
    """A capture host in a subprocess, with its own state directory and socket."""

    def __init__(self, name: str):
        self.dir = Path(tempfile.mkdtemp(prefix=f"{name}-", dir=STATE_ROOT))
        self.socket = self.dir / "control.sock"
        self.log = self.dir / "host.log"
        self.process = subprocess.Popen(
            [sys.executable, HOST, "--socket", str(self.socket), "host",
             "--state-dir", str(self.dir / "state"),
             "--gamescope-pid", str(ABSENT_PID)],
            stdout=self.log.open("wb"), stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and not self.socket.exists():
            if self.process.poll() is not None:
                raise RuntimeError("the host exited during startup: "
                                   + self.log.read_text())
            time.sleep(0.2)
        if not self.socket.exists():
            raise RuntimeError("the host socket did not appear")

    def command(self, command: str, args: dict | None = None,
                timeout: float = 60.0) -> dict:
        return host_module.request(self.socket, command, args, timeout=timeout)

    def raw(self, line: str, timeout: float = 60.0) -> dict:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stream:
            stream.settimeout(timeout)
            stream.connect(str(self.socket))
            stream.sendall(line.encode("utf-8"))
            buffered = b""
            while b"\n" not in buffered:
                chunk = stream.recv(65536)
                if not chunk:
                    break
                buffered += chunk
        return json.loads(buffered.split(b"\n", 1)[0].decode("utf-8"))

    def stop(self) -> None:
        if self.process.poll() is None:
            try:
                self.command("quit", {}, timeout=30)
            except Exception:
                pass
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()


class ProtocolTests(unittest.TestCase):
    """The command socket behaves as the caller expects."""

    host: HostProcess

    @classmethod
    def setUpClass(cls):
        STATE_ROOT.mkdir(parents=True, exist_ok=True)
        cls.host = HostProcess("protocol")

    @classmethod
    def tearDownClass(cls):
        cls.host.stop()
        shutil.rmtree(cls.host.dir, ignore_errors=True)

    def test_status_reports_idle(self):
        # A host reports idle before it is asked to do anything. Its own host,
        # because this class shares one and a failed start leaves a phase behind.
        host = HostProcess("status-idle")
        try:
            result = host.command("status")
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["phase"], "idle")
            self.assertIsNone(result["recording"])
        finally:
            host.stop()
            shutil.rmtree(host.dir, ignore_errors=True)

    def test_socket_is_private(self):
        self.assertEqual(os.stat(self.host.socket).st_mode & 0o777, 0o600)
        self.assertEqual(os.stat(self.host.dir).st_mode & 0o777, 0o700)

    def test_unknown_command_is_an_error(self):
        reply = self.host.raw('{"command": "explode", "args": {}}\n')
        self.assertFalse(reply["ok"])
        self.assertIn("unknown command", reply["error"])

    def test_malformed_request_is_an_error(self):
        reply = self.host.raw('{"args": {}}\n')
        self.assertFalse(reply["ok"])

    def test_preflight_reports_a_missing_source(self):
        # No compositor is required to exist for this test: the point is that an
        # absent source is reported, never silently replaced by another one.
        result = self.host.command("preflight")
        self.assertEqual(result["status"], "completed")
        if not result["target_resolved"]:
            self.assertIn("error", result)

    def test_record_stop_without_a_recording_is_an_error(self):
        with self.assertRaises(host_module.HostError) as caught:
            self.host.command("record_stop")
        self.assertIn("no recording is active", str(caught.exception))

    def test_long_command_acknowledges_before_its_result(self):
        # The acknowledgement must arrive before the result even when the result
        # is a refusal, and the service must stay answerable while it refuses.
        # No compositor node exists here, so the refusal names the missing source
        # rather than a client that never appeared: the source is knowable first.
        out = self.host.dir / "ack-attempt"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stream:
            stream.settimeout(60)
            stream.connect(str(self.host.socket))
            stream.sendall(json.dumps({
                "command": "record_start",
                "args": {"out": str(out), "wait_game": True,
                         "wait_timeout_s": 3, "until_exit": False,
                         "seconds": 5},
            }).encode("utf-8") + b"\n")
            reader = stream.makefile("rb")
            first = json.loads(reader.readline().decode("utf-8"))
            self.assertTrue(first["ok"])
            self.assertEqual(first["result"]["status"], "accepted")
            second = json.loads(reader.readline().decode("utf-8"))
            self.assertFalse(second["ok"])
            self.assertIn("expected one gamescope video source", second["error"])

    def test_existing_recording_is_never_overwritten(self):
        out = self.host.dir / "already-recorded"
        out.mkdir(parents=True, exist_ok=True)
        (out / "capture.mkv").write_bytes(b"not really a video")
        with self.assertRaises(host_module.HostError) as caught:
            self.host.command("record start",
                              {"out": str(out), "wait_game": False})
        self.assertIn("never overwrites", str(caught.exception))
        self.assertEqual((out / "capture.mkv").read_bytes(), b"not really a video")

    def test_tmp_output_is_refused(self):
        with self.assertRaises(host_module.HostError) as caught:
            self.host.command("record start",
                              {"out": "/tmp/wws-refused", "wait_game": False})
        self.assertIn("/tmp", str(caught.exception))

    def test_snapshot_requires_a_source(self):
        # A named process that cannot own the node must fail closed rather than
        # writing an image from whatever source happens to be present.
        with self.assertRaises(host_module.HostError):
            self.host.command("snapshot", {
                "out": str(self.host.dir / "s.png"),
                "gamescope_pid": 2 ** 22 - 1})


class ContainerTests(unittest.TestCase):
    """Container facts and arrival-time checks on a generated recording."""

    @classmethod
    def setUpClass(cls):
        cls.dir = Path(tempfile.mkdtemp(prefix="container-", dir=STATE_ROOT))
        cls.video = cls.dir / "capture.mkv"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
             "testsrc=size=64x64:rate=10:duration=2", "-c:v",
             "libx264", "-pix_fmt", "yuv420p", str(cls.video)],
            check=True)
        cls.sidecar = cls.dir / "frames.jsonl"
        cls.sidecar.write_text("".join(
            json.dumps({"index": i, "monotonic_ns": 1000 + i, "since_start_ns": i,
                        "pipeline_pts_ns": None, "bytes": 10}) + "\n"
            for i in range(10)), encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_fast_verify_reads_first_pts_without_counting_frames(self):
        meta = host_module.verify_recording(self.video, self.sidecar, "fast")
        self.assertEqual(meta["first_pts_s"], 0.0)
        self.assertFalse(meta["frame_count_checked"])
        self.assertAlmostEqual(meta["duration_s"], 2.0, places=1)
        self.assertEqual(meta["problems"], [])

    def test_full_verify_counts_frames_and_reports_a_mismatch(self):
        meta = host_module.verify_recording(self.video, self.sidecar, "full")
        self.assertTrue(meta["frame_count_checked"])
        # The generated clip holds 20 frames while this sidecar claims 10, so
        # the mismatch must be reported: without it the frame-to-time mapping
        # would look sound when it is not.
        self.assertIn(meta["frames"], (20, 21))
        self.assertEqual(len(meta["problems"]), 1)
        self.assertIn("frames.jsonl has 10 entries", meta["problems"][0])


class BattleGateTests(unittest.TestCase):
    """The gate that keeps a client stuck on login out of the recordings."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="battle-gate-", dir=STATE_ROOT))
        self.pylog = self.dir / "python.log"
        self.out = self.dir / "out"
        self.out.mkdir()
        self.host = host_module.CaptureHost.__new__(host_module.CaptureHost)
        self.host.pending_cancel = __import__("threading").Event()
        self.host.set_phase = lambda *a, **k: None

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_waits_for_a_new_play_line(self):
        self.pylog.write_text(
            "[Script] [BattleReplay] play, ('/older/replay')\n", encoding="utf-8")
        import threading

        def start_replay():
            time.sleep(1.0)
            with self.pylog.open("a", encoding="utf-8") as stream:
                stream.write("[Script] [BattleReplay] play, ('/the/one')\n")

        threading.Thread(target=start_replay, daemon=True).start()
        result = self.host._await_replay_start(self.out, str(self.pylog), 20, set())
        gate = (self.out / "battle_gate.txt").read_text(encoding="utf-8")
        self.assertEqual(result, set())
        self.assertIn("replay started:", gate)
        self.assertIn("/the/one", gate)

    def test_an_earlier_play_line_does_not_satisfy_the_wait(self):
        self.pylog.write_text(
            "[Script] [BattleReplay] play, ('/older/replay')\n", encoding="utf-8")
        with self.assertRaises(host_module.HostError) as caught:
            self.host._await_replay_start(self.out, str(self.pylog), 3, set())
        self.assertIn("never reported [BattleReplay] play", str(caught.exception))

    def test_missing_log_fails_closed(self):
        with self.assertRaises(host_module.HostError):
            self.host._await_replay_start(self.out, str(self.pylog), 2, set())

    def test_the_gate_records_the_line_it_matched(self):
        # Observed in a live run: the delta began with a bare newline, so the
        # first line of the delta was empty and an empty string is what the gate
        # recorded. The recorded line must be the one that matched.
        self.pylog.write_text("[Script] [Vehicle] id = 1\n", encoding="utf-8")

        def start_replay():
            time.sleep(1.0)
            with self.pylog.open("a", encoding="utf-8") as stream:
                stream.write("\n[Script] [BattleReplay] play, ('/the/one')\n")

        threading.Thread(target=start_replay, daemon=True).start()
        self.host._await_replay_start(self.out, str(self.pylog), 20, set())
        gate = (self.out / "battle_gate.txt").read_text(encoding="utf-8")
        self.assertIn("replay started: [Script] [BattleReplay] play, ('/the/one')",
                      gate)

    def test_every_candidate_is_watched_not_just_the_newest(self):
        """The regression of 2026-10-14 07:29.

        `record start` is issued BEFORE the game is launched, so the log the client will
        write may not exist yet -- and the two routes write different files. Choosing one
        candidate up front left the gate watching the workspace log while the `play` line
        sat in the install's log, and the recording stood at `waiting_for_battle` while
        the battle played. The gate now follows every candidate and re-reads the list, so
        a log that only appears later is picked up.
        """
        late = self.dir / "install" / "python.log"
        late.parent.mkdir()
        original = host_module.PYLOG_CANDIDATES
        host_module.PYLOG_CANDIDATES = (late, self.pylog)
        self.addCleanup(lambda: setattr(host_module, "PYLOG_CANDIDATES", original))
        self.pylog.write_text("[Script] [Vehicle] id = 1\n", encoding="utf-8")

        def start_replay():
            time.sleep(1.5)
            late.write_text("[Script] [BattleReplay] play, ('/the/late/one')\n",
                            encoding="utf-8")

        threading.Thread(target=start_replay, daemon=True).start()
        self.host._await_replay_start(self.out, None, 20, set())
        gate = (self.out / "battle_gate.txt").read_text(encoding="utf-8")
        self.assertIn("/the/late/one", gate)
        self.assertIn(str(late), gate)

    def test_a_play_line_already_in_the_newest_candidate_still_does_not_count(self):
        """Widening the watch must not weaken the wait: every candidate is read from its
        own current length, so a line written before the gate was armed cannot satisfy
        it -- the property the original single-file version already had."""
        other = self.dir / "workspace" / "python.log"
        other.parent.mkdir()
        other.write_text("[Script] [BattleReplay] play, ('/older/replay')\n",
                         encoding="utf-8")
        original = host_module.PYLOG_CANDIDATES
        host_module.PYLOG_CANDIDATES = (self.pylog, other)
        self.addCleanup(lambda: setattr(host_module, "PYLOG_CANDIDATES", original))
        with self.assertRaises(host_module.HostError):
            self.host._await_replay_start(self.out, None, 3, set())


class StartGateTests(unittest.TestCase):
    """What a start refuses, and how long it takes to refuse it."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="start-gate-", dir=STATE_ROOT))

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def make_host(self, gamescope_pid: int | None) -> "host_module.CaptureHost":
        host = host_module.CaptureHost.__new__(host_module.CaptureHost)
        host.state_dir = self.dir
        host.socket_path = self.dir / "control.sock"
        host.started_at = None
        host.gamescope_pid = gamescope_pid
        host.pending_cancel = threading.Event()
        host.pending_out = None
        host.recorder = None
        host.finished_metrics = None
        host.lifecycle = threading.RLock()
        host.phase = "idle"
        host.detail = ""
        return host

    def test_a_start_without_a_source_fails_before_it_waits(self):
        # The compositor node exists before the client does, so a source that
        # will never appear is knowable now. Reporting it as a client timeout
        # would name the wrong cause and cost the whole timeout.
        host = self.make_host(2 ** 22 - 1)
        started = time.monotonic()
        with self.assertRaises(host_module.HostError) as caught:
            host.record_start({"out": str(self.dir / "out"), "wait_game": True,
                               "wait_timeout_s": 30, "until_exit": False})
        self.assertIn("expected one gamescope video source",
                      str(caught.exception))
        self.assertEqual(host.phase, "failed")
        self.assertLess(time.monotonic() - started, 10)

    def test_the_wait_for_a_client_reports_a_timeout(self):
        host = self.make_host(None)
        # The baseline holds every client that is already running, so only a new
        # one can end the wait: the test does not depend on the machine being idle.
        with self.assertRaises(host_module.HostError) as caught:
            host._await_game(host_module.game_pids(), None, 1.0)
        self.assertIn("no game client process appeared", str(caught.exception))


class GuardTests(unittest.TestCase):
    """Pure guards, with no host process."""

    def test_game_pids_is_a_set_of_integers(self):
        self.assertTrue(all(isinstance(pid, int)
                            for pid in host_module.game_pids()))

    def test_a_signal_late_in_the_delta_is_still_found(self):
        # A single read was bounded at 1 MB, so a signal past that point was
        # never reported: the gate timed out instead of starting a recording.
        directory = Path(tempfile.mkdtemp(prefix="log-read-", dir=STATE_ROOT))
        try:
            log = directory / "python.log"
            log.write_bytes(b"x" * (2 << 20) + b"\n"
                            + b"[Script] [BattleReplay] play, ('/r')\n")
            gained = host_module.read_log_since(log, 0)
            self.assertIn(b"[BattleReplay] play", gained)
            self.assertEqual(
                host_module.matching_line(gained, b"[BattleReplay] play"),
                "[Script] [BattleReplay] play, ('/r')")
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_the_matching_line_is_not_the_first_line(self):
        # The delta starts mid-line, so its first line is a fragment. Reporting
        # the fragment names the wrong line and hides which replay started.
        data = (b"rial of an earlier line\n[Vehicle] id = 1, teamId = 0\n"
                b"[Script] [BattleReplay] play, ('/the/one')\n")
        self.assertEqual(host_module.matching_line(data, b"[BattleReplay] play"),
                         "[Script] [BattleReplay] play, ('/the/one')")
        self.assertEqual(host_module.matching_line(data, b"absent"), "")

    def test_replay_match_uses_the_command_line(self):
        host = host_module.CaptureHost.__new__(host_module.CaptureHost)
        own = host_module.game_pids() or {os.getpid()}
        self.assertEqual(host._match_game(own, None), own)

    def test_resolve_target_fails_closed_without_a_source(self):
        # An explicit dead process id cannot be satisfied by another node.
        with self.assertRaises(Exception):
            host_module.resolve_target(2 ** 22 - 1, STATE_ROOT)


class GrabTimestampTests(unittest.TestCase):
    def test_pts_is_mapped_through_segment_and_pipeline_clock(self):
        from types import SimpleNamespace as NS
        from unittest.mock import Mock, patch
        raw = bytes(2560 * 1600 * 3)
        buffer = NS(pts=2_000_000_000, get_size=lambda: len(raw),
                    extract_dup=lambda offset, size: raw)
        segment = NS(to_running_time=lambda fmt, pts: pts - 1_000_000_000)
        sample = NS(get_buffer=lambda: buffer, get_caps=lambda: None,
                    get_segment=lambda: segment)
        grabber = host_module.Grabber(1, 2, 3)
        grabber.ensure = lambda: None
        grabber.sink = NS(emit=lambda *args: sample)
        grabber.Gst = NS(SECOND=1_000_000_000, CLOCK_TIME_NONE=2**64-1,
                         Format=NS(TIME=3))
        grabber.GstVideo = NS(VideoInfo=NS(new_from_caps=lambda caps:
                                         NS(width=2560, height=1600, stride=[7680])))
        grabber.pipeline = NS(get_clock=lambda: NS(get_time=lambda: 101_200_000_000),
                              get_base_time=lambda: 100_000_000_000)
        out = Mock()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'frame.ppm'
            with patch.object(host_module.time, 'monotonic_ns', return_value=900_000_000_000):
                self.assertEqual(grabber.pull(path), len(raw))
            meta = json.loads(Path(str(path) + '.json').read_text())
        self.assertEqual(meta['age_ns'], 200_000_000)
        self.assertEqual(meta['captured_monotonic_ns'], 899_800_000_000)


if __name__ == "__main__":
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    unittest.main(verbosity=2)
