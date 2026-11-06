#!/usr/bin/env python3
"""Tests for the gamescope recorder's selection and verification logic.

Run: /usr/bin/python3 tools/capture/test_gamescope_record.py

The selection tests matter most: a wrong node records the wrong thing, so every
rejection path is asserted.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import gamescope_record as rec

UID = 1000


def alive(*_args) -> bool:
    """Stub process check: tests do not depend on /proc state."""
    return True


def dump(*nodes: tuple) -> list[dict]:
    """Build a PipeWire dump from (node_id, name, serial, client_id) tuples."""
    entries = [{"id": 1, "type": "PipeWire:Interface:Metadata"}]
    for node_id, name, serial, client_id in nodes:
        entries.append({"id": node_id, "type": "PipeWire:Interface:Node",
                        "info": {"props": {"media.class": "Video/Source",
                                           "node.name": name,
                                           "object.serial": serial,
                                           "client.id": client_id}}})
        if client_id is not None:
            entries.append({"id": client_id, "type": "PipeWire:Interface:Client",
                            "info": {"props": {
                                "application.process.binary": "gamescope",
                                "application.process.id": 4242,
                                "pipewire.sec.pid": 4242,
                                "pipewire.sec.uid": UID}}})
    return entries


class Discovery(unittest.TestCase):
    def test_joins_node_to_client(self) -> None:
        nodes = rec.video_source_nodes(dump((179, "gamescope", 3859, 203)))
        self.assertEqual(1, len(nodes))
        self.assertEqual("gamescope", nodes[0]["binary"])
        self.assertEqual(4242, nodes[0]["process_id"])

    def test_ignores_non_video_classes(self) -> None:
        entries = dump((179, "gamescope", 3859, 203))
        entries[0]["info"] = {"props": {"media.class": "Audio/Sink"}}
        self.assertEqual(1, len(rec.video_source_nodes(entries)))

    def test_selects_the_single_verified_node(self) -> None:
        nodes = rec.video_source_nodes(dump((179, "gamescope", 3859, 203)))
        self.assertEqual(3859, rec.select_target(nodes, UID, process_check=alive)["serial"])

    def test_absent_node_is_an_error(self) -> None:
        with self.assertRaises(rec.CaptureError):
            rec.select_target([], UID, process_check=alive)

    def test_ambiguous_node_is_an_error(self) -> None:
        nodes = rec.video_source_nodes(
            dump((179, "gamescope", 3859, 203), (180, "gamescope", 3860, 204)))
        with self.assertRaises(rec.CaptureError) as caught:
            rec.select_target(nodes, UID, process_check=alive)
        self.assertIn("found 2", str(caught.exception))

    def test_rejects_foreign_uid(self) -> None:
        entries = dump((179, "gamescope", 3859, 203))
        for entry in entries:
            if entry["type"] == "PipeWire:Interface:Client":
                entry["info"]["props"]["pipewire.sec.uid"] = UID + 1
        with self.assertRaises(rec.CaptureError):
            rec.select_target(rec.video_source_nodes(entries), UID, process_check=alive)

    def test_rejects_client_that_is_not_the_compositor(self) -> None:
        entries = dump((179, "gamescope", 3859, 203))
        for entry in entries:
            if entry["type"] == "PipeWire:Interface:Client":
                entry["info"]["props"]["application.process.binary"] = "wireplumber"
        with self.assertRaises(rec.CaptureError):
            rec.select_target(rec.video_source_nodes(entries), UID, process_check=alive)

    def test_rejects_mismatched_peer_pid(self) -> None:
        entries = dump((179, "gamescope", 3859, 203))
        for entry in entries:
            if entry["type"] == "PipeWire:Interface:Client":
                entry["info"]["props"]["pipewire.sec.pid"] = 1
        with self.assertRaises(rec.CaptureError):
            rec.select_target(rec.video_source_nodes(entries), UID, process_check=alive)

    def test_stale_node_is_rejected(self) -> None:
        # A node outlives its process; selecting it would record nothing.
        nodes = rec.video_source_nodes(dump((179, "gamescope", 3859, 203)))
        with self.assertRaises(rec.CaptureError):
            rec.select_target(nodes, UID, process_check=lambda *_: False)
        self.assertEqual(3859, rec.select_target(nodes, UID, process_check=alive)["serial"])

    def test_process_check_rejects_a_foreign_pid(self) -> None:
        checked = []

        def check(pid, uid):
            checked.append((pid, uid))
            return True

        rec.select_target(rec.video_source_nodes(dump((179, "gamescope", 3859, 203))),
                          UID, process_check=check)
        self.assertEqual([(4242, UID)], checked)

    def test_rejects_node_without_serial(self) -> None:
        with self.assertRaises(rec.CaptureError):
            rec.select_target(rec.video_source_nodes(dump((179, "gamescope", None, 203))), UID, process_check=alive)

    def test_other_node_name_is_rejected(self) -> None:
        nodes = rec.video_source_nodes(
            dump((140, "v4l2_input.usb-0", 140, 33), (179, "gamescope", 3859, 203)))
        self.assertEqual(3859, rec.select_target(nodes, UID, process_check=alive)["serial"])

    def test_pid_filter_selects_one_of_several(self) -> None:
        entries = dump((179, "gamescope", 3859, 203))
        entries.append({"id": 180, "type": "PipeWire:Interface:Node",
                        "info": {"props": {"media.class": "Video/Source",
                                           "node.name": "gamescope",
                                           "object.serial": 3860,
                                           "client.id": 204}}})
        entries.append({"id": 204, "type": "PipeWire:Interface:Client",
                        "info": {"props": {
                            "application.process.binary": "gamescope",
                            "application.process.id": 777,
                            "pipewire.sec.pid": 777,
                            "pipewire.sec.uid": UID}}})
        nodes = rec.video_source_nodes(entries)
        self.assertEqual(3860, rec.select_target(nodes, UID, process_check=alive, wanted_pid=777)["serial"])
        self.assertEqual(3859, rec.select_target(nodes, UID, process_check=alive, wanted_pid=4242)["serial"])
        with self.assertRaises(rec.CaptureError):
            rec.select_target(nodes, UID, process_check=alive, wanted_pid=1234)


class OutputGuard(unittest.TestCase):
    def test_absent_outputs_report_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual([], rec.existing_outputs(Path(directory)))

    def test_previous_run_outputs_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "capture.mkv").write_bytes(b"x")
            (path / "session.json").write_text("{}", encoding="utf-8")
            self.assertEqual(["capture.mkv", "session.json"], rec.existing_outputs(path))


class Pipeline(unittest.TestCase):
    def test_targets_the_serial_not_a_path(self) -> None:
        text = rec.build_pipeline(3859, rec.Path("/tmp/x.mkv"), "nvav1enc", 0)
        self.assertIn("target-object=3859", text)
        self.assertNotIn("path=", text)

    def test_stamps_arrival_time(self) -> None:
        # The stream's own timestamps are unusable, so arrival time is stamped.
        text = rec.build_pipeline(1, rec.Path("/tmp/x.mkv"), "av1enc", 0)
        self.assertIn("do-timestamp=true", text)

    def test_uses_one_caps_filter(self) -> None:
        for fps in (0, 60):
            text = rec.build_pipeline(1, rec.Path("/tmp/x.mkv"), "av1enc", fps)
            self.assertEqual(1, text.count("video/x-raw,"))

    def test_fps_cap_is_applied(self) -> None:
        self.assertIn("videorate", rec.build_pipeline(1, rec.Path("/tmp/x.mkv"), "av1enc", 60))

    def test_zero_fps_keeps_the_stream_rate(self) -> None:
        self.assertNotIn("videorate", rec.build_pipeline(1, rec.Path("/tmp/x.mkv"), "av1enc", 0))
        self.assertEqual(0, rec.DEFAULT_FPS)

    def test_every_encoded_frame_is_counted(self) -> None:
        text = rec.build_pipeline(1, rec.Path("/tmp/x.mkv"), "av1enc", 0)
        self.assertLess(text.index("identity"), text.index("matroskamux"))

    def test_cbr_is_the_default_and_caps_the_rate(self) -> None:
        self.assertEqual("cbr", rec.DEFAULT_RC_MODE)
        text = rec.build_pipeline(1, rec.Path("/tmp/x.mkv"), "nvav1enc", 0)
        self.assertIn("rc-mode=cbr", text)
        # Constant rate must not also carry a variable-rate ceiling.
        self.assertNotIn("max-bitrate", text)

    def test_vbr_keeps_headroom_when_requested(self) -> None:
        text = rec.build_pipeline(1, rec.Path("/tmp/x.mkv"), "nvav1enc", 0, 4000, "vbr")
        self.assertIn("rc-mode=vbr", text)
        self.assertIn("max-bitrate=8000", text)

    def test_bitrate_is_applied(self) -> None:
        text = rec.build_pipeline(1, rec.Path("/tmp/x.mkv"), "nvav1enc", 0, 4000, "vbr")
        self.assertIn("bitrate=4000", text)
        self.assertIn("max-bitrate=8000", text)

    def test_default_bitrate_is_modest(self) -> None:
        # The point of the target rate is a small file, so assert the default.
        self.assertLessEqual(rec.DEFAULT_BITRATE_KBPS, 6000)
        text = rec.build_pipeline(1, rec.Path("/tmp/x.mkv"), "nvav1enc", rec.DEFAULT_FPS)
        self.assertIn(f"bitrate={rec.DEFAULT_BITRATE_KBPS}", text)

    def test_non_nvenc_encoder_gets_no_nvenc_options(self) -> None:
        text = rec.build_pipeline(1, rec.Path("/tmp/x.mkv"), "x264enc", 0, 4000)
        self.assertNotIn("bitrate=4000", text)
        self.assertNotIn("rc-mode", text)

    def test_output_path_is_used(self) -> None:
        self.assertIn("location=/tmp/out.mkv",
                      rec.build_pipeline(1, rec.Path("/tmp/out.mkv"), "av1enc", 0))


class Timeline(unittest.TestCase):
    def test_monotonic_timeline_has_no_problems(self) -> None:
        self.assertEqual([], rec.check_timestamps([0.0, 0.016, 0.033, 1.5]))

    def test_regression_is_detected(self) -> None:
        problems = rec.check_timestamps([0.0, 0.5, 0.25])
        self.assertEqual(1, len(problems))
        self.assertIn("precedes", problems[0])

    def test_empty_recording_is_detected(self) -> None:
        self.assertEqual(["the recording contains no frames"], rec.check_timestamps([]))

    def test_negative_first_timestamp_is_detected(self) -> None:
        self.assertTrue(any("negative" in p for p in rec.check_timestamps([-1.0, 0.0])))

    def test_repeated_timestamps_are_allowed(self) -> None:
        self.assertEqual([], rec.check_timestamps([0.0, 0.0, 0.016]))

    def test_frame_count_mismatch_is_an_error(self) -> None:
        self.assertEqual([], rec.check_frame_count(10, 10))
        problems = rec.check_frame_count(10, 9)
        self.assertEqual(1, len(problems))
        self.assertIn("cannot be trusted", problems[0])

    def test_sidecar_must_be_monotonic(self) -> None:
        self.assertEqual([], rec.check_sidecar([1, 2, 2, 3]))
        self.assertTrue(rec.check_sidecar([1, 0]))
        self.assertEqual(["frames.jsonl is empty"], rec.check_sidecar([]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
