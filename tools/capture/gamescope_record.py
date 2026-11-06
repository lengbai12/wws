#!/usr/bin/env python3
"""Record the gamescope screen-cast stream with per-frame arrival timestamps.

The capture target is the gamescope compositor's PipeWire ``Video/Source`` node,
not a desktop portal screen cast. Selection is by verified process identity and
fails closed: an absent or ambiguous node is an error, never a silent fallback to
another video source.

Two facts drive the pipeline shape:

* The stream's own buffer timestamps are unusable. They arrive as zero or unset,
  which produced a corrupt container, so ``do-timestamp`` stamps each buffer with
  the pipeline clock when it arrives.
* The timestamp that matters is when the frame arrived here, because that is what
  a label must be aligned against. Every encoded frame is therefore also written
  to ``frames.jsonl`` with a local monotonic arrival time.

The frame count in the container and the line count in ``frames.jsonl`` must
agree, otherwise the frame-to-time mapping is unsound and the run is reported as
failed.

Run with the system interpreter, which provides the GStreamer bindings:

    /usr/bin/python3 tools/capture/gamescope_record.py --out DIR --seconds 20 --verify
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Client binaries that identify the compositor which owns the target node.
GAMESCOPE_BINARIES = ("gamescope", ".gamescope-wrapped")
DEFAULT_NODE_NAME = "gamescope"
# A frame rate cap is off by default: capture keeps the real stream rate and any
# subsampling happens later, where the arrival times are still available.
DEFAULT_FPS = 0
# Target rate in kbit/s. A recording is a working artefact, so a modest rate is
# preferable to a large one. NVENC overshoots its constant-rate target by about
# 1.57x on fast footage (measured: 4.7 Mbit/s against a 3 Mbit/s setting), so the
# target is set low enough that the result stays near 3 Mbit/s.
DEFAULT_BITRATE_KBPS = 2000
# Constant rate holds the file to a predictable size. Variable rate was measured
# at 1.95x its target (7.8 Mbit/s against a 4 Mbit/s setting), so it is not used
# by default.
DEFAULT_RC_MODE = "cbr"
DEFAULT_ENCODERS = ("nvav1enc", "nvh264enc", "av1enc", "x264enc")


class CaptureError(RuntimeError):
    """A condition that makes the recording meaningless if ignored."""


# --- target discovery (pure functions, unit tested) ------------------------

def video_source_nodes(dump: list[dict]) -> list[dict]:
    """Video/Source nodes joined with their owning client's process properties."""
    clients: dict[int, dict] = {}
    for entry in dump:
        if entry.get("type") != "PipeWire:Interface:Client":
            continue
        clients[entry["id"]] = (entry.get("info") or {}).get("props") or {}
    nodes = []
    for entry in dump:
        if entry.get("type") != "PipeWire:Interface:Node":
            continue
        props = (entry.get("info") or {}).get("props") or {}
        if props.get("media.class") != "Video/Source":
            continue
        client_id = props.get("client.id")
        client = clients.get(int(client_id)) if client_id is not None else {}
        nodes.append({
            "id": entry.get("id"),
            "name": props.get("node.name"),
            "serial": props.get("object.serial"),
            "client_id": client_id,
            "binary": client.get("application.process.binary"),
            "process_id": client.get("application.process.id"),
            "sec_pid": client.get("pipewire.sec.pid"),
            "sec_uid": client.get("pipewire.sec.uid"),
        })
    return nodes


def process_alive(pid: object, uid: int) -> bool:
    """Whether ``pid`` is a live process owned by ``uid``.

    A node can outlive its process, so a stale node must not be selected.
    """
    try:
        path = Path(f"/proc/{int(pid)}")
        return path.is_dir() and path.stat().st_uid == uid
    except (TypeError, ValueError, OSError):
        return False


def select_target(nodes: list[dict], uid: int, node_name: str = DEFAULT_NODE_NAME,
                  wanted_pid: int | None = None, process_check=process_alive) -> dict:
    """The single compositor node, verified by process identity.

    Raises when the node is absent or ambiguous. A fallback to another video
    source would record the wrong thing, so it is not permitted.
    """
    rejected = []
    candidates = []
    for node in nodes:
        reason = None
        if node.get("serial") is None:
            reason = "no object.serial"
        elif node.get("name") != node_name:
            reason = f"node name {node.get('name')!r}"
        elif node.get("binary") not in GAMESCOPE_BINARIES:
            reason = f"client binary {node.get('binary')!r}"
        elif str(node.get("process_id")) != str(node.get("sec_pid")):
            reason = (f"process id {node.get('process_id')} differs from "
                      f"peer process id {node.get('sec_pid')}")
        elif str(node.get("sec_uid")) != str(uid):
            reason = f"peer uid {node.get('sec_uid')} is not {uid}"
        elif not process_check(node.get("process_id"), uid):
            reason = f"process {node.get('process_id')} is not alive or not owned by {uid}"
        elif wanted_pid is not None and str(node.get("process_id")) != str(wanted_pid):
            reason = f"process id {node.get('process_id')} is not {wanted_pid}"
        if reason:
            rejected.append(f"node {node.get('id')}: {reason}")
            continue
        candidates.append(node)
    if len(candidates) != 1:
        listing = ", ".join(f"{c.get('name')}#{c.get('id')}" for c in candidates) or "none"
        shown = rejected[:6]
        more = f" (+{len(rejected) - len(shown)} more)" if len(rejected) > len(shown) else ""
        raise CaptureError(
            f"expected one {node_name} video source, found {len(candidates)} ({listing}); "
            f"rejected: {'; '.join(shown) or 'none'}{more}")
    return candidates[0]


def read_pipewire_dump() -> list[dict]:
    if shutil.which("pw-dump") is None:
        raise CaptureError("pw-dump is not installed")
    # Discovery consumes JSON, not terminal formatting. Keep the full fresh
    # registry snapshot (including clients) without color/indentation overhead.
    result = subprocess.run(["pw-dump", "--no-colors", "--raw"],
                            capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise CaptureError(f"pw-dump failed: {result.stderr.strip()}")
    return json.loads(result.stdout)


# --- pipeline --------------------------------------------------------------

def build_pipeline(serial: object, out: Path, encoder: str, fps: int,
                   bitrate_kbps: int = DEFAULT_BITRATE_KBPS,
                   rc_mode: str = DEFAULT_RC_MODE) -> str:
    """Pipeline text. One caps filter: two adjacent filters parse as an element.

    ``rc_mode`` selects how the target rate is honoured. Constant rate holds the
    file to a predictable size; variable rate overshoots its target.
    """
    caps = "video/x-raw,format=BGRA" + (f",framerate={fps}/1" if fps > 0 else "")
    rate = "videorate ! " if fps > 0 else ""
    tuned = ""
    if encoder.startswith("nv") and bitrate_kbps > 0:
        if rc_mode == "cbr":
            tuned = f" bitrate={bitrate_kbps} rc-mode=cbr gop-size=120"
        else:
            tuned = (f" bitrate={bitrate_kbps} rc-mode=vbr "
                     f"max-bitrate={bitrate_kbps * 2} gop-size=120")
    return (f"pipewiresrc target-object={serial} client-name=wws-capture "
            f"do-timestamp=true ! videoconvert ! {rate}{caps} ! "
            f"{encoder}{tuned} ! identity name=stamp signal-handoffs=true ! "
            f"matroskamux ! filesink location={out}")


def available_element(name: str) -> bool:
    result = subprocess.run(["gst-inspect-1.0", name], capture_output=True, text=True)
    return result.returncode == 0


def choose_encoder(preferred: list[str]) -> str:
    for name in preferred:
        if available_element(name):
            return name
    raise CaptureError(f"no usable encoder among {preferred}")


def record(serial: object, out: Path, frames_path: Path, encoder: str, fps: int,
           seconds: float, bitrate_kbps: int = DEFAULT_BITRATE_KBPS,
           rc_mode: str = DEFAULT_RC_MODE) -> dict:
    """Run the capture pipeline, writing frames.jsonl. Raises on pipeline failure."""
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import GLib, Gst

    Gst.init(None)
    pipeline_text = build_pipeline(serial, out, encoder, fps, bitrate_kbps, rc_mode)
    pipeline = Gst.parse_launch(pipeline_text)
    stamp = pipeline.get_by_name("stamp")
    loop = GLib.MainLoop()
    failure: list[str] = []
    frames_seen = 0
    # The sidecar is written and flushed per frame: an interrupted recording must
    # not lose the arrival times that give the video its meaning.
    sidecar = frames_path.open("w", encoding="utf-8")
    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def on_error(_bus, message):
        error, debug = message.parse_error()
        failure.append(f"{error.message}" + (f" ({debug.splitlines()[0]})" if debug else ""))
        loop.quit()

    def on_handoff(_identity, buffer, *_rest):
        # Arrival time of this frame, the value a label is aligned against.
        nonlocal frames_seen
        now = time.monotonic_ns()
        pts = buffer.pts
        sidecar.write(json.dumps({
            "index": frames_seen,
            "monotonic_ns": now,
            "since_start_ns": now - started_mono_ns,
            "pipeline_pts_ns": pts if pts is not None and pts >= 0 else None,
            "bytes": buffer.get_size(),
        }) + "\n")
        sidecar.flush()
        frames_seen += 1

    import signal

    def on_signal():
        # A signalled stop must still finalise the container and the sidecar.
        # The handler runs on the main loop, because a plain Python signal
        # handler does not run while the loop is blocked in native code.
        pipeline.send_event(Gst.Event.new_eos())
        return True

    GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGINT, on_signal)
    GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGTERM, on_signal)

    bus.connect("message::error", on_error)
    bus.connect("message::eos", lambda *_: loop.quit())
    stamp.connect("handoff", on_handoff)

    started_mono_ns = time.monotonic_ns()
    started_wall = time.time()
    started_mono = time.monotonic()
    pipeline.set_state(Gst.State.PLAYING)
    GLib.timeout_add(int(seconds * 1000),
                     lambda: (pipeline.send_event(Gst.Event.new_eos()), False)[1])
    loop.run()
    pipeline.set_state(Gst.State.NULL)
    sidecar.close()
    stopped_mono = time.monotonic()

    meta = {
        "pipeline": pipeline_text,
        "encoder": encoder,
        "fps_cap": fps,
        "bitrate_kbps": bitrate_kbps,
        "rc_mode": rc_mode,
        "seconds_requested": seconds,
        "wall_start": started_wall,
        "monotonic_start_ns": started_mono_ns,
        "monotonic_stop_ns": time.monotonic_ns(),
        "elapsed_s": round(stopped_mono - started_mono, 3),
        "sidecar_frames": frames_seen,
        "frames_jsonl": str(frames_path),
    }
    if failure:
        raise CaptureError("pipeline failed: " + "; ".join(failure))
    if not out.is_file() or out.stat().st_size == 0:
        raise CaptureError(f"no output written to {out}")
    if not frames_seen:
        raise CaptureError("the pipeline produced no frames")
    return meta


# --- verification (pure functions, unit tested) ----------------------------

def check_timestamps(pts: list[float], tolerance_s: float = 0.0) -> list[str]:
    """Problems with a recorded frame timeline."""
    problems = []
    if not pts:
        return ["the recording contains no frames"]
    for index in range(1, len(pts)):
        if pts[index] < pts[index - 1] - tolerance_s:
            problems.append(
                f"frame {index} timestamp {pts[index]} precedes frame {index - 1} "
                f"at {pts[index - 1]}")
    if pts[0] < 0.0:
        problems.append(f"first frame timestamp {pts[0]} is negative")
    return problems


def check_frame_count(sidecar_frames: int, container_frames: int) -> list[str]:
    """The frame-to-time mapping holds only when both counts agree."""
    if sidecar_frames != container_frames:
        return [f"frames.jsonl has {sidecar_frames} entries but the container holds "
                f"{container_frames} frames, so frame times cannot be trusted"]
    return []


def check_sidecar(times_ns: list[int]) -> list[str]:
    """Problems with the arrival-time series."""
    if not times_ns:
        return ["frames.jsonl is empty"]
    problems = []
    for index in range(1, len(times_ns)):
        if times_ns[index] < times_ns[index - 1]:
            problems.append(f"arrival time {index} precedes its predecessor")
    return problems


def probe(path: Path) -> dict:
    """Frame count, duration and per-frame timestamps of a recording."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "frame=pts_time", "-show_entries", "format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise CaptureError(f"ffprobe failed: {result.stderr.strip()}")
    data = json.loads(result.stdout)
    pts = [float(f["pts_time"]) for f in data.get("frames", []) if "pts_time" in f]
    return {"frames": len(pts), "first_pts_s": pts[0] if pts else None,
            "last_pts_s": pts[-1] if pts else None,
            "duration_s": float(data.get("format", {}).get("duration", 0.0)),
            "pts": pts}


def probe_metadata(path: Path) -> dict:
    """Duration of a recording without decoding it.

    Counting frames or reading every timestamp makes ffprobe decode the whole
    file, which costs minutes on a long recording. A caller that only needs the
    duration must not pay that, as long as it states what it did not check.
    """
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise CaptureError(f"ffprobe failed: {result.stderr.strip()}")
    data = json.loads(result.stdout)
    return {"duration_s": float(data.get("format", {}).get("duration", 0.0))}


def read_sidecar(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise CaptureError(f"{path.name} line {lineno}: {error}") from error
    return rows


def existing_outputs(out: Path) -> list[str]:
    """Output files already present in ``out``.

    A directory holding output from an earlier run must not be reused silently:
    a run that outlives its directory replacement can leave files from two
    different recordings side by side.
    """
    return [name for name in ("capture.mkv", "frames.jsonl", "session.json")
            if (out / name).exists()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True,
                        help="output directory; writes capture.mkv and frames.jsonl")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS,
                        help="frame rate cap, 0 keeps the stream rate")
    parser.add_argument("--bitrate-mbps", type=float,
                        default=DEFAULT_BITRATE_KBPS / 1000,
                        help="target bit rate in Mbit/s")
    parser.add_argument("--rc-mode", choices=["cbr", "vbr"], default=DEFAULT_RC_MODE,
                        help="constant rate holds the size; variable rate overshoots")
    parser.add_argument("--node-name", default=DEFAULT_NODE_NAME)
    parser.add_argument("--gamescope-pid", type=int,
                        help="require this gamescope process id")
    parser.add_argument("--serial", help="use this PipeWire object.serial directly")
    parser.add_argument("--encoder", choices=[*DEFAULT_ENCODERS, "auto"], default="auto")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace output that an earlier run left in the directory")
    parser.add_argument("--verify-fast", action="store_true",
                        help="check duration and arrival times only; the frame "
                             "count and per-frame PTS are recorded as unchecked")
    parser.add_argument("--verify", action="store_true",
                        help="probe the recording and check the frame timeline")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    video = args.out / "capture.mkv"
    sidecar = args.out / "frames.jsonl"
    present = existing_outputs(args.out)
    if present and not args.overwrite:
        raise CaptureError(
            f"{args.out} already holds {', '.join(present)} "
            "from an earlier run; pass --overwrite to replace them")

    if args.serial:
        target = {"id": None, "name": args.node_name, "serial": args.serial,
                  "process_id": args.gamescope_pid}
    else:
        nodes = video_source_nodes(read_pipewire_dump())
        target = select_target(nodes, os.getuid(), args.node_name, args.gamescope_pid)
    encoder = args.encoder if args.encoder != "auto" else choose_encoder(list(DEFAULT_ENCODERS))
    print(f"target: node {target.get('id')} serial {target['serial']} "
          f"pid {target.get('process_id')} encoder {encoder} "
          f"bitrate {args.bitrate_mbps} Mbit/s")

    meta = record(target["serial"], video, sidecar, encoder, args.fps, args.seconds,
                  int(args.bitrate_mbps * 1000), args.rc_mode)
    meta["target"] = target
    meta["video"] = str(video)
    meta["video_bytes"] = video.stat().st_size
    problems: list[str] = []

    if args.verify_fast:
        # Metadata only: no decode, so minutes are saved on a long recording.
        # What was skipped is recorded here rather than left to be assumed.
        probed = probe_metadata(video)
        rows = read_sidecar(sidecar)
        meta["duration_s"] = probed["duration_s"]
        meta["frame_count_checked"] = False
        meta["pts_checked"] = False
        problems = check_sidecar([int(r["monotonic_ns"]) for r in rows])
        meta["problems"] = problems
    elif args.verify:
        probed = probe(video)
        rows = read_sidecar(sidecar)
        meta["frames"] = probed["frames"]
        meta["duration_s"] = probed["duration_s"]
        meta["first_pts_s"] = probed["first_pts_s"]
        meta["last_pts_s"] = probed["last_pts_s"]
        meta["frame_count_checked"] = True
        meta["pts_checked"] = True
        meta["frames_per_second"] = round(
            probed["frames"] / meta["elapsed_s"], 2) if meta["elapsed_s"] else None
        problems = (check_frame_count(len(rows), probed["frames"])
                    + check_timestamps(probed["pts"])
                    + check_sidecar([int(r["monotonic_ns"]) for r in rows]))
        meta["problems"] = problems
    (args.out / "session.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"wrote {meta['video']} ({meta['video_bytes']} bytes, "
          f"{meta['elapsed_s']}s, {meta['sidecar_frames']} frames timed)")
    if args.verify_fast:
        print(f"container duration {meta['duration_s']}s "
              f"(frame count NOT checked, PTS NOT checked)")
        for problem in problems:
            print(f"ERROR {problem}")
        if problems:
            sys.exit(1)
        print("arrival times ok")
    elif args.verify:
        print(f"container frames {meta['frames']} duration {meta['duration_s']}s "
              f"rate {meta['frames_per_second']}/s")
        for problem in problems:
            print(f"ERROR {problem}")
        if problems:
            sys.exit(1)
        print("timeline ok")
    print(f"wrote {args.out / 'session.json'} and {sidecar}")


if __name__ == "__main__":
    try:
        main()
    except CaptureError as error:
        raise SystemExit(f"capture error: {error}") from error
