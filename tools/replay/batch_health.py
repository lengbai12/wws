#!/usr/bin/env python3
"""Report the health of the unattended 15.8 recording batch.

The batch runs for hours and its failures are quiet. The driver can die, the
Steam client the direct route needs for Steamworks can be closed, or a session
can stall behind a game process that never exits. Every check therefore states
the evidence it used, and the exit code is non-zero when a check fails, so a
caller can branch on it without parsing prose.

Usage:
  /usr/bin/python3 tools/replay/batch_health.py
  /usr/bin/python3 tools/replay/batch_health.py --json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
# The batch log moves with the corpus being built. Reading the wrong one makes the
# check blind: it then reports the previous run's session as the one in flight and
# can mistake that run's finish for this one's. The newest of the known names wins,
# and --log overrides it.
LOG_CANDIDATES = (
    WORKSPACE / "replays/parsed/ops-batch.log",
    WORKSPACE / "replays/parsed/auto-learn-158-batch.log",
    WORKSPACE / "replays/parsed/auto-learn-158-batch2.log",
)
LOG = max((p for p in LOG_CANDIDATES if p.is_file()),
          key=lambda p: p.stat().st_mtime, default=LOG_CANDIDATES[0])
REPLAYS = WORKSPACE / "replays"
RECORDINGS = WORKSPACE / "recordings"
PARSED = WORKSPACE / "replays/parsed"

# Each session writes about 2 GB and needs about 6 GB of memory while its game
# runs. The limits leave room for the work that follows the batch instead of
# discovering the shortage at the last session.
MIN_FREE_DISK_GB = 30
MIN_FREE_MEMORY_GB = 4
# A recording legitimately pauses between sessions: the client's own start-up,
# the recorder's full verify, and the per-session label stages add up to several
# minutes with no new frames. Only a pause this long means something is wrong.
STALL_AFTER_S = 12 * 60
GROWTH_SAMPLE_S = 20
# The panel rectangle is asserted by the pipeline and never measured, so it is
# measured here instead, once per recording. A recording measured at another
# rectangle must not reach the corpus: its crops and its labels would describe a
# different input than the model was trained on.
RECT_CACHE = PARSED / "minimap-rect-verify.json"
RECT_TOOL = WORKSPACE / "tools/replay/verify_minimap_rect.py"
HUD_CALIBRATION = PARSED / "calibration/hud-2560x1600-minimapmax.json"
# Recordings already known to disagree with the calibration. They stay listed so
# the check reports them as known instead of alarming every 30 minutes.
RECT_QUARANTINE = PARSED / "minimap-rect-quarantine.txt"
# The measurement rule this checker uses. It is recorded with every cached
# verdict, so changing a threshold re-measures instead of trusting an old answer.
RECT_RULE = {"ratio": 3.0, "share": 2 / 3, "frames": 7}
# A deliberate stop is written here by the stop watcher, so that a batch which was
# stopped on purpose is not reported as a driver that vanished.
STOP_MARKER = PARSED / "batch-stop.json"


def read_stop_marker(driver_pids_seen: list[int] | None = None) -> dict:
    """The deliberate-stop record, when it still describes the current run.

    A stop record outlives the run it describes, and a later resume starts a new
    driver. Comparing the two times is what tells them apart: a driver that began
    AFTER the recorded stop supersedes it, and the record must not then be read as
    "the stop failed".
    """
    if not STOP_MARKER.is_file():
        return {}
    try:
        marker = json.loads(STOP_MARKER.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    stopped = marker.get("stopped_at")
    drivers = driver_pids_seen or []
    if stopped and drivers:
        try:
            stopped_epoch = time.mktime(time.strptime(stopped[:19], "%Y-%m-%dT%H:%M:%S"))
        except ValueError:
            return marker
        newest_start = 0.0
        for pid in drivers:
            try:
                newest_start = max(newest_start, Path(f"/proc/{pid}").stat().st_mtime)
            except OSError:
                continue
        if newest_start and newest_start > stopped_epoch:
            return {}
    return marker


def rect_rule() -> str:
    """The measurement rule, as a string that invalidates a stale cache.

    The thresholds decide the verdict, so a cached verdict from other thresholds
    is not evidence about this one.
    """
    return (f"ratio>={RECT_RULE['ratio']};share={RECT_RULE['share']};"
            f"frames={RECT_RULE['frames']}")

BATCH_START = re.compile(r"^=== .*批量开始.*?(\d{4}-\d\d-\d\dT[\d:+-]+) ===")
SESSION_LINE = re.compile(r"^\[(\d\d:\d\d:\d\d)\] === (\S+) \(recording (\S+)\)")
DONE_MARKER = "--skip-merge given; stopping after the per-session stages"


def driver_pids() -> list[int]:
    """Process ids running the batch driver, excluding this process.

    The command line is matched, so the pattern is bracketed and this process is
    skipped: a pattern is otherwise found in the checker's own invocation.
    """
    pids = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        joined = b" ".join(argv).decode("utf-8", "replace")
        if "auto_learn" in joined and "--skip-merge" in joined:
            pids.append(int(entry.name))
    return pids


def process_names() -> set[str]:
    names = set()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            names.add((entry / "comm").read_text(encoding="utf-8").strip())
        except OSError:
            continue
    return names


def usable_recordings() -> dict[str, str]:
    """Replay path -> recording directory, for finalised recordings only."""
    found = {}
    for directory in sorted(RECORDINGS.glob("*")):
        marker = directory / "replay_path.txt"
        if not directory.is_dir() or not marker.exists():
            continue
        if not (directory / "session.json").exists():
            continue
        found[str(Path(marker.read_text(encoding="utf-8").strip()).resolve())] = \
            directory.name
    return found


def free_gb() -> tuple[float, float]:
    """(available memory GB, free disk GB on the workspace's filesystem)."""
    memory = 0.0
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            memory = int(line.split()[1]) / (1024 * 1024)
            break
    disk = subprocess.run(["df", "-BG", "--output=avail", str(WORKSPACE)],
                          capture_output=True, text=True).stdout.splitlines()
    return memory, float(disk[-1].strip().rstrip("G"))


def frame_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    with path.open("rb") as stream:
        return sum(1 for _ in stream)


def session_reports() -> list[dict]:
    """Integrity facts of every recording finalised after the batch started."""
    reports = []
    for directory in sorted(RECORDINGS.glob("*")):
        session = directory / "session.json"
        if not directory.is_dir() or not session.exists():
            continue
        try:
            meta = json.loads(session.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        problems = meta.get("problems")
        if not problems and not meta.get("failed"):
            continue
        reports.append({
            "recording": directory.name,
            "frames": meta.get("frames"),
            "sidecar_frames": meta.get("sidecar_frames"),
            "failed": meta.get("failed"),
            "problems": problems,
        })
    return reports


def log_markers() -> dict:
    """Batch start, the session in flight, and stage failures since the start."""
    if not LOG.is_file():
        return {"started": None, "last_session": None, "elapsed_s": None,
                "failures": [], "finished": False, "tail": []}
    lines = LOG.read_text(encoding="utf-8", errors="replace").splitlines()
    started_index = None
    started_wall = None
    for index, line in enumerate(lines):
        match = BATCH_START.match(line)
        if match:
            started_index, started_wall = index, match.group(1)
    scope = lines[started_index:] if started_index is not None else lines
    last_session = None
    for line in scope:
        match = SESSION_LINE.match(line)
        if match:
            last_session = {"at": match.group(1), "session": match.group(2),
                            "recording": match.group(3)}
    failures = [line for line in scope if "FAILED" in line]
    elapsed = None
    if started_wall:
        try:
            elapsed = round(time.time() - time.mktime(
                time.strptime(started_wall[:19], "%Y-%m-%dT%H:%M:%S")))
        except ValueError:
            elapsed = None
    return {"started": started_wall, "last_session": last_session,
            "elapsed_s": elapsed, "failures": failures,
            "finished": bool(lines) and DONE_MARKER in lines[-1],
            "tail": lines[-4:]}


def read_quarantine() -> set[str]:
    if not RECT_QUARANTINE.is_file():
        return set()
    return {line.strip() for line in
            RECT_QUARANTINE.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")}


def verify_panel_rects() -> dict:
    """Measure every finalised recording's minimap panel once, and remember it.

    Only the recordings not measured before under the current rule cost a decode,
    so a health check run stays short however long the batch is.
    """
    rule = rect_rule()
    cache = {}
    if RECT_CACHE.is_file():
        try:
            cache = json.loads(RECT_CACHE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cache = {}
    cache = {name: entry for name, entry in cache.items()
             if isinstance(entry, dict) and entry.get("rule") == rule}
    added = False
    for directory in sorted(RECORDINGS.glob("*")):
        if not (directory / "session.json").is_file():
            continue
        # Directories kept as evidence of a failed attempt are not candidates for
        # the corpus, so measuring them reports a difference that means nothing.
        if directory.name.startswith("_"):
            continue
        if directory.name in cache:
            continue
        video = directory / "capture.mkv"
        if not video.is_file():
            continue
        completed = subprocess.run(
            ["/usr/bin/python3", str(RECT_TOOL), "--video", str(video),
             "--calibration", str(HUD_CALIBRATION), "--json",
             "--frames", str(RECT_RULE["frames"])],
            capture_output=True, text=True)
        try:
            entry = json.loads(completed.stdout)
        except json.JSONDecodeError:
            entry = {"verdict": "UNREADABLE",
                     "error": (completed.stderr or "no output").strip()[-300:]}
        entry["rule"] = rule
        cache[directory.name] = entry
        added = True
    if added:
        RECT_CACHE.write_text(json.dumps(cache, indent=2, ensure_ascii=False) + "\n",
                              encoding="utf-8")
    return cache


def collect() -> dict:
    replays = sorted(REPLAYS.glob("*.wowsreplay"))
    usable = usable_recordings()
    done = [p.stem for p in replays if str(p.resolve()) in usable]
    pending = [p.stem for p in replays if str(p.resolve()) not in usable]
    markers = log_markers()
    names = process_names()
    memory_gb, disk_gb = free_gb()

    driver = driver_pids()
    current = None
    if markers["last_session"]:
        recording = RECORDINGS / markers["last_session"]["recording"]
        frames_path = recording / "frames.jsonl"
        first = frame_count(frames_path)
        time.sleep(GROWTH_SAMPLE_S)
        second = frame_count(frames_path)
        per_second = None
        if first is not None and second is not None and second >= first:
            per_second = round((second - first) / GROWTH_SAMPLE_S, 1)
        current = {
            "session": markers["last_session"]["session"],
            "recording": markers["last_session"]["recording"],
            "frames": second,
            "frames_per_second": per_second,
            "finalised": (recording / "session.json").exists(),
        }

    problems = []
    notes = []
    stopped = read_stop_marker(driver)
    if stopped:
        if driver:
            # The stop was requested but the driver is still there: either the stop
            # has not fired yet or it failed, and both are worth knowing.
            problems.append(
                "a deliberate stop was recorded at "
                f"{stopped.get('stopped_at')} but the batch driver is still running "
                f"(pids {driver})")
        else:
            notes.append(
                "the batch was stopped on purpose at "
                f"{stopped.get('stopped_at')} (deadline {stopped.get('deadline')}): "
                f"{stopped.get('reason')}. Resume with "
                "`./.venv/bin/python tools/replay/auto_learn.py --skip-merge` "
                "after starting the Steam client.")
    if not driver and not stopped:
        if not markers["finished"]:
            problems.append(
                "the batch driver is gone but the run never reached its end marker"
                f" (last log line: {markers['tail'][-1] if markers['tail'] else 'none'})")
    if not driver and not markers.get("started"):
        problems.append("no batch has been started in this workspace's log")
    if driver and "steam" not in names:
        problems.append("no Steam client is running: the direct route needs it for "
                        "Steamworks, so the next launch would exit within a second")
    if disk_gb < MIN_FREE_DISK_GB:
        problems.append(f"only {disk_gb:.0f} GB of disk left (limit "
                        f"{MIN_FREE_DISK_GB} GB): the remaining sessions will not fit")
    if memory_gb < MIN_FREE_MEMORY_GB:
        problems.append(f"only {memory_gb:.1f} GB of memory available (limit "
                        f"{MIN_FREE_MEMORY_GB} GB): the game needs about 6 GB")
    if driver and current and current["frames_per_second"] == 0 \
            and not current["finalised"]:
        # Between sessions the driver is legitimately busy with the recorder's
        # finalise pass or the per-session label stages, and neither writes frames.
        # Only "nothing is running and no frame has arrived for a long time" is a
        # stall, so the test is on the sidecar's own age, not on a log line.
        busy = {"WorldOfWarships", "ffprobe", "gamescope-wl"} & names
        sidecar = RECORDINGS / current["recording"] / "frames.jsonl"
        quiet = None
        if sidecar.is_file():
            quiet = (time.time() - sidecar.stat().st_mtime) / 60
        if not busy and quiet is not None and quiet > STALL_AFTER_S / 60:
            problems.append(
                f"session {current['session']} has written no frame for "
                f"{quiet:.0f} min and nothing is running: the game, the recorder's "
                "verify pass and the pipeline are all absent")

    panels = verify_panel_rects()
    quarantine = read_quarantine()
    quarantined = sorted(name for name, result in panels.items()
                         if result.get("verdict") != "PASS" and name in quarantine)
    for name, result in sorted(panels.items()):
        if result.get("verdict") == "PASS":
            continue
        detail = result.get("error") or next(
            (sample["reason"] for sample in result.get("samples", [])
             if sample.get("reason")), "no reason reported")
        message = (
            f"recording {name}: the recorded minimap panel does not match the "
            f"calibration {result.get('rect')} (verdict {result.get('verdict')}): "
            f"{detail}")
        if name in quarantine:
            continue
        problems.append(message)

    return {
        "driver_pids": driver,
        "steam_running": "steam" in names,
        "game_running": "WorldOfWarships" in names,
        "recorded": len(done),
        "pending": len(pending),
        "total": len(replays),
        "current": current,
        "markers": markers,
        "memory_available_gb": round(memory_gb, 1),
        "disk_free_gb": round(disk_gb, 1),
        "finalised_with_problems": session_reports(),
        "notes": notes,
        "minimap_panel": {
            "checked": len(panels),
            "passed": sum(1 for r in panels.values() if r.get("verdict") == "PASS"),
            "quarantined": quarantined,
            "new_failures": sorted(
                name for name, r in panels.items()
                if r.get("verdict") != "PASS" and name not in quarantine),
        },
        "problems": problems,
    }


def report(state: dict) -> None:
    markers = state["markers"]
    print(f"batch started {markers['started']}"
          + (f" ({markers['elapsed_s'] / 3600:.1f} h ago)" if markers["elapsed_s"] else ""))
    print(f"recorded {state['recorded']}/{state['total']}; "
          f"pending {state['pending']}")
    current = state["current"]
    if current:
        shape = ("finalised" if current["finalised"]
                 else f"{current['frames']} frames, {current['frames_per_second']}/s")
        print(f"current session {current['session']} ({current['recording']}): {shape}")
    print(f"driver pids {state['driver_pids'] or 'none'}; "
          f"steam={'yes' if state['steam_running'] else 'NO'}; "
          f"game={'yes' if state['game_running'] else 'no'}")
    print(f"memory available {state['memory_available_gb']} GB; "
          f"disk free {state['disk_free_gb']} GB")
    panel = state["minimap_panel"]
    print(f"minimap panel verified {panel['passed']}/{panel['checked']}"
          + (f"; NEW FAILURES: {', '.join(panel['new_failures'])}"
             if panel["new_failures"] else "")
          + (f"; quarantined (known): {', '.join(panel['quarantined'])}"
             if panel["quarantined"] else ""))
    if markers["failures"]:
        print(f"stage failures in this batch: {len(markers['failures'])}")
        for line in markers["failures"][-4:]:
            print("  " + line.strip())
    for item in state["finalised_with_problems"]:
        print(f"recording {item['recording']} reported problems: "
              f"failed={item['failed']} problems={item['problems']}")
    for note in state.get("notes", []):
        print("NOTE: " + note)
    if state["problems"]:
        print("VERDICT PROBLEM")
        for item in state["problems"]:
            print("  - " + item)
    else:
        print("VERDICT OK")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--log", type=Path,
                        help="batch log to read (default: the newest known one)")
    args = parser.parse_args()
    global LOG
    if args.log:
        LOG = args.log.resolve()
    state = collect()
    state["log"] = str(LOG)
    if args.json:
        print(json.dumps(state, indent=2, ensure_ascii=False))
    else:
        report(state)
    return 1 if state["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
