"""Chain the whole minimap pipeline over every replay, unattended and resumable.

The reference project runs its stages by hand and never re-runs training on new
data. Its discipline is worth keeping and this driver keeps it: every stage is a
headless CLI, every artifact is written once and never overwritten, a stage whose
outputs already exist is skipped, a session that fails is recorded and the loop
moves on, and training only ever sees sessions whose gates passed.

Stages, in order, per replay:

  record     record the replay to recordings/<session>/ (unless already recorded)
  align      own-position stream, then anchors from the client log, then the fit
  project    per-MAP affine world -> minimap pixels, fitted against replay truth
  dataset    minimap crops + normalised own-ship labels
  inventory  every minimap marker class, with its colour gate
  heatmap    multi-class heatmap targets for training

Then, once per run: merge every session into one corpus with SESSION-LEVEL splits
and train on it. That split is the reason to run several sessions at all: a
single battle split into blocks still shares its map, ship and UI with itself.

State lives in ``replays/parsed/auto-learn-state.json`` and is rewritten after
every stage, so a run can be interrupted and resumed without redoing work.

Usage:
  auto_learn.py [--only ID[,ID...]] [--limit N] [--no-record] [--skip-train]
                [--epochs N] [--state PATH] [--dry-run] [--force STAGE]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
TOOLS = WORKSPACE / "tools"
PARSED = WORKSPACE / "replays/parsed"
PYTHON = str(WORKSPACE / ".venv/bin/python")
SYSTEM_PYTHON = "/usr/bin/python3"

# The HUD calibration is per UI configuration. The minimap size setting moves and
# rescales the panel, so the ROI, the zone pixels and the ring scale are all tied to
# it; a calibration made at one size cannot be used at another.
HUD_CALIBRATION = PARSED / "calibration/hud-2560x1600-minimapmax.json"
ZONE_DIR = PARSED / "calibration"
GRID = "176"
SIGMA = "1.0"
# The channel count is part of the artifact name: adding a class changes the model's
# output shape, and an old dataset must never be silently reused for a new one.
CLASS_COUNT = "10"
STAGES = ("recorded", "aligned", "projected", "dataset", "inventory", "heatmap")


def log(message):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), message), flush=True)


def run(argv, stage, state, key, dry_run=False):
    """Run one CLI, record its outcome, and never hide a failure."""
    printable = " ".join(str(part) for part in argv)
    log("  %-9s %s" % (stage, printable))
    if dry_run:
        return True
    started = time.time()
    completed = subprocess.run([str(part) for part in argv], capture_output=True, text=True)
    entry = {
        "command": printable,
        "exit": completed.returncode,
        "seconds": round(time.time() - started, 1),
        "tail": (completed.stdout or "").strip().splitlines()[-6:],
    }
    if completed.returncode:
        entry["tail"] = ((completed.stderr or "").strip().splitlines() or entry["tail"])[-8:]
    state["sessions"].setdefault(key, {})[stage] = entry
    write_state(state)
    if completed.returncode:
        log("  %-9s FAILED (exit %d)" % (stage, completed.returncode))
        for line in entry["tail"]:
            log("      %s" % line)
        return False
    return True


def write_state(state):
    state["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    path = Path(state["state_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def match_groups():
    groups = {}
    path = PARSED / "summary.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            groups[Path(row.get("file", "")).stem] = (row.get("meta") or {}).get("map") or "unknown"
    return groups


def find_recording(replay):
    """Return an existing recording of THIS replay, or the canonical new path.

    A recording directory is matched by the replay path it stored, not by its
    name: earlier recordings carry suffixes such as -fullrun01, and re-recording a
    battle that is already on disk would waste twenty minutes of the GPU.
    """
    canonical = WORKSPACE / "recordings" / replay.stem
    usable = []
    for directory in sorted((WORKSPACE / "recordings").glob("*")):
        marker = directory / "replay_path.txt"
        if not directory.is_dir() or not marker.exists():
            continue
        if marker.read_text(encoding="utf-8").strip() != str(replay):
            continue
        # Only a FINALISED recording counts: an interrupted attempt leaves an empty
        # or unfinalised container, and treating that as done would silently train on
        # nothing.
        if not (directory / "session.json").exists():
            continue
        video = directory / "capture.mkv"
        usable.append((video.stat().st_size if video.exists() else 0, directory))
    if not usable:
        return canonical
    usable.sort(reverse=True)
    return usable[0][1]


def client_log_for(replay):
    """The client log that actually carries THIS replay's [BattleReplay] play line.

    The two playback routes write to different files: the direct-Proton route (15.8)
    writes ``profile/python.log`` beside the workspace, and a client started by Steam
    (15.9) writes into the game installation. Handing the align stage the wrong one does
    not fail as "wrong log": it fails as ``no [BattleReplay] play line for <session>``,
    which names the replay and not the path. So the log is picked by CONTENT -- the same
    rule the capture host's battle gate follows.
    """
    needle = "[BattleReplay] play, ('%s'" % replay
    candidates = [
        # The direct-Proton route (15.8) writes beside the workspace...
        WORKSPACE / "profile/python.log",
        # ...and a client started by Steam (15.9) writes into the installation.
        Path.home() / ".local/share/Steam/steamapps/common/World of Warships"
        / "profile/python.log",
    ]
    for path in candidates:
        if not path.exists():
            continue
        if needle in path.read_text(encoding="utf-8", errors="replace"):
            return path
    return candidates[0]


def events_aggregate(export_path):
    """The per-build events aggregate the align and projection stages read.

    The aggregate holds every session's replay-side deaths for ONE build, and that is
    why it is keyed by build: a 15.8 file carries no row for a 15.9 replay, so reading
    the wrong one fails as "no death rows for <session>" -- an error that names the
    symptom instead of the cause. The build comes from the session's OWN export row.

    This function only SELECTS the file. It never appends: a shared aggregate that grows
    from two writers is how it drifts, so the project rebuilds and certifies it instead
    (tools/replay/certify_events_rebuild.py). A missing file fails in the stage that
    reads it, naming the path.
    """
    build = ""
    path = Path(export_path)
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("event") == "session":
                    build = str(row.get("build") or "")
                    break
    # The original 15.8.0 lane's aggregate is the bare name; every other build gets its
    # own file, which is also how the earlier 15.9 work wrote it.
    if not build or build.startswith("15.8"):
        return PARSED / "events.jsonl"
    return PARSED / ("events-%s.jsonl" % build)


def map_name_from_export(path):
    """Return a filesystem-safe map name from an exported session row."""
    if not Path(path).exists():
        return None
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") == "session" and row.get("map"):
                return str(row["map"]).replace("spaces/", "").replace("/", "_").strip()
    return None


def record(session, replay, out_dir, state, args):
    if (out_dir / "session.json").exists() and args.force not in ("record", "*"):
        log("  record    already recorded: %s" % out_dir.name)
        return True
    # A recording for this replay may already be running: the queue can be restarted
    # while a session is in flight, and launching a second client would corrupt both.
    in_flight = out_dir / "recorder.pid"
    if in_flight.exists() and Path("/proc/%s" % in_flight.read_text().strip()).exists():
        log("  record    a recording of this replay is already running (%s); waiting for it"
            % in_flight.read_text().strip())
        if args.dry_run:
            return True
    else:
        driver = TOOLS / "capture/start_workspace_replay_recording.sh"
        ok = run([driver, out_dir, replay, "until-exit"], "record", state, session, args.dry_run)
        if not ok or args.dry_run:
            return ok
    recorder = (out_dir / "recorder.pid")
    if recorder.exists():
        pid = recorder.read_text().strip()
        log("  record    waiting for recorder %s to finalise" % pid)
        while Path("/proc/%s" % pid).exists():
            time.sleep(5)
    # A finished replay does NOT close the client: it returns to the harbour and keeps
    # running, which is why the window still looks busy after the battle ends. Leaving
    # it alive would stack the next session's game on top of this one, so the client is
    # stopped here by the pid the supervisor recorded.
    client = out_dir / "wows.pid"
    if client.exists():
        client_pid = client.read_text().strip()
        if Path("/proc/%s" % client_pid).exists():
            log("  record    stopping the client %s left in the harbour" % client_pid)
            subprocess.run(["kill", "-TERM", client_pid])
            for _ in range(60):
                if not Path("/proc/%s" % client_pid).exists():
                    break
                time.sleep(2)
    return (out_dir / "session.json").exists()


def acquire_heavy_lock():
    """Take the shared heavy-job lock, or explain why not and return None.

    Every heavy entry point takes this one lock (tools/replay/heavy_lock.sh does the same
    with flock(1)); auto_learn is a Python entry point, so it takes it here, in-process,
    and holds it for the whole batch. The marker is exported so the per-session recording
    script - which is invoked by this process - does not try to take it a second time and
    refuse itself.
    """
    import fcntl
    import os
    import subprocess
    import time

    if os.environ.get("HEAVY_LOCK_HELD"):
        return "inherited"
    if subprocess.run(["pgrep", "-x", "WorldOfWarships"],
                      capture_output=True).returncode == 0:
        print("refusing: WorldOfWarships is already running; the game and a batch of "
              "recordings never overlap", file=sys.stderr)
        return None
    root = Path(__file__).resolve().parent.parent.parent
    lock_path = root / "state/heavy.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "a+", encoding="utf-8")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.seek(0)
        print("refusing: another heavy job holds the lock:\n%s" % handle.read(),
              file=sys.stderr)
        return None
    os.environ["HEAVY_LOCK_HELD"] = "1"
    handle.seek(0)
    handle.truncate()
    handle.write("label=auto_learn\npid=%d\nsince=%s\n" % (os.getpid(), time.strftime("%F %T")))
    handle.flush()
    return handle


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", help="comma separated session ids")
    parser.add_argument("--replay-dir", default="replays",
                        help="workspace-relative directory the replays come from. The "
                             "Steam-route 15.9 replays live in replays/steam159 under "
                             "their own version tag; putting them in replays/ instead "
                             "would let a 15.8 batch pick them up and drive the wrong "
                             "build's stages over them.")
    parser.add_argument("--limit", type=int, default=0, help="process at most this many replays")
    parser.add_argument("--no-record", action="store_true",
                        help="only process replays that already have a recording")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-merge", action="store_true",
                        help="stop after the per-session stages, before the corpus "
                             "merge (use this when only recording or labelling)")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--hud-calibration", type=Path, default=HUD_CALIBRATION)
    parser.add_argument("--state", type=Path, default=PARSED / "auto-learn-state.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", default="", help="stage name, or * for everything")
    args = parser.parse_args(argv)

    # One heavy job at a time: this batch takes the shared lock before doing anything.
    lock = acquire_heavy_lock()
    if lock is None:
        return 3

    state = {"state_path": str(args.state), "sessions": {}, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if args.state.exists():
        state = json.loads(args.state.read_text(encoding="utf-8"))
        state["state_path"] = str(args.state)
        state.setdefault("sessions", {})
    calibration = json.loads(args.hud_calibration.read_text(encoding="utf-8"))
    panel = calibration["elements"]["minimap"]["rect"]
    panel_width = int(panel[2])
    log("HUD calibration %s  minimap panel %s (%d px wide)"
        % (args.hud_calibration.name, panel, panel_width))
    groups = match_groups()
    replays = sorted((WORKSPACE / args.replay_dir).glob("*.wowsreplay"))
    if args.only:
        wanted = {item.strip() for item in args.only.split(",")}
        replays = [path for path in replays if path.stem in wanted]
    if args.limit:
        replays = replays[:args.limit]
    log("%d replays to consider: %s" % (len(replays), [p.stem for p in replays]))

    ready = []
    for replay in replays:
        session = replay.stem
        out_dir = find_recording(replay)
        log("=== %s (recording %s)" % (session, out_dir.name))
        if not args.no_record:
            if not record(session, replay, out_dir, state, args):
                log("  record    unavailable; skipping this session")
                continue
        elif not (out_dir / "session.json").exists():
            log("  record    no recording and --no-record was given; skipping")
            continue
        if not (out_dir / "session.json").exists():
            log("  record    still no session.json; skipping")
            continue

        # Decide the UI configuration BEFORE running any stage. The panel size is part
        # of the input contract, so a recording made at another size must be skipped
        # rather than re-processed: its projection, zones and crops would all be
        # computed for the wrong rectangle and would look plausible.
        marker = out_dir / "panel_rect.txt"
        recorded_panel = None
        if marker.exists():
            recorded_panel = [int(v) for v in marker.read_text().split(",")]
        else:
            existing = PARSED / ("datasets/minimap-%s.npz" % session)
            if existing.exists():
                recorded_panel = [int(v) for v in np.load(existing, allow_pickle=False)["minimap_rect"]]
        if recorded_panel and recorded_panel != panel:
            log("  ui        EXCLUDED: recorded with minimap panel %s, this run uses %s"
                % (recorded_panel, panel))
            continue
        if not marker.exists() and not args.dry_run:
            marker.write_text(",".join(str(v) for v in panel) + "\n", encoding="utf-8")

        exported = PARSED / ("minimap-%s.jsonl" % session)
        position = PARSED / ("position-%s.jsonl" % session)
        anchors = PARSED / ("alignment/%s.anchors.csv" % session)
        alignment = PARSED / ("alignment/%s.json" % session)
        dataset = PARSED / ("datasets/minimap-%s.npz" % session)
        inventory = PARSED / ("inventory/%s.inventory.jsonl" % session)
        # The configuration is part of the file name so that "the output exists"
        # really does mean "the output this run wants": a grid change would
        # otherwise be skipped silently and the trainer would read old targets.
        heatmap = PARSED / ("datasets/heatmap-%s-g%s-s%s-c%s.npz"
                            % (session, GRID, SIGMA, CLASS_COUNT))

        if not exported.exists() or args.force in ("align", "*"):
            if not run([PYTHON, TOOLS / "replay/export_events.py", replay,
                        "--kinds", "position,track,spotted,zone,visibility,death,squadron,shot,waypoints",
                        "--out", exported], "align", state, session, args.dry_run):
                continue
        if args.dry_run and not exported.exists():
            log("  align     dry run: the map name and the projection path follow the export, "
                "so the rest of the plan is only known at run time")
            continue
        map_name = map_name_from_export(exported) or groups.get(session, "unknown")
        # The affine maps world coordinates to ABSOLUTE minimap pixels, so it depends
        # on the panel size as much as on the map. Both are in the name, otherwise a
        # larger minimap would silently reuse the smaller one's transform.
        projected = PARSED / ("calibration/minimap-%s-%dx%d.json"
                              % (map_name, panel[2], panel[3]))
        # Zone pixels are calibrated per (map, panel size): the panel rescales about
        # its bottom-right corner, so a file made at another size would place A/B/C in
        # the wrong spot. A missing file means the class is reported unavailable.
        zones = ZONE_DIR / ("minimap-zones-%s-%d.json" % (map_name, panel_width))
        log("  align     map %s -> projection %s" % (map_name, projected.name))
        if not alignment.exists() or args.force in ("align", "*"):
            if not run([PYTHON, TOOLS / "replay/anchors_from_pylog.py",
                        "--pylog", client_log_for(replay), "--recording", out_dir,
                        "--events", events_aggregate(exported), "--session", replay.name,
                        "--out", anchors,
                        "--report", str(anchors).replace(".csv", ".report.json")],
                       "align", state, session, args.dry_run):
                continue
            if not run([PYTHON, TOOLS / "replay/align_timeline.py", anchors,
                        "--out", alignment], "align", state, session, args.dry_run):
                continue
        if not (position.exists()) or args.force in ("align", "*"):
            run([PYTHON, TOOLS / "replay/export_events.py", replay, "--kinds", "position",
                 "--out", position], "align", state, session, args.dry_run)

        if not projected.exists() or args.force in ("project", "*"):
            if not run([PYTHON, TOOLS / "replay/calibrate_minimap.py", "fit",
                        "--video", out_dir / "capture.mkv",
                        "--hud-calibration", args.hud_calibration, "--position", position,
                        "--alignment", alignment, "--events", events_aggregate(exported),
                        "--session", replay.name, "--frames", "60", "--settle-s", "30",
                        "--out", projected,
                        "--samples-out", PARSED / ("calibration/minimap-samples-%s.jsonl" % session)],
                       "project", state, session, args.dry_run):
                continue

        if not dataset.exists() or args.force in ("dataset", "*"):
            if not run([PYTHON, TOOLS / "replay/build_minimap_dataset.py",
                        "--video", out_dir / "capture.mkv", "--hud-calibration", args.hud_calibration,
                        "--minimap-calibration", projected, "--position", position,
                        "--alignment", alignment, "--events", events_aggregate(exported),
                        "--session", replay.name, "--step-s", "1.0",
                        "--out", dataset,
                        "--labels-out", str(dataset).replace(".npz", ".labels.jsonl"),
                        "--summary", str(dataset).replace(".npz", ".summary.json")],
                       "dataset", state, session, args.dry_run):
                continue

        zone_arg = ["--zones", zones] if zones.exists() else []
        # The lock marker is client-side, so its label is the agent's own lock action.
        # A session without that log reports the class UNAVAILABLE rather than empty.
        lock_log = PARSED / ("locks-%s.jsonl" % session)
        lock_arg = ["--locks", lock_log] if lock_log.exists() else []
        if not inventory.exists() or args.force in ("inventory", "*"):
            if not run([PYTHON, TOOLS / "replay/build_minimap_inventory.py",
                        "--export", exported, "--alignment", alignment,
                        "--hud-calibration", args.hud_calibration, "--minimap-calibration", projected,
                        *zone_arg, *lock_arg, "--dataset", dataset, "--out", inventory,
                        "--summary", str(inventory).replace(".jsonl", ".summary.json"),
                        "--terrain-out", str(inventory).replace(".jsonl", ".terrain.png"),
                        "--overlay-out", str(inventory).replace(".jsonl", ".overlay-verify.png"),
                        "--check"], "inventory", state, session, args.dry_run):
                continue

        if not heatmap.exists() or args.force in ("heatmap", "*"):
            if not run([PYTHON, TOOLS / "replay/build_heatmap_dataset.py",
                        "--inventory", inventory, "--dataset", dataset, "--out", heatmap,
                        "--summary", str(heatmap).replace(".npz", ".summary.json"),
                        "--grid", GRID, "--sigma", SIGMA],
                       "heatmap", state, session, args.dry_run):
                continue
        # Only sessions recorded with THIS UI configuration can share a corpus: the
        # crop rectangle is part of the input contract, so a session recorded at a
        # different minimap size is a different input space.
        recorded_rect = [int(v) for v in np.load(dataset, allow_pickle=False)["minimap_rect"]]
        if recorded_rect != panel:
            log("  corpus    EXCLUDED: recorded with minimap rect %s, this run uses %s"
                % (recorded_rect, panel))
            continue
        ready.append((session, heatmap, dataset))

    log("sessions ready for training: %s" % [item[0] for item in ready])
    if args.skip_merge:
        # Recording a batch only feeds the per-session stages. The merge rewrites the
        # corpus every earlier run produced and training reads all of it, so both are
        # deliberate steps of their own: running them from a recording batch would
        # redo the heaviest work (and its memory peak) on every session.
        log("--skip-merge given; stopping after the per-session stages")
        return 0
    if len(ready) < 3:
        log("fewer than three sessions carry a heatmap dataset; a session-level "
            "holdout needs more, so training is skipped this run")
        return 0

    pairs = ["%s:%s" % (heat, image) for _session, heat, image in ready]
    merged_heat = PARSED / "datasets/heatmap-merged.npz"
    merged_images = PARSED / "datasets/minimap-merged.npz"
    merge_command = [PYTHON, TOOLS / "replay/merge_heatmap_datasets.py"]
    for pair in pairs:
        merge_command += ["--pair", pair]
    merge_command += ["--summary", PARSED / "summary.jsonl",
                      "--out-heatmaps", merged_heat, "--out-images", merged_images,
                      "--out-summary", PARSED / "datasets/heatmap-merged.summary.json"]
    if not run(merge_command, "merge", state, "corpus", args.dry_run):
        return 1
    if args.skip_train:
        log("--skip-train given; stopping after the merge")
        return 0
    if not run([PYTHON, TOOLS / "replay/train_minimap_heatmap.py",
                "--heatmaps", merged_heat, "--images", merged_images,
                "--model-out", PARSED / "artifacts/minimap-heatmap-corpus.pt",
                "--manifest-out", PARSED / "artifacts/minimap-heatmap-corpus.json",
                "--epochs", str(args.epochs)], "train", state, "corpus", args.dry_run):
        return 1
    return 0 if run([PYTHON, TOOLS / "replay/train_minimap_heatmap.py", "--validate",
                     "--model", PARSED / "artifacts/minimap-heatmap-corpus.pt",
                     "--manifest", PARSED / "artifacts/minimap-heatmap-corpus.json"],
                    "validate", state, "corpus", args.dry_run) else 1


if __name__ == "__main__":
    sys.exit(main())
