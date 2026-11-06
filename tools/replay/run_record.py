#!/usr/bin/env python3
"""One run's record: who started it, whether it succeeded, and what it wrote.

Why this exists: failure used to be able to masquerade as success. A training script with
no exit-code checks still printed "=== done" and returned 0; the collector treated "the
process is gone" as "it finished" and deleted the per-session datasets either way; and
nothing recorded which artifacts a run actually produced, so a stale model could be
validated as if it were new. This is the record that makes those three impossible:

  * a run has an id and a status, and success is only ever written by COMMIT;
  * COMMIT verifies every artifact exists and hashes to what the run recorded;
  * nothing else is allowed to claim the run finished.

    run_record.py start <name> [--artifact PATH ...]   -> prints the run id
    run_record.py note  <run_id> <key=value> ...
    run_record.py fail  <run_id> [--reason TEXT]
    run_record.py commit <run_id> [--artifact PATH ...]
    run_record.py status <run_id>                      -> prints the status, exit 1 unless ok
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from fsutil import sha256


RUNS = Path("state/runs")
TERMINAL = ("ok", "failed")


def path_of(run_id):
    return RUNS / ("%s.json" % run_id)


def load(run_id):
    path = path_of(run_id)
    if not path.exists():
        raise SystemExit("run_record: no such run %s" % run_id)
    return json.loads(path.read_text(encoding="utf-8"))


def save(record):
    RUNS.mkdir(parents=True, exist_ok=True)
    path = path_of(record["run_id"])
    record["updated"] = time.strftime("%F %T")
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def fingerprint(paths):
    out = {}
    for path in paths:
        item = Path(path)
        out[str(item)] = sha256(item) if item.is_file() else None
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("start", "note", "fail", "commit", "status"))
    ap.add_argument("run_id", nargs="?", help="the run id, or the NAME when starting")
    ap.add_argument("--name", default=None,
                    help="override the run's name; by default the positional argument of "
                         "`start` is the name, which is how the scripts call it")
    ap.add_argument("--artifact", action="append", default=[])
    ap.add_argument("--reason", default="")
    ap.add_argument("--note", action="append", default=[])
    args = ap.parse_args(argv)

    RUNS.mkdir(parents=True, exist_ok=True)

    if args.action == "start":
        name = args.name or args.run_id or "run"
        run_id = "%s-%s-%d" % (name, time.strftime("%Y%m%dT%H%M%S"), os.getpid())
        record = {"run_id": run_id, "name": name, "status": "running",
                  "pid": os.getpid(), "started": time.strftime("%F %T"),
                  "artifacts_before": fingerprint(args.artifact), "notes": {}}
        save(record)
        print(run_id)
        return 0

    record = load(args.run_id)
    if args.action == "status":
        print(record["status"])
        return 0 if record["status"] == "ok" else 1
    if record["status"] in TERMINAL:
        raise SystemExit("run_record: run %s is already %s"
                         % (record["run_id"], record["status"]))
    if args.action == "note":
        for item in args.note:
            key, _, value = item.partition("=")
            record["notes"][key] = value
        save(record)
        return 0
    if args.action == "fail":
        record["status"] = "failed"
        record["reason"] = args.reason or "unspecified"
        save(record)
        print("run %s FAILED: %s" % (record["run_id"], record["reason"]), file=sys.stderr)
        return 1

    # commit: the only path to "ok", and it refuses unless every artifact is there and
    # every artifact that already existed has changed (a run that wrote nothing new is
    # not a run that succeeded).
    wanted = list(args.artifact) or list(record.get("artifacts_expected") or [])
    if not wanted:
        record["status"] = "failed"
        record["reason"] = "commit called with no artifacts, so success cannot be claimed"
        save(record)
        print("run %s FAILED: %s" % (record["run_id"], record["reason"]), file=sys.stderr)
        return 1
    missing = [path for path in wanted if not Path(path).is_file()]
    if missing:
        record["status"] = "failed"
        record["reason"] = "artifacts missing at commit: %s" % ", ".join(missing)
        save(record)
        print("run %s FAILED: %s" % (record["run_id"], record["reason"]), file=sys.stderr)
        return 1
    after = fingerprint(wanted)
    before = record.get("artifacts_before") or {}
    # An artifact that existed before with exactly this hash was NOT written by this run:
    # that is the "a stale model gets validated as if it were new" hazard. It is always a
    # failure, not a flag, because there is no honest reason to commit an unchanged file.
    unchanged = [path for path, digest in after.items() if before.get(path) == digest]
    if unchanged:
        record["status"] = "failed"
        record["reason"] = ("artifacts unchanged since the run started (stale, not "
                            "produced by this run): %s" % ", ".join(unchanged))
        save(record)
        print("run %s FAILED: %s" % (record["run_id"], record["reason"]), file=sys.stderr)
        return 1
    record["artifacts"] = after
    record["status"] = "ok"
    record["finished"] = time.strftime("%F %T")
    save(record)
    print("run %s committed: %s" % (record["run_id"], ", ".join(wanted)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
