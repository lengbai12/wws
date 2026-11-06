#!/usr/bin/env python3
"""Certify a rebuilt events aggregate against the one it replaces.

`replays/parsed/events.jsonl` has no second writer by design: when a new queue needs
sessions it does not cover, the whole file is rebuilt with the same exporter and the
rebuild is admitted only by proving that every session the old file already had is
unchanged, counter by counter. "The new file exists" is not a certificate; this is.

    ./.venv/bin/python tools/replay/certify_events_rebuild.py \
        replays/parsed/events.jsonl replays/parsed/events.jsonl.new

Exits non-zero on a missing session, a changed counter, or a session whose rows are
fewer than before. Sessions only present in the new file are reported as added.
"""

import argparse
import collections
import json
import sys
from pathlib import Path


def blocks(path):
    """Group records by the session row they follow, and count kinds per session."""
    per_session = collections.Counter()
    counts = collections.defaultdict(collections.Counter)
    order = []
    current = None
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("event") == "session":
                current = record.get("replay")
                if current not in counts:
                    order.append(current)
                counts[current]["session"] += 1
                continue
            if current is None:
                raise SystemExit("%s: a %s row appears before any session row"
                                 % (path, record.get("event")))
            per_session[current] += 1
            counts[current][record.get("event", "?")] += 1
    return order, counts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("old")
    ap.add_argument("new")
    args = ap.parse_args()

    old_order, old = blocks(args.old)
    new_order, new = blocks(args.new)
    print("old %d sessions, new %d sessions" % (len(old_order), len(new_order)))

    problems = []
    for session in old_order:
        if session not in new:
            problems.append("%s: missing from the rebuild" % session)
            continue
        if new[session] != old[session]:
            keys = sorted(set(old[session]) | set(new[session]))
            detail = ", ".join("%s %d->%d" % (k, old[session][k], new[session][k])
                               for k in keys if old[session][k] != new[session][k])
            problems.append("%s: counters changed (%s)" % (session, detail))

    added = [session for session in new_order if session not in old]
    print("unchanged: %d" % (len(old_order) - len(problems)))
    print("added: %d%s" % (len(added), ("  " + ", ".join(added)) if added else ""))

    if problems:
        print("\nVERDICT FAIL")
        for problem in problems:
            print("  " + problem)
        return 1
    print("\nVERDICT pass: every pre-existing session is byte-for-byte the same counts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
