"""Run a command and report its peak memory, split into anonymous and file pages.

The machine has no swap, so every heavy stage has to be measured rather than
guessed. ``VmHWM`` alone is misleading here for two reasons: the trainer
memory-maps the dataset twins, and a faulted-in file page counts as RSS even
though the kernel can drop it under pressure; and the pipeline runs its stages as
subprocesses, whose memory the parent's own counters never see. ``RssAnon`` is the
part that cannot be reclaimed -- that is what decides whether an OOM is possible --
and ``RssFile`` is the reclaimable working set. Both are summed over the whole
process tree and sampled from /proc while the job runs.

  ./.venv/bin/python tools/replay/peak_rss.py <command> [args...]
  ./.venv/bin/python tools/replay/peak_rss.py --trace FILE -- <command> [args...]

With --trace the samples are also written as ``elapsed_s anon_gb file_gb``, so a
long run can be attributed to its phases afterwards (a merge, forty epochs and a
validation pass all report one peak otherwise).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

FIELDS = ("VmHWM", "RssAnon", "RssFile")
SAMPLE_S = 0.1


def parent_of(pid):
    """Return {pid: ppid} for every process, tolerating comms with spaces."""
    parents = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open("/proc/%s/stat" % entry, encoding="utf-8") as handle:
                data = handle.read()
            parents[int(entry)] = int(data[data.rindex(")") + 2:].split()[1])
        except (OSError, ValueError, IndexError):
            continue
    return parents


def tree(root):
    parents = parent_of(root)
    pids, changed = {root}, True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in pids and pid not in pids:
                pids.add(pid)
                changed = True
    return pids


def sample(root):
    totals = dict.fromkeys(FIELDS, 0)
    for pid in tree(root):
        try:
            with open("/proc/%d/status" % pid, encoding="utf-8") as handle:
                for line in handle:
                    key = line.split(":")[0]
                    if key in FIELDS:
                        totals[key] += int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            continue
    return totals


def main(argv):
    trace = None
    if argv and argv[0] == "--trace":
        trace, argv = argv[1], argv[2:]
        if argv and argv[0] == "--":
            argv = argv[1:]
    child = subprocess.Popen(argv)
    started = time.time()
    peak = dict.fromkeys(FIELDS, 0)
    handle = open(trace, "w", encoding="utf-8") if trace else None
    try:
        while child.poll() is None:
            for key, value in sample(child.pid).items():
                peak[key] = max(peak[key], value)
            if handle:
                handle.write("%.1f %.3f %.3f\n" % (time.time() - started,
                                                  peak["RssAnon"] / 2 ** 30,
                                                  peak["RssFile"] / 2 ** 30))
                handle.flush()
            time.sleep(SAMPLE_S)
    finally:
        if handle:
            handle.close()
    print("[peak-rss] exit %d  tree peak anon %.2f GB  tree peak file %.2f GB  "
          "tree peak RSS %.2f GB"
          % (child.returncode, peak["RssAnon"] / 2 ** 30, peak["RssFile"] / 2 ** 30,
             peak["VmHWM"] / 2 ** 30), flush=True)
    return child.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
