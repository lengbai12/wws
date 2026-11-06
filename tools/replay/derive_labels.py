"""Derive pixel-domain labels from replay truth plus a HUD calibration.

A label here is the value a HUD element MUST show at a video time, computed from
the replay, not a human annotation. Every row says which replay fact produced it
and which calibrated rectangle it belongs to, so a row can be recomputed and
audited long after the recording is gone.

Occupancy is stated per row, because the three cases are not interchangeable:

  present        the element is on screen and the value is known
  no_information the replay carries no value yet for this time
  absent         the replay proves the element cannot be on screen, such as the
                 own ship's health bar after the own ship died

A row with ``no_information`` or ``absent`` has ``value: null``. Those rows are
still written, so a training stage can see that the case exists instead of
treating a missing sample as a zero.

Usage:
  derive_labels.py --events replays/parsed/events.jsonl \
      --session 236066_random_Kurama_XI.wowsreplay \
      --alignment replays/parsed/alignment/<recording>.json \
      --calibration replays/parsed/calibration/hud-2560x1600.json \
      --recording recordings/<session> --out replays/parsed/labels/<session>.own_hp.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PRESENT = "present"
NO_INFORMATION = "no_information"
ABSENT = "absent"

EVIDENCE = "replay_health_event"


def read_events(path):
    """Yield (replay, row) for every row, including each leading session row.

    The session row carries the recording player's name, and the own vehicle is
    only reachable through that name, so the row is handed on rather than dropped.
    """
    current = None
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") == "session":
                current = row.get("replay")
            if current is None:
                continue
            yield current, row


def own_vehicle(events_path, session):
    """Return (player_name, vehicle_id, max_health) for the recording player.

    The own ship is found through the roster, because the export states the
    recording player by name and the entity ids by vehicle.

    On a build whose roster cannot LINK the recording player's row to a vehicle the id
    comes from the session row instead: measured on the 15.9 Operations sessions, the
    own roster row carries `max_health` (45500) with `vehicle_id` null, while the session
    row carries `own_vehicle_id` (310572, read from the own-position packet). Without the
    fallback the whole dataset stage died with a bare
    `TypeError: int() argument must be ... not 'NoneType'`, which names neither the field
    nor the build.
    """
    player = None
    own_id = None
    roster = []
    for replay, row in read_events(events_path):
        if replay != session:
            continue
        if row.get("event") == "session":
            player = row.get("recording_player") or player
            own_id = row.get("own_vehicle_id") or own_id
        elif row.get("event") == "roster":
            roster.append(row)
    if player is None:
        raise SystemExit("no session row for %s in %s" % (session, events_path))
    for row in roster:
        if row.get("name") == player:
            vehicle = row.get("vehicle_id")
            if vehicle is None:
                vehicle = own_id
            if vehicle is None:
                raise SystemExit(
                    "the roster row for %s carries no vehicle id and the session row "
                    "carries no own_vehicle_id in %s" % (player, events_path))
            return player, int(vehicle), float(row["max_health"])
    raise SystemExit("roster holds no row for %s in %s" % (player, session))


def health_series(events_path, session, vehicle_id):
    """Return the folded health steps of one vehicle, ordered by clock.

    Packet repetition makes exact duplicate rows normal, so equal consecutive
    steps collapse; the value is unchanged and the step list stays readable.
    """
    steps = []
    for replay, row in read_events(events_path):
        if replay != session or row.get("event") != "health":
            continue
        if int(row["vehicle"]) != vehicle_id:
            continue
        step = (float(row["clock"]), float(row["value"]))
        if steps and steps[-1] == step:
            continue
        steps.append(step)
    steps.sort(key=lambda item: item[0])
    return steps


def death_clock(events_path, session, vehicle_id):
    """Return the clock of this vehicle's death, or None when it survived."""
    for replay, row in read_events(events_path):
        if replay != session or row.get("event") != "death":
            continue
        if int(row["victim"]) == vehicle_id:
            return float(row["clock"])
    return None


def hp_at(steps, clock):
    """Return the health in force at a clock, or None before the first step.

    Health arrives as steps, so the value in force is the latest step at or
    before the clock. Nothing is interpolated: a value between two steps is not
    a value the replay ever reported.
    """
    value = None
    for step_clock, step_value in steps:
        if step_clock > clock:
            break
        value = step_value
    return value


def load_alignment(path):
    """Return the clock-to-video mapping of an alignment record."""
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    mapping = record.get("replay_to_video")
    if not isinstance(mapping, dict):
        raise SystemExit("%s holds no replay_to_video mapping" % path)
    return float(mapping["rate"]), float(mapping["offset_s"]), record


def video_time(rate, offset_s, clock):
    """Map a replay clock to a container video time."""
    return rate * clock + offset_s


def build_rows(session, steps, max_hp, dead_at, rate, offset_s, rect,
               start_s, end_s, step_s, element="own_hp_bar", recording=None,
               resolution=None, settle_s=0.0, hud_start_s=None):
    """Return one label row per sampled video time.

    The sample grid is stated in video time because that is what a frame loader
    asks for, and each row keeps the replay clock it was derived from.

    ``settle_s`` marks a row unsettled when a health step lands within that many
    seconds after it. The replay value is still correct there, but the rendered
    bar leads the packet that carries the step, so comparing the two within that
    window measures the renderer, not the label.

    ``hud_start_s`` is the first video time at which the calibrated element was
    MEASURED to be on screen. Earlier rows hold a replay value the screen does
    not show yet, so they carry no usable target.
    """
    rows = []
    index = 0
    video = start_s
    step_clocks = [step_clock for step_clock, _ in steps]
    while video <= end_s + 1e-9:
        clock = (video - offset_s) / rate
        caveats = []
        settled = True
        if hud_start_s is not None and video < hud_start_s:
            value, occupancy = None, NO_INFORMATION
            caveats.append("hud_not_on_screen_at_this_time")
        elif clock < 0:
            value, occupancy = None, NO_INFORMATION
            caveats.append("replay_clock_negative_before_battle_start")
        elif dead_at is not None and clock > dead_at:
            value, occupancy = None, ABSENT
            caveats.append("own_ship_dead_hud_bar_absent")
        else:
            hp = hp_at(steps, clock)
            if hp is None:
                value, occupancy = None, NO_INFORMATION
                caveats.append("no_health_step_at_or_before_this_clock")
            else:
                value = {"hp": hp, "max_hp": max_hp, "fraction": hp / max_hp}
                occupancy = PRESENT
                if settle_s > 0 and any(clock <= step_clock <= clock + settle_s
                                        for step_clock in step_clocks):
                    settled = False
                    caveats.append("within_health_step_window")
        row = {
            "sample_id": "%s:own_hp:%05d" % (Path(session).stem, index),
            "session_id": Path(session).stem,
            "kind": "own_hp",
            "element": element,
            "rect": list(rect),
            "replay_clock_s": round(clock, 6),
            "video_time_s": round(video, 6),
            "value": value,
            "occupancy": occupancy,
            "settled": settled,
            "evidence": EVIDENCE,
            "caveats": caveats,
        }
        if recording:
            row["recording"] = str(recording)
        if resolution:
            row["resolution"] = list(resolution)
        rows.append(row)
        index += 1
        video += step_s
    return rows


def find_first_readable(sample_at, lo, hi, iterations=14):
    """Return the earliest video time whose element is on screen.

    ``sample_at(t)`` returns the measured fraction, or None when the element
    cannot be read. The readable set is taken to be monotone in time, which holds
    for a HUD that is absent during the opening sequence and present afterwards.
    Bisection costs a handful of frame reads; scanning every sample would cost
    thousands. Returns None when neither end is readable, because then no bound
    can be stated.
    """
    if sample_at(lo) is not None:
        return lo
    if sample_at(hi) is None:
        return None
    for _ in range(iterations):
        mid = (lo + hi) / 2.0
        if sample_at(mid) is None:
            lo = mid
        else:
            hi = mid
    return hi


def summarise(rows):
    """Return counts per occupancy plus the value range, for a quick check."""
    counts = {PRESENT: 0, NO_INFORMATION: 0, ABSENT: 0}
    fractions = []
    for row in rows:
        counts[row["occupancy"]] = counts.get(row["occupancy"], 0) + 1
        if row["value"]:
            fractions.append(row["value"]["fraction"])
    unsettled = sum(1 for row in rows if row.get("settled") is False)
    summary = {
        "rows": len(rows),
        "occupancy": counts,
        "present_rows": counts[PRESENT],
        "settled_rows": len(rows) - unsettled,
        "unsettled_rows": unsettled,
        "fraction_min": min(fractions) if fractions else None,
        "fraction_max": max(fractions) if fractions else None,
    }
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--session", required=True,
                        help="replay file name as it appears in the event stream")
    parser.add_argument("--alignment", type=Path, required=True,
                        help="alignment record for THIS recording")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--step-s", type=float, default=0.5,
                        help="spacing of the sample grid in video seconds")
    parser.add_argument("--recording", type=Path,
                        help="recordings/<session> directory, recorded in each row")
    parser.add_argument("--element", default="own_hp_bar")
    parser.add_argument("--settle-s", type=float, default=0.3,
                        help="mark a row unsettled when a health step lands within "
                             "this many seconds after it; the rendered bar leads the "
                             "packet, so the comparison is ambiguous inside the window")
    parser.add_argument("--probe-video", type=Path,
                        help="measured recordings/<session>/capture.mkv; rows before the "
                             "first frame where the element is on screen carry no target")
    parser.add_argument("--summary", type=Path, help="optional JSON summary path")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import hud_calibration as hud

    calibration = hud.load_calibration(args.calibration)
    rect = hud.element_rect(calibration, args.element)
    if rect is None:
        raise SystemExit("element %r is not calibrated in %s"
                         % (args.element, args.calibration))
    resolution = calibration.get("resolution")

    player, vehicle_id, max_hp = own_vehicle(args.events, args.session)
    steps = health_series(args.events, args.session, vehicle_id)
    if not steps:
        raise SystemExit("no health steps for %s (vehicle %d)" % (player, vehicle_id))
    dead_at = death_clock(args.events, args.session, vehicle_id)
    rate, offset_s, alignment = load_alignment(args.alignment)

    start_s = video_time(rate, offset_s, steps[0][0])
    span_end_clock = dead_at if dead_at is not None else steps[-1][0]
    end_s = video_time(rate, offset_s, span_end_clock)

    hud_start_s = None
    if args.probe_video:
        import calibrate_hud
        direction = element.get("fill_direction") or "left_to_right"

        def sample_at(second):
            rgb, width, height = calibrate_hud.read_frame(args.probe_video, second)
            return hud.measure_fill_fraction(rgb, width, height, rect, direction)

        hud_start_s = find_first_readable(sample_at, start_s, end_s)
        if hud_start_s is None:
            print("WARNING the element could not be read at either end of the span; "
                  "no on-screen bound is stated")
        else:
            print("element first readable at video %.3f s (bisected from the pixels)"
                  % hud_start_s)

    rows = build_rows(args.session, steps, max_hp, dead_at, rate, offset_s, rect,
                      start_s, end_s, args.step_s, element=args.element,
                      recording=args.recording, resolution=resolution,
                      settle_s=args.settle_s, hud_start_s=hud_start_s)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = summarise(rows)
    summary.update({
        "session": args.session,
        "player": player,
        "vehicle_id": vehicle_id,
        "max_hp": max_hp,
        "health_steps": len(steps),
        "own_death_clock": dead_at,
        "alignment_rate": rate,
        "alignment_offset_s": offset_s,
        "sample_step_s": args.step_s,
        "settle_s": args.settle_s,
        "hud_first_readable_video_s": hud_start_s,
        "video_span_s": [round(start_s, 3), round(end_s, 3)],
        "calibration": str(args.calibration),
        "element_rect": rect,
    })
    print("session %s  player %s  vehicle %d  max_hp %.0f  health steps %d  own death %s"
          % (args.session, player, vehicle_id, max_hp, len(steps), dead_at))
    print("video span %.3f..%.3f s  step %.3f s" % (start_s, end_s, args.step_s))
    print("rows %d  present %d  no_information %d  absent %d  settled %d  unsettled %d"
          % (summary["rows"], summary["occupancy"][PRESENT],
             summary["occupancy"][NO_INFORMATION], summary["occupancy"][ABSENT],
             summary["settled_rows"], summary["unsettled_rows"]))
    print("fraction range %s..%s"
          % (summary["fraction_min"], summary["fraction_max"]))
    print("wrote %s" % args.out)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n",
                                encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
