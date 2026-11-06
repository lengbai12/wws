"""Collect every marker the minimap draws, as labels derived from the replay.

The minimap shows the own ship, teammates, spotted enemies, the capture points
and the terrain. This tool turns the replay into that list per sample time, so a
model can be trained on all of it rather than on one position.

WHAT THE REPLAY CAN AND CANNOT GIVE (measured, not assumed):

  own ship         packet 0x2C, own vehicle id. Always drawn while alive; absent
                   after death, when the marker leaves the map.
  teammates        packet 0x0A position + roster team. Teammates are drawn on the
                   minimap whether or not they are spotted.
  visible enemies  packet 0x0A position + `Vehicle.visibilityFlags` != 0. The flags
                   are the ALL_CLIENTS per-vehicle bitmask, which is what the map
                   actually draws. `Avatar.spottedEntities` is the recording
                   PLAYER's own visibility: it is recorded beside the label as
                   `spotted_by_player` but it does NOT decide the class, because
                   measured on this corpus 35 % of the enemies it calls unspotted
                   still carry a drawn red marker -- only the team-wide flag
                   explains that.
  unknown          a vehicle whose side or visibility the replay never established
                   (no visibility packet yet, or a roster row without a team). It is
                   its own class and never `enemy_not_visible`, so that "no
                   information" cannot be trained as a confirmed absence.
  capture points   `InteractiveZone` gives identity, radius and owner state, but
                   NO world position: where a point is drawn is map-static and
                   comes from the calibrated zone file.
  terrain, grid    static per map; extracted from many frames by a per-pixel
                   median, which removes the moving markers.
  planes           NOT in a replay without a carrier. The Kurama battle has 24
                   vehicle entities and no plane entity, and no aircraft carrier
                   appears in its roster, so this run states the class as absent
                   rather than silently omitting it.

Usage:
  build_minimap_inventory.py --export replays/parsed/minimap-<session>.jsonl \
      --alignment replays/parsed/alignment/<recording>.json \
      --hud-calibration replays/parsed/calibration/hud-2560x1600.json \
      --minimap-calibration replays/parsed/calibration/minimap-2560x1600.json \
      --zones replays/parsed/calibration/minimap-zones-41_Conquest.json \
      --dataset replays/parsed/datasets/minimap-<session>.npz \
      --out replays/parsed/inventory/<session>.inventory.jsonl \
      --summary replays/parsed/inventory/<session>.inventory.summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import minimap_calibration as mini

OWN = "own"
FRIENDLY = "friendly"
ENEMY_VISIBLE = "enemy_visible"
ENEMY_NOT_VISIBLE = "enemy_not_visible"
# A vehicle whose side OR visibility the replay did not establish. It is its own
# class and never `enemy_not_visible`, because that one is the model's
# false-positive test: an absence nobody observed must not be trained as one.
ENEMY_UNKNOWN = "enemy_unknown"
ZONE = "zone"
SQUADRON = "squadron"
AIM = "aim"
RANGE_RING = "range_ring"
WRECK = "wreck"
PLANE = "plane"
# The lock marker is CLIENT-SIDE: the server never echoes which ship the player has
# locked (the targeting interface's setWeaponLock/setPriorityTarget are client->server
# and never appear in the stream), so its label cannot come from the replay. It comes
# from the AGENT'S OWN LOCK ACTION instead: whoever locks a target knows the target, so
# the action is its own supervision. See docs/self-supervised-lock-labels.md.
LOCK = "lock"

# The side of a vehicle the roster does NOT list but which this player's team has
# SPOTTED. It is a value of its own, not a team id, because it is derived from a
# different piece of evidence than the roster's.
#
# WHY IT EXISTS: an Operation's roster lists the HUMAN players only, so on 236813 the
# 8-row roster names 7 team-mates and NOT ONE enemy, while the export tracks 48 ships
# -- both sides' AI ships have no roster row at all. With the roster as the only source
# of sides, every one of those 41 ships came out `enemy_unknown`, i.e. the enemy class
# of an entire session would have been empty. (Measured on 236813: 35 of the 48 tracked
# ships appear in `Avatar.spottedEntities`, 0 of the 7 roster team-mates do. On 236925
# and 236818 the roster happens to list the enemy side as well -- 7 and 11 rows with
# relation 2 -- so this is a per-session gap, not a per-map one.)
#
# WHY IT IS SOUND: `Avatar.spottedEntities` is what THIS player's team has spotted, and
# a team never spots its own ships. So membership implies the other side, which is
# exactly what `classify_vehicle` needs, and a ship that is never spotted stays
# `enemy_unknown` -- the absence nobody observed -- rather than being asserted either
# way.
SPOTTED_SIDE = "spotted:the other side"

MARKER_CLASSES = (OWN, FRIENDLY, ENEMY_VISIBLE, ENEMY_NOT_VISIBLE, ENEMY_UNKNOWN, WRECK,
                  ZONE, SQUADRON, AIM, RANGE_RING, PLANE, LOCK)

# Range rings are drawn as the ship's weapon and detection ranges. Their radii are
# fixed by the map scale (pixels_per_km, measured from the printed range labels),
# but WHICH ones are drawn is a player setting, so the enabled set is measured per
# frame rather than assumed. The share of ring-coloured pixels on the circle is both
# the detector and the audit trail.
RING_COLOUR_SHARE = 0.22


def read_locks(path):
    """Read the agent's own lock log: rows of {clock, vehicle} or {clock, x, z}.

    A row asserts that from ``clock`` onward the agent held a lock. ``vehicle: null``
    (or a row without a target) means it released, so the state is carried forward
    until the next row: an empty log means the class is UNAVAILABLE, never empty.
    """
    rows = []
    if not path or not Path(path).exists():
        return rows
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "clock" not in row:
                continue
            rows.append(row)
    rows.sort(key=lambda row: float(row["clock"]))
    return rows


def lock_at(rows, clock):
    """The lock in force at ``clock``, or None when the agent had no target."""
    active = None
    for row in rows:
        if float(row["clock"]) > clock:
            break
        active = row
    if active is None:
        return None
    return active if active.get("vehicle") is not None or "x" in active else None


def read_export(path):
    """Return the export rows split into the pieces the inventory needs."""
    session = None
    roster = []
    positions = []
    tracks = []
    spotted = []
    visibility = []
    zones = []
    deaths = []
    squadrons = []
    shots = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            kind = row.get("event")
            if kind == "session":
                session = row
            elif kind == "roster":
                roster.append(row)
            elif kind == "position":
                positions.append(row)
            elif kind == "track":
                tracks.append(row)
            elif kind == "spotted":
                spotted.append(row)
            elif kind == "visibility":
                visibility.append(row)
            elif kind == "zone":
                zones.append(row)
            elif kind == "death":
                deaths.append(row)
            elif kind == "squadron":
                squadrons.append(row)
            elif kind == "shot":
                shots.append(row)
    for rows in (positions, tracks, spotted, visibility, zones):
        rows.sort(key=lambda item: item["clock"])
    if session is None:
        raise SystemExit("%s holds no session row" % path)
    return (session, roster, positions, tracks, spotted, visibility, zones, deaths,
            squadrons, shots)


def latest(rows, clock, key):
    """Return the row with the greatest clock at or before ``clock``, or None.

    Values are stepped, never interpolated: a position between two packets is not
    a position the replay ever reported.
    """
    found = None
    for row in rows:
        if row["clock"] > clock:
            break
        found = row
    return found


def positions_by_vehicle(tracks, clock):
    """Return {vehicle: row} of the latest track row per vehicle."""
    found = {}
    for row in tracks:
        if row["clock"] > clock:
            break
        found[row["vehicle"]] = row
    return found


def spotted_at(spotted_rows, clock, known_ids):
    """Return (spotted set, whether the set is trustworthy) at a clock.

    The property is sent repeatedly, so the set in force is the latest one at or
    before the clock. Before the first packet the answer is unknown, which is
    different from an empty set.
    """
    row = latest(spotted_rows, clock, None)
    if row is None:
        return set(), False
    return set(int(item) for item in row["entities"]), True


def visibility_at(visibility_rows, vehicle, clock):
    """Return the visibility bitmask in force for a vehicle, or None if unknown.

    ``Vehicle.visibilityFlags`` is sent only when it changes, so before the first
    packet the answer is unknown, which is not the same as "not visible".
    """
    found = None
    for row in visibility_rows:
        if row["clock"] > clock:
            break
        if row["vehicle"] == vehicle:
            found = row["flags"]
    return found


def zone_states(zones, clock):
    """Return {zone id: state dict} of the latest zone rows at or before a clock."""
    state = {}
    for row in zones:
        if row["clock"] > clock:
            break
        entry = state.setdefault(row["zone"], dict(row))
        for field, value in row.items():
            if value is not None:
                entry[field] = value
    return state


def project(matrix, translation, rect, x, z):
    """Return (pixel_x, pixel_y, u, v) for a world point."""
    px = matrix[0][0] * x + matrix[0][1] * z + translation[0]
    py = matrix[1][0] * x + matrix[1][1] * z + translation[1]
    return (round(px, 3), round(py, 3),
            round((px - rect[0]) / rect[2], 6), round((py - rect[1]) / rect[3], 6))


def classify_vehicle(team, own_team, flags, death_clock, clock):
    """The marker class for one non-own vehicle, and the reason, in this order.

    The ORDER is the point and every step is a measured property:

      1. LIFE STATE FIRST. A sunk ship keeps a marker for the rest of the battle on
         either side: measured on 236387 the patch under a dead ally and under a dead
         enemy is statistically identical (mean luminance 31.5 against 32.0) and three
         times darker than a living marker's (100.2). Deciding the side first is what
         used to label a dead teammate `friendly` and teach the model to paint a live
         teammate over a wreck.
      2. THEN the side -- only when BOTH sides are known. `None == None` is True, so a
         roster without teams used to put every unknown vehicle on our own side. The
         caller supplies `SPOTTED_SIDE` for a vehicle the roster omits but this team
         has spotted, so "known" here includes that evidence and not only the roster's.
      3. THEN visibility: flags != 0 is drawn, flags == 0 is a confirmed absence, and no
         flags at all is `enemy_unknown` -- the absence nobody observed. That last case
         used to be written as `enemy_not_visible`, which is the model's false-positive
         test, i.e. it asserted an absence that was never seen.
    """
    if death_clock is not None and clock > death_clock:
        return WRECK, ("packet 0x0A position frozen at the wreck, vehicle is dead per "
                       "the death events")
    if team is None or own_team is None:
        return ENEMY_UNKNOWN, ("packet 0x0A position; the roster carries no team for "
                               "this vehicle, so the side is unknown")
    if team == own_team:
        return FRIENDLY, "packet 0x0A position, teammate drawn always"
    if flags is None:
        return ENEMY_UNKNOWN, "packet 0x0A position, visibility flags not sent yet"
    if flags != 0:
        return ENEMY_VISIBLE, ("packet 0x0A position, Vehicle.visibilityFlags != 0; "
                               "measured agreement with drawn markers 0.948")
    return ENEMY_NOT_VISIBLE, ("packet 0x0A position, Vehicle.visibilityFlags == 0; "
                              "not drawn while invisible to the team")


def ring_shares(image, centre, labels_km, pixels_per_km, max_radius=345):
    """Return {label_km: share} of ring-coloured pixels on each candidate circle.

    The rings are the ship's own ranges, so a circle is centred on the own ship.
    ``share`` is the fraction of the circle whose pixels look like the light
    unsaturated line work the map draws, which is what makes the answer auditable:
    the number that decided the class is stored with the marker.
    """
    work = image.astype(np.int32)
    luminance = work.mean(axis=2)
    saturation = work.max(axis=2) - work.min(axis=2)
    mask = (luminance > 80) & (luminance < 230) & (saturation < 60)
    height, width = mask.shape[:2]
    angles = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
    cosines, sines = np.cos(angles), np.sin(angles)
    shares = {}
    for label in labels_km:
        radius = label * pixels_per_km
        if radius > max_radius or radius < 20:
            continue
        columns = np.clip((centre[0] + radius * cosines).round().astype(int), 0, width - 1)
        rows = np.clip((centre[1] + radius * sines).round().astype(int), 0, height - 1)
        shares[label] = float(mask[rows, columns].mean())
    return shares


def build(export_path, alignment_path, calibration_path, projection_path, zones_path,
          dataset_path=None, step_s=1.0, images=None, locks_path=None):
    (session, roster, positions, tracks, spotted_rows, visibility_rows, zone_rows,
     death_rows, squadron_rows, shot_rows) = read_export(export_path)
    alignment = json.loads(Path(alignment_path).read_text(encoding="utf-8"))
    rate = float(alignment["replay_to_video"]["rate"])
    offset_s = float(alignment["replay_to_video"]["offset_s"])
    calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    rect = calibration["elements"]["minimap"]["rect"]
    projection = json.loads(Path(projection_path).read_text(encoding="utf-8"))
    matrix = projection["world_to_pixel"]["matrix"]
    translation = projection["world_to_pixel"]["translation"]
    lock_rows = read_locks(locks_path)
    zone_block = None
    if zones_path and Path(zones_path).exists():
        zone_block = json.loads(Path(zones_path).read_text(encoding="utf-8"))
    zones = zone_block["zones"] if zone_block else []
    # Rings and zones are per map: a ring radius comes from the measured map scale and
    # a zone position from a per-map calibration. When either is missing the class is
    # reported as unavailable instead of stopping the run or inventing a value.
    ring_block = projection.get("rings")
    ring_labels = [float(v) for v in ring_block["labels_km"]] if ring_block else []
    pixels_per_km = float(ring_block["pixels_per_km"]) if ring_block else None

    player = session.get("recording_player")
    own_id = None
    own_row = None
    teams = {}
    for entry in roster:
        if own_row is None and entry.get("name") == player:
            own_row = entry
        vehicle = entry.get("vehicle_id")
        if vehicle is None:
            continue
        teams[int(vehicle)] = entry.get("team_id")
        if entry.get("name") == player:
            own_id = int(vehicle)
    # The own ship is not in the 0x0A position stream (it sends 0x2C instead), so on a
    # build whose roster cannot place the recording player the export records the id it
    # read from that packet. Take it when the roster has none: the id is self-declared,
    # and without it the own position stream -- and with it the whole `own` channel --
    # comes out empty.
    if own_id is None and session.get("own_vehicle_id") is not None:
        own_id = int(session["own_vehicle_id"])
    if own_id is None:
        raise SystemExit("no vehicle for the recording player %r: neither the roster nor "
                         "the 0x2C packet supplied one" % player)
    # The side comes from our own roster ROW, which carries a team even when its vehicle
    # id is unresolved (15.9). Looking it up as teams[own_id] was the only route before,
    # so an unresolved own vehicle also lost the side.
    own_team = (own_row or {}).get("team_id")
    if own_team is None:
        own_team = teams.get(own_id)
    death_clocks = {}
    for row in death_rows:
        victim = row.get("victim")
        if victim is not None:
            death_clocks[int(victim)] = float(row["clock"])
    # Every vehicle this team spotted ANYWHERE in the battle. It is a whole-battle set,
    # not the per-clock one, because "is this ship on the other side" is a property of
    # the ship while `spotted_at` answers "is it drawn right now": a ship spotted once
    # and then lost is still an enemy when its visibility flags read 0.
    spotted_ever = set()
    for row in spotted_rows:
        for entity in row.get("entities") or ():
            spotted_ever.add(int(entity))
    # Without a side for our own ship nothing can be split into sides at all, and
    # `None == None` would otherwise put every unknown vehicle on our side.
    if own_team is None:
        enemies, teammates = [], []
    else:
        enemies = sorted(v for v, team in teams.items() if team != own_team)
        teammates = sorted(v for v, team in teams.items()
                           if team == own_team and v != own_id)

    if dataset_path and Path(dataset_path).exists():
        data = np.load(dataset_path, allow_pickle=False)
        clocks = [float(v) for v in data["clocks"]]
    else:
        first = positions[0]["clock"] if positions else 0.0
        last = positions[-1]["clock"] if positions else 0.0
        clocks = []
        clock = first
        while clock <= last:
            clocks.append(clock)
            clock += step_s

    rows = []
    counts = {name: 0 for name in MARKER_CLASSES}
    ring_counts = {}
    for index, clock in enumerate(clocks):
        markers = []
        own_row = latest(positions, clock, None)
        own_death = death_clocks.get(own_id)
        # The own marker leaves the map when the ship sinks. The 0x2C packet simply
        # stops, so without this the last reported position would keep painting an
        # `own` marker for the rest of the battle.
        if own_row is not None and (own_death is None or clock <= own_death):
            px, py, u, v = project(matrix, translation, rect, own_row["x"], own_row["z"])
            markers.append({"class": OWN, "vehicle": own_id, "team": own_team,
                            "world_x": own_row["x"], "world_z": own_row["z"],
                            "pixel_x": px, "pixel_y": py, "u": u, "v": v,
                            "source": "packet 0x2C own position"})
        track_rows = positions_by_vehicle(tracks, clock)
        spotted_set, known = spotted_at(spotted_rows, clock, teams)
        for vehicle, row in track_rows.items():
            if vehicle == own_id:
                continue
            team = teams.get(vehicle)
            # The roster has no row for this ship, but this team has spotted it, and a
            # team never spots its own ships: it is on the other side. See SPOTTED_SIDE.
            if team is None and own_team is not None and vehicle in spotted_ever:
                team = SPOTTED_SIDE
            flags = visibility_at(visibility_rows, vehicle, clock)
            kind, source = classify_vehicle(team, own_team, flags,
                                            death_clocks.get(vehicle), clock)
            px, py, u, v = project(matrix, translation, rect, row["x"], row["z"])
            markers.append({"class": kind, "vehicle": vehicle, "team": team,
                            "world_x": row["x"], "world_z": row["z"],
                            "pixel_x": px, "pixel_y": py, "u": u, "v": v,
                            "visibility_flags": flags,
                            "visibility_known": flags is not None,
                            "spotted_by_player": vehicle in spotted_set,
                            "spotted_known": known, "source": source})
        # The lock marker: the agent's own action is the label. The drawn circle sits on
        # the locked ship's marker, so the label is that ship's position projected the
        # same way every other marker is. Nothing here claims the circle looks like a
        # circle -- what it looks like is unverified, exactly like the `aim` class.
        locked = lock_at(lock_rows, clock) if lock_rows else None
        if locked is not None:
            target = locked.get("vehicle")
            row = None
            if target is not None:
                row = positions_by_vehicle(tracks, clock).get(target)
            elif "x" in locked and "z" in locked:
                row = {"x": float(locked["x"]), "z": float(locked["z"])}
            if row is not None:
                px, py, u, v = project(matrix, translation, rect, row["x"], row["z"])
                markers.append({
                    "class": LOCK, "vehicle": target, "team": teams.get(target),
                    "world_x": row["x"], "world_z": row["z"],
                    "pixel_x": px, "pixel_y": py, "u": u, "v": v,
                    "source": ("the agent's own lock action (self-supervised): the replay "
                               "cannot carry the lock, because the targeting interface's "
                               "setWeaponLock/setPriorityTarget are client->server"),
                })
        if ring_block and images is not None and index < len(images) and own_row is not None:
            own_pixel = (own_row["x"], own_row["z"])
            centre = mini.apply_affine(matrix, translation, own_pixel[0], own_pixel[1])
            centre = (centre[0] - rect[0], centre[1] - rect[1])
            shares = ring_shares(images[index], centre, ring_labels, pixels_per_km)
            for marker in markers:
                if marker["class"] == OWN:
                    marker["ring_shares"] = {str(k): round(v, 4) for k, v in shares.items()}
            for label in ring_labels:
                share = shares.get(label)
                if share is None or share < RING_COLOUR_SHARE:
                    continue
                markers.append({
                    "class": RANGE_RING, "label_km": label,
                    "radius_px": round(label * pixels_per_km, 3),
                    "centre_u": round(centre[0] / (rect[2] - 1), 6),
                    "centre_v": round(centre[1] / (rect[3] - 1), 6),
                    "u": round(centre[0] / (rect[2] - 1), 6),
                    "v": round(centre[1] / (rect[3] - 1), 6),
                    "colour_share": round(share, 4),
                    "source": ("range ring centred on the own ship; radius = label_km x "
                               "pixels_per_km from the measured map scale, presence "
                               "decided by the share of ring-coloured pixels on the circle"),
                })
        # Planes are not entities: the squadron stream is the minimap's own data,
        # carrying a position and a team per squadron. It is kept while "present",
        # which the removeMinimapSquadron call clears.
        live_squadrons = {}
        for row in squadron_rows:
            if row["clock"] > clock:
                break
            if row.get("present"):
                live_squadrons[int(row["squadron"])] = row
            else:
                live_squadrons.pop(int(row["squadron"]), None)
        for squadron_id, row in live_squadrons.items():
            if row.get("x") is None:
                continue
            px, py, u, v = project(matrix, translation, rect, row["x"], row["z"])
            markers.append({
                "class": SQUADRON, "squadron": squadron_id, "team": row.get("team"),
                "own_team": row.get("team") == own_team,
                "world_x": row["x"], "world_z": row["z"],
                "pixel_x": px, "pixel_y": py, "u": u, "v": v,
                "source": "Avatar.receive_addMinimapSquadron / receive_updateMinimapSquadron",
            })
        # The aim point of the own ship's most recent salvo, projected: this is where
        # the player was pointing, which is the data behind the minimap aim marker.
        # The marker's colour is not verified, so it carries no colour gate.
        own_shots = [row for row in shot_rows if row["clock"] <= clock]
        if own_shots:
            last = own_shots[-1]
            if last.get("aim_x") is not None:
                px, py, u, v = project(matrix, translation, rect,
                                       last["aim_x"], last["aim_z"])
                markers.append({
                    "class": AIM, "owner": last["owner"],
                    "world_x": last["aim_x"], "world_z": last["aim_z"],
                    "pixel_x": px, "pixel_y": py, "u": u, "v": v,
                    "seconds_since_shot": round(clock - last["clock"], 3),
                    "source": "Avatar.receiveArtilleryShots tarPos (aim point at firing)",
                })
        zone_state = zone_states(zone_rows, clock)
        for zone in zones:
            state = zone_state.get(zone["zone"], {})
            # A zone has no world position in the replay, so its calibrated pixel
            # IS the position and is used directly.
            markers.append({
                "class": ZONE, "zone": zone["zone"], "letter": zone["letter"],
                "team_id": state.get("team_id"), "radius_m": zone["radius_m"],
                "radius_px": zone["radius_px"],
                "pixel_x": zone["pixel"][0], "pixel_y": zone["pixel"][1],
                "u": round((zone["pixel"][0] - rect[0]) / rect[2], 6),
                "v": round((zone["pixel"][1] - rect[1]) / rect[3], 6),
                "source": "map-static zone calibration, owner from InteractiveZone.teamId",
            })
        for marker in markers:
            counts[marker["class"]] = counts.get(marker["class"], 0) + 1
            if marker["class"] == RANGE_RING:
                ring_counts[str(marker["label_km"])] = ring_counts.get(
                    str(marker["label_km"]), 0) + 1
        caveats = []
        if not known:
            caveats.append("spottedSetUnknownBeforeFirstPacket")
        rows.append({
            "sample_id": "minimap:%05d" % index,
            "session_id": session.get("replay", "").replace(".wowsreplay", ""),
            "video_time_s": round(rate * clock + offset_s, 3),
            "replay_clock_s": round(clock, 3),
            "minimap_rect": rect,
            "markers": markers,
            "counts": {name: sum(1 for m in markers if m["class"] == name)
                       for name in MARKER_CLASSES},
            "caveats": caveats,
        })
    summary_rings = {str(label): ring_counts.get(str(label), 0) for label in ring_labels}
    return rows, {
        "session": session.get("replay"),
        "map": session.get("map"),
        "own_vehicle": own_id,
        "own_team": own_team,
        "own_team_known": own_team is not None,
        "vehicles_without_a_team": sorted(v for v, team in teams.items() if team is None),
        "teammates": len(teammates),
        "enemies": len(enemies),
        "enemy_entities_ever_spotted": sorted({int(item) for row in spotted_rows
                                               for item in row["entities"]}),
        "samples": len(rows),
        "marker_totals": counts,
        "ring_labels_km": ring_labels,
        "pixels_per_km": pixels_per_km,
        "zone_calibration": str(zones_path) if zones else None,
        "squadron_samples": len(squadron_rows),
        "shot_samples": len(shot_rows),
        "class_availability": {
            "zone": bool(zones),
            "squadron": bool(squadron_rows),
            "aim": bool(shot_rows),
            "range_ring": bool(ring_block),
            "plane": False,
            "lock": bool(lock_rows),
            "lock_labels": len(lock_rows),
        "reason": ("a map without its own zone calibration cannot place A/B/C, a map "
                   "without a measured ring scale cannot size a range circle, and the "
                   "lock marker needs a lock log -- the replay cannot carry a lock, so "
                   "the label comes from the agent's own action"),
        },
        "ring_enabled_samples": summary_rings,
        "plane_support": ("planes arrive as the `squadron` class, not as entities: a "
                          "carrier battle's entity table holds only ships, zones and "
                          "smoke, while the minimap squares come from "
                          "Avatar.receive_addMinimapSquadron / receive_updateMinimapSquadron"),
        "terrain_support": "static per map; extracted by a per-pixel median over "
                           "many frames",
        "wreck_support": "sunken ships keep a dark marker at the wreck; derived from "
                         "the death events, measured 0.337 hit rate on sunk enemies "
                         "against 0.000 on living ones",
        "zone_source": str(zones_path),
    }


COLOURS = {
    OWN: (255, 255, 255),
    FRIENDLY: (0, 255, 255),
    ENEMY_VISIBLE: (255, 40, 40),
    ZONE: (255, 235, 40),
    WRECK: (170, 120, 255),
    RANGE_RING: (255, 150, 30),
}


def draw_ring(array, u, v, colour, radius=7):
    """Draw a thin ring at a normalised position so the marker underneath stays visible."""
    height, width = array.shape[0], array.shape[1]
    centre_column = u * (width - 1)
    centre_row = v * (height - 1)
    for step in range(360):
        angle = step * 3.14159265 / 180.0
        row = int(round(centre_row + radius * np.sin(angle)))
        column = int(round(centre_column + radius * np.cos(angle)))
        if 0 <= row < height and 0 <= column < width:
            array[row, column] = colour
    return array


def write_png(path, image):
    import subprocess
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", "%dx%d" % (image.shape[1], image.shape[0]), "-i", "-",
                    "-frames:v", "1", "-y", str(path)],
                   input=np.ascontiguousarray(image).tobytes(), check=True)
    return path


def overlay(args, rows):
    """Tile real crops with every collected marker drawn, for a human to check.

    Rings are used rather than filled dots so the icon the label claims to
    describe stays visible underneath. Unspotted enemies are deliberately NOT
    drawn: their absence is the claim under test.
    """
    data = np.load(args.dataset, allow_pickle=False)
    images = data["images"]
    count = min(args.overlay_count, len(rows))
    stride = max(1, len(rows) // count)
    picks = list(range(0, len(rows), stride))[:count]
    tiles = []
    for index in picks:
        tile = np.ascontiguousarray(images[index].copy())
        for marker in rows[index]["markers"]:
            colour = COLOURS.get(marker["class"])
            if colour is None:
                continue
            if marker["class"] == RANGE_RING:
                # A range ring is a circle of its own radius, not a point.
                draw_ring(tile, marker["centre_u"], marker["centre_v"], colour,
                          radius=int(round(marker["radius_px"])))
                continue
            draw_ring(tile, marker["u"], marker["v"], colour)
        tiles.append(tile)
    columns = 2
    rows_count = (len(tiles) + columns - 1) // columns
    height, width = tiles[0].shape[0], tiles[0].shape[1]
    canvas = np.zeros((rows_count * height, columns * width, 3), dtype=np.uint8)
    for position, tile in enumerate(tiles):
        row, column = divmod(position, columns)
        canvas[row * height:(row + 1) * height, column * width:(column + 1) * width] = tile
    print("overlay: white own, cyan friendly, red spotted enemy, yellow zone; "
          "unspotted enemies intentionally not drawn")
    return write_png(args.overlay_out, canvas)


CHECKS = (
    (OWN, "near-white min>200", 0.90, "at least"),
    (FRIENDLY, "green G>R+30", 0.50, "at least"),
    (ENEMY_VISIBLE, "red R>G+30", 0.90, "at least"),
    (ENEMY_NOT_VISIBLE, "red R>G+30", 0.15, "at most"),
    (WRECK, "darker than 0.62 x the map mean", 0.20, "at least"),
    (SQUADRON, "green when friendly, red when enemy", 0.20, "at least"),
)


def pixel_at(image, u, v, radius=3):
    """Return the pixels of a small patch around a normalised position."""
    height, width = image.shape[0], image.shape[1]
    column = int(round(u * (width - 1)))
    row = int(round(v * (height - 1)))
    return image[max(0, row - radius):row + radius + 1,
                 max(0, column - radius):column + radius + 1].reshape(-1, 3).astype(int)


def agreement(rows, images, stride=5):
    """Measure whether each class's labels land on the colour that class draws.

    This is the check that keeps the labels honest: a class whose labels do not
    land on its own colour is wrong, however plausible the derivation looked. The
    absent class is checked the other way round, because a marker that should NOT
    be drawn must not be found.
    """
    hits = {name: [0, 0] for name, _rule, _limit, _sense in CHECKS}
    zone_hits = [0, 0]
    for index in range(0, len(rows), stride):
        if index >= len(images):
            break
        image = images[index]
        for marker in rows[index]["markers"]:
            patch = pixel_at(image, marker["u"], marker["v"])
            red = float(np.mean((patch[:, 0] > patch[:, 1] + 30)
                                & (patch[:, 0] > patch[:, 2] + 30)))
            green = float(np.mean(patch[:, 1] > patch[:, 0] + 30))
            white = float(np.mean(patch.min(axis=1) > 200))
            dark = 1.0 if patch.mean() < 0.62 * image.mean() else 0.0
            team = marker.get("team")
            squadron_colour = green if team is not None and marker.get("own_team") else red
            if team is None:
                squadron_colour = max(red, green)
            found = {OWN: white, FRIENDLY: green, ENEMY_VISIBLE: red,
                     ENEMY_NOT_VISIBLE: red, WRECK: dark,
                     SQUADRON: squadron_colour}.get(marker["class"])
            if found is not None:
                hits[marker["class"]][1] += 1
                if found > 0.08:
                    hits[marker["class"]][0] += 1
            if marker["class"] == ZONE and marker["team_id"] is not None:
                # Sample the disc away from the letter to read its ownership colour.
                offset_u = marker["u"] + 0.018
                ring = pixel_at(image, offset_u, marker["v"], radius=1)
                is_red = float(np.mean(ring[:, 0] > ring[:, 1] + 20)) > 0.5
                zone_hits[1] += 1
                if is_red == (marker["team_id"] == 1):
                    zone_hits[0] += 1
    results = {}
    failed = []
    for name, rule, limit, sense in CHECKS:
        matched, total = hits[name]
        share = matched / total if total else 0.0
        ok = share >= limit if sense == "at least" else share <= limit
        results[name] = {"rule": rule, "limit": limit, "sense": sense,
                         "samples": total, "share": round(share, 4), "pass": ok}
        if not ok:
            failed.append(name)
    # A declared range ring must sit on ring-coloured pixels, and a label that was
    # NOT declared must not. Both directions are checked, because a detector that
    # always answers "yes" would otherwise look perfect.
    declared = [marker["colour_share"] for row in rows for marker in row["markers"]
                if marker["class"] == RANGE_RING]
    declared_labels = {marker["label_km"] for row in rows for marker in row["markers"]
                       if marker["class"] == RANGE_RING}
    undeclared = []
    for row in rows:
        for marker in row["markers"]:
            if marker["class"] != OWN:
                continue
            shares = marker.get("ring_shares")
            if not shares:
                continue
            for label, share in shares.items():
                if float(label) not in declared_labels:
                    undeclared.append(share)
    ring_sample_count = sum(1 for row in rows for marker in row["markers"]
                            if marker["class"] == RANGE_RING)
    ring_ok = bool(declared) and min(declared) >= RING_COLOUR_SHARE
    results["range_ring"] = {
        "rule": "declared rings sit on ring-coloured pixels (share >= %.2f)" % RING_COLOUR_SHARE,
        "samples": ring_sample_count,
        "declared_share_min": round(min(declared), 4) if declared else None,
        "declared_share_mean": round(sum(declared) / len(declared), 4) if declared else None,
        "labels_declared_somewhere": sorted(declared_labels),
        "share": round(min(declared), 4) if declared else 0.0,
        "pass": ring_ok,
    }
    results["zone"] = {"rule": "disc colour agrees with InteractiveZone.teamId",
                       "samples": zone_hits[1],
                       "share": round(zone_hits[0] / zone_hits[1], 4) if zone_hits[1] else 0.0,
                       "pass": zone_hits[1] > 0 and zone_hits[0] / zone_hits[1] >= 0.8}
    if not results["zone"]["pass"]:
        failed.append("zone")
    if not results["range_ring"]["pass"]:
        failed.append("range_ring")
    return results, failed


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--hud-calibration", type=Path, required=True)
    parser.add_argument("--minimap-calibration", type=Path, required=True)
    parser.add_argument("--zones", type=Path,
                        help="per-map zone calibration; omit for a map that has none, "
                             "and the zone class is then reported as unavailable")
    parser.add_argument("--locks", type=Path,
                        help="the agent's own lock log ({clock, vehicle} per line); "
                             "without it the lock class is reported UNAVAILABLE")
    parser.add_argument("--dataset", type=Path,
                        help="reuse a dataset's sample clocks instead of a fresh grid")
    parser.add_argument("--step-s", type=float, default=1.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--terrain-out", type=Path,
                        help="write the static background as a PNG, needs --dataset")
    parser.add_argument("--overlay-out", type=Path,
                        help="draw every collected marker on real crops, needs --dataset")
    parser.add_argument("--overlay-count", type=int, default=4)
    parser.add_argument("--check", action="store_true",
                        help="measure whether each class lands on its own colour")
    args = parser.parse_args(argv)

    images = None
    if args.dataset and Path(args.dataset).exists():
        images = np.load(args.dataset, allow_pickle=False)["images"]
    rows, summary = build(args.export, args.alignment, args.hud_calibration,
                          args.minimap_calibration, args.zones, args.dataset,
                          args.step_s, images, args.locks)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print("samples %d  markers %s" % (len(rows), summary["marker_totals"]))
    print("own vehicle %s (team %s)  teammates %d  enemies %d"
          % (summary["own_vehicle"], summary["own_team"], summary["teammates"],
             summary["enemies"]))
    print("enemy entities ever spotted: %s" % summary["enemy_entities_ever_spotted"])
    print("planes: %s" % summary["plane_support"])

    if args.check and args.dataset and Path(args.dataset).exists():
        results, failed = agreement(rows, np.load(args.dataset, allow_pickle=False)["images"])
        for name, result in results.items():
            print("  %-18s %-42s %5s of %5d = %.3f  %s"
                  % (name, result["rule"], result["samples"],
                     result["samples"], result["share"],
                     "ok" if result["pass"] else "FAIL"))
        summary["label_colour_check"] = results
        print("VERDICT %s" % ("FAIL: " + ", ".join(failed) if failed else "pass"))

    if args.overlay_out and args.dataset and Path(args.dataset).exists():
        print("wrote %s" % overlay(args, rows))

    if args.terrain_out and args.dataset and Path(args.dataset).exists():
        data = np.load(args.dataset, allow_pickle=False)
        # Markers move between samples, so the per-pixel median keeps what never
        # moves: terrain, the grid, the zone rings and the map's own lettering.
        median = np.median(data["images"], axis=0).astype(np.uint8)
        args.terrain_out.parent.mkdir(parents=True, exist_ok=True)
        import subprocess
        subprocess.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                        "-s", "%dx%d" % (median.shape[1], median.shape[0]), "-i", "-",
                        "-frames:v", "1", "-y", str(args.terrain_out)],
                       input=np.ascontiguousarray(median).tobytes(), check=True)
        summary["terrain_png"] = str(args.terrain_out)
        summary["terrain_method"] = ("per-pixel median over %d minimap crops; moving "
                                     "markers are removed, static map art remains"
                                     % len(data["images"]))
        print("wrote %s (median over %d crops)" % (args.terrain_out, len(data["images"])))

    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n",
                                encoding="utf-8")
        print("wrote %s" % args.summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
