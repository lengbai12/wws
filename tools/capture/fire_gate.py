#!/usr/bin/env python3
"""The fire decision: one truth source for "may this shot leave the barrel".

The servo used to answer that question in several places (the per-step loop, the
dispatch check, the confirmation frame, the scope gate), each with its own set of
booleans. A disagreement between them fired or withheld a shot for reasons no
report explained. This module holds the whole decision, so a caller only collects
evidence and reads the answer:

    decision = fire_decision(Evidence(...))
    if decision['allow']:
        trigger()
    else:
        report['fire_decision'] = decision        # every blocker, named

The gates are the nine the owner enumerated. Eight are hard, and the ninth (the
scope pair) is hard only when the scope was requested:

 1. ``battle_active``        the battle HUD itself is drawn
 2. ``own_alive``            the shooter's own ship is alive and in control
 3. ``main_guns_selected``   the selected weapon is the main battery
 4. ``reload_ready``         the main battery is loaded
 5. ``enemy_plate_current``  an enemy plate exists and comes from this frame
 6. ``enemy_plate_alive``    the plate reads hp > 0 and has red pixels
 7. ``aim_error``            the measured error is inside the firing tolerance
 8. ``identity_focus``       the game identity and window focus are verified
 9. ``scope``                if asked for: scope evidenced and target visible

Two readings are deliberately **not** gates. A range gap beyond its tolerance is
a warning that records ``expected_short_or_long_km`` and never stops a shot, and
``body_visible`` is recorded and never consulted: a hull detector missing the
hull is not evidence that the ship is not there, and the owner's nine conditions
do not include it.

Every value is compared strictly. ``None`` means "not observed" and is a blocker,
never a pass: an unknown battle state or an unread nameplate is exactly the case
where the old code guessed.
"""
from dataclasses import dataclass, fields
import math
from typing import Any, Mapping
import weakref

GUN = "gun"

#: Gate names in evaluation order. A report can iterate these.
GATES = (
    "battle_active",
    "own_alive",
    "main_guns_selected",
    "reload_ready",
    "enemy_plate_current",
    "enemy_plate_alive",
    "aim_error",
    "identity_focus",
    "scope",
)

BATTLE_NOT_ACTIVE = "battle_not_active"
OWN_SHIP_NOT_ALIVE = "own_ship_not_alive"
WEAPON_NOT_GUN = "weapon_not_gun"
RELOAD_NOT_READY = "reload_not_ready"
ENEMY_PLATE_MISSING = "enemy_plate_missing"
ENEMY_PLATE_NOT_CURRENT = "enemy_plate_not_current"
ENEMY_PLATE_HP_UNKNOWN = "enemy_plate_hp_unknown"
ENEMY_PLATE_DEPLETED = "enemy_plate_depleted"
ENEMY_PLATE_NO_RED_PIXELS = "enemy_plate_no_red_pixels"
AIM_ERROR_UNKNOWN = "aim_error_unknown"
FIRE_TOLERANCE_MISSING = "fire_tolerance_missing"
AIM_ERROR_OUT_OF_TOLERANCE = "aim_error_out_of_tolerance"
IDENTITY_NOT_VERIFIED = "identity_not_verified"
FOCUS_NOT_CONFIRMED = "focus_not_confirmed"
SCOPE_NOT_EVIDENCED = "scope_not_evidenced"
SCOPE_TARGET_NOT_VISIBLE = "scope_target_not_visible"

TERRAIN_BLOCKS_SHOT = "terrain_blocks_shot"
RANGE_GAP_OUTSIDE_TOLERANCE = "range_gap_outside_tolerance"
RANGE_GAP_WITHOUT_TOLERANCE = "range_gap_without_tolerance"
BODY_NOT_VISIBLE = "body_not_visible"


@dataclass
class Evidence:
    """Everything the nine gates read, and nothing else.

    ``enemy_plate`` is the plate evidence for the current frame: a mapping with a
    ``bbox`` (four numbers), ``hp`` and ``red_px``. The field aliases used by the
    detectors in this repo (``nameplate_bbox``, ``current``, ``red_pixels``) are
    accepted so a caller can pass its own candidate dict unchanged.
    """

    battle_active: Any = None
    own_alive: Any = None
    weapon_kind: Any = None
    reload_ready: Any = None
    enemy_plate: Any = None
    frame_fresh: Any = None
    aim_error_px: Any = None
    fire_tolerance_px: Any = None
    identity_ok: Any = None
    focus_ok: Any = None
    scope_requested: Any = None
    scope_evidenced: Any = None
    scope_target_visible: Any = None
    range_gap_km: Any = None
    range_tolerance_km: Any = None
    #: Recorded in the decision, never consulted by a gate.
    body_visible: Any = None
    #: Alternative evidence basis. ``minimap_contact`` lets a fresh minimap
    #: contact stand in for a nameplate (a submarine or a cyclone-hidden target
    #: never draws one); every other gate still applies unchanged.
    #: Line-of-fire terrain test from the minimap: True clear, False land on
    #: the line, None no map evidence. Only an explicit False blocks a shot.
    terrain_clear: Any = None
    terrain_detail: Any = None
    shot_basis: Any = 'plate'
    minimap_contact: Any = None
    minimap_contact_fresh: Any = None

    def as_dict(self):
        return {item.name: getattr(self, item.name) for item in fields(self)}


def _number(value):
    """A finite float, or None. Booleans are not numbers and NaN is not a value."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _plate_value(plate, *names):
    if not isinstance(plate, Mapping):
        return None
    for name in names:
        if plate.get(name) is not None:
            return plate[name]
    return None


def _plate_bbox(plate):
    box = _plate_value(plate, "bbox", "nameplate_bbox")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    values = [_number(value) for value in box]
    if any(value is None for value in values):
        return None
    if values[2] <= 0 or values[3] <= 0:
        return None
    return values


def fire_decision(evidence):
    """Decide one shot from one frame's evidence.

    Returns ``{"allow", "blockers", "warnings", "gates", "status", "missing",
    "evidence"}``. ``gates[name]["status"]`` is one of ``pass``, ``fail``,
    ``unknown`` or ``not_requested``; every non-passing gate contributes a named
    blocker, so a refused shot is explained by the decision itself.
    """
    if not isinstance(evidence, Evidence):
        evidence = Evidence(**evidence)
    ev = evidence
    gates = {}
    blockers = []
    warnings = []

    def record(name, ok, blocker=None, status=None, detail=None):
        status = status or ("pass" if ok else "fail")
        gates[name] = {"status": status, "ok": bool(ok),
                       "blocker": None if ok else blocker, "detail": detail}
        if not ok:
            blockers.append(blocker)

    def observed(value):
        """pass/fail/unknown for a strict boolean reading."""
        if value is True:
            return True, "pass"
        if value is None:
            return False, "unknown"
        return False, "fail"

    # 1. battle_active
    ok, status = observed(ev.battle_active)
    record("battle_active", ok, BATTLE_NOT_ACTIVE, status,
           detail=None if ok else {"value": ev.battle_active})

    # 2. own_alive
    ok, status = observed(ev.own_alive)
    record("own_alive", ok, OWN_SHIP_NOT_ALIVE, status,
           detail=None if ok else {"value": ev.own_alive})

    # 3. main_guns_selected
    kind = None if ev.weapon_kind is None else str(ev.weapon_kind).strip().lower()
    ok = kind == GUN
    status = "pass" if ok else ("unknown" if kind in (None, "", "unknown") else "fail")
    record("main_guns_selected", ok, WEAPON_NOT_GUN, status,
           detail={"weapon_kind": ev.weapon_kind})

    # 4. reload_ready
    ok, status = observed(ev.reload_ready)
    record("reload_ready", ok, RELOAD_NOT_READY, status,
           detail=None if ok else {"value": ev.reload_ready})

    # 5. enemy_plate_current
    bbox = _plate_bbox(ev.enemy_plate)
    if bbox is None and getattr(ev, 'shot_basis', None) == 'minimap_contact' \
            and getattr(ev, 'minimap_contact_fresh', None) is True:
        # The operator allows a shot once the target's position is good enough.
        # A submarine (or a target hidden by the cyclone) draws a minimap
        # contact but no nameplate at all, so the plate gate could never be
        # satisfied and no shot would ever be possible. The contact is real
        # measured evidence -- the minimap model's marker plus the range
        # closure -- so it satisfies this gate, and the decision records that
        # basis instead of pretending a plate was seen.
        record("enemy_plate_current", True, status="pass",
               detail={"basis": "minimap_contact",
                       "contact": getattr(ev, 'minimap_contact', None)})
    elif bbox is None:
        record("enemy_plate_current", False, ENEMY_PLATE_MISSING,
               "unknown" if ev.enemy_plate is None else "fail",
               detail={"enemy_plate": None if ev.enemy_plate is None else "without_bbox"})
    else:
        fresh = ev.frame_fresh
        if fresh is None:
            fresh = _plate_value(ev.enemy_plate, "frame_fresh")
        if fresh is True:
            record("enemy_plate_current", True, status="pass", detail={"bbox": bbox})
        else:
            record("enemy_plate_current", False, ENEMY_PLATE_NOT_CURRENT,
                   "unknown" if fresh is None else "fail",
                   detail={"bbox": bbox, "frame_fresh": fresh})

    # 6. enemy_plate_alive -- needs the plate gate 5 found
    if bbox is None:
        gates["enemy_plate_alive"] = {"status": "unknown", "ok": False,
                                      "blocker": None,
                                      "detail": {"reason": "no_enemy_plate"}}
    else:
        hp = _number(_plate_value(ev.enemy_plate, "hp", "current", "current_hp"))
        red_px = _number(_plate_value(ev.enemy_plate, "red_px", "red_pixels"))
        alive_blockers = []
        if hp is None:
            # The operator's rule is "a roughly right solution may fire"; the
            # enemy's exact HP is not needed for that, it only exists to avoid
            # shooting a dead ship. Measured 2026-10-23: a plate whose digits
            # were too small to read blocked a 7.5 px solution. Unknown HP is a
            # warning; hp <= 0 and no red pixels still block.
            warnings.append({"code": ENEMY_PLATE_HP_UNKNOWN,
                             "detail": "plate alive but its HP digits are unreadable"})
        elif hp <= 0:
            alive_blockers.append(ENEMY_PLATE_DEPLETED)
        if red_px is None or red_px <= 0:
            alive_blockers.append(ENEMY_PLATE_NO_RED_PIXELS)
        alive_ok = not alive_blockers
        alive_status = "pass" if alive_ok else (
            "fail" if hp is not None and hp <= 0 else "unknown")
        record("enemy_plate_alive", alive_ok, alive_blockers[0] if alive_blockers else None,
               alive_status, detail={"hp": hp, "red_px": red_px})
        for code in alive_blockers[1:]:
            blockers.append(code)
        if alive_blockers and ENEMY_PLATE_DEPLETED in alive_blockers:
            gates["enemy_plate_alive"]["blocker"] = ENEMY_PLATE_DEPLETED

    # 7. aim_error
    error = _number(ev.aim_error_px)
    tolerance = _number(ev.fire_tolerance_px)
    aim_blockers = []
    if tolerance is None or tolerance <= 0:
        aim_blockers.append(FIRE_TOLERANCE_MISSING)
    if error is None:
        aim_blockers.append(AIM_ERROR_UNKNOWN)
    elif tolerance is not None and tolerance > 0 and error > tolerance:
        aim_blockers.append(AIM_ERROR_OUT_OF_TOLERANCE)
    aim_ok = not aim_blockers
    record("aim_error", aim_ok, aim_blockers[0] if aim_blockers else None,
           "pass" if aim_ok else ("unknown" if error is None or tolerance is None
                                  or tolerance <= 0 else "fail"),
           detail={"aim_error_px": error, "fire_tolerance_px": tolerance})
    if len(aim_blockers) > 1:
        blockers.extend(aim_blockers[1:])
        gates["aim_error"]["blocker"] = AIM_ERROR_OUT_OF_TOLERANCE \
            if AIM_ERROR_OUT_OF_TOLERANCE in aim_blockers else aim_blockers[0]

    # 8. identity_focus
    identity_ok, identity_status = observed(ev.identity_ok)
    focus_ok, focus_status = observed(ev.focus_ok)
    pair_ok = identity_ok and focus_ok
    detail = {"identity_ok": ev.identity_ok, "focus_ok": ev.focus_ok}
    if pair_ok:
        record("identity_focus", True, status="pass", detail=detail)
    else:
        status = "unknown" if "unknown" in (identity_status, focus_status) else "fail"
        gates["identity_focus"] = {"status": status, "ok": False,
                                   "blocker": None, "detail": detail}
        if not identity_ok:
            blockers.append(IDENTITY_NOT_VERIFIED)
        if not focus_ok:
            blockers.append(FOCUS_NOT_CONFIRMED)
        gates["identity_focus"]["blocker"] = (IDENTITY_NOT_VERIFIED if not identity_ok
                                              else FOCUS_NOT_CONFIRMED)

    # 9. scope -- only when the caller asked for it
    if ev.scope_requested is not True:
        record("scope", True, status="not_requested",
               detail={"scope_requested": ev.scope_requested})
    else:
        scope_blockers = []
        if ev.scope_evidenced is not True:
            scope_blockers.append(SCOPE_NOT_EVIDENCED)
        if ev.scope_target_visible is not True:
            scope_blockers.append(SCOPE_TARGET_NOT_VISIBLE)
        scope_ok = not scope_blockers
        record("scope", scope_ok, scope_blockers[0] if scope_blockers else None,
               "pass" if scope_ok else ("unknown" if ev.scope_evidenced is None
                                        or ev.scope_target_visible is None else "fail"),
               detail={"scope_evidenced": ev.scope_evidenced,
                       "scope_target_visible": ev.scope_target_visible})
        if len(scope_blockers) > 1:
            blockers.extend(scope_blockers[1:])
            gates["scope"]["blocker"] = SCOPE_TARGET_NOT_VISIBLE

    # 6b. terrain_clear
    # Land between us and the target is the one occlusion the minimap can prove.
    # Unknown stays a warning: without a contact there is no line to test, and
    # refusing on "no evidence" would re-introduce the refusals that cost the
    # fire rate. The hull-render flag alone never vetoes (measured 2026-10-23:
    # the body detector false-negatives on distant, backlit hulls while the
    # nameplate and the hits say the ship is there); it is recorded, and when
    # the terrain test also fails the pair blocks.
    terrain = getattr(ev, 'terrain_clear', None)
    if terrain is False:
        record("terrain_clear", False, TERRAIN_BLOCKS_SHOT, "fail",
               detail=getattr(ev, 'terrain_detail', None))
    elif terrain is True:
        record("terrain_clear", True, status="pass",
               detail=getattr(ev, 'terrain_detail', None))
    else:
        record("terrain_clear", True, status="unknown",
               detail={"reason": "no_terrain_evidence",
                       "terrain": getattr(ev, 'terrain_detail', None)})


    # Warnings: a range gap is a number to watch, never a veto. The caption
    # distance under the plate is target km and the reticle readout is aim km,
    # so a positive gap means the crosshair is ranged short of the target.
    gap = _number(ev.range_gap_km)
    gap_tolerance = _number(ev.range_tolerance_km)
    if gap is not None:
        warning = {"code": RANGE_GAP_OUTSIDE_TOLERANCE if gap_tolerance is not None
                   else RANGE_GAP_WITHOUT_TOLERANCE,
                   "range_gap_km": round(gap, 2),
                   "range_tolerance_km": gap_tolerance,
                   "expected_short_or_long_km": round(abs(gap), 2),
                   "miss_direction": "short" if gap > 0 else ("long" if gap < 0 else "none")}
        if gap_tolerance is None:
            warnings.append(warning)
        elif abs(gap) > gap_tolerance:
            warnings.append(warning)

    if ev.body_visible is False:
        # Never a gate: an unresolved hull is recorded so a miss can be read,
        # but the nine conditions do not include it.
        warnings.append({"code": BODY_NOT_VISIBLE, "body_visible": False,
                         "note": "recorded only; never blocks a shot"})

    status_map = {name: gates[name]["status"] for name in GATES}
    missing = [name for name in GATES if gates[name]["status"] in ("unknown", "not_requested")
               and not gates[name]["ok"]]
    return {
        "allow": not blockers,
        "blockers": blockers,
        "warnings": warnings,
        "gates": gates,
        "status": status_map,
        "missing": missing,
        "evidence": ev.as_dict(),
    }


def clear_evidence(**overrides):
    """Evidence that passes every gate, for auxiliary uses and tests.

    The scope pair is "not requested" by default; a caller that wants to know
    whether the scope gate alone blocks passes ``scope_requested`` and its
    evidence.
    """
    base = dict(
        battle_active=True,
        own_alive=True,
        weapon_kind=GUN,
        reload_ready=True,
        enemy_plate={"bbox": [100, 200, 120, 17], "hp": 30000, "red_px": 1500},
        frame_fresh=True,
        aim_error_px=10.0,
        fire_tolerance_px=60.0,
        identity_ok=True,
        focus_ok=True,
        scope_requested=False,
        scope_evidenced=True,
        scope_target_visible=True,
        range_gap_km=0.1,
        range_tolerance_km=0.5,
    )
    base.update(overrides)
    return Evidence(**base)


def scope_blocks(scope_requested, scope_evidenced, scope_target_visible=True):
    """The scope pair alone, so callers share the gate instead of re-testing it."""
    decision = fire_decision(clear_evidence(scope_requested=scope_requested,
                                            scope_evidenced=scope_evidenced,
                                            scope_target_visible=scope_target_visible))
    return not decision["gates"]["scope"]["ok"]


class RangeReader:
    """Target range for one frame: the plate caption first, the minimap last.

    The caption under the nameplate is the game's own number for the ship being
    aimed at and costs ~58 ms; the minimap recogniser costs ~4.5 s uncached and
    ~45 ms cached, so it is only read when the caption is unreadable. The result
    is memoised against the frame itself, so every gate in one step reuses one
    reading instead of re-running OCR.

    Readers are injected (``(frame, bbox) -> km`` and ``(frame) -> km``) so a
    caller can supply its own and a test can count calls.
    """

    def __init__(self, caption_reader=None, minimap_reader=None, cache_size=8):
        self.caption_reader = caption_reader or _default_caption_km
        self.minimap_reader = minimap_reader or _default_minimap_km
        self.cache_size = max(1, int(cache_size))
        self.calls = {"caption": 0, "minimap": 0}
        self.hits = 0
        self._cache = {}

    @staticmethod
    def _key(frame, plate_bbox):
        shape = getattr(frame, "shape", None)
        if shape is None or len(shape) < 2:
            return None
        height, width = int(shape[0]), int(shape[1])
        y, x = min(64, height - 1), min(64, width - 1)
        try:
            fingerprint = (bytes(frame[y, x]) + bytes(frame[-1 - y, -1 - x])
                           + bytes(frame[height // 2, width // 2]))
        except (TypeError, IndexError):
            fingerprint = b""
        box = None
        if plate_bbox is not None:
            try:
                box = tuple(int(round(float(value))) for value in plate_bbox[:4])
            except (TypeError, ValueError, IndexError):
                box = None
        return (id(frame), shape, fingerprint, box)

    def read(self, frame, plate_bbox):
        """Return ``{target_km, source, cached, caption_attempted}`` for a frame."""
        key = self._key(frame, plate_bbox)
        entry = self._cache.get(key) if key is not None else None
        if entry is not None:
            holder, result = entry
            if holder() is frame:
                self.hits += 1
                return dict(result, cached=True)
            # The array behind this id was freed; the id may belong to another
            # frame now, which is why the cache holds a weak reference.
            del self._cache[key]
        source = "unavailable"
        target_km = None
        if plate_bbox is not None:
            self.calls["caption"] += 1
            try:
                value = self.caption_reader(frame, tuple(plate_bbox))
            except (TypeError, ValueError, OSError, IndexError):
                value = None
            number = _number(value)
            if number is not None:
                target_km, source = number, "nameplate_caption"
        if target_km is None:
            self.calls["minimap"] += 1
            try:
                value = self.minimap_reader(frame)
            except (TypeError, ValueError, OSError, IndexError):
                value = None
            number = _number(value)
            if number is not None:
                target_km, source = number, "minimap"
        result = {"target_km": target_km, "source": source, "cached": False,
                  "caption_attempted": plate_bbox is not None}
        if key is not None:
            if len(self._cache) >= self.cache_size:
                self._cache.pop(next(iter(self._cache)))
            try:
                holder = weakref.ref(frame)
            except TypeError:
                holder = lambda: frame  # noqa: E731 - unweakrefable object
            self._cache[key] = (holder, result)
        return dict(result)

    def reset(self):
        self._cache.clear()
        self.calls.update(caption=0, minimap=0)
        self.hits = 0


def _default_caption_km(frame, plate_bbox):
    import aim_servo
    return aim_servo.hud_distance_km(frame, plate_bbox)


def _default_minimap_km(frame):
    import aim_servo
    return aim_servo.minimap_target_km(frame)
