"""Gates for the vehicle classification and the roster/own-ship identity rules.

Run with the project environment -- it is the only interpreter that has pycryptodome,
which ``export_events`` needs:

    ./.venv/bin/python -B -m unittest tools/replay/test_label_identity.py

Why these are pinned here rather than eyeballed on a corpus: these booleans decide the
`friendly`, `enemy_visible`, `enemy_not_visible`, `enemy_unknown` and `wreck` channels,
and the corpus that would show them is ~20 GB. Each branch of the order is a claim about
what the replay means, so each one is stated once, in the open.

The classification is verified against real frames elsewhere (the inventory's own
`--check` colour audit); what is verified here is the decision logic, including the two
`None` traps that silently turned "unknown" into a label.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_minimap_inventory as inv  # noqa: E402
import build_heatmap_dataset as heat  # noqa: E402

try:
    import export_events as export
except ImportError as exc:  # pragma: no cover - reported as a skip, not a failure
    export = None
    EXPORT_ERROR = str(exc)
else:
    EXPORT_ERROR = ""

needs_project_env = unittest.skipUnless(
    export is not None,
    "export_events needs pycryptodome: run this with ./.venv/bin/python")


class Packet:
    """The two fields the 0x2C reader looks at."""

    def __init__(self, entity, payload_length=export.POSITION_PAYLOAD if export else 32):
        self.type = export.POSITION_PACKET if export else 0x2C
        header = int(entity).to_bytes(4, "little", signed=True)
        self.payload = header + bytes(payload_length - 4)


class VehicleClassification(unittest.TestCase):
    """The ORDER of the decisions, which is what the corpus was getting wrong."""

    def test_a_living_teammate_is_friendly(self):
        kind, _source = inv.classify_vehicle(team=0, own_team=0, flags=0,
                                             death_clock=None, clock=500.0)
        self.assertEqual(inv.FRIENDLY, kind)

    def test_a_living_visible_enemy_is_enemy_visible(self):
        kind, _source = inv.classify_vehicle(team=1, own_team=0, flags=1,
                                             death_clock=None, clock=500.0)
        self.assertEqual(inv.ENEMY_VISIBLE, kind)

    def test_a_living_invisible_enemy_is_a_confirmed_absence(self):
        kind, _source = inv.classify_vehicle(team=1, own_team=0, flags=0,
                                             death_clock=None, clock=500.0)
        self.assertEqual(inv.ENEMY_NOT_VISIBLE, kind)

    def test_a_dead_teammate_is_a_wreck_not_a_teammate(self):
        """The defect this fixes: the side used to be decided before the life state, so
        a sunk teammate stayed `friendly` and the model was taught to paint a live
        teammate over a wreck."""
        kind, _source = inv.classify_vehicle(team=0, own_team=0, flags=1,
                                             death_clock=100.0, clock=101.0)
        self.assertEqual(inv.WRECK, kind)

    def test_a_dead_enemy_is_a_wreck(self):
        kind, _source = inv.classify_vehicle(team=1, own_team=0, flags=1,
                                             death_clock=100.0, clock=101.0)
        self.assertEqual(inv.WRECK, kind)

    def test_death_outranks_an_unknown_side(self):
        """A wreck is a world object, so it does not need a side to be recognised."""
        kind, _source = inv.classify_vehicle(team=None, own_team=0, flags=None,
                                             death_clock=100.0, clock=101.0)
        self.assertEqual(inv.WRECK, kind)

    def test_the_ship_is_still_alive_at_the_death_clock_itself(self):
        """The comparison is strict, so the boundary belongs to the living side."""
        kind, _source = inv.classify_vehicle(team=0, own_team=0, flags=1,
                                             death_clock=100.0, clock=100.0)
        self.assertEqual(inv.FRIENDLY, kind)

    def test_missing_visibility_flags_are_unknown_not_an_absence(self):
        """This used to be `enemy_not_visible`, which is the model's false-positive
        test: it asserted an absence nobody had observed."""
        kind, _source = inv.classify_vehicle(team=1, own_team=0, flags=None,
                                             death_clock=None, clock=500.0)
        self.assertEqual(inv.ENEMY_UNKNOWN, kind)

    def test_a_vehicle_without_a_team_is_unknown(self):
        kind, _source = inv.classify_vehicle(team=None, own_team=0, flags=1,
                                             death_clock=None, clock=500.0)
        self.assertEqual(inv.ENEMY_UNKNOWN, kind)

    def test_an_unknown_side_for_our_own_ship_makes_everyone_unknown(self):
        """Even an exact team match cannot mean anything when our own side is unknown."""
        kind, _source = inv.classify_vehicle(team=0, own_team=None, flags=1,
                                             death_clock=None, clock=500.0)
        self.assertEqual(inv.ENEMY_UNKNOWN, kind)

    def test_none_never_equals_none_into_a_friendly(self):
        """The trap: `team == own_team` was True for two `None`s, so a roster without
        teams put every unknown vehicle on our own side."""
        kind, _source = inv.classify_vehicle(team=None, own_team=None, flags=None,
                                             death_clock=None, clock=500.0)
        self.assertNotEqual(inv.FRIENDLY, kind)
        self.assertEqual(inv.ENEMY_UNKNOWN, kind)

    def test_every_class_returned_is_a_declared_marker_class(self):
        for kind in (inv.WRECK, inv.ENEMY_UNKNOWN, inv.FRIENDLY, inv.ENEMY_VISIBLE,
                     inv.ENEMY_NOT_VISIBLE):
            self.assertIn(kind, inv.MARKER_CLASSES)

    def test_the_unknown_class_is_not_the_false_positive_class(self):
        """`enemy_not_visible` is fed to the model as the absence test, so the unknown
        class must be a different name or the fix is cosmetic."""
        self.assertNotEqual(inv.ENEMY_NOT_VISIBLE, inv.ENEMY_UNKNOWN)


@needs_project_env
class OwnVehicleFromPackets(unittest.TestCase):
    """0x2C carries a phantom id beside the ship; the entity table tells them apart."""

    def test_the_known_entity_wins_even_when_it_is_not_the_most_frequent(self):
        """Measured on 236387: 371584 arrives 7909 times and 371585 7907, and only
        371585 is a vehicle the parser's entity table knows. Taking the first or the
        most frequent packet chose the phantom."""
        packets = [Packet(371584)] * 3 + [Packet(371585)] * 2
        self.assertEqual(371585, export.own_entity_from_packets(packets, {371585}))

    def test_the_most_frequent_candidate_is_the_fallback(self):
        packets = [Packet(7)] * 3 + [Packet(9)] * 1
        self.assertEqual(7, export.own_entity_from_packets(packets, None))

    def test_no_candidate_at_all_is_none(self):
        self.assertIsNone(export.own_entity_from_packets([], {1}))

    def test_a_payload_of_another_length_is_ignored(self):
        packets = [Packet(11, payload_length=45)]
        self.assertIsNone(export.own_entity_from_packets(packets, {11}))

    def test_an_unusable_entity_table_still_answers(self):
        packets = [Packet(5)] * 2
        self.assertEqual(5, export.own_entity_from_packets(packets, set()))


@needs_project_env
class RosterUsability(unittest.TestCase):
    """When the packet roster may be replaced by the toalba one."""

    def test_a_named_team_carrying_roster_that_places_us_is_usable(self):
        self.assertEqual("", export.roster_problem({1: {"name": "a", "teamId": 0}}, 5))

    def test_an_empty_roster_is_unusable(self):
        self.assertTrue(export.roster_problem({}, 5))

    def test_a_roster_without_a_single_name_is_unusable(self):
        self.assertTrue(export.roster_problem({1: {"name": None, "teamId": 0}}, 5))

    def test_a_roster_without_a_single_team_is_unusable(self):
        """15.9: names arrive, teams do not, and the inventory separates sides by team."""
        self.assertTrue(export.roster_problem({1: {"name": "a", "teamId": None}}, 5))

    def test_a_roster_that_cannot_place_us_is_unusable(self):
        """Without our own vehicle the own position stream -- and the `own` channel --
        comes out empty, which is the 15.9 case."""
        self.assertTrue(export.roster_problem({1: {"name": "a", "teamId": 0}}, None))

    def test_a_partially_resolved_roster_is_still_usable(self):
        """One unresolved row is normal (only the recording player may be missing from
        the 0x0A family); it must not throw the whole table away."""
        entries = {1: {"name": "a", "teamId": 0}, 2: {"name": None, "teamId": 1}}
        self.assertEqual("", export.roster_problem(entries, 5))


class HeatmapChannelContract(unittest.TestCase):
    """The consumer side: an unknown marker may be neither painted nor counted absent.

    `enemy_not_visible` is not a heatmap channel on purpose -- those are the model's
    false-positive test. The whole point of splitting `enemy_unknown` out of it is that
    "no information" must not reach EITHER side of that contract, so the two properties
    are asserted here instead of being read off the corpus.
    """

    def setUp(self):
        self.row = {
            "sample_id": "minimap:00000", "session_id": "synthetic",
            "markers": [
                {"class": inv.ENEMY_UNKNOWN, "u": 0.25, "v": 0.25},
                {"class": inv.ENEMY_NOT_VISIBLE, "u": 0.75, "v": 0.75},
                {"class": inv.ENEMY_VISIBLE, "u": 0.5, "v": 0.5},
            ],
        }

    def test_the_unknown_class_is_not_a_heatmap_channel(self):
        self.assertNotIn(inv.ENEMY_UNKNOWN, heat.CLASSES)

    def test_the_unknown_class_is_not_the_negative_class(self):
        self.assertNotEqual(inv.ENEMY_UNKNOWN, heat.NEGATIVE_CLASS)
        self.assertEqual(inv.ENEMY_NOT_VISIBLE, heat.NEGATIVE_CLASS)

    def test_an_unknown_marker_paints_nothing(self):
        only_unknown = {"markers": [{"class": inv.ENEMY_UNKNOWN, "u": 0.25, "v": 0.25}]}
        heatmaps, truth, _rings = heat.targets_for(
            only_unknown, heat.CLASSES, 32, 1.0, [0, 0, 32, 32])
        self.assertEqual(0, len(truth))
        self.assertEqual(0.0, float(heatmaps.sum()))

    def test_a_known_class_still_paints(self):
        """The companion to the test above: the harness would also pass if NOTHING
        painted, so a real channel is checked on the same path."""
        visible = {"markers": [{"class": inv.ENEMY_VISIBLE, "u": 0.5, "v": 0.5}]}
        heatmaps, truth, _rings = heat.targets_for(
            visible, heat.CLASSES, 32, 1.0, [0, 0, 32, 32])
        self.assertEqual(1, len(truth))
        self.assertGreater(float(heatmaps.sum()), 0.0)

    def test_only_the_confirmed_absence_becomes_a_negative(self):
        negatives = [(m["u"], m["v"]) for m in self.row["markers"]
                     if m["class"] == heat.NEGATIVE_CLASS]
        self.assertEqual([(0.75, 0.75)], negatives)


if __name__ == "__main__":
    unittest.main(verbosity=2)
