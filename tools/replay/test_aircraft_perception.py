import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aircraft_perception as aircraft


def blank():
    return np.full(aircraft.FRAME_SHAPE, 96, np.uint8)


def stamp(frame, relative, x=900, y=500, scale=1.0, tint=None):
    rgba = np.asarray(Image.open(aircraft.ASSET_ROOT / relative).convert("RGBA")).copy()
    if tint is not None:
        rgba[..., :3] = tint
    if scale != 1:
        rgba = cv2.resize(rgba, (round(rgba.shape[1]*scale), round(rgba.shape[0]*scale)),
                          interpolation=cv2.INTER_LINEAR)
    h, w = rgba.shape[:2]
    alpha = rgba[..., 3:4].astype(np.float32)/255
    background = frame[y:y+h, x:x+w]
    frame[y:y+h, x:x+w] = (rgba[..., :3]*alpha + background*(1-alpha)).astype(np.uint8)
    return [x, y, w, h]


FIXTURES = Path(__file__).resolve().parent / "aircraft_fixtures"


def place_crop(path, x=800, y=480):
    crop = np.asarray(Image.open(path).convert("RGB"))
    frame = blank()
    h, w = crop.shape[:2]
    frame[y:y+h, x:x+w] = crop
    return frame


class AircraftRecognition(unittest.TestCase):
    def test_native_enemy_bomber_torpedo_fighter_and_skip_families(self):
        # These are native glyph placements, not claims of real-frame recall.
        families = (
            "controllable/bomber_ap_enemy.png", "controllable/bomber_he_enemy.png",
            "controllable/torpedo_regular_enemy.png", "controllable/torpedo_deepwater_enemy.png",
            "controllable/fighter_ap_enemy.png", "controllable/fighter_he_enemy.png",
            "controllable/skip_ap_enemy.png", "controllable/skip_he_enemy.png",
            "controllable/auxiliary_enemy.png", "consumables/smoke_enemy.png",
            "consumables/fighter_enemy.png", "consumables/bomber_enemy.png",
            "consumables/scout_enemy.png", "consumables/fighter_upgrade_enemy.png",
            "airsupport/bomber_depth_charge_enemy.png", "airsupport/bomber_he_enemy.png",
            "airsupport/bomber_mine_enemy.png",
        )
        for family in families:
            with self.subTest(family=family):
                frame = blank()
                box = stamp(frame, "plane/" + family)
                markers = aircraft.detect(frame)
                self.assertEqual(1, len(markers))
                marker = markers[0]
                self.assertEqual("aircraft", marker["class"])
                self.assertEqual("enemy", marker["team"])
                self.assertEqual(box, marker["bbox"])
                self.assertIn("plane/" + family, marker["matched_assets"])
                self.assertTrue(np.isfinite(marker["score"]))
                self.assertGreaterEqual(marker["color_support_fraction"], .8)
                self.assertIsNone(marker["world_velocity"])
                self.assertEqual("screen_hud_marker_only", marker["observation_semantics"])

    def test_ally_team_is_measured(self):
        frame = blank()
        stamp(frame, "plane/controllable/torpedo_regular_ally.png")
        markers = aircraft.detect(frame)
        self.assertEqual(1, len(markers))
        marker = markers[0]
        self.assertEqual("ally", marker["team"])
        r, g, b = marker["measured_color_rgb"]
        self.assertGreater(g, r+65)
        self.assertGreater(b, r+35)

    def test_other_team_palettes_abstain(self):
        for team in ("own", "division", "teamkiller"):
            with self.subTest(team=team):
                frame = blank()
                stamp(frame, f"plane/controllable/torpedo_regular_{team}.png")
                self.assertEqual([], aircraft.detect(frame))

    def test_multiple_markers_are_stable_and_deduplicated(self):
        frame = blank()
        stamp(frame, "plane/controllable/torpedo_regular_enemy.png", x=700)
        stamp(frame, "plane/consumables/fighter_enemy.png", x=1000)
        markers = aircraft.detect(frame)
        self.assertEqual(2, len(markers))
        self.assertEqual(markers, aircraft.detect(frame))
        self.assertEqual([700, 1000], [m["bbox"][0] for m in markers])

    def test_capture_scale_is_supported(self):
        for scale in aircraft.SCALES:
            with self.subTest(scale=scale):
                frame = blank()
                box = stamp(frame, "plane/controllable/torpedo_regular_enemy.png", scale=scale)
                found = aircraft.detect(frame)
                self.assertEqual(1, len(found))
                self.assertEqual(box, found[0]["bbox"])

    def test_nearby_markers_survive_overlapping_proposals_and_group_boundary(self):
        # Both separations produce overlapping 45px proposal margins. Moving
        # across a shared grouping boundary must not lose either full glyph.
        for left in (740, 775, 820):
            with self.subTest(left=left):
                frame = blank()
                boxes = [stamp(frame, "plane/controllable/torpedo_regular_enemy.png", x=left),
                         stamp(frame, "plane/consumables/fighter_enemy.png", x=left+60)]
                markers = aircraft.detect(frame)
                self.assertEqual(boxes, [marker["bbox"] for marker in markers])
                self.assertEqual(["enemy", "enemy"], [marker["team"] for marker in markers])

    def test_empty_and_ordinary_orange_fire_projectile_shapes_abstain(self):
        self.assertEqual([], aircraft.detect(blank()))
        frame = blank()
        cv2.circle(frame, (850, 500), 17, (255, 165, 20), -1)
        cv2.line(frame, (950, 500), (1000, 475), (255, 155, 30), 4)
        frame[500:530, 1100:1130] = (255, 145, 10)
        self.assertEqual([], aircraft.detect(frame))
        for tint in ((255, 160, 20), (255, 220, 120), (200, 200, 200)):
            frame = blank()
            stamp(frame, "plane/consumables/fighter_enemy.png", tint=tint)
            self.assertEqual([], aircraft.detect(frame))

    def test_solid_enemy_color_is_not_an_aircraft(self):
        frame = blank()
        frame[500:525, 900:925] = (255, 64, 0)
        self.assertEqual([], aircraft.detect(frame))

    def test_ship_mine_fire_and_torpedo_symbols_are_negative_classes(self):
        for relative in (
            "ship/icon_enemy_destroyer.png", "ship/icon_enemy_battleship.png",
            "ship/icon_enemy_submarine.png", "ship/icon_enemy_aircarrier.png",
            "mines/mine_enemy.png", "mines/mine_enemy_small.png",
            "state_markers/icon_fire.png", "torpedoes/icon_torpedo_normal_enemy.png",
        ):
            for scale in (1.0, 1.25, 1.5):
                with self.subTest(asset=relative, scale=scale):
                    frame = blank()
                    stamp(frame, relative, scale=scale)
                    self.assertEqual([], aircraft.detect(frame))

    def test_hud_minimap_letterbox_and_additional_mask_are_excluded(self):
        for x, y in ((900, 0), (900, 1540), (100, 100), (100, 1300),
                     (2200, 500), (1100, 1400), (2200, 1000)):
            with self.subTest(x=x, y=y):
                frame = blank()
                stamp(frame, "plane/consumables/fighter_enemy.png", x=x, y=y)
                self.assertEqual([], aircraft.detect(frame))
        frame = blank()
        box = stamp(frame, "plane/consumables/fighter_enemy.png")
        excluded = np.zeros(aircraft.FRAME_SHAPE[:2], bool)
        x, y, w, h = box
        excluded[y, x] = True  # Even a corner of the full sprite box excludes it.
        self.assertEqual([], aircraft.detect(frame, excluded))

    def test_hud_boundary_is_half_open(self):
        frame = blank()
        stamp(frame, "plane/consumables/fighter_enemy.png", x=900, y=75)
        self.assertEqual(1, len(aircraft.detect(frame)))
        frame = blank()
        stamp(frame, "plane/consumables/fighter_enemy.png", x=900, y=74)
        self.assertEqual([], aircraft.detect(frame))

    def test_frame_edge_requires_complete_glyph_and_matching_context(self):
        frame = blank()
        stamp(frame, "plane/consumables/fighter_enemy.png", x=0, y=500)
        self.assertEqual([], aircraft.detect(frame))

    def test_missing_native_assets_remain_unknown(self):
        frame = blank()
        stamp(frame, "plane/consumables/fighter_enemy.png")
        with patch.object(aircraft, "_templates", return_value=((), ())):
            self.assertEqual([], aircraft.detect(frame))

    def test_ambiguous_negative_shape_abstains(self):
        frame = blank()
        stamp(frame, "plane/consumables/fighter_enemy.png")
        with patch.object(aircraft, "_negative_score", return_value=1.0):
            self.assertEqual([], aircraft.detect(frame))

    def test_indistinguishable_aircraft_subtypes_stay_unknown(self):
        from dataclasses import replace
        planes, negatives = aircraft._templates()
        template = next(t for t in planes
                        if t.asset == "plane/consumables/fighter_enemy.png" and t.scale == 1.0)
        alternative = replace(template, symbol="unresolved_variant", asset="same_shape_other_variant")
        frame = blank()
        stamp(frame, "plane/consumables/fighter_enemy.png")
        with patch.object(aircraft, "_templates", return_value=((template, alternative), negatives)):
            markers = aircraft.detect(frame)
        self.assertEqual(1, len(markers))
        self.assertEqual("unknown", markers[0]["aircraft_type"])
        self.assertEqual(2, len(markers[0]["matched_assets"]))

    def test_real_scout_family_across_adjacent_frames_and_positions(self):
        for name in ("scout-guided.png", "scout-adjacent-02.png",
                     "scout-adjacent-03.png", "scout-adjacent-04.png"):
            for x, y in ((800, 480), (1450, 850)):
                with self.subTest(crop=name, position=(x, y)):
                    markers = aircraft.detect(place_crop(FIXTURES / name, x, y))
                    self.assertEqual(1, len(markers))
                    marker = markers[0]
                    self.assertEqual("ally", marker["team"])
                    self.assertEqual("scout", marker["aircraft_type"])
                    self.assertEqual("consumables", marker["marker_family"])
                    self.assertIn("plane/consumables/scout_ally.png", marker["matched_assets"])
                    self.assertEqual("screen_hud_marker_only", marker["observation_semantics"])
                    self.assertIsNone(marker["world_velocity"])

    def test_real_rotating_warning_is_distinct_from_plane_marker(self):
        for name in ("danger-guided.png", "danger-adjacent-01.png", "danger-adjacent-02.png",
                     "danger-adjacent-03.png", "danger-adjacent-04.png",
                     "resume-danger-amber.png", "resume-danger-red.png"):
            for x, y in ((800, 480), (1450, 850)):
                with self.subTest(crop=name, position=(x, y)):
                    markers = aircraft.detect(place_crop(FIXTURES / name, x, y))
                    self.assertEqual(1, len(markers))
                    marker = markers[0]
                    self.assertEqual("aircraft", marker["class"])
                    self.assertEqual("central_dangers", marker["marker_family"])
                    self.assertEqual("unknown", marker["aircraft_type"])
                    self.assertEqual("unknown", marker["team"])
                    self.assertEqual("screen_hud_danger_indicator_only", marker["observation_semantics"])
                    self.assertTrue(all(asset.startswith("central_dangers/")
                                        for asset in marker["matched_assets"]))
                    self.assertIsNone(marker["world_velocity"])
                    for unsupported in ("count", "distance", "body_bbox", "world_position"):
                        self.assertNotIn(unsupported, marker)

    def test_danger_palette_does_not_promote_text_ship_or_torpedo_shapes(self):
        for tint in ((255, 54, 0), (255, 138, 0)):
            with self.subTest(tint=tint):
                frame = blank()
                cv2.putText(frame, "SOUTH 12345", (800, 500), cv2.FONT_HERSHEY_SIMPLEX,
                            .8, tint, 2, cv2.LINE_AA)
                self.assertEqual([], aircraft.detect(frame))
            for relative in ("ship/icon_enemy_destroyer.png", "ship/icon_enemy_battleship.png",
                             "torpedoes/icon_torpedo_normal_enemy.png"):
                with self.subTest(tint=tint, asset=relative):
                    frame = blank()
                    stamp(frame, relative, scale=4/3, tint=tint)
                    self.assertEqual([], aircraft.detect(frame))

    def test_recorded_building_detected_alert_and_torpedo_confusions_abstain(self):
        paths = [FIXTURES / "ship-label-native-context.png", FIXTURES / "aerodrome-native-context.png"]
        torpedoes = Path(__file__).resolve().parent / "torpedo_fixtures"
        paths += [torpedoes / name for name in ("alert-lookalike-236374-410s.png",
                  "alert-lookalike-aimprobe.png", "normal-captured-236374.png",
                  "normal-recording-236374-410s.png")]
        for path in paths:
            with self.subTest(crop=path.name):
                self.assertEqual([], aircraft.detect(place_crop(path)))

    def test_partial_real_assets_cannot_establish_family(self):
        for name in ("scout-guided.png", "danger-guided.png",
                     "resume-danger-amber.png", "resume-danger-red.png"):
            with self.subTest(crop=name):
                frame = place_crop(FIXTURES / name)
                # Remove half the actual glyph rather than lowering confidence
                # on a remaining arc/wing. Full native shape evidence is required.
                frame[480:560, 800:845] = 96
                self.assertEqual([], aircraft.detect(frame))

    def test_new_renderings_honor_exclusion_mask(self):
        for name in ("scout-guided.png", "danger-guided.png",
                     "resume-danger-amber.png", "resume-danger-red.png"):
            with self.subTest(crop=name):
                frame = place_crop(FIXTURES / name)
                excluded = np.zeros(aircraft.FRAME_SHAPE[:2], bool)
                excluded[480:560, 800:890] = True
                self.assertEqual([], aircraft.detect(frame, excluded))

    def test_invalid_frame_and_mask_are_rejected(self):
        for frame in (np.zeros((100, 100, 3), np.uint8),
                      np.zeros((1600, 2560), np.uint8),
                      np.zeros(aircraft.FRAME_SHAPE, np.float32)):
            with self.subTest(shape=frame.shape, dtype=str(frame.dtype)):
                with self.assertRaises(ValueError):
                    aircraft.detect(frame)
        for mask in (np.zeros((100, 100), bool), np.zeros((1600, 2560), np.uint8)):
            with self.assertRaises(ValueError):
                aircraft.detect(blank(), mask)


class PaletteBoundaries(unittest.TestCase):
    def test_integer_channels_preserve_float_palette_boundaries(self):
        # All red/green pairs include float-rounding boundaries at 170/180
        # for .7. Blue still exercises both sides of its independent gate.
        r, g = np.meshgrid(np.arange(256, dtype=np.int16),
                           np.arange(256, dtype=np.int16))
        for b in (np.zeros_like(r), g, np.minimum(g+25, 255),
                  np.minimum(g+26, 255), np.full_like(r, 255)):
            enemy = ((r >= 145) & (r-g >= 70) &
                     (g <= r*.46) & (b <= g+25))
            danger = ((r >= 145) & (r-g >= 60) &
                      (g <= r*.7) & (b <= g+25))
            np.testing.assert_array_equal(
                enemy, aircraft._team_pixel_channels(r, g, b, "enemy"))
            np.testing.assert_array_equal(
                danger, aircraft._danger_pixel_channels(r, g, b))


class NegativeSignalBoundaries(unittest.TestCase):
    def test_bounded_signals_preserve_negative_scores_and_frame_rejection(self):
        # Exercise real template size parities: rounding a half-pixel center
        # after an odd crop translation can otherwise shift registration.
        _, negatives = aircraft._templates()
        self.assertTrue(negatives, "actual negative assets required")
        frame = np.random.default_rng(871).integers(
            0, 256, aircraft.FRAME_SHAPE, dtype=np.uint8)
        r, g, b = (frame[..., i].astype(np.int16) for i in range(3))
        for team, rendered in (("enemy", False), ("ally", False), ("ally", True)):
            team_templates = [t for t in negatives if t.team == team]
            self.assertTrue(team_templates)
            representatives = {}
            for template in sorted(team_templates, key=lambda t: t.shape.size):
                th, tw = template.shape.shape
                representatives.setdefault((th % 2, tw % 2), template)
            nh = max(t.shape.shape[0] for t in team_templates)
            nw = max(t.shape.shape[1] for t in team_templates)
            # The reference deliberately uses the original full-frame float
            # signal expression, independently of the channel crop helpers.
            a = frame.astype(np.float32)
            if rendered:
                full = np.maximum(0, a[..., 1] - a[..., 0] - 40)
            elif team == "enemy":
                full = np.maximum(0, a[..., 0] - np.maximum(a[..., 1], a[..., 2]))
            else:
                full = np.maximum(0, np.minimum(a[..., 1], a[..., 2]) - a[..., 0])
            del a
            for w, h in ((5, 5), (5, 6), (6, 5), (6, 6)):
                positions = ((0, 0), (1, 1), (5, 6), (0, 500), (900, 0),
                             (2560-w, 1600-h), (2559-w, 1599-h), (901, 501))
                for x, y in positions:
                    cx, cy = x+w/2, y+h/2
                    # Same bound used by detect; score/reference domains below
                    # are derived independently for every actual template.
                    nx = max(0, int(cx-nw/2)-4) & ~1
                    ny = max(0, int(cy-nh/2)-4) & ~1
                    right = min(frame.shape[1], int(cx+nw/2)+5)
                    bottom = min(frame.shape[0], int(cy+nh/2)+5)
                    channels = (r[ny:bottom, nx:right], g[ny:bottom, nx:right],
                                b[ny:bottom, nx:right])
                    bounded = (aircraft._rendered_signal_channels(*channels, team) if rendered
                               else aircraft._signal_channels(*channels, team))
                    np.testing.assert_array_equal(full[ny:bottom, nx:right], bounded)
                    for template in representatives.values():
                        with self.subTest(team=team, rendered=rendered, box=(x, y, w, h),
                                          template=template.asset, shape=template.shape.shape):
                            th, tw = template.shape.shape
                            fx, fy = int(round(cx-tw/2))-2, int(round(cy-th/2))-2
                            bx = int(round(x-nx+w/2-tw/2))-2
                            by = int(round(y-ny+h/2-th/2))-2
                            self.assertEqual((fx, fy), (bx+nx, by+ny))
                            full_valid = (fx >= 0 and fy >= 0 and
                                          fx+tw+4 <= 2560 and fy+th+4 <= 1600)
                            bounded_valid = (bx >= 0 and by >= 0 and
                                             bx+tw+4 <= bounded.shape[1] and
                                             by+th+4 <= bounded.shape[0])
                            self.assertEqual(full_valid, bounded_valid)
                            reference = aircraft._negative_score(
                                full, [x, y, w, h], team, (template,))
                            actual = aircraft._negative_score(
                                bounded, [x-nx, y-ny, w, h], team, (template,))
                            self.assertEqual(reference, actual)
                            if not full_valid:
                                self.assertEqual(-1.0, actual)


class SupportedCropBoundaries(unittest.TestCase):
    def template(self, padded_size=7):
        core = np.ones((3, 3), bool)
        return aircraft._Template("test", "test", "test", "enemy", 1.0,
                                  np.ones((padded_size, padded_size), np.float32), core, 3, 3)

    def test_equivalent_cores_keep_full_local_maximum_context(self):
        from dataclasses import replace
        crop = np.arange(400, dtype=np.float32).reshape(20, 20)
        colored = np.zeros_like(crop)
        colored[8:11, 9:12] = 1
        cache = {}
        template = self.template()
        for candidate in (template, replace(template, asset="other", shape=template.shape * 2)):
            supported, x, y = aircraft._supported_crop(crop, colored, candidate, cache)
            # The only feasible core origin is (9,8), hence score origin (7,6).
            # Its one-score-pixel halo must survive reuse by another template.
            self.assertEqual((6, 5), (x, y))
            np.testing.assert_array_equal(crop[5:14, 6:15], supported)

    def test_rejected_core_does_not_poison_another_core_or_proposal(self):
        from dataclasses import replace
        crop = np.zeros((20, 20), np.float32)
        colored = np.zeros_like(crop)
        colored[8:10, 9:12] = 1
        template = self.template()
        cache = {}
        self.assertIsNone(aircraft._supported_crop(crop, colored, template, cache))
        smaller_core = replace(template, core=np.ones((2, 3), bool), height=2)
        self.assertIsNotNone(aircraft._supported_crop(crop, colored, smaller_core, cache))
        colored[10, 9:12] = 1
        # Support caches are proposal-local; a fresh proposal must reconsider.
        self.assertIsNotNone(aircraft._supported_crop(crop, colored, template, {}))

    def test_equal_cores_with_different_score_domains_do_not_share_envelopes(self):
        crop = np.zeros((20, 20), np.float32)
        colored = np.zeros_like(crop)
        colored[15:18, 15:18] = 1
        cache = {}
        self.assertIsNotNone(aircraft._supported_crop(crop, colored, self.template(), cache))
        # A wider padded score domain cannot translate far enough to reach
        # that core, even though its unpadded mask is identical.
        self.assertIsNone(aircraft._supported_crop(crop, colored, self.template(9), cache))

    def test_shared_rectangle_envelopes_keep_core_and_proposal_domains(self):
        from dataclasses import replace
        shared_color = np.zeros((40, 45), np.float32)
        # Feasible origins reach all four edges of the shared rectangle-count
        # domain. Interior partial cores exercise different color thresholds.
        for y, x in ((0, 0), (0, 42), (37, 0), (37, 42), (9, 12), (24, 29)):
            shared_color[y:y+3, x:x+3] = 1
        shared_color[9, 12] = 0
        shared_color[24, 31] = 0
        full = self.template()
        missing_corner = full.core.copy()
        missing_corner[0, 0] = False
        other_corner = full.core.copy()
        other_corner[0, 2] = False
        candidates = (full, replace(full, core=missing_corner),
                      replace(full, core=other_corner), self.template(9))
        shared_cache = {}
        for dx, dy, width, height in ((0, 0, 20, 20), (10, 7, 20, 20),
                                      (25, 20, 20, 20), (0, 0, 45, 40)):
            crop = np.zeros((height, width), np.float32)
            colored = shared_color[dy:dy+height, dx:dx+width]
            cache = {"shared": (shared_color, shared_cache, dx, dy)}
            for template in candidates:
                with self.subTest(origin=(dx, dy), core=template.core.tolist(),
                                  score_domain=template.shape.shape):
                    self.assertEqual(
                        aircraft._supported_bounds(crop, colored, template, {}),
                        aircraft._supported_bounds(crop, colored, template, cache))

    def test_proposal_color_count_keeps_exact_threshold_and_shared_neighbors(self):
        core = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
        template = aircraft._Template("test", "test", "test", "enemy", 1.0,
                                      np.ones((7, 7), np.float32), core, 3, 3)
        shared = np.zeros((40, 40), np.float32)
        shared[28:31, 28:31] = 1  # Ink outside this proposal must not supply it.
        shared[8:11, 8:11] = core
        shared[8, 9] = shared[9, 8] = 0
        crop = shared[:20, :20]
        self.assertIsNone(aircraft._supported_bounds(
            crop, crop, template, {"shared": (shared, {}, 0, 0)}))
        shared[8, 9] = 1  # Exactly four of five core pixels satisfy .8 support.
        local = aircraft._supported_bounds(crop, crop, template, {})
        self.assertIsNotNone(local)
        self.assertEqual(local, aircraft._supported_bounds(
            crop, crop, template, {"shared": (shared, {}, 0, 0)}))

    def test_shared_rectangle_ink_does_not_establish_core_support(self):
        # The rectangle contains enough ink for the sparse core, but its only
        # pixel cannot cover that core's corner in the valid score domain.
        core = np.zeros((3, 3), bool)
        core[0, 0] = True
        template = aircraft._Template("test", "test", "test", "enemy", 1.0,
                                      np.ones((7, 7), np.float32), core, 3, 3)
        colored = np.zeros((20, 20), np.float32)
        colored[-1, -1] = 1
        cache = {"shared": (colored, {}, 0, 0)}
        self.assertIsNone(aircraft._supported_bounds(colored, colored, template, cache))

    def test_shared_envelope_keeps_gaps_and_exact_window_edges(self):
        # Two supported islands have an unsupported gap inside their envelope.
        # Local colored noise can meet the count gate without matching a core.
        colored = np.zeros((60, 60), np.float32)
        colored[8:11, 8:11] = 1
        colored[43:46, 43:46] = 1
        colored[22, 22:35] = 1
        template = self.template()
        shared_cache = {}
        windows = ((0, 0, 60, 60), (0, 0, 20, 20), (20, 20, 20, 20),
                   (35, 35, 25, 25), (10, 0, 20, 20), (6, 6, 20, 20))
        for dx, dy, width, height in windows:
            crop = colored[dy:dy+height, dx:dx+width]
            counts = cv2.matchTemplate(crop, template.color_core, cv2.TM_CCORR)
            ys, xs = np.where(counts[2:height-4, 2:width-4] >= template.required_color-.1)
            expected = None if not xs.size else (
                max(0, int(xs.min())-1), max(0, int(ys.min())-1),
                min(width, int(xs.max())+8), min(height, int(ys.max())+8))
            with self.subTest(origin=(dx, dy), size=(width, height)):
                self.assertEqual(expected, aircraft._supported_bounds(
                    crop, crop, template, {"shared": (colored, shared_cache, dx, dy)}))

    def test_zero_core_borders_preserve_threshold_and_edge_translations(self):
        rng = np.random.default_rng(764)
        for h, w in ((5, 6), (6, 5), (7, 7)):
            core = np.zeros((h, w), bool)
            core[1:-1, 2:-1] = True
            template = aircraft._Template("test", "test", "test", "enemy", 1.0,
                np.ones((h+4, w+4), np.float32), core, w, h)
            colored = (rng.random((32, 35)) > .75).astype(np.float32)
            for y, x in ((2, 2), (26-h, 29-w)):
                colored[y:y+h, x:x+w] = core
            # Independent full-core correlation supplies the untrimmed score
            # domain and the one-pixel local-maximum halo.
            counts = cv2.matchTemplate(colored, core.astype(np.float32), cv2.TM_CCORR)
            ys, xs = np.where(counts[2:32-h-1, 2:35-w-1] >= .8*core.sum()-.1)
            expected = (max(0, int(xs.min())-1), max(0, int(ys.min())-1),
                        min(35, int(xs.max())+w+5), min(32, int(ys.max())+h+5))
            with self.subTest(core_shape=(h, w)):
                self.assertEqual(expected, aircraft._supported_bounds(
                    colored, colored, template, {}))
                self.assertEqual(expected, aircraft._supported_bounds(
                    colored, colored, template, {"shared": (colored, {}, 0, 0)}))


if __name__ == "__main__":
    unittest.main()
