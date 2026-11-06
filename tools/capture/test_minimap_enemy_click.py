#!/usr/bin/env python3
"""Offline tests for tools/capture/minimap_enemy_click.py.

Runs with plain `python3` (numpy only, plus whatever decodes PNG: Pillow when
present, ffmpeg otherwise) -- the icon fixtures and the synthetic icons are
both rasterised from the game's own SVG glyph paths, so nothing here needs the
running client:

    python3 tools/capture/test_minimap_enemy_click.py

What is covered, and what is not, is spelled out in the module docstring of
minimap_enemy_click.py: only the battleship icon could be cut from real client
frames, so the other four classes are proven against synthetic icons.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import minimap_enemy_click as M

FIXTURES = HERE / 'minimap_fixtures'
MAP4_FRAME = M.ROOT / 'state' / 'cn360-flow-20261021' / 'map4' / 'frame-000.png'
SEA = (6, 40, 60)
GLYPH_RED = (225, 55, 55)


def synthetic_icon(ship_class, scale, angle, canvas=None, noise=2.0, seed=2):
    """Rasterise one of the five glyphs at `scale` px/unit, rotated by `angle`.

    Same polygon tables the classifier uses for its templates, but sampled on a
    plain pixel grid: this is the synthetic half of the evidence.
    """
    canvas = canvas or int(round(16 * scale)) + 12
    radians = math.radians(angle)
    cos, sin = math.cos(radians), math.sin(radians)
    rows, columns = np.mgrid[0:canvas, 0:canvas]
    x = (columns - canvas / 2 + .5) / scale
    y = (rows - canvas / 2 + .5) / scale
    # rotate the sample point back into the icon's own design frame
    design_x = cos * x + sin * y + 4.5
    design_y = -sin * x + cos * y + 8.
    glyph = np.zeros((canvas, canvas), bool)
    for polygon in M.ICON_POLYGONS[ship_class]:
        points = np.asarray(polygon, np.float64)
        inside = np.zeros((canvas, canvas), bool)
        for index in range(len(points)):
            x1, y1 = points[index]
            x2, y2 = points[(index + 1) % len(points)]
            crosses = (y1 > design_y) != (y2 > design_y)
            with np.errstate(divide='ignore', invalid='ignore'):
                crossing = (x2 - x1) * (design_y - y1) / (y2 - y1) + x1
            inside ^= crosses & (design_x < crossing)
        glyph |= inside
    image = np.zeros((canvas, canvas, 3), np.uint8)
    image[:] = SEA
    image[glyph] = GLYPH_RED
    rng = np.random.default_rng(seed)
    noisy = image.astype(np.int16) + rng.normal(0, noise, image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def fixture(name):
    return M.read_png(FIXTURES / name)


class SyntheticClassIcons(unittest.TestCase):
    """One test per class: five classes, two scales, four headings each."""

    scales = (1.4, 2.0)
    angles = (0, 37, 118, 249)

    def check(self, ship_class):
        seen = []
        for scale in self.scales:
            for angle in self.angles:
                patch = synthetic_icon(ship_class, scale, angle)
                label, score = M.classify_icon(patch)
                seen.append((scale, angle, label, round(score, 3)))
                self.assertEqual(label, ship_class,
                                 f'{ship_class} at scale {scale}, heading {angle} '
                                 f'classified as {label}: {seen}')

    def test_cv(self):
        self.check('cv')

    def test_bb(self):
        self.check('bb')

    def test_ca(self):
        self.check('ca')

    def test_dd(self):
        self.check('dd')

    def test_ss(self):
        # The submarine is the destroyer triangle plus a bar across the stern;
        # below ~1.2 px per design unit that 1 px gap closes and the two
        # classes are genuinely the same picture, so only >=1.4 is asserted.
        self.check('ss')

    def test_rejects_when_heading_is_ambiguous(self):
        """A hull seen end-on has no class evidence: it must not be guessed."""
        for ship_class in M.CLASSES:
            patch = synthetic_icon(ship_class, 2.0, 0, canvas=26)
            squashed = np.repeat(patch[:, :4], 3, axis=1)[:, :26]
            label, _ = M.classify_icon(squashed)
            self.assertEqual(label, 'unknown', f'{ship_class} squashed was accepted')


class UnknownShapes(unittest.TestCase):
    """Non-ship shapes must come back as 'unknown', never as a class."""

    def setUp(self):
        self.canvas = np.zeros((26, 26, 3), np.uint8)
        self.canvas[:] = SEA

    def test_empty_sea(self):
        self.assertEqual(M.classify_icon(self.canvas)[0], 'unknown')

    def test_noisy_sea(self):
        rng = np.random.default_rng(5)
        noisy = np.clip(self.canvas.astype(int) + rng.normal(0, 3, self.canvas.shape),
                        0, 255).astype(np.uint8)
        self.assertEqual(M.classify_icon(noisy)[0], 'unknown')

    def test_red_disc(self):
        disc = self.canvas.copy()
        rows, columns = np.mgrid[0:26, 0:26]
        disc[(columns - 13) ** 2 + (rows - 13) ** 2 <= 36] = GLYPH_RED
        self.assertEqual(M.classify_icon(disc)[0], 'unknown')

    def test_thin_red_line(self):
        line = self.canvas.copy()
        line[6:20, 11:14] = GLYPH_RED
        self.assertEqual(M.classify_icon(line)[0], 'unknown')

    def test_grey_ring_and_white_arrow(self):
        ring = self.canvas.copy()
        rows, columns = np.mgrid[0:26, 0:26]
        distance = (columns - 13) ** 2 + (rows - 13) ** 2
        ring[(distance >= 16) & (distance <= 36)] = (170, 170, 170)
        self.assertEqual(M.classify_icon(ring)[0], 'unknown')
        self.assertEqual(M.classify_icon(fixture('map4-own.png'))[0], 'unknown')
        self.assertEqual(M.classify_icon(fixture('course6-own.png'))[0], 'unknown')

    def test_rejects_non_rgb_input(self):
        with self.assertRaises(ValueError):
            M.classify_icon(np.zeros((26, 26), np.uint8))
        with self.assertRaises(ValueError):
            M.classify_icon(np.zeros((6, 6, 3), np.uint8))


class CoursePoint(unittest.TestCase):
    """The waypoint sits on the own->enemy line, short of the enemy, on the map."""

    rect = (546, 74, 1364, 926)

    def test_fraction_is_measured_from_own(self):
        own, enemy = (600., 300.), (1000., 700.)
        point = M.course_point(own, enemy, self.rect, fraction=.95)
        self.assertEqual(point, (980, 680))
        to_own = math.dist(point, own)
        to_enemy = math.dist(point, enemy)
        self.assertGreater(to_own, to_enemy)
        self.assertAlmostEqual(to_enemy / (to_own + to_enemy), .05, places=4)

    def test_minimum_gap_keeps_the_click_off_the_icon(self):
        own, enemy = (600., 300.), (650., 300.)
        point = M.course_point(own, enemy, self.rect, fraction=.99)
        self.assertEqual(point, (632, 300))
        self.assertGreaterEqual(math.dist(point, enemy), M.MIN_GAP_PX)
        self.assertGreater(math.dist(point, own), 0.)

    def test_short_leg_does_not_overshoot(self):
        point = M.course_point((600., 300.), (610., 300.), self.rect)
        self.assertLessEqual(math.dist(point, (610., 300.)), 5.0001)
        self.assertGreaterEqual(point[0], 600)

    def test_result_stays_inside_the_map(self):
        point = M.course_point((1350., 90.), (1360., 80.), self.rect, min_gap_px=0)
        x0, y0, x1, y1 = self.rect
        self.assertTrue(x0 < point[0] < x1 and y0 < point[1] < y1)

    def test_sea_callback_pulls_the_point_back(self):
        own, enemy = (600., 300.), (1000., 300.)
        # Everything east of x=880 is "not sea" (a nameplate, say).
        point = M.course_point(own, enemy, self.rect, fraction=.95,
                               is_sea=lambda candidate: candidate[0] <= 880)
        self.assertLessEqual(point[0], 880)
        self.assertGreater(point[0], 600)

    def test_rejects_impossible_arguments(self):
        with self.assertRaises(ValueError):
            M.course_point((600., 300.), (1000., 700.), self.rect, fraction=0.)
        with self.assertRaises(ValueError):
            M.course_point((600., 300.), (600., 300.), self.rect)
        with self.assertRaises(ValueError):
            M.course_point((100., 300.), (1000., 700.), self.rect)


class RealFrameFixtures(unittest.TestCase):
    """Crops cut from the three saved client frames, classified as recorded."""

    def test_manifest_crops_match_their_labels(self):
        manifest = json.loads((FIXTURES / 'manifest.json').read_text())
        self.assertGreaterEqual(len(manifest['crops']), 6)
        for entry in manifest['crops']:
            patch = fixture(entry['file'])
            self.assertEqual(patch.dtype, np.uint8)
            self.assertEqual(tuple(patch.shape[1::-1]), tuple(entry['size']))
            label, score = M.classify_icon(patch)
            self.assertEqual(label, entry['expect'],
                             f"{entry['name']} ({entry['source']} {entry['crop_box']}) "
                             f'classified as {label} with score {score:.3f}')

    def test_enemy_icons_score_high(self):
        for name in ('map4-enemy.png', 'map2-enemy.png', 'course6-enemy.png',
                     'course6-minimap-enemy.png'):
            label, score = M.classify_icon(fixture(name))
            self.assertEqual(label, 'bb', name)
            self.assertGreater(score, .85, name)

    def test_source_frames_are_still_the_ones_recorded(self):
        manifest = json.loads((FIXTURES / 'manifest.json').read_text())
        import hashlib
        for name, path in manifest['frames'].items():
            source = M.ROOT / path
            if not source.is_file():
                continue
            recorded = next(entry['source_sha256'] for entry in manifest['crops']
                            if entry['source'] == path)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            self.assertEqual(digest, recorded, f'{name} changed after the crops were cut')


class FrameDetection(unittest.TestCase):
    """Marker and icon detection on the real tactical-map frame."""

    def setUp(self):
        if not MAP4_FRAME.is_file():
            self.skipTest(f'{MAP4_FRAME} is not present')

    def test_own_marker_and_enemy_icon(self):
        rgb = M.read_png(MAP4_FRAME)
        own, reason = M.find_own_marker(rgb, M.DEFAULT_MAP_RECT)
        self.assertIsNotNone(own, reason)
        self.assertLess(math.dist(own['pixel'], (697, 375)), 4)
        enemies = M.find_enemy_icons(rgb, M.DEFAULT_MAP_RECT)
        accepted = [entry for entry in enemies if entry['ship_class'] != 'unknown']
        self.assertEqual(len(accepted), 1, enemies)
        self.assertEqual(accepted[0]['ship_class'], 'bb')
        self.assertLess(math.dist(accepted[0]['pixel'], (937, 639)), 4)

    def test_click_point_lands_on_flat_sea(self):
        rgb = M.read_png(MAP4_FRAME)
        own, _ = M.find_own_marker(rgb, M.DEFAULT_MAP_RECT)
        enemies = [entry for entry in M.find_enemy_icons(rgb, M.DEFAULT_MAP_RECT)
                   if entry['ship_class'] != 'unknown']
        point = M.course_point(own['pixel'], enemies[0]['pixel'], M.DEFAULT_MAP_RECT,
                               is_sea=lambda candidate: M.sea_flat(rgb, candidate))
        self.assertTrue(M.sea_flat(rgb, point), point)
        self.assertGreater(math.dist(point, enemies[0]['pixel']), M.MIN_GAP_PX)


def synthetic_route(route_deg=40., enemy_deg=0., route_len=140., with_route=True,
                    size=(420, 320), own=(120., 160.)):
    """A tactical-map frame with a drawn autopilot route, in the game's style.

    Own marker: a filled neutral disc (same bright/dark-contrast signature as
    the arrow, and too large to be a waypoint marker). Route: 3x3 white dashes
    every 12 px, ending at a hollow 15x15 diamond. The enemy is a red disc; the
    callers pass its exact pixel so the synthetic frame does not have to fake a
    classified icon.
    """
    width, height = size
    frame = np.zeros((height, width, 3), np.uint8)
    frame[:] = SEA
    rows, columns = np.mgrid[0:height, 0:width]
    frame[(columns - own[0]) ** 2 + (rows - own[1]) ** 2 <= 9 ** 2] = (225, 225, 225)
    if with_route:
        radians = math.radians(route_deg)
        cosine, sine = math.cos(radians), math.sin(radians)
        for radius in np.arange(25., route_len - 18., 12.):
            centre_x = own[0] + cosine * radius
            centre_y = own[1] + sine * radius
            frame[int(round(centre_y)) - 1:int(round(centre_y)) + 2,
                  int(round(centre_x)) - 1:int(round(centre_x)) + 2] = (235, 235, 235)
        waypoint = (own[0] + cosine * route_len, own[1] + sine * route_len)
        distance = (np.abs(columns - waypoint[0]) + np.abs(rows - waypoint[1]))
        frame[(distance >= 5) & (distance <= 7)] = (230, 230, 230)
    enemy_radians = math.radians(enemy_deg)
    # Keep the red disc clear of the waypoint diamond, even when the two
    # bearings coincide (a real icon sits on the ship, not on the waypoint).
    enemy = (own[0] + math.cos(enemy_radians) * (route_len + 40.),
             own[1] + math.sin(enemy_radians) * (route_len + 40.))
    frame[(columns - enemy[0]) ** 2 + (rows - enemy[1]) ** 2 <= 6 ** 2] = (210, 60, 60)
    return frame, own, enemy


class CourseDirection(unittest.TestCase):
    """The drawn route is measured on a post-click frame and compared to the bearing."""

    rect = (10, 10, 410, 310)

    def test_measures_the_drawn_angle(self):
        frame, own, enemy = synthetic_route(route_deg=40., enemy_deg=0.)
        measured = M.measure_course(frame, self.rect, own_pixel=own, enemy_pixel=enemy)
        self.assertIsNotNone(measured)
        self.assertAlmostEqual(measured['angle_deg'], 40., delta=4.)
        self.assertAlmostEqual(measured['course_deg'], 40., delta=4.)
        self.assertAlmostEqual(measured['expected_deg'], 0., delta=2.)
        self.assertEqual(measured['dashes'] >= M.CHAIN_MIN_DASHES, True)
        self.assertAlmostEqual(M.course_direction(frame, self.rect, own_pixel=own,
                                                  enemy_pixel=enemy),
                               measured['angle_deg'], places=6)

    def test_hint_finds_the_own_marker(self):
        frame, own, enemy = synthetic_route(route_deg=110., enemy_deg=110.)
        measured = M.measure_course(frame, self.rect, own_hint=own, enemy_pixel=enemy)
        self.assertIsNotNone(measured)
        self.assertLess(measured['angle_deg'], 5.)

    def test_threshold_is_fifteen_degrees_inclusive(self):
        self.assertTrue(M.angle_within(0.))
        self.assertTrue(M.angle_within(14.999, 15.))
        self.assertFalse(M.angle_within(15.001, 15.))
        self.assertFalse(M.angle_within(None))

    def test_route_free_frame_is_unmeasurable(self):
        frame, own, enemy = synthetic_route(with_route=False)
        self.assertIsNone(M.course_direction(frame, self.rect, own_pixel=own,
                                             enemy_pixel=enemy))


class VerifyAndCorrect(unittest.TestCase):
    """The verification loop: one correction, guards, never a blind second click."""

    rect = (10, 10, 410, 310)

    def plan(self, own, enemy, click):
        return dict(own=own, enemy=enemy, click_frame=click)

    def capture_list(self, frames):
        queue = list(frames)

        def capture():
            return queue.pop(0)

        return capture

    def frame_block(self, frame, own, enemy, verified=True, age_s=0.):
        return dict(rgb=frame, map_rect=self.rect, own_hint=own, enemy_hint=enemy,
                    map_verified=verified, age_s=age_s, path='synthetic')

    def test_angle_above_tolerance_corrects_once_and_improves(self):
        # Error model: the game draws the route 45 degrees clockwise of where
        # the click pointed (a constant rotation about the own ship). The first
        # click points east, so the first drawn route is at 45 degrees. The
        # corrected click must point 45 degrees anti-clockwise, which under the
        # same model draws the route at the enemy bearing (0 degrees).
        before, own, enemy = synthetic_route(route_deg=45., enemy_deg=0.)
        clicks = []

        def click(point):
            clicks.append(point)
            return dict(reached=True, window=[int(point[0]), int(point[1])])

        def capture():
            if not clicks:
                return self.frame_block(before, own, enemy)
            bearing = math.degrees(math.atan2(clicks[0][1] - own[1],
                                              clicks[0][0] - own[0]))
            drawn = bearing + 45.
            frame, _, _ = synthetic_route(route_deg=drawn, enemy_deg=0.)
            return self.frame_block(frame, own, enemy)

        verification = M.verify_and_correct(
            self.plan(own, enemy, (own[0] + 140., own[1])),
            capture, click, iterations=2)
        self.assertEqual(verification['state'], 'ok', verification)
        self.assertTrue(verification['corrected'])
        self.assertEqual(verification['clicks'], 2)
        self.assertEqual(len(clicks), 1)
        self.assertAlmostEqual(verification['measurements'][0]['angle_deg'], 45., delta=4.)
        self.assertLess(verification['angle_deg'], 8.)
        self.assertLess(verification['measurements'][1]['angle_deg'],
                        verification['measurements'][0]['angle_deg'])
        # The drawn route landed 45 degrees clockwise of the requested bearing,
        # so the correction rotates the click 45 degrees the other way.
        bearing = math.degrees(math.atan2(clicks[0][1] - own[1],
                                          clicks[0][0] - own[0])) % 360.
        self.assertLess(abs((bearing + 45. + 180.) % 360. - 180.), 8.,
                        f'corrected click bearing {bearing:.1f} deg')

    def test_unmeasurable_frame_never_sends_a_second_click(self):
        frame, own, enemy = synthetic_route(with_route=False)
        clicks = []
        verification = M.verify_and_correct(
            self.plan(own, enemy, (own[0] + 100., own[1])),
            self.capture_list([self.frame_block(frame, own, enemy)]),
            lambda point: clicks.append(point) or dict(reached=True),
            iterations=1)
        self.assertEqual(verification['state'], 'unverifiable', verification)
        self.assertFalse(verification['corrected'])
        self.assertEqual(verification['clicks'], 1)
        self.assertEqual(clicks, [])

    def test_clicks_stay_capped_at_two(self):
        wrong, own, enemy = synthetic_route(route_deg=45., enemy_deg=0.)
        clicks = []
        verification = M.verify_and_correct(
            self.plan(own, enemy, (own[0] + 140., own[1])),
            self.capture_list([self.frame_block(wrong, own, enemy)] * 3),
            lambda point: clicks.append(point) or dict(reached=True),
            iterations=3)
        self.assertEqual(verification['clicks'], 2)
        self.assertEqual(len(clicks), 1)
        self.assertEqual(verification['corrected'], True)
        self.assertEqual(verification['state'], 'failed', verification)

    def test_refused_frame_does_not_click(self):
        wrong, own, enemy = synthetic_route(route_deg=45., enemy_deg=0.)
        clicks = []
        verification = M.verify_and_correct(
            self.plan(own, enemy, (own[0] + 140., own[1])),
            self.capture_list([self.frame_block(wrong, own, enemy, verified=False)]),
            lambda point: clicks.append(point) or dict(reached=True),
            iterations=1)
        self.assertEqual(verification['state'], 'refused', verification)
        self.assertEqual(clicks, [])

    def test_stale_frame_is_refused(self):
        wrong, own, enemy = synthetic_route(route_deg=45., enemy_deg=0.)
        clicks = []
        verification = M.verify_and_correct(
            self.plan(own, enemy, (own[0] + 140., own[1])),
            self.capture_list([self.frame_block(wrong, own, enemy, age_s=99.)]),
            lambda point: clicks.append(point) or dict(reached=True),
            iterations=1, max_age_s=3.)
        self.assertEqual(verification['state'], 'refused', verification)
        self.assertEqual(clicks, [])


class RealCourseFrames(unittest.TestCase):
    """Stored client frames: what the measurement actually proves on real pixels."""

    COURSE3 = M.ROOT / 'state' / 'cn360-flow-20261021' / 'course3' / 'after_click.png'
    COURSE2 = M.ROOT / 'state' / 'cn360-flow-20261021' / 'course2' / 'after_click.png'
    STANDOFF = (M.ROOT / 'state' / 'standoff-session' / 'course-toward-enemy'
                / 'after_click.png')
    LM3 = M.ROOT / 'state' / 'cn360-flow-20261021' / 'lm3' / 'frame-000.png'
    RECT = (726, 194, 1815, 1328)

    def test_course3_route_is_measured_as_wildly_wrong(self):
        if not self.COURSE3.is_file():
            self.skipTest(f'{self.COURSE3} is not present')
        rgb = M.read_png(self.COURSE3)
        self.assertEqual(M.content_rect(rgb), (0, 96, 2560, 1504))
        measured = M.measure_course(rgb, self.RECT, own_pixel=(946, 698),
                                    enemy_pixel=(1247, 946))
        self.assertIsNotNone(measured)
        self.assertGreater(measured['angle_deg'], 100., measured)
        self.assertAlmostEqual(measured['waypoint'][0], 933.6, delta=3.)
        self.assertAlmostEqual(measured['waypoint'][1], 591.4, delta=4.)

    def test_course2_route_is_measured_as_wrong(self):
        if not self.COURSE2.is_file():
            self.skipTest(f'{self.COURSE2} is not present')
        rgb = M.read_png(self.COURSE2)
        measured = M.measure_course(rgb, self.RECT, own_pixel=(1303, 542),
                                    enemy_pixel=(1247, 946))
        self.assertIsNotNone(measured)
        self.assertGreater(measured['angle_deg'], 40., measured)

    def test_standoff_route_is_measured_as_aligned(self):
        if not self.STANDOFF.is_file():
            self.skipTest(f'{self.STANDOFF} is not present')
        rgb = M.read_png(self.STANDOFF)
        measured = M.measure_course(rgb, self.RECT, own_pixel=(1520, 1030),
                                    enemy_pixel=(1617, 1025))
        self.assertIsNotNone(measured)
        self.assertLess(measured['angle_deg'], 5., measured)
        self.assertTrue(M.angle_within(measured['angle_deg']))

    def test_hud_banner_does_not_win_the_own_marker(self):
        lm2 = M.ROOT / 'state' / 'cn360-flow-20261021' / 'lm2' / 'frame-000.png'
        if not lm2.is_file():
            self.skipTest(f'{lm2} is not present')
        rgb = M.read_png(lm2)
        own, reason = M.find_own_marker(rgb, (545, 76, 1361, 945))
        self.assertIsNotNone(own, reason)
        # A white banner ("侦查距离降低到15公里！") sits at y=197 and used to win
        # the fill vote; the own arrow is at (977, 341).
        self.assertLess(math.dist(own['pixel'], (977, 341)), 4, own)

    def test_frame_without_a_route_is_unmeasurable(self):
        if not self.LM3.is_file():
            self.skipTest(f'{self.LM3} is not present')
        rgb = M.read_png(self.LM3)
        self.assertIsNone(M.course_direction(rgb, M.DEFAULT_MAP_RECT,
                                             own_pixel=(977, 341),
                                             enemy_pixel=(846, 622)))


class DryRunCli(unittest.TestCase):
    """The documented command line works and never clicks in a dry run."""

    def setUp(self):
        if not MAP4_FRAME.is_file():
            self.skipTest(f'{MAP4_FRAME} is not present')

    def test_dry_run_prints_and_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(HERE / 'minimap_enemy_click.py'),
                 '--frame', str(MAP4_FRAME), '--out', directory],
                capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((Path(directory) / 'report.json').read_text())
        self.assertFalse(report['execute'])
        self.assertEqual(report['status'], 'dry-run')
        self.assertIsNone(report['failure_reason'])
        self.assertEqual(report['frame_source']['size'], [1924, 1058])
        self.assertEqual(report['chosen_target']['ship_class'], 'bb')
        x0, y0, x1, y1 = report['map']['rect']
        click_x, click_y = report['click']['window']
        self.assertTrue(x0 < click_x < x1 and y0 < click_y < y1)
        self.assertIn('would click window', result.stdout)
        self.assertIn('class=bb', result.stdout)


if __name__ == '__main__':
    unittest.main(verbosity=2)
