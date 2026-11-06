#!/usr/bin/env python3
"""Native HUD name text identity, independent of camera scale and replay truth.

The immutable anchor is the bottom text row above the distance-selected health
bar. Health, distance, class icons, reticle rings and screen proximity are not
identity evidence. Thresholds deliberately refuse weak or competing appearances;
this producer does not establish BowNet or firing acceptance.
"""
from __future__ import annotations

import hashlib
import json

import cv2
import numpy as np
from target_confirmation import confirm_target

METHOD = 'native_name_header_visible_contrast.v2'
MIN_SCORE = .92
MIN_MARGIN = .12


def _occluding_ring(rgb, ring):
    """Register an existing lock circle on its upper arc, away from name text."""
    cx, cy = ring['center']
    radius = ring['radius_px']
    angles = np.linspace(np.pi, 2*np.pi, 96, endpoint=False)
    cosine, sine = np.cos(angles), np.sin(angles)
    dx, dy, dr = np.meshgrid(np.arange(-5,6), np.arange(-5,6),
                             np.arange(-5,5.1,.5), indexing='ij')
    centres_x, centres_y = cx+dx.ravel(), cy+dy.ravel()
    radii = radius+dr.ravel()
    values = []
    for offset in (-4,0,4):
        xs = np.rint(centres_x[:,None]+(radii[:,None]+offset)*cosine).astype(int)
        ys = np.rint(centres_y[:,None]+(radii[:,None]+offset)*sine).astype(int)
        if xs.min()<0 or ys.min()<0 or xs.max()>=rgb.shape[1] or ys.max()>=rgb.shape[0]:
            return cx, cy, radius
        values.append(rgb[ys,xs].min(axis=2).astype(np.float32))
    scores = np.clip(values[1]-(values[0]+values[2])/2,-50,50).mean(axis=1)
    index = int(np.argmax(scores))
    return float(centres_x[index]), float(centres_y[index]), float(radii[index])


def header_signature(rgb, candidate, *, _register_ring=False):
    """Extract native text strokes, never rescale the HUD or use the hull."""
    frame = np.asarray(rgb)
    x, y, w, h = map(int, candidate['nameplate_bbox'])
    left, top, right, bottom = x + w // 2 - 150, y - 32, x + w // 2 + 150, y - 3
    evidence = dict(method=METHOD, bbox=[left, top, 300, 29], supported=False,
                    reason='header_clipped', stroke_pixels=0, glyph_components=0)
    if top < 0 or bottom > frame.shape[0] or right <= 0 or left >= frame.shape[1]:
        return None, evidence
    # Missing search margin is not missing name text. Keep native registration
    # and mark off-screen pixels unknown; never synthesize readable strokes.
    lo, hi = max(0, left), min(right, frame.shape[1])
    crop = np.zeros((29, 300, 3), dtype=frame.dtype)
    crop[:, lo-left:hi-left] = frame[top:bottom, lo:hi, :3]
    gray = crop.min(axis=2).astype(np.float32)
    # Exclude the confirmed circle from the local background as well: Gaussian
    # subtraction otherwise creates false dark gaps around occluded letters.
    # A confirmed UI circle is an occluder, never name evidence. Preserve its
    # unknown pixels for pairwise matching instead of reconstructing letters.
    visible = np.zeros(gray.shape, dtype=bool)
    visible[:, lo-left:hi-left] = True
    valid = visible.copy()
    ring = None
    if frame.shape == (1600, 2560, 3):
        ring = confirm_target(frame, [candidate]).get('ring')
        if ring is not None:
            yy, xx = np.ogrid[top:bottom, left:right]
            cx, cy, radius = (_occluding_ring(frame, ring) if _register_ring else
                              (*ring['center'], ring['radius_px']))
            valid &= abs(np.hypot(xx-cx, yy-cy)-radius) > 3
            evidence['occlusion'] = dict(kind='confirmed_ui_lock_ring',
                                         center=[cx,cy], radius_px=radius)
    weights = cv2.GaussianBlur(valid.astype(np.float32), (13, 13), 2)
    background = cv2.GaussianBlur(gray*valid, (13, 13), 2) / np.maximum(weights, 1e-6)
    mask = ((gray > 110) & (gray-background > 14) & valid).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    # A circle/antialiasing gap can split one glyph vertically. Merge only
    # its grouping geometry; supervision and matching retain original ink.
    grouped = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,1), np.uint8))
    _, _, grouped_stats, _ = cv2.connectedComponentsWithStats(grouped)
    glyphs = [(i, xx, yy, ww, hh, area) for i, (xx, yy, ww, hh, area) in enumerate(grouped_stats[1:], 1)
              if yy > 0 and (4 if _register_ring else 6) <= hh <= 22 and
              1 <= ww <= 26 and area >= 5]
    # Colons are two disconnected dots, not missing letters. Keep both native
    # components as one grouping box; matching below still uses original ink.
    dots = [tuple(map(int, s)) for s in stats[1:]
            if 1 <= s[2] <= 4 and 1 <= s[3] <= 4 and s[4] >= 2]
    for a in dots:
        for b in dots:
            ax, ay, aw, ah, aa = a
            bx, by, bw, bh, ba = b
            if (abs((ax+aw/2)-(bx+bw/2)) <= 1 and
                    4 <= by-ay <= 10 and 7 <= by+bh-ay <= 14):
                glyphs.append((-1, min(ax,bx), ay,
                               max(ax+aw,bx+bw)-min(ax,bx), by+bh-ay, aa+ba))
    # Antialiased adjacent letters can share a one-pixel bridge. Split only
    # grouping geometry at a narrow interior ink valley, retaining all original
    # pixels for the unchanged contrast/support identity comparison.
    if ring is not None:
        separated = []
        for glyph in glyphs:
            i,gx,gy,gw,gh,area = glyph
            if i > 0 and gw >= 12 and gw >= gh:
                columns = mask[gy:gy+gh,gx:gx+gw].sum(axis=0)
                cut = min(range(4,gw-3),key=lambda k:columns[k])
                if columns[cut] <= 1 and all(mask[gy:gy+gh,a:b].sum()>=10
                        for a,b in ((gx,gx+cut),(gx+cut,gx+gw))):
                    separated.extend(((i,gx,gy,cut,gh,area),
                                      (i,gx+cut,gy,gw-cut,gh,area)))
                    continue
            separated.append(glyph)
        glyphs = separated
    # A flat baseline is interrupted by the lock circle; use the row's vertical
    # overlap. Four full-height letters can survive when both colons are hidden;
    # the unchanged ink/span and pairwise contrast checks still apply.
    groups = [[g for g in glyphs if g[2] <= baseline and baseline-g[2] <= 18
               and g[2]+g[4] >= baseline-8]
              for baseline in sorted({g[2]+g[4] for g in glyphs})
              if baseline >= mask.shape[0]-10]
    viable = [group for group in groups if
              (len(group) >= 5 or (ring is not None and len(group) >= 4 and
                                   all(g[4] >= 6 for g in group))) and
              (not _register_ring or sum(g[4] >= 6 for g in group) >= 3)]
    if not viable:
        if ring is not None and not _register_ring:
            return header_signature(rgb,candidate,_register_ring=True)
        evidence['reason'] = 'insufficient_name_text'
        return None, evidence
    group = max(viable, key=lambda g: (len(g), sum(v[5] for v in g)))
    x0, y0 = min(g[1] for g in group), min(g[2] for g in group)
    x1, y1 = max(g[1] + g[3] for g in group), max(g[2] + g[4] for g in group)
    # Keep fragments cut by the circle, but only inside the independently
    # supported text row. Never reconstruct pixels hidden by the circle.
    baseline = max(g[2]+g[4] for g in group)
    fragments = [i for i, (xx, yy, ww, hh, area) in enumerate(stats[1:], 1)
                 if x0-12 <= xx and xx+ww <= x1+12 and yy >= baseline-18
                 and yy+hh <= baseline+4 and ww <= 26 and area >= 2]
    strokes = np.isin(labels, fragments).astype(np.uint8)
    ys, xs = np.nonzero(strokes)
    if not len(xs):
        evidence['reason'] = 'insufficient_name_text'
        return None, evidence
    x0, x1 = int(xs.min()), int(xs.max())+1
    y0, y1 = int(ys.min()), int(ys.max())+1
    if x0 <= lo-left or x1 >= hi-left:
        evidence['reason'] = 'name_text_touches_frame_edge'
        return None, evidence
    support = int(strokes.sum())
    evidence.update(bbox=[int(left+x0), int(top+y0), int(x1-x0), int(y1-y0)], stroke_pixels=support,
                    glyph_components=len(group), reason='insufficient_name_text')
    if not (30 <= x1-x0 <= 240 and 7 <= y1-y0 <= 22 and support >= 75):
        return None, evidence
    # Normalize subpixel antialiasing within observed pixels only. The 0.5 px
    # kernel does not rescale HUD text or fill an unknown circle segment.
    weights = cv2.GaussianBlur(valid.astype(np.float32), (3, 3), .5)
    gray = cv2.GaussianBlur(gray*valid, (3, 3), .5) / np.maximum(weights, 1e-6)
    text_band = np.zeros_like(valid)
    text_band[max(0, baseline-18):min(29, baseline+4),
              max(0, x0-1):min(300, x1+1)] = True
    valid &= text_band
    evidence.update(supported=True, reason='native_name_text',
                    pixel_sha256=hashlib.sha256(crop.tobytes()+valid.tobytes()).hexdigest())
    return dict(strokes=strokes, valid=valid, gray=gray), evidence


def appearance_score(a, b):
    """Contrast agreement of common visible name pixels, never circle pixels.

    Registration is two-stage. The name crop is anchored on the health-bar box,
    which jitters by several pixels between frames, so a two-pixel search fails
    to find the same name. A coarse two-pixel grid over +-8 px finds the basin
    and a quarter-pixel refinement of +-1 px inside it restores the original
    subpixel precision. Thresholds and the per-window conflict veto are
    unchanged; only which translations are tried changed.
    """
    if a is None or b is None:
        return 0.0
    shape = (a['gray'].shape[1], a['gray'].shape[0])

    def score_at(dx, dy):
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        vb = cv2.warpAffine(b['valid'].astype(np.uint8), matrix, shape,
                            flags=cv2.INTER_NEAREST).astype(bool)
        common = a['valid'] & vb
        ac = a['strokes']*common
        bc = cv2.warpAffine(b['strokes'], matrix, shape,
                            flags=cv2.INTER_NEAREST)*common
        # Absolute common support is required on both sides. Occluded
        # letters are unknown, not evidence of a different name.
        if min(int(ac.sum()), int(bc.sum())) < 75:
            return -1.0
        if any(np.ptp(np.nonzero(strokes)[1]) < 29 for strokes in (ac, bc)):
            return -1.0
        aa = a['gray']
        bb = cv2.warpAffine(b['gray'], matrix, shape)
        weights = cv2.GaussianBlur(common.astype(np.float32), (13, 13), 2)
        denominator = np.maximum(weights, 1e-6)
        ca = np.maximum(aa-cv2.GaussianBlur(aa*common, (13, 13), 2)/denominator, 0)*common
        cb = np.maximum(bb-cv2.GaussianBlur(bb*common, (13, 13), 2)/denominator, 0)*common
        norm = float(np.linalg.norm(ca)*np.linalg.norm(cb))
        if norm <= 0:
            return -1.0
        similarity = min(1.0, float(np.sum(ca*cb))/norm)
        if similarity < MIN_SCORE:
            return -1.0
        # A changed letter must not be hidden by the rest of a long name.
        # Only windows with enough observed ink on both sides may veto;
        # an occluded or blank window cannot claim a glyph disagreement.
        columns = np.nonzero(np.any((ac > 0) | (bc > 0), axis=0))[0]
        for x in range(max(0, int(columns[0])-7), int(columns[-1])+1, 4):
            if min(ac[:, x:x+8].sum(), bc[:, x:x+8].sum()) < 10:
                continue
            pa, pb = ca[:, x:x+8], cb[:, x:x+8]
            local_norm = float(np.linalg.norm(pa)*np.linalg.norm(pb))
            if local_norm <= 0 or float(np.sum(pa*pb))/local_norm < .75:
                return -1.0
        return similarity

    # Baseline: the original subpixel registration. Always evaluated, so this
    # function can never score lower than it did before the wider search.
    best = 0.0
    for dy in np.arange(-2, 2.01, .25):
        for dx in np.arange(-2, 2.01, .25):
            best = max(best, score_at(dx, dy))
    # Wider basin: the bar-anchored crop can shift by several pixels between
    # frames. Only a coarse grid is affordable there; refine where it wins.
    coarse = None
    for dy in np.arange(-8, 8.01, 2.0):
        for dx in np.arange(-8, 8.01, 2.0):
            value = score_at(dx, dy)
            if coarse is None or value > coarse[2]:
                coarse = (dx, dy, value)
    if coarse is not None and coarse[2] > best:
        cx, cy = coarse[0], coarse[1]
        for dy in np.arange(cy-1, cy+1.0001, .25):
            for dx in np.arange(cx-1, cx+1.0001, .25):
                best = max(best, score_at(dx, dy))
    return best

class CrossZoomIdentity:
    """One source generation and immutable selected-name appearance per action."""
    def __init__(self):
        self.source = None
        self.anchor = None
        self.anchor_evidence = None
        self.anchor_registered = None
        self.target_id = None
        self.generation = 0

    def reset(self):
        self.source = self.anchor = self.anchor_evidence = self.target_id = None
        self.anchor_registered = None

    def observe_source(self, source, *, battle_active):
        source = json.dumps(source, sort_keys=True, default=str) if source is not None else None
        lost = not battle_active or source is None or (self.source is not None and source != self.source)
        had_anchor = self.anchor is not None
        if lost:
            self.reset()
        if battle_active and source is not None:
            self.source = source
        return lost and had_anchor

    def identify(self, rgb, candidates, *, selected=None, frame_source=None):
        # Caller-supplied identities cannot override current image evidence.
        rows = [{k: v for k, v in c.items() if k not in
                 ('target_id', 'track_id', 'association_id', 'identity', 'identity_key')}
                for c in candidates]
        signatures = [header_signature(rgb, row) for row in rows]
        evidence = dict(method=METHOD, source=self.source, frame_source=frame_source,
                        replay_truth_used=False, minimum_score=MIN_SCORE, minimum_margin=MIN_MARGIN,
                        anchor=self.anchor_evidence, score=None, margin=None, status='unconfirmed')
        for row, (_, detail) in zip(rows, signatures):
            row['identity_evidence'] = detail
        if self.source is None:
            evidence['reason'] = 'current_battle_source_missing'
            return rows, evidence
        establishing = self.anchor is None
        if len(rows) > 1 and any(signature is None for signature, _ in signatures):
            evidence['reason'] = 'insufficient_competing_name_text'
            return rows, evidence
        if establishing:
            indices = [i for i, c in enumerate(candidates) if selected is not None and
                       c['nameplate_bbox'] == selected['nameplate_bbox']]
            if len(indices) != 1:
                evidence['reason'] = 'distance_selected_anchor_missing_or_ambiguous'
                return rows, evidence
            index = indices[0]
            anchor, detail = signatures[index]
            if anchor is None:
                evidence['reason'] = 'insufficient_anchor_name_text'
                return rows, evidence
            scores = [appearance_score(anchor, signature) for signature, _ in signatures]
        else:
            scores = [appearance_score(self.anchor, signature) for signature, _ in signatures]
            # Hough's lower arc can merge with text while still passing the
            # glyph-count gate. Retry the same immutable anchor and current
            # name with upper-arc registration; never replace the anchor with
            # a subsequent frame or relax contrast/support thresholds.
            if max(scores, default=0.0) < MIN_SCORE and self.anchor_registered is not None:
                for i, row in enumerate(rows):
                    signature, detail = header_signature(rgb, row, _register_ring=True)
                    score = appearance_score(self.anchor_registered, signature)
                    if score > scores[i]:
                        scores[i] = score
                        row['identity_evidence'] = dict(detail, registration='upper_arc')
            index = max(range(len(rows)), key=lambda i: scores[i]) if rows else None
        best = scores[index] if index is not None else 0.0
        runner = max((s for i, s in enumerate(scores) if i != index), default=0.0)
        margin = best - runner
        evidence.update(score=best, margin=margin, candidate_scores=scores)
        if best < MIN_SCORE or margin < MIN_MARGIN:
            evidence['reason'] = 'appearance_ambiguous' if margin < MIN_MARGIN else 'appearance_not_matched'
            return rows, evidence
        if establishing:
            self.generation += 1
            self.anchor, self.anchor_evidence = anchor.copy(), dict(detail, frame_source=frame_source)
            self.anchor_registered = header_signature(rgb, rows[index], _register_ring=True)[0]
            self.target_id = f'header-{self.generation}-{detail["pixel_sha256"][:20]}'
        evidence.update(status='confirmed', reason='unique_native_name_appearance',
                        target_id=self.target_id, anchor=self.anchor_evidence,
                        matched_index=index, established=establishing)
        rows[index].update(target_id=self.target_id, identity_evidence={
            **rows[index]['identity_evidence'], **evidence})
        return rows, evidence


def main(argv=None):
    """Read saved RGB frames only; never attach to a game or create input."""
    import argparse
    from pathlib import Path
    from PIL import Image
    from main_view_perception import detect_nameplates
    from target_range_selection import inspect_candidates
    from fire_hud import FireHudReader

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wide', type=Path, required=True)
    parser.add_argument('--scoped', type=Path, required=True)
    parser.add_argument('--max-range-km', type=float, required=True)
    args = parser.parse_args(argv)
    producer = CrossZoomIdentity()
    producer.observe_source({'offline_pair': [str(args.wide), str(args.scoped)]}, battle_active=True)
    report = dict(mode='offline_saved_pixels', input_executed=False, frames=[])
    hud_reader = FireHudReader()
    for path, scoped in ((args.wide, False), (args.scoped, True)):
        frame = np.asarray(Image.open(path).convert('RGB'))
        if frame.shape == (1440, 2560, 3):
            frame = np.pad(frame, ((80, 80), (0, 0), (0, 0)))
        view = hud_reader.detect(frame).get('aiming_view') or {}
        scope_confirmed = view.get('status') == 'observed' and view.get('mode') == 'scope_ruler'
        candidates = [dict(c, trackable=True) for c in detect_nameplates(frame, scope_observed=scoped)]
        selection = inspect_candidates(frame, candidates, max_range_km=args.max_range_km)
        rows, evidence = producer.identify(frame, candidates,
                                          selected=None if scoped else selection['range_selected'],
                                          frame_source=str(path))
        matched = next((row for row in rows if row.get('target_id') == producer.target_id
                        and producer.target_id is not None), None)
        range_confirmed = bool(matched is not None and inspect_candidates(
            frame, [matched], max_range_km=args.max_range_km)['range_selected'] is not None)
        report['frames'].append(dict(frame=str(path), identity=evidence, candidates=rows,
                                     range_selection=selection, range_confirmed=range_confirmed,
                                     scope_ruler_confirmed=scope_confirmed))
    report['cross_zoom_scope_confirmed'] = (not report['frames'][0]['scope_ruler_confirmed']
                                            and report['frames'][1]['scope_ruler_confirmed'])
    print(json.dumps(report, indent=2))
    return 0 if (report['cross_zoom_scope_confirmed'] and
                 all(row['identity']['status'] == 'confirmed' and row['range_confirmed']
                     for row in report['frames'])) else 1


if __name__ == '__main__':
    raise SystemExit(main())
