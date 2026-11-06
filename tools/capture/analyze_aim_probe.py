#!/usr/bin/env python3
"""Analyze saved bounded aim probes without launching a game or sending input.

Usage: .venv/bin/python tools/capture/analyze_aim_probe.py --report
state/aimprobe-paused-live-v2/report.json --out state/aimprobe-screen-response.json
Repeat --report for repeated horizontal/vertical probes in one unchanged view.
Only local scene response is measured. No output is a FireControl calibration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

# Half-open native-frame rectangles. The marker belt includes reticle, labels,
# torpedo warnings and healthbars. The lower region stops before weapon HUD.
REGIONS = {'scene': (384, 320, 2176, 1220),
           'lower_scene': (384, 900, 2100, 1220)}
EXCLUSIONS = ((0, 0, 2560, 320), (0, 1220, 2560, 1600),
              (0, 0, 384, 1600), (2176, 0, 2560, 1600),
              (0, 650, 2560, 830), (770, 450, 900, 555),
              (1510, 450, 1670, 555))
POLICY = {'native_shape': [1600, 2560, 3], 'native_dtype': 'uint8',
          'regions_xyxy': REGIONS, 'excluded_xyxy': EXCLUSIONS,
          'feature_patch_change_mean_min': 1.0, 'forward_backward_max_px': 2.0,
          'max_track_displacement_px': 400.0, 'min_tracks': 12,
          'min_spatial_tiles': 3, 'coherent_radius_px': 3.0,
          'min_coherent_fraction': .65, 'min_motion_px': 1.0,
          'baseline_max_changed_fraction': .002,
          'response_min_changed_fraction': .01,
          'rollback_max_changed_fraction': .005,
          'rollback_max_response_mae_ratio': .1,
          'fixed_scene': {'orb_features': 4000, 'orb_fast_threshold': 8,
                          'ratio_test': .78, 'ransac_reprojection_px': 3.0,
                          'min_matches': 20, 'min_inliers': 12,
                          'min_inlier_fraction': .6, 'max_residual_p90_px': 2.0,
                          'min_spatial_tiles': 3}}

FIXED_SCENE_POLICY = {'orb_features': 4000, 'orb_fast_threshold': 8,
                      'ratio_test': .78, 'ransac_reprojection_px': 3.0,
                      'min_matches': 20, 'min_inliers': 12,
                      'min_inlier_fraction': .6, 'max_residual_p90_px': 2.0,
                      'min_spatial_tiles': 3, 'min_motion_px': 1.0}

# This native-frame anchor was observed in saved v3 evidence. It is supplied
# geometry, not an image-confirmed reticle or camera/zoom observation.
RETICLE_LOCAL_POLICY = {
    'anchor_xy_px': [1280.5, 805.0], 'roi_xyxy': [1000, 560, 1560, 1050],
    'anchor_status': 'supplied_saved_v3_geometry_not_detected_per_frame',
    'reticle_exclusion_xyxy': [1269, 796, 1292, 814],
    'mask_boundary_erosion_px': 9, 'highpass_sigma_px': 3.0,
    'min_highpass_rms': .005, 'min_gradient_eigenvalue_ratio': .02,
    'min_phase_response': .5, 'max_secondary_phase_peak_ratio': .8,
    'phase_peak_exclusion_radius_px': 5.0, 'min_ecc_correlation': .9,
    'max_phase_ecc_disagreement_px': 1.0, 'min_overlap_fraction': .5,
    'min_overlap_ncc': .9, 'max_translation_px': 160.0,
    'max_forward_inverse_error_px': 5.0, 'max_zero_net_error_px': 1.0,
    'min_response_px': 1.0, 'ecc_iterations': 60, 'ecc_epsilon': 1e-5}
POLICY['reticle_local'] = RETICLE_LOCAL_POLICY


def reticle_local_motion(before, after):
    """Diagnose translation on a fixed, HUD-masked ROI around a supplied anchor.

    Phase correlation supplies a translation seed; ECC can refine translation
    only (no rotation, affine or homography degrees of freedom). Independent
    pair estimates and sequence checks are required before accepting diagnostic
    samples. Even a successful fit cannot establish camera mode, zoom, actual
    reticle detection, rigid scene geometry or post-input settling.
    """
    policy = RETICLE_LOCAL_POLICY
    rect = policy['roi_xyxy']
    out = {'valid': False, 'sufficient': False, 'calibration_valid': False,
           'world_calibration_valid': False,
           'semantics': 'hud_masked_reticle_roi_translation_screen_pixels_only',
           'method': 'highpass_phase_correlation_cross_checked_translation_only_ecc',
           'anchor_xy_px': policy['anchor_xy_px'], 'anchor_status': policy['anchor_status'],
           'roi_xyxy': rect, 'excluded_xyxy': [list(r) for r in EXCLUSIONS] +
           [policy['reticle_exclusion_xyxy']], 'policy': policy}
    if not valid_image(before) or not valid_image(after):
        return dict(out, reason='requires two native uint8 1600x2560 RGB images')
    x0, y0, x1, y1 = rect
    mask = region_mask(rect)[y0:y1, x0:x1].copy()
    a, b, c, d = policy['reticle_exclusion_xyxy']
    mask[max(0, b-y0):min(y1-y0, d-y0), max(0, a-x0):min(x1-x0, c-x0)] = 0
    radius = policy['mask_boundary_erosion_px']
    mask = cv2.erode(mask, np.ones((2*radius+1, 2*radius+1), np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0)
    included = mask != 0
    out.update(valid=True, included_pixels=int(included.sum()),
               mask_boundary_erosion_px=radius)
    filtered = []
    texture = []
    for image in (before, after):
        gray = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY).astype(np.float32)/255
        high = gray-cv2.GaussianBlur(gray, (0, 0), policy['highpass_sigma_px'])
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)[included]
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)[included]
        eigen = np.linalg.eigvalsh([[np.mean(gx*gx), np.mean(gx*gy)],
                                   [np.mean(gx*gy), np.mean(gy*gy)]])
        texture.append({'highpass_rms': float(np.sqrt(np.mean(high[included]**2))),
                        'gradient_eigenvalue_ratio': float(max(0, eigen[0])/max(eigen[1], 1e-12))})
        filtered.append(high*(mask/255).astype(np.float32))
    out['texture_before_after'] = texture
    if any(t['highpass_rms'] < policy['min_highpass_rms'] or
           t['gradient_eigenvalue_ratio'] < policy['min_gradient_eigenvalue_ratio'] for t in texture):
        return dict(out, reason='blank, weak or directionally ambiguous local texture')
    aa, bb = filtered
    window = cv2.createHanningWindow((x1-x0, y1-y0), cv2.CV_32F)
    # OpenCV applies the window in place; preserve unwindowed arrays for ECC.
    shift, response = cv2.phaseCorrelate(aa.copy(), bb.copy(), window)
    phase = np.asarray(shift, dtype=float)
    out['phase'] = {'screen_dx_dy_px': phase.tolist(), 'response': float(response)}
    if not np.isfinite(phase).all() or not math.isfinite(response):
        out['phase'] = None
        return dict(out, reason='nonfinite phase-correlation result')
    # A competing peak outside the main lobe rejects periodic/repeated texture.
    cross = np.fft.fft2(bb*window)*np.conj(np.fft.fft2(aa*window))
    cross /= np.maximum(np.abs(cross), 1e-12)
    correlation = np.fft.ifft2(cross).real
    py, px = np.unravel_index(np.argmax(correlation), correlation.shape)
    yy, xx = np.indices(correlation.shape)
    dx = np.minimum(abs(xx-px), correlation.shape[1]-abs(xx-px))
    dy = np.minimum(abs(yy-py), correlation.shape[0]-abs(yy-py))
    secondary = correlation[np.hypot(dx, dy) > policy['phase_peak_exclusion_radius_px']].max()
    ratio = float(max(0, secondary)/max(correlation[py, px], 1e-12))
    out['phase']['secondary_peak_ratio'] = ratio
    warp = np.float32([[1, 0, phase[0]], [0, 1, phase[1]]])
    try:
        cc, warp = cv2.findTransformECC(
            aa, bb, warp, cv2.MOTION_TRANSLATION,
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
             policy['ecc_iterations'], policy['ecc_epsilon']), mask, 5)
    except cv2.error as error:
        return dict(out, reason='translation ECC did not converge', ecc_error=str(error))
    displacement = warp[:, 2].astype(float)
    if not np.isfinite(displacement).all() or not math.isfinite(cc):
        return dict(out, reason='nonfinite translation ECC result')
    disagreement = float(np.linalg.norm(displacement-phase))
    out['ecc'] = {'motion_model': 'translation_only', 'correlation': float(cc),
                  'screen_dx_dy_px': displacement.tolist(),
                  'phase_disagreement_px': disagreement}
    # Evaluate only source pixels whose translated destination is also unmasked.
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    aligned = cv2.warpAffine(bb, warp, (x1-x0, y1-y0), flags=flags)
    aligned_mask = cv2.warpAffine(mask, warp, (x1-x0, y1-y0),
                                 flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
    overlap = included & (aligned_mask != 0)
    fraction = float(overlap.sum()/included.sum())
    ncc = None
    if overlap.sum() >= 2:
        left, right = aa[overlap].astype(float), aligned[overlap].astype(float)
        left -= left.mean()
        right -= right.mean()
        denominator = float(np.linalg.norm(left)*np.linalg.norm(right))
        if denominator > 1e-12:
            ncc = float(np.dot(left, right)/denominator)
    out['match'] = {'overlap_pixels': int(overlap.sum()), 'overlap_fraction': fraction,
                    'overlap_ncc': ncc}
    failed = []
    for passed, name in (
            (response >= policy['min_phase_response'], 'phase_response'),
            (ratio <= policy['max_secondary_phase_peak_ratio'], 'phase_peak_uniqueness'),
            (cc >= policy['min_ecc_correlation'], 'ecc_correlation'),
            (disagreement <= policy['max_phase_ecc_disagreement_px'], 'phase_ecc_agreement'),
            (np.linalg.norm(phase) <= policy['max_translation_px'] and
             np.linalg.norm(displacement) <= policy['max_translation_px'], 'translation_bound'),
            (fraction >= policy['min_overlap_fraction'], 'overlap_fraction'),
            (ncc is not None and ncc >= policy['min_overlap_ncc'], 'overlap_ncc')):
        if not passed:
            failed.append(name)
    out.update(screen_dx_px=float(displacement[0]), screen_dy_px=float(displacement[1]),
               sufficient=not failed, failed_bounds=failed,
               reason='bounded local translation diagnostic' if not failed else
                      'local translation failed quality bounds: '+', '.join(failed))
    return out


def reticle_local_consistency(pairs, source_valid):
    """Check independently fitted forward, inverse, baseline and zero-net pairs."""
    names = ('baseline->before', 'before->after', 'after->rollback', 'before->rollback')
    local = {name: pairs.get(name, {}).get('reticle_local', {}) for name in names}
    out = {'accepted': False, 'calibration_valid': False,
           'independently_estimated_pairs': list(names), 'failed_conditions': []}
    failed = out['failed_conditions']
    if not source_valid:
        failed.append('source_or_transport_evidence_invalid')
    for name, value in local.items():
        if not value.get('sufficient'):
            failed.append(name+': translation_quality_rejected_or_missing')
    if any(not all(number(value.get(k)) for k in ('screen_dx_px', 'screen_dy_px'))
           for value in local.values()):
        return out
    vectors = {name: np.asarray([v['screen_dx_px'], v['screen_dy_px']])
               for name, v in local.items()}
    inverse_error = float(np.linalg.norm(vectors['before->after']+vectors['after->rollback']))
    zero_error = float(np.linalg.norm(vectors['before->rollback']))
    baseline_error = float(np.linalg.norm(vectors['baseline->before']))
    motion = float(np.linalg.norm(vectors['before->after']))
    out.update(forward_inverse_sum_px=(vectors['before->after']+vectors['after->rollback']).tolist(),
               forward_inverse_error_px=inverse_error, zero_net_error_px=zero_error,
               baseline_motion_px=baseline_error, forward_motion_px=motion)
    policy = RETICLE_LOCAL_POLICY
    if inverse_error > policy['max_forward_inverse_error_px']:
        failed.append('forward_inverse_disagreement')
    if zero_error > policy['max_zero_net_error_px']:
        failed.append('zero_net_not_restored')
    if baseline_error > policy['max_zero_net_error_px']:
        failed.append('no_input_baseline_moved')
    if motion < policy['min_response_px']:
        failed.append('no_resolved_input_response')
    out['accepted'] = not failed
    return out


def valid_image(image):
    return isinstance(image, np.ndarray) and image.shape == (1600, 2560, 3) and image.dtype == np.uint8


def fixed_scene_motion(before, after, rect=None):
    """Estimate static-scene screen displacement with masked ORB/RANSAC.

    The source and destination features must agree on one homography over
    several scene tiles.  HUD geometry is excluded before feature detection;
    changed-patch optical flow is intentionally not used here because it
    cannot measure a camera transform when the scene remains visually stable.
    This reports screen displacement only and never implies world angles.
    """
    rect = rect or REGIONS['scene']
    out = {'valid': False, 'sufficient': False, 'world_calibration_valid': False,
           'semantics': 'static_scene_screen_pixels', 'rect_xyxy': list(rect)}
    if not valid_image(before) or not valid_image(after):
        return dict(out, reason='requires two native uint8 1600x2560 RGB images')
    mask = region_mask(rect)
    gray_a = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(after, cv2.COLOR_RGB2GRAY)
    detector = cv2.ORB_create(nfeatures=FIXED_SCENE_POLICY['orb_features'],
                              fastThreshold=FIXED_SCENE_POLICY['orb_fast_threshold'],
                              edgeThreshold=15, patchSize=31)
    key_a, desc_a = detector.detectAndCompute(gray_a, mask)
    key_b, desc_b = detector.detectAndCompute(gray_b, mask)
    out.update(valid=True, keypoints_before=len(key_a), keypoints_after=len(key_b))
    if desc_a is None or desc_b is None:
        return dict(out, reason='static scene has no ORB descriptors', matches=0, inliers=0)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(desc_a, desc_b, k=2)
    good = [first for first, second in pairs
            if first.distance < FIXED_SCENE_POLICY['ratio_test'] * second.distance]
    out['matches'] = len(good)
    if len(good) < FIXED_SCENE_POLICY['min_matches']:
        return dict(out, reason='too few ratio-tested static-scene matches', inliers=0)
    starts = np.float32([key_a[m.queryIdx].pt for m in good])
    ends = np.float32([key_b[m.trainIdx].pt for m in good])
    homography, inlier_mask = cv2.findHomography(
        starts, ends, cv2.RANSAC, FIXED_SCENE_POLICY['ransac_reprojection_px'])
    if homography is None or inlier_mask is None:
        return dict(out, reason='static-scene homography RANSAC failed', inliers=0)
    inliers = inlier_mask.ravel().astype(bool)
    projected = cv2.perspectiveTransform(starts.reshape(-1, 1, 2), homography).reshape(-1, 2)
    residual = np.linalg.norm(projected - ends, axis=1)
    inlier_starts = starts[inliers]
    tiles = {(int(x // 200), int(y // 160)) for x, y in inlier_starts}
    sample_points = np.float32([[[rect[0] + (rect[2] - rect[0]) * fx,
                                  rect[1] + (rect[3] - rect[1]) * fy]]
                                for fy in (.25, .5, .75) for fx in (.25, .5, .75)])
    sample_projected = cv2.perspectiveTransform(sample_points, homography).reshape(-1, 2)
    displacement = sample_projected - sample_points.reshape(-1, 2)
    median_displacement = np.median(displacement, axis=0)
    p90 = float(np.quantile(residual[inliers], .9)) if np.any(inliers) else math.inf
    inlier_fraction = float(np.mean(inliers))
    sufficient = bool(
        int(np.count_nonzero(inliers)) >= FIXED_SCENE_POLICY['min_inliers'] and
        inlier_fraction >= FIXED_SCENE_POLICY['min_inlier_fraction'] and
        len(tiles) >= FIXED_SCENE_POLICY['min_spatial_tiles'] and
        p90 <= FIXED_SCENE_POLICY['max_residual_p90_px'])
    out.update(inliers=int(np.count_nonzero(inliers)),
               inlier_fraction=inlier_fraction, spatial_tiles=len(tiles),
               residual_p90_px=p90, median_screen_dx_px=float(median_displacement[0]),
               median_screen_dy_px=float(median_displacement[1]),
               displacement_samples_px=displacement.tolist(), sufficient=sufficient,
               homography=homography.tolist(),
               reason='repeatable static-scene homography' if sufficient else
                      'static-scene matches fail spatial consensus or residual bound')
    return out

def region_mask(rect):
    mask = np.zeros((1600, 2560), np.uint8)
    x0, y0, x1, y1 = rect
    mask[y0:y1, x0:x1] = 255
    for a, b, c, d in EXCLUSIONS:
        mask[b:d, a:c] = 0
    return mask


def measure_pair(before, after, rect):
    """Changed scene patches only: stationary overlays cannot vote for zero flow.

    This deliberately censors stationary features. The surviving median is local
    moving-scene flow, never an estimate of whole-camera or world displacement.
    Difference metrics retain all unmasked pixels, including stationary scene.
    """
    out = {'valid': False, 'sufficient': False, 'world_calibration_valid': False,
           'semantics': 'local_changed_scene_screen_pixels', 'rect_xyxy': list(rect)}
    if not valid_image(before) or not valid_image(after):
        return dict(out, reason='requires two native uint8 1600x2560 RGB images')
    mask = region_mask(rect)
    a = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY)
    b = cv2.cvtColor(after, cv2.COLOR_RGB2GRAY)
    difference = cv2.absdiff(before, after)
    pixels = difference[mask != 0]
    out.update(valid=True, mean_abs_rgb=float(pixels.mean()),
               fraction_pixels_delta_over_10=float(np.mean(pixels.max(axis=1) > 10)),
               included_pixels=int(len(pixels)))
    # A source feature needs a changed 9x9 neighborhood. This excludes unchanged
    # HUD even inside the conservative geometry, without treating motion as aim.
    change = cv2.boxFilter(difference.max(axis=2).astype(np.float32), -1, (9, 9))
    eligible = mask & np.where(change > POLICY['feature_patch_change_mean_min'], 255, 0).astype(np.uint8)
    small_a = cv2.resize(a, (1280, 800), interpolation=cv2.INTER_AREA)
    small_b = cv2.resize(b, (1280, 800), interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(eligible, (1280, 800), interpolation=cv2.INTER_NEAREST)
    # Equal feature budgets per tile prevent one bright label/landmark dominating.
    points = []
    for y in range(rect[1] // 2, rect[3] // 2, 80):
        for x in range(rect[0] // 2, rect[2] // 2, 100):
            height = min(80, rect[3] // 2-y)
            width = min(100, rect[2] // 2-x)
            found = cv2.goodFeaturesToTrack(small_a[y:y+height, x:x+width], 8, .02, 6,
                                          mask=small_mask[y:y+height, x:x+width])
            if found is not None:
                points.extend(found.reshape(-1, 2) + [x, y])
    out['candidate_tracks'] = len(points)
    if not points:
        return dict(out, reason='no changed scene features', tracks=0)
    points = np.asarray(points, np.float32).reshape(-1, 1, 2)
    moved, status, _ = cv2.calcOpticalFlowPyrLK(small_a, small_b, points, None,
                                              winSize=(31, 31), maxLevel=5)
    if moved is None:
        return dict(out, reason='forward optical flow failed', tracks=0)
    back, reverse, _ = cv2.calcOpticalFlowPyrLK(small_b, small_a, moved, None,
                                             winSize=(31, 31), maxLevel=5)
    if back is None:
        return dict(out, reason='reverse optical flow failed', tracks=0)
    starts = points.reshape(-1, 2) * 2
    ends = moved.reshape(-1, 2) * 2
    delta = ends-starts
    good = (status.ravel() != 0) & (reverse.ravel() != 0)
    good &= np.linalg.norm((back-points).reshape(-1, 2)*2, axis=1) < 2
    good &= np.linalg.norm(delta, axis=1) <= 400
    good &= np.isfinite(ends).all(axis=1)
    safe_ends = np.nan_to_num(ends, nan=-1, posinf=-1, neginf=-1)
    ix = np.clip(safe_ends[:, 0], 0, 2559).astype(int)
    iy = np.clip(safe_ends[:, 1], 0, 1599).astype(int)
    good &= (ends[:, 0] >= 0) & (ends[:, 0] < 2560) & (ends[:, 1] >= 0) & (ends[:, 1] < 1600)
    good &= mask[iy, ix] != 0
    # Endpoint patches must change too; a moving patch landing on a static label
    # is not eligible merely because its starting patch changed.
    good &= change[iy, ix] > POLICY['feature_patch_change_mean_min']
    stationary = np.linalg.norm(delta, axis=1) < POLICY['min_motion_px']
    out['stationary_tracks_excluded'] = int(np.count_nonzero(good & stationary))
    good &= ~stationary
    starts, ends, delta = starts[good], ends[good], delta[good]
    out['tracks'] = len(delta)
    out['track_evidence'] = [{'from_px': p.tolist(), 'to_px': q.tolist()} for p, q in zip(starts, ends)]
    if len(delta) < 12:
        return dict(out, reason='fewer than 12 forward/backward-consistent scene tracks')
    median = np.median(delta, axis=0)
    residual = np.linalg.norm(delta-median, axis=1)
    coherent = residual < 3
    tiles = {(int(p[0]//200), int(p[1]//160)) for p in starts[coherent]}
    observed = bool(np.linalg.norm(median) >= 1 and coherent.mean() >= .65 and len(tiles) >= 3)
    out.update(median_scene_dx_px=float(median[0]), median_scene_dy_px=float(median[1]),
               coherent_fraction=float(coherent.mean()), coherent_spatial_tiles=len(tiles),
               residual_p90_px=float(np.quantile(residual, .9)), sufficient=observed,
               reason='bounded coherent local scene response' if observed else 'local flow stationary, spatially narrow or perspective-dependent')
    return out


def scene_motion(before, after):
    """Compatibility entry point for the bounded capture probe."""
    return measure_pair(before, after, REGIONS['scene'])


def file_identity(path):
    data = path.read_bytes()
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def vector(value):
    return isinstance(value, list) and len(value) == 2 and all(number(v) for v in value)


def analyze_report(path):
    path = path.resolve()
    report = json.loads(path.read_text())
    result = {'report': file_identity(path), 'identity': report.get('identity'),
              'capture_source': report.get('source'), 'images': {}, 'pairs': {},
              'rejections': [], 'local_response_valid': False,
              'calibration_valid': False, 'world_calibration_valid': False}
    reject = result['rejections']
    if report.get('status') != 'completed' or report.get('execute') is not True or report.get('identity_verified') is not True:
        reject.append('report is not a completed identity-verified executed probe')
    identity, source = report.get('identity', {}), report.get('source', {})
    if not isinstance(identity, dict) or any(not identity.get(k) for k in ('pid', 'display', 'start_ticks')):
        reject.append('recorded original game identity is missing')
    if not isinstance(source, dict) or any(k not in source for k in ('id', 'process_id', 'serial', 'name')):
        reject.append('recorded capture node/process/serial identity is missing')
    if not isinstance(identity, dict):
        identity = {}
    if not isinstance(source, dict):
        source = {}
    checks = report.get('focus_checks', [])
    if not checks or any(c.get('pid') != identity.get('pid') for c in checks) or len({c.get('window') for c in checks}) != 1:
        reject.append('recorded focus/window identity is inconsistent')
    if report.get('rollback_input_sent') is not True or report.get('unreleased_own_keys') != [] or report.get('cleanup_errors'):
        reject.append('rollback transport or owned-key cleanup was not confirmed')
    if report.get('limits', {}).get('no_mouse_buttons') is not True:
        reject.append('probe did not record the no-mouse-buttons transport bound')
    seconds = report.get('limits', {}).get('seconds')
    elapsed = report.get('elapsed_s')
    if not number(seconds) or not 0 < seconds <= 30 or not number(elapsed) or not 0 <= elapsed <= seconds:
        reject.append('recorded probe exceeds or lacks its finite <=30s total bound')
    snapshots = {}
    arrays = {}
    # Use the actual HUD consumer; saved key events do not confirm a camera view.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'replay'))
    from fire_hud import FireHudReader
    hud_reader = FireHudReader()
    for snapshot in report.get('snapshots', []):
        name = snapshot.get('name')
        if name in snapshots:
            reject.append(f'duplicate snapshot name: {name}')
            continue
        snapshots[name] = snapshot
        recorded = Path(snapshot.get('path', ''))
        image_path = recorded if recorded.is_absolute() else path.parent / recorded
        evidence = {'recorded': snapshot, 'file': None, 'valid': False}
        result['images'][name] = evidence
        if not image_path.is_file():
            reject.append(f'image is missing: {name}: {image_path}')
            continue
        evidence['file'] = file_identity(image_path)
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None or image.shape not in ((1600, 2560, 3), (1600, 2560, 4)) or image.dtype != np.uint8:
            reject.append(f'image is not native uint8 1600x2560 color: {name}')
            continue
        # Capture-host PNGs carry alpha (including transparent letterboxing).
        # Analyze stored color bytes, as the live probe's IMREAD_COLOR path does;
        # do not composite alpha onto a new background or alter source evidence.
        conversion = cv2.COLOR_BGRA2RGB if image.shape[2] == 4 else cv2.COLOR_BGR2RGB
        arrays[name] = cv2.cvtColor(image, conversion)
        evidence['valid'] = True
        hud = hud_reader.detect(arrays[name])
        evidence['firing_view'] = {key: hud[key] for key in
                                  ('weapon', 'reticle', 'battle', 'aiming_view')}
        if snapshot.get('status') != 'completed' or snapshot.get('node') != source.get('id') or str(snapshot.get('process_id')) != str(source.get('process_id')):
            reject.append(f'snapshot capture identity mismatch: {name}')
        if snapshot.get('bytes') != evidence['file']['bytes']:
            reject.append(f'snapshot byte count mismatch: {name}')
        if not all(number(snapshot.get(k)) for k in ('request_elapsed_s', 'completed_elapsed_s')) or snapshot['request_elapsed_s'] > snapshot['completed_elapsed_s']:
            reject.append(f'snapshot request/completion interval invalid: {name}')
    for a, b in [('baseline', 'before'), ('before', 'after'), ('after', 'rollback'), ('before', 'rollback')]:
        if a in arrays and b in arrays:
            result['pairs'][f'{a}->{b}'] = {
                name: dict(measure_pair(arrays[a], arrays[b], rect),
                           fixed_scene=fixed_scene_motion(arrays[a], arrays[b], rect))
                for name, rect in REGIONS.items()}
            result['pairs'][f'{a}->{b}']['reticle_local'] = reticle_local_motion(arrays[a], arrays[b])
        else:
            reject.append(f'missing required image pair: {a}->{b}')
    movements = [item for item in report.get('inputs', []) if 'relative' in item]
    relative = movements[0].get('relative') if movements else None
    result['input_relative'] = relative
    if len(movements) != 2 or any(not vector(item.get('relative')) or max(abs(v) for v in item['relative']) > 200 for item in movements):
        reject.append('requires exactly one nonzero bounded relative input and its inverse')
    elif relative == [0, 0] or movements[1]['relative'] != [-v for v in relative]:
        reject.append('relative rollback is not the exact inverse')
    else:
        for index, (a, b) in enumerate([('before', 'after'), ('after', 'rollback')]):
            elapsed = movements[index].get('elapsed_s')
            start = snapshots.get(a, {}).get('completed_elapsed_s')
            end = snapshots.get(b, {}).get('request_elapsed_s')
            if not all(number(v) for v in (elapsed, start, end)) or not start <= elapsed <= end:
                reject.append(f'recorded input is not bounded between snapshot intervals: {a}->{b}')
        baseline_end = snapshots.get('baseline', {}).get('completed_elapsed_s')
        before_start = snapshots.get('before', {}).get('request_elapsed_s')
        rollback_end = snapshots.get('rollback', {}).get('completed_elapsed_s')
        before_end = snapshots.get('before', {}).get('completed_elapsed_s')
        if not all(number(v) for v in (baseline_end, before_start, rollback_end, before_end)):
            reject.append('missing baseline/no-input interval evidence')
        elif baseline_end > before_start or any(number(item.get('elapsed_s')) and
                (baseline_end < item['elapsed_s'] < before_start or before_end < item['elapsed_s'] < rollback_end)
                for item in report.get('inputs', []) if 'relative' not in item):
            reject.append('view-changing key input or invalid baseline interval in measurement window')
    result['source_valid'] = not reject
    firing_frames = [result['images'].get(name, {}).get('firing_view', {})
                     for name in ('baseline', 'before', 'after', 'rollback')]
    ruler_confirmed = all(frame.get('aiming_view', {}).get('mode') == 'scope_ruler'
                          for frame in firing_frames)
    centers = [frame.get('reticle', {}).get('center') for frame in firing_frames]
    reticle_confirmed = all(vector(center) for center in centers)
    result['firing_view_evidence'] = {
        'required_frames': ['baseline', 'before', 'after', 'rollback'],
        'scope_ruler_present_in_all_frames': ruler_confirmed,
        'selected_gun_in_all_frames': all(frame.get('weapon', {}).get('kind') == 'gun'
                                          for frame in firing_frames),
        'reticle_observed_in_all_frames': reticle_confirmed,
        'reticle_centers_px': centers,
        'max_reticle_center_change_px': (float(max(np.linalg.norm(np.asarray(center)-centers[0])
                                                   for center in centers))
                                         if reticle_confirmed else None),
        'zoom_confirmed': None, 'stable': None,
        'target_anchor_association': 'unknown', 'calibration_valid': False}
    checks = {}
    for region in REGIONS:
        pairs = result['pairs']
        if not all(k in pairs for k in ('baseline->before', 'before->after', 'before->rollback')):
            continue
        baseline = pairs['baseline->before'][region]
        response = pairs['before->after'][region]
        rollback = pairs['before->rollback'][region]
        stable = baseline['fraction_pixels_delta_over_10'] <= .002 and baseline['mean_abs_rgb'] <= .5
        changed = response['fraction_pixels_delta_over_10'] >= .01 and response['mean_abs_rgb'] > max(.5, 5*baseline['mean_abs_rgb'])
        recovered = rollback['fraction_pixels_delta_over_10'] <= .005 and rollback['mean_abs_rgb'] <= max(.5, .1*response['mean_abs_rgb'])
        checks[region] = {'no_input_baseline_stable': stable, 'response_above_baseline': changed,
                          'rollback_image_recovered': recovered,
                          'rollback_to_response_mae_ratio': rollback['mean_abs_rgb']/max(response['mean_abs_rgb'], 1e-9),
                          'local_flow_coherent': response['sufficient'],
                          'reversible_scene_change_valid': bool(stable and changed and recovered and result['source_valid'])}
    result['checks'] = checks
    result['reticle_local_checks'] = reticle_local_consistency(result['pairs'], result['source_valid'])
    result['local_response_valid'] = any(c['reversible_scene_change_valid'] for c in checks.values())
    if not result['local_response_valid']:
        result['reticle_local_checks']['accepted'] = False
        result['reticle_local_checks']['failed_conditions'].append('reversible_scene_image_checks_failed')
    result['interpretation'] = ('reversible local scene/perspective response' if result['local_response_valid'] else 'local response rejected or unobserved')
    result['timestamp_semantics'] = 'host snapshot request/completion intervals; not exposure timestamps or settling samples'
    return result


def aggregate(results):
    """Describe repeatability without fitting across unconfirmed camera states."""
    missing = ['image-confirmed unchanged camera mode and zoom are unavailable; '
               'responses cannot establish one shared input mapping']
    keys = [json.dumps([r['identity'], r['capture_source']], sort_keys=True) for r in results]
    if len(set(keys)) != 1:
        missing.append('all sequences must retain one game generation and capture node/serial')
    hashes = [r['report'].get('sha256') for r in results]
    if any(h is None for h in hashes) or len(set(hashes)) != len(hashes):
        missing.append('repeatability requires distinct source reports, not reused or unreadable evidence')
    image_sequences = [tuple(r.get('images', {}).get(name, {}).get('file', {}).get('sha256')
                             if r.get('images', {}).get(name, {}).get('file') else None
                             for name in ('baseline', 'before', 'after', 'rollback')) for r in results]
    if len(image_sequences) != len(set(image_sequences)):
        missing.append('reused identical image sequences cannot establish independent repeated observations')
    samples = []
    for r in results:
        flow = r['pairs'].get('before->after', {}).get('lower_scene', {})
        if (r.get('source_valid') and
                r.get('checks', {}).get('lower_scene', {}).get('reversible_scene_change_valid')
                and flow.get('sufficient') and vector(r.get('input_relative'))):
            samples.append((r['input_relative'], [flow['median_scene_dx_px'], flow['median_scene_dy_px']]))
    groups = {}
    for input_vector, response in samples:
        groups.setdefault(tuple(input_vector), []).append(response)
    repeatability = []
    for command, response in groups.items():
        response = np.asarray(response)
        scatter = float(np.max(np.linalg.norm(response-response.mean(axis=0), axis=1)))
        passed = len(response) >= 2 and scatter <= 2.0
        repeatability.append({'relative': list(command), 'samples': len(response),
                              'max_deviation_px': scatter, 'valid': passed})
    if not repeatability or not all(g['valid'] for g in repeatability):
        missing.append('at least two independently recorded coherent responses per input vector with <=2px deviation')
    inputs = np.asarray([s[0] for s in samples], dtype=float).reshape(-1, 2)
    rank = int(np.linalg.matrix_rank(inputs)) if len(inputs) else 0
    if rank < 2:
        missing.append('independent horizontal and vertical scene responses; observed input rank is below two')
    return {'status': 'diagnostic_only' if samples else 'invalid',
            'semantics': 'changed_lower_scene_pixels; camera_state_unconfirmed; not_aim_anchor_response',
            'input_rank': rank, 'accepted_samples': len(samples), 'repeatability': repeatability,
            'local_scene_px_per_input_unit': None, 'max_fit_error_px': None,
            'missing_conditions': missing}


def summarize_reticle_local(results):
    """Count diagnostic samples without promoting them to a response matrix."""
    samples = []
    rejected = []
    for result in results:
        checks = result.get('reticle_local_checks', {})
        if result.get('source_valid') and checks.get('accepted'):
            fit = result['pairs']['before->after']['reticle_local']
            samples.append({'report': result['report']['path'],
                            'relative': result['input_relative'],
                            'screen_dx_dy_px': [fit['screen_dx_px'], fit['screen_dy_px']]})
        else:
            rejected.append({'report': result['report']['path'],
                             'failed_conditions': checks.get('failed_conditions',
                                                            ['local diagnostic unavailable'])})
    return {'status': 'measured_diagnostic_only' if samples else 'rejected',
            'semantics': 'hud_masked_reticle_roi_translation_screen_pixels_only',
            'accepted_samples': len(samples), 'rejected_samples': len(rejected),
            'samples': samples, 'rejections': rejected, 'policy': RETICLE_LOCAL_POLICY,
            'calibration_valid': False, 'world_calibration_valid': False,
            'screen_px_per_mouse_unit': None,
            'limitations': ['supplied anchor geometry is not per-frame reticle confirmation',
                            'phase and ECC share image evidence; quality bounds are not proof of rigid scene motion',
                            'local translation is an ROI diagnostic, not a pointwise Jacobian or world-angle mapping',
                            'camera mode, zoom identity and post-input settling remain unconfirmed']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, action='append', required=True)
    parser.add_argument('--out', type=Path, required=True, help='new JSON output; existing files are not overwritten')
    args = parser.parse_args(argv)
    results = []
    for path in args.report:
        try:
            results.append(analyze_report(path))
        except (OSError, ValueError, TypeError, KeyError, AttributeError, cv2.error) as error:
            results.append({'report': {'path': str(path.resolve())}, 'identity': None,
                            'capture_source': None, 'pairs': {}, 'local_response_valid': False,
                            'calibration_valid': False, 'world_calibration_valid': False,
                            'rejections': [f'{type(error).__name__}: {error}']})
    local = aggregate(results)
    reticle_local = summarize_reticle_local(results)
    fixed_samples = []
    for result in results:
        fixed = result.get('pairs', {}).get('before->after', {}).get('scene', {}).get('fixed_scene', {})
        if result.get('source_valid') and fixed.get('sufficient'):
            fixed_samples.append({'report': result['report'],
                                  'identity': result['identity'],
                                  'capture_source': result['capture_source'],
                                  'camera_state': 'unconfirmed',
                                  'relative': result.get('input_relative'),
                                  'screen_dx_px': fixed.get('median_screen_dx_px'),
                                  'screen_dy_px': fixed.get('median_screen_dy_px')})
    fixed_scene = {'status': 'measured_screen_displacement' if fixed_samples else 'invalid',
                   'semantics': 'static_scene_homography_screen_pixels_only',
                   'accepted_samples': len(fixed_samples), 'samples': fixed_samples,
                   'world_calibration_valid': False,
                   'missing_conditions': ([] if fixed_samples else
                                          ['static-scene homography failed acceptance bounds']) +
                                         ['camera mode and zoom are unknown; screen pixels cannot map to world angles']}
    output = {'schema_version': 3, 'mode': 'offline_saved_probe_analysis', 'input_injected': False,
              'policy': POLICY, 'sequences': results, 'local_response': local,
              'fixed_scene_response': fixed_scene,
              'reticle_local_response': reticle_local,
              'calibration_valid': False, 'world_calibration_valid': False,
              'mouse_response': {'status': 'invalid', 'camera_mode': None,
                                 'screen_px_per_mouse_unit': None, 'settled_tolerance_px': None,
                                 'missing_conditions': local['missing_conditions'] + [
                                     'image-confirmed firing camera mode and zoom, unchanged through every probe',
                                     'confirmed firing reticle and fixed aim/hull anchor association in every source image',
                                     'time-resolved no-input controls and repeated positive/negative probes on both axes in that firing view',
                                     'observed post-input settling duration and <=5px settled aim tolerance',
                                     'validated response at the reticle/aim anchor; own-ship or background perspective flow is insufficient']},
              'limits': ['source identities are recorded evidence, not reverified live identities',
                         'changed-patch masking excludes zero-flow features and can bias local medians',
                         'single snapshots cannot establish absence of transient overshoot or camera animation',
                         'no pixel-to-angle, world velocity, target identity, ballistic or fire authorization is inferred']}
    with args.out.open('x') as handle:
        json.dump(output, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'out': str(args.out.resolve()), 'local_response_status': local['status'],
                      'reversible_scene_changes': sum(r['local_response_valid'] for r in results),
                      'reticle_local_accepted_samples': reticle_local['accepted_samples'],
                      'calibration_valid': False, 'rejections': [r['rejections'] for r in results]}, indent=2))
    return 1 if any(r['rejections'] for r in results) else 0


if __name__ == '__main__':
    raise SystemExit(main())
