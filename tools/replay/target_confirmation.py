"""White target-lock ring evidence bound to an unambiguous health bar.

A ring proves UI acquisition, not visible hull, range or ammunition readiness.
All positions are native 2560x1600 frame pixels; bboxes are [x, y, width, height].
No replay identity or OCR identity is consumed. Candidate associations include
every compatible detected bar, so an overlap remains ambiguous.
"""
import math
import cv2
import numpy as np

from main_view_perception import FRAME_H, FRAME_W

_RING_ANGLES = np.linspace(0, 2*math.pi, 128, endpoint=False)
_RING_COS = np.cos(_RING_ANGLES)
_RING_SIN = np.sin(_RING_ANGLES)


def confirm_target(rgb, detections):
    if not isinstance(rgb, np.ndarray) or rgb.shape != (FRAME_H, FRAME_W, 3) or rgb.dtype != np.uint8:
        raise ValueError(f"frame must be {FRAME_W}x{FRAME_H} uint8 RGB")
    # Grayscale is pointwise: converting each exact Hough region preserves its
    # pixels and boundary behavior without converting the rest of the frame.
    candidates = []
    for index, detection in enumerate(detections):
        x, y, w, h = detection['nameplate_bbox']
        # Native HUD ring surrounds the class/name above the health bar.
        left, top = max(0, x-100), max(0, y-175)
        right, bottom = min(rgb.shape[1], x+w+100), y
        region = rgb[top:bottom, left:right]
        if region.shape[0] < 40 or region.shape[1] < 40:
            continue
        crop = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        circles = cv2.HoughCircles(crop, cv2.HOUGH_GRADIENT, 1, 25,
                                  param1=100, param2=24, minRadius=22, maxRadius=58)
        if circles is None:
            continue
        for cx, cy, radius in circles[0]:
            cx, cy = float(cx+left), float(cy+top)
            if not x-15 <= cx <= x+w+15 or not 25 <= y-cy <= 140:
                continue
            # Hough estimates can land on the outer antialias edge. Retry a
            # nearby common radius, never a different radius per quadrant.
            # Each candidate must still pass the original full-circle gates.
            accepted = None
            for correction in (0, -1, 1, -2, 2):
                fitted_radius = radius + correction
                if not 22 <= fitted_radius <= 58:
                    continue
                support = np.zeros(128, dtype=bool)
                for delta in (-2, -1, 0, 1, 2):
                    xx = np.rint(cx+(fitted_radius+delta)*_RING_COS).astype(int)
                    yy = np.rint(cy+(fitted_radius+delta)*_RING_SIN).astype(int)
                    if xx.min()<0 or yy.min()<0 or xx.max()>=rgb.shape[1] or yy.max()>=rgb.shape[0]:
                        continue
                    pixels = rgb[yy, xx]
                    low = pixels.min(axis=1)
                    high = pixels.max(axis=1)
                    support |= (low > 155) & (high-low < 55)
                quarters = support.reshape(4,32).mean(axis=1)
                if float(support.mean()) >= .63 and np.min(quarters) >= .4:
                    accepted = (fitted_radius, support, quarters)
                    break
            if accepted is None:
                continue
            radius, support, quarters = accepted
            if any(math.hypot(cx-c['center'][0], cy-c['center'][1]) < 15 for c in candidates):
                continue
            ring_left, ring_top = math.floor(cx-radius), math.floor(cy-radius)
            ring_right, ring_bottom = math.ceil(cx+radius)+1, math.ceil(cy+radius)+1
            candidates.append(dict(center=[cx,cy], radius_px=float(radius),
                                   bbox=[ring_left, ring_top, ring_right-ring_left, ring_bottom-ring_top],
                                   support=float(support.mean()), quarters=quarters.tolist()))
    matches=[]
    for circle in candidates:
        cx,cy=circle['center']
        compatible=[]
        for index,detection in enumerate(detections):
            x,y,w,h=detection['nameplate_bbox']
            if x-circle['radius_px']<=cx<=x+w+circle['radius_px'] and 25<=y-cy<=140:
                compatible.append(index)
        circle['compatible_detection_indices'] = compatible
        circle['association'] = ('unique_healthbar' if len(compatible) == 1 else
                                 'overlapping_healthbars' if compatible else 'unknown')
        if len(compatible)==1:
            matches.append(dict(circle, detection_index=compatible[0]))
    result=dict(confirmed=False, state='unknown', target_association='unknown',
                detection_index=None, evidence='white_ring_quadrant_support',
                candidates=candidates, reason='no_unique_lock_ring',
                coordinate_system='native_frame_xy_pixels',
                identity_association='unknown', body_association='unverified')
    if len(matches)==1 and len(candidates)==1:
        result.update(confirmed=True,state='confirmed',target_association='unique_healthbar',
                      detection_index=matches[0]['detection_index'],ring=matches[0],reason='unique_ui_lock')
    elif candidates:
        result.update(state='ambiguous',reason='overlapping_targets_or_multiple_rings')
    return result
