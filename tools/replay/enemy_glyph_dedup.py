"""Conservative enemy deduplication using the red pixels of one fitted glyph."""
import cv2
import numpy as np


MIN_ICON_SCORE = .75
MIN_COMPONENT_AREA = 12
MAX_CENTRE_DISTANCE = 2


def deduplicate(rgb, detections, matcher):
    """Keep the strongest candidate only when both explain one isolated red glyph.

    Unknown species may still have a useful template fit. A fit alone is not
    identity: candidates must share red component identities near their points,
    and each template must cover at least 90% of their pixels (one pixel
    antialiasing tolerance). Multiple components require distinct matching
    template lobes, preserving native internal gaps without joining fragments.
    Large/touching glyphs, weak fits, conflicting species, and chains stay separate.
    The original winning detection, including its coordinates, is returned.
    """
    if len(detections) < 2:
        return detections
    colour = rgb.astype(np.int16)
    red = colour[..., 0] - np.maximum(colour[..., 1], colour[..., 2]) >= 40
    _, labels, stats, _ = cv2.connectedComponentsWithStats(red.astype(np.uint8), connectivity=8)
    templates = {(species, angle, scale): template
                 for species, angle, scale, template in matcher.templates}
    height, width = rgb.shape[:2]

    def evidence(detection):
        if detection['icon_score'] < MIN_ICON_SCORE:
            return None
        x = round(detection['u'] * (width-1))
        y = round(detection['v'] * (height-1))
        left, top = max(0, x-4), max(0, y-4)
        region = labels[top:min(height, y+5), left:min(width, x+5)]
        yy, xx = np.indices(region.shape)
        ids = np.unique(region[(xx+left-x)**2 + (yy+top-y)**2 <= 16])
        ids = ids[ids != 0]
        if not len(ids):
            return None
        boxes = stats[ids]
        bx, by = boxes[:, :2].min(axis=0)
        right, bottom = (boxes[:, :2] + boxes[:, 2:4]).max(axis=0)
        bw, bh = right-bx, bottom-by
        area = boxes[:, 4].sum()
        if area < MIN_COMPONENT_AREA or max(bw, bh) > 25:
            return None
        # A clipped glyph does not provide complete identity evidence.
        if bx == 0 or by == 0 or right == width or bottom == height:
            return None
        template = templates.get((detection['icon_candidate'], detection['icon_angle_deg'],
                                  detection['icon_scale']))
        if template is None:
            return None
        cx, cy = np.asarray([x, y]) + detection['icon_offset_px']
        cx, cy = int(cx), int(cy)
        support = cv2.dilate((template >= .15).astype(np.uint8), np.ones((3, 3), np.uint8))
        region = labels[by:bottom, bx:right]
        yy, xx = np.where(np.isin(region, ids))
        pixel_ids = region[yy, xx]
        xx, yy = xx+bx-cx+12, yy+by-cy+12
        inside = (xx >= 0) & (xx < 25) & (yy >= 0) & (yy < 25)
        covered = np.count_nonzero(support[yy[inside], xx[inside]])
        if covered / area < .9:
            return None
        if len(ids) > 1:
            # Cruiser/carrier gaps can split one glyph into intentional lobes.
            # Only allow that topology when every observed component matches a
            # different template lobe; arbitrary fragments are still ambiguous.
            count, lobes, lobe_stats, _ = cv2.connectedComponentsWithStats(
                (template >= .15).astype(np.uint8), connectivity=8)
            if count-1 != len(ids):
                return None
            assigned = set()
            for identity in ids:
                selected = inside & (pixel_ids == identity)
                matches = []
                for lobe in range(1, count):
                    mask = cv2.dilate((lobes == lobe).astype(np.uint8), np.ones((3, 3), np.uint8))
                    hits = np.count_nonzero(mask[yy[selected], xx[selected]])
                    if hits / stats[identity, 4] >= .9 and hits >= .7*lobe_stats[lobe, 4]:
                        matches.append(lobe)
                if len(matches) != 1 or matches[0] in assigned:
                    return None
                assigned.add(matches[0])
        return tuple(int(identity) for identity in ids), detection['icon_candidate'], cx, cy

    def rank(detection):
        return (-detection['score'], -detection['heatmap_score'], -detection['icon_score'],
                detection['u'], detection['v'], detection['species'],
                detection['icon_candidate'], detection['icon_angle_deg'], detection['icon_scale'],
                tuple(detection['icon_offset_px']), -detection.get('icon_margin', 0.))

    kept, groups = [], []
    for detection in sorted(detections, key=rank):
        proof = evidence(detection)
        if proof is not None:
            for group in groups:
                if all(proof[:2] == other[:2] and
                       (proof[2]-other[2])**2 + (proof[3]-other[3])**2 <= MAX_CENTRE_DISTANCE**2
                       for other in group):
                    group.append(proof)
                    break
            else:
                groups.append([proof])
                kept.append(detection)
        else:
            kept.append(detection)
    kept_ids = {id(detection) for detection in kept}
    return [detection for detection in detections if id(detection) in kept_ids]
