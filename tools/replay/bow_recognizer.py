#!/usr/bin/env python3
"""Image-only bow/stern keypoints from reviewed BowNet weights, never fire authority.

A missing model is an error, not a request to fall back to bar geometry. Crops
are square-padded top-left and resized to 256 RGB pixels. Aspect bins describe
camera azimuth about the ship, not the screen bow-to-stern vector or velocity.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

SIZE = 256
PREPROCESS = 'nameplate-8w-by-6w-square-topleft-rgb-area256-v1'


def crop_box(bar, shape):
    if len(shape) != 3 or shape[2] != 3 or len(bar) != 4:
        raise ValueError('requires RGB frame and xywh nameplate')
    if any(isinstance(v, bool) or not math.isfinite(float(v)) for v in bar):
        raise ValueError('nonfinite nameplate')
    x, y, w, h = map(float, bar)
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x+w > shape[1] or y+h > shape[0]:
        raise ValueError('nameplate outside frame')
    cx = x+w/2
    return (max(0, math.floor(cx-4*w)), max(0, math.floor(y+h)),
            min(shape[1], math.ceil(cx+4*w)), min(shape[0], math.ceil(y+h+6*w)))


def prepare_crop(frame, bar):
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
        raise ValueError('requires uint8 RGB')
    x0, y0, x1, y1 = crop_box(bar, frame.shape)
    if min(x1-x0, y1-y0) < 16:
        raise ValueError('crop has insufficient image support')
    from bow_dataset import source_crop
    roi = [x0, y0, x1-x0, y1-y0]
    return source_crop(frame, roi), dict(
        source_roi=roi, side=max(roi[2:]), preprocessing=PREPROCESS)


def decode_prediction(mask, bow, stern, orientation, *, min_mask=.75,
                      min_keypoint_mass=.25, min_orientation=.75, orientation_supervised=True):
    """Reject absent, ambiguous, unsupported or padded keypoints.

    Keypoint confidence is probability mass within a radius of five crop pixels,
    not the maximum of a 65536-cell softmax. Thresholds are provisional until
    measured on independent labelled sessions; outputs remain non-authoritative.
    """
    arrays = [np.asarray(v) for v in (mask, bow, stern, orientation)]
    mask, bow, stern, orientation = arrays
    if any(not np.isfinite(v).all() for v in arrays):
        return None
    if mask.shape != (SIZE, SIZE) or bow.shape != mask.shape or stern.shape != mask.shape or orientation.shape != (16,):
        raise ValueError('unexpected BowNet output shape')
    binary = mask >= .5
    if int(binary.sum()) < 32:
        return None
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8))
    components = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 32]
    if len(components) != 1:
        return None
    support = labels == components[0]
    segmentation_confidence = float(mask[support].mean())
    orientation_bin = int(orientation.argmax()) if orientation_supervised else "unknown"
    orientation_confidence = float(orientation[orientation_bin]) if orientation_supervised else None
    if segmentation_confidence < min_mask or (orientation_supervised and orientation_confidence < min_orientation):
        return None
    points, confidences = [], []
    yy, xx = np.ogrid[:SIZE, :SIZE]
    for heat in (bow, stern):
        y, x = np.unravel_index(int(heat.argmax()), heat.shape)
        disk = (xx-x)**2+(yy-y)**2 <= 25
        confidence = float(heat[disk].sum())
        if confidence < min_keypoint_mass or not np.any(support & disk):
            return None
        points.append([float(x), float(y)])
        confidences.append(confidence)
    if math.dist(*points) < 5:
        return None
    return dict(bow_point=points[0], stern_point=points[1],
        bow_confidence=confidences[0], stern_confidence=confidences[1],
        segmentation_confidence=segmentation_confidence,
        orientation_bin=orientation_bin, orientation_confidence=orientation_confidence,
        orientation_degrees=orientation_bin*22.5 if orientation_supervised else None,
        fire_authorized=False, world_velocity=None, lead_compensation=False)


class BowRecognizer:
    def __init__(self, model_path, device='cpu'):
        import torch
        from train_bow_model import BowNet, MODEL_TYPE
        self.path = Path(model_path)
        if not self.path.is_file():
            raise ValueError(f'bow model unavailable: {self.path}')
        self.model_sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        payload = torch.load(self.path, map_location='cpu', weights_only=True)
        from bow_dataset import CROP_CONVENTION, ORIENTATION_CONVENTION
        if not isinstance(payload, dict):
            raise ValueError('BowNet checkpoint must contain its input contract')
        required = dict(schema_version=1, input_channels='RGB',
                        input_scale='uint8 / 255.0', crop_transform=CROP_CONVENTION,
                        orientation_convention=ORIENTATION_CONVENTION)
        if any(payload.get(key) != value for key, value in required.items()):
            raise ValueError('BowNet preprocessing or orientation contract mismatch')
        if payload.get('model_type') != MODEL_TYPE or payload.get('input_size') != SIZE:
            raise ValueError('not a reviewed BowNet checkpoint; legacy hull masks have no bow labels')
        if payload.get('orientation_bins') != 16:
            raise ValueError('orientation contract mismatch')
        self.orientation_supervised = payload.get('orientation_supervised', True)
        if not isinstance(self.orientation_supervised, bool):
            raise ValueError('orientation_supervised must be boolean')
        self.model = BowNet(payload.get('base', 16))
        self.model.load_state_dict(payload['state_dict'])
        self.device = torch.device(device)
        self.model.to(self.device).eval()

    def predict_crop(self, image):
        """Predict a reviewed-dataset crop, without inventing source coordinates."""
        import torch
        if image.shape != (SIZE, SIZE, 3) or image.dtype != np.uint8:
            raise ValueError('requires exact uint8 RGB 256x256 crop')
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.device).float()/255
        with torch.inference_mode():
            mask, bow, stern, orientation = self.model(tensor)
            outputs = [torch.sigmoid(mask)[0, 0].cpu().numpy(),
                torch.softmax(bow.flatten(2), dim=2)[0, 0].reshape(SIZE, SIZE).cpu().numpy(),
                torch.softmax(stern.flatten(2), dim=2)[0, 0].reshape(SIZE, SIZE).cpu().numpy(),
                torch.softmax(orientation, dim=1)[0].cpu().numpy()]
        return decode_prediction(*outputs, orientation_supervised=self.orientation_supervised)

    def detect(self, frame, bar):
        image, transform = prepare_crop(frame, bar)
        result = self.predict_crop(image)
        if result is None:
            return None
        x0, y0, w, h = transform['source_roi']
        for key in ('bow_point', 'stern_point'):
            px, py = result[key]
            # OpenCV pixel-centre transform; reject black square padding.
            x, y = (px+.5)*transform['side']/SIZE-.5, (py+.5)*transform['side']/SIZE-.5
            if not (0 <= x < w and 0 <= y < h):
                return None
            result[key] = [x0+x, y0+y]
        result.update(transform, model_sha256=self.model_sha256,
                      evidence='learned_bow_stern_candidate; thresholds_not_heldout_validated')
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--bar', nargs=4, type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        parser.error('image unreadable')
    try:
        result = BowRecognizer(args.model).detect(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), args.bar)
        report = dict(status='candidate' if result else 'rejected', result=result, fire_authorized=False)
    except (ValueError, OSError) as error:
        report = dict(status='unavailable', reason=str(error), result=None, fire_authorized=False)
    with args.out.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps(report, allow_nan=False))
    return 0 if report['status'] == 'candidate' else 1


if __name__ == '__main__':
    raise SystemExit(main())
