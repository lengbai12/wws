#!/usr/bin/env python3
"""Raw RGB minimap perception: unique own position plus distinct enemy candidates."""
import argparse
import json

import numpy as np

from enemy_marker_classifier import EnemyRecognizer
from own_ship_locator import OwnShipLocator


class MinimapRecognizer:
    def __init__(self, package, device='cpu'):
        self.enemy = EnemyRecognizer(package, device)
        self.own = OwnShipLocator()

    def detect(self, rgb):
        if rgb.dtype != np.uint8 or list(rgb.shape) != self.enemy.package['input']['shape']:
            raise ValueError('expected uint8 RGB minimap crop of shape '+str(self.enemy.package['input']['shape']))
        return dict(own=self.own.detect(rgb), enemies=self.enemy.detect(rgb))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--package', required=True)
    ap.add_argument('--rgb-npy', required=True, help='uint8 RGB minimap crop; package defines ROI and size')
    args = ap.parse_args()
    recognizer = MinimapRecognizer(args.package)
    print(json.dumps(recognizer.detect(np.load(args.rgb_npy, allow_pickle=False))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
