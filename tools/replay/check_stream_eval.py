#!/usr/bin/env python3
"""Prove that the streaming evaluation equals the whole-split one.

evaluate(..., stream=True) decodes one forward chunk at a time instead of building the
split's logits as one tensor; the training split alone is 12.4 GB that way, and building
it after the last epoch is what OOM-killed a 40-epoch run. The claim is that decoding
chunk by chunk is the same measurement, because decode() pads each sample's own edges
and never crosses a sample boundary.

Tested on a real slice of the corpus at a dense threshold (0.0, where every local
maximum is a peak and the comparison is at its strictest) as well as at the operating
point. Heavy: it reads the corpus and runs the model, so do not run it beside a
training on this machine.

    ./.venv/bin/python tools/replay/check_stream_eval.py [SAMPLES]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_minimap_heatmap as T

HEATMAPS = "replays/parsed/datasets/heatmap-merged-14s-g176.npz"
IMAGES = "replays/parsed/datasets/minimap-merged-14s-g176.npz"


def main():
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    data = T.load_all(HEATMAPS, IMAGES, 2)
    index = list(T.select(data, "validation"))[:count]
    model = T.MinimapHeatmap(len(data["classes"]), len(data["ring_labels"]), data["grid"])
    print("slice %d samples  grid %d  classes %d" % (len(index), data["grid"],
                                                     len(data["classes"])))
    failed = False
    for threshold in (0.0, T.PEAK_THRESHOLD):
        streamed, _p, heatmaps = T.evaluate(model, index, data, threshold=threshold,
                                            stream=True)
        assert heatmaps is None, "the streamed path must not return the split's tensor"
        whole, _p2, _h2 = T.evaluate(model, index, data, threshold=threshold)
        same = streamed == whole
        failed |= not same
        print("threshold %.2f   metrics identical: %s" % (threshold, same))
        for name in streamed:
            if streamed[name] != whole[name]:
                print("   differs:", name)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
