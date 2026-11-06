#!/usr/bin/env python3
"""Export the trained Operations model to ONNX and write the Rust parity fixture.

The reference project does exactly this split: training stays in Python and the runtime
compiles an ONNX graph. The Rust runtime agent therefore needs an ONNX file plus a
fixture that proves the Rust inference agrees with PyTorch on the same bytes - not
"it runs", but "it produces the same logits and the same peaks".

The fixture carries real validation samples: the prepared float input the model actually
takes (N, 3, 361, 360), the heatmap and ring logits PyTorch produced, and the peaks the
reference decode() finds in sigmoid(heatmaps) at the manifest's threshold. It also
records the hashes of the model and the corpus it was drawn from, because a parity report
that cannot name its inputs cannot be re-checked later.

Everything about WHERE the data comes from is read from the manifest, not written here:
the manifest already names the corpus, the model and the state dict. Hard-coded corpus
names (the 14-session ones this file used to carry) silently keep exporting the previous
corpus after a re-merge, and the fixture then proves parity for a model nobody ships.

    ./.venv/bin/python tools/replay/export_onnx.py \
        --manifest replays/parsed/artifacts/minimap-heatmap-ops-18s.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_minimap_heatmap as T

def canonical(peaks):
    return sorted(peaks, key=lambda p: (-p[2], p[1], p[0]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True,
                    help="the training manifest: it names the model, its state dict and "
                         "the corpus the fixture is drawn from, so none of them are "
                         "repeated here where they could go stale")
    ap.add_argument("--model", default=None,
                    help="override the scripted model (default: the manifest's artifact)")
    ap.add_argument("--onnx-out", default=None,
                    help="default: the model path with an .onnx suffix")
    ap.add_argument("--fixture", default="rust/wws-inference/tests/fixtures/inference")
    ap.add_argument("--samples", type=int, default=2)
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    model_path = Path(args.model or manifest["artifact"]["path"])
    state_path = Path(manifest["artifact"]["state_dict"])
    state = torch.load(state_path, map_location="cpu")
    classes = list(state["classes"])
    rings = list(state["ring_labels"])
    grid = int(manifest["output"]["heatmaps"]["grid"][0])
    threshold = float(manifest["decoding"]["peak_threshold"])
    max_peaks = int(manifest["decoding"]["max_peaks_per_class"])
    print("architecture %s  classes %d  rings %d  grid %d  threshold %.2f"
          % (state["architecture"], len(classes), len(rings), grid, threshold))

    model = T.MinimapHeatmap(len(classes), len(rings), grid)
    model.load_state_dict(state["model"])
    model.eval()

    # Export. `dynamo=False` keeps the classic TorchScript-based exporter, which needs no
    # extra runtime and emits a graph any ONNX runtime can load.
    onnx_out = Path(args.onnx_out or model_path.with_suffix(".onnx"))
    onnx_out.parent.mkdir(parents=True, exist_ok=True)
    # The input shape comes from the manifest too: it is part of the model's contract,
    # and a dummy of the wrong size would export a graph the runtime cannot be fed.
    input_shape = [int(v) for v in manifest["input"]["shape"]]
    dummy = torch.zeros(max(1, input_shape[0]), *input_shape[1:], dtype=torch.float32)
    torch.onnx.export(
        model, (dummy,), str(onnx_out),
        input_names=["image"], output_names=["heatmaps", "rings"],
        dynamic_axes={"image": {0: "batch"}, "heatmaps": {0: "batch"}, "rings": {0: "batch"}},
        opset_version=17, dynamo=False)
    import onnx
    graph = onnx.load(str(onnx_out))
    shapes = {value.name: [d.dim_value or d.dim_param for d in value.type.tensor_type.shape.dim]
              for value in list(graph.graph.input) + list(graph.graph.output)}
    print("onnx %.1f KB  ir=%s  opset=%s" % (onnx_out.stat().st_size / 1024,
                                             graph.ir_version,
                                             [p.version for p in graph.opset_import]))
    for name, shape in shapes.items():
        print("   %-10s %s" % (name, shape))

    # Fixture: real validation samples, through the same normal path the loader uses.
    data = T.load_all(manifest["dataset"]["heatmaps"], manifest["dataset"]["images"],
                      manifest["training"]["downsample"])
    # data["downsample"] is the EFFECTIVE factor, not the requested one: load_all switches
    # to the half-resolution member when the corpus has one, and then the stored images are
    # already downsampled. Passing the manifest's factor would pool them a second time
    # (361x360 -> 180x180) and the fixture would stop reproducing the training input.
    print("effective downsample %d (manifest asked for %d); corpus %s"
          % (data["downsample"], manifest["training"]["downsample"],
             Path(manifest["dataset"]["heatmaps"]).name))
    if list(data["images"].shape[1:3]) != input_shape[2:4]:
        raise SystemExit("the manifest declares input %s but the corpus supplies %s; the "
                         "fixture cannot reproduce the trained input"
                         % (input_shape, list(data["images"].shape[1:3])))
    index = list(T.select(data, "validation"))[:args.samples]
    prepared = T.prepare_images(data["images"][np.asarray(index)], torch.device("cpu"),
                                data["downsample"])
    with torch.no_grad():
        heat_logits, ring_logits = model(prepared)
    peaks = T.decode(torch.sigmoid(heat_logits), threshold)

    out = Path(args.fixture)
    out.mkdir(parents=True, exist_ok=True)
    arrays = {
        "input.f32": prepared.numpy(),
        "heatmaps.f32": heat_logits.numpy(),
        "rings.f32": ring_logits.numpy(),
    }
    for name, array in arrays.items():
        (out / name).write_bytes(np.ascontiguousarray(array, dtype=np.float32).tobytes())
        print("   fixture %-14s %s  %.2f MB"
              % (name, array.shape, array.nbytes / 1e6))
    (out / "expected.json").write_text(json.dumps({
        "input_shape": list(prepared.shape),
        "heatmaps_shape": list(heat_logits.shape),
        "rings_shape": list(ring_logits.shape),
        "threshold": threshold,
        "max_peaks_per_class": max_peaks,
        # What this comparison is ABOUT. A parity report that cannot name the model and
        # the corpus it used cannot be re-checked, and the runtime must be able to refuse
        # a model it was not verified against.
        "identity": {
            "manifest": str(args.manifest),
            "manifest_sha256": T.sha256(args.manifest),
            "model": str(model_path),
            "model_sha256": T.sha256(model_path),
            "state_dict": str(state_path),
            "state_dict_sha256": T.sha256(state_path),
            "corpus": str(manifest["dataset"]["heatmaps"]),
            "corpus_sha256": manifest["dataset"]["heatmaps_sha256"],
            "images": str(manifest["dataset"]["images"]),
            "classes": classes,
            "ring_labels": [float(v) for v in rings],
            "effective_downsample": int(data["downsample"]),
        },
        "peaks": [[[list(p) for p in canonical(channel)] for channel in sample]
                  for sample in peaks],
    }, indent=1) + "\n", encoding="utf-8")
    print("wrote %s and %s" % (onnx_out, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
