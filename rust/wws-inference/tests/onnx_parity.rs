//! Parity gate: the Rust runtime must produce what PyTorch produced.
//!
//! `tools/replay/export_onnx.py` writes the ONNX graph and a fixture of real validation
//! samples: the prepared float input, the heatmap and ring logits PyTorch produced, and
//! the peaks the Python reference decodes from `sigmoid(heatmaps)`. This test runs the
//! same input through ONNX Runtime in Rust and checks both levels.
//!
//! On the exported model the two runtimes differ by about 1.4e-04 on the logits (they
//! use different CPU kernels), and no pixel of the fixture sits within 2e-04 of the
//! threshold, so the logits comparison is a band and the peaks comparison is exact.
//! A peak that flips because a value sits right on the threshold would show up here as
//! a failure, which is the intended behaviour: the runtime must not silently disagree
//! with the model that was validated.

use ndarray::{Array2, Array4};
use serde::Deserialize;
use std::path::PathBuf;
use wws_inference::Model;

#[derive(Deserialize)]
struct Expected {
    input_shape: [usize; 4],
    heatmaps_shape: [usize; 4],
    rings_shape: [usize; 2],
    threshold: f32,
    max_peaks_per_class: usize,
    peaks: Vec<Vec<Vec<[f32; 3]>>>,
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/inference")
}

fn read_raw(path: &PathBuf) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn model_path() -> PathBuf {
    root().join("replays/parsed/artifacts/minimap-heatmap-ops-14s.onnx")
}

#[test]
fn onnx_inference_matches_pytorch() {
    let dir = fixture();
    let expected: Expected = serde_json::from_str(
        &std::fs::read_to_string(dir.join("expected.json")).expect("fixture expected.json"),
    )
    .expect("fixture parses");

    let onnx = model_path();
    assert!(
        onnx.exists(),
        "{} is missing; run: ./.venv/bin/python tools/replay/export_onnx.py",
        onnx.display()
    );

    let input = Array4::from_shape_vec(
        expected.input_shape,
        read_raw(&dir.join("input.f32")),
    )
    .expect("input shape matches the element count");
    let torch_heatmaps = Array4::from_shape_vec(
        expected.heatmaps_shape,
        read_raw(&dir.join("heatmaps.f32")),
    )
    .expect("heatmaps shape matches the element count");
    let torch_rings = Array2::from_shape_vec(
        expected.rings_shape,
        read_raw(&dir.join("rings.f32")),
    )
    .expect("rings shape matches the element count");

    let mut model = Model::load(&onnx.to_string_lossy(), expected.heatmaps_shape[2])
        .expect("the ONNX graph loads");
    let (heatmaps, rings) = model.logits(input.view()).expect("inference runs");

    assert_eq!(heatmaps.shape(), torch_heatmaps.shape(), "heatmap shape");
    assert_eq!(rings.shape(), torch_rings.shape(), "ring shape");

    let heat_delta = heatmaps
        .iter()
        .zip(torch_heatmaps.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let ring_delta = rings
        .iter()
        .zip(torch_rings.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("max|delta| heatmaps {heat_delta:.3e}  rings {ring_delta:.3e}");
    assert!(heat_delta < 5e-4, "heatmap logits differed by {heat_delta:e}");
    assert!(ring_delta < 5e-4, "ring logits differed by {ring_delta:e}");

    // The decode is the perception step a live frame would run.
    let probabilities = heatmaps.mapv(|value| 1.0 / (1.0 + (-value).exp()));
    let peaks = wws_perception::decode(
        probabilities.view(),
        expected.threshold,
        expected.max_peaks_per_class,
    );

    // Two ONNX Runtime builds do not agree bit for bit with PyTorch either; on this
    // fixture the logits differ by about 1.4e-04. A peak whose score sits that close to
    // the threshold may therefore appear on one side and not the other, which says
    // nothing about the port. Everything further from the threshold has to match
    // exactly, and the count of threshold-adjacent peaks is reported rather than hidden.
    const BAND: f32 = 2e-4;
    let stable = |peaks: &Vec<[f32; 3]>| -> Vec<[f32; 3]> {
        peaks
            .iter()
            .filter(|peak| (peak[2] - expected.threshold).abs() >= BAND)
            .cloned()
            .collect()
    };

    assert_eq!(peaks.len(), expected.peaks.len(), "sample count");
    let mut total = 0;
    let mut adjacent = 0;
    for (sample, (got, want)) in peaks.iter().zip(expected.peaks.iter()).enumerate() {
        assert_eq!(got.len(), want.len(), "sample {sample}: channel count");
        for (channel, (got, want)) in got.iter().zip(want.iter()).enumerate() {
            let (got_stable, want_stable) = (stable(got), stable(want));
            adjacent += got.len() - got_stable.len() + want.len() - want_stable.len();
            assert_eq!(
                got_stable.len(),
                want_stable.len(),
                "sample {sample} channel {channel}: stable peak count"
            );
            for (index, (got, want)) in got_stable.iter().zip(want_stable.iter()).enumerate() {
                // Position must be exact; the score is the sigmoid value at that cell,
                // so it carries the same ~1e-04 runtime difference as the logits.
                assert_eq!(
                    (got[0], got[1]),
                    (want[0], want[1]),
                    "sample {sample} channel {channel} stable peak {index}: position"
                );
                assert!(
                    (got[2] - want[2]).abs() < 5e-4,
                    "sample {sample} channel {channel} stable peak {index}: score {} vs {}",
                    got[2],
                    want[2]
                );
            }
            total += got_stable.len();
        }
    }
    println!(
        "parity holds: {} samples, {total} stable peaks identical to the reference, \
         {adjacent} threshold-adjacent peaks allowed to differ",
        peaks.len()
    );
}
