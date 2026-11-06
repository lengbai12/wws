//! Perception primitives for the Operations agent.
//!
//! The Python pipeline in `tools/replay/` stays the source of truth for anything that
//! reads or writes a corpus: replay parsing leans on third-party parsers with per-build
//! schemas, and training is PyTorch. This crate is the other half - the part that has to
//! run every frame inside a live game, which is the layer the reference project
//! (`fps-vision-main`) is built around and the layer where Rust's lack of a GIL and lack
//! of interpreter overhead is worth something.
//!
//! It starts at the decode step, because that is the most testable point in the chain:
//! pure arithmetic, no device and no capture, and it can be checked against the Python
//! reference on real data by [`tests/decode_parity.rs`](../tests/decode_parity.rs).

use ndarray::{Array4, ArrayView4, s};
use std::path::Path;

/// One detected marker, in grid coordinates: `[column, row, score]`.
///
/// Column indexes the heatmap's width and row its height, matching the order the Python
/// reference reads out of `torch.nonzero`.
pub type Peak = [f32; 3];

/// Local maxima of a 3x3 window at or above `threshold`, capped per channel.
///
/// A faithful port of `decode()` in `tools/replay/train_minimap_heatmap.py`:
///
/// * the window is the sample's own 3x3 neighbourhood with the outside treated as
///   `-inf`, which is what `max_pool2d(3, stride=1, padding=1)` does for float input,
///   so a border cell can qualify;
/// * `>= pooled - 1e-6` keeps the tolerance the Python side uses for ties, so a plateau
///   is reported as several peaks rather than none;
/// * `>= threshold` is applied to the value itself, before any pooling;
/// * the per-channel cap keeps the highest scores, and ties are broken by row then
///   column so the result does not depend on the sort implementation's stability.
///
/// Returns `[sample][channel] -> peaks`, the same nesting the reference produces.
pub fn decode(
    heatmaps: ArrayView4<'_, f32>,
    threshold: f32,
    max_peaks_per_class: usize,
) -> Vec<Vec<Vec<Peak>>> {
    let (samples, channels, height, width) = heatmaps.dim();
    let mut out = Vec::with_capacity(samples);
    for sample in 0..samples {
        let mut per_sample = Vec::with_capacity(channels);
        for channel in 0..channels {
            let plane = heatmaps.slice(s![sample, channel, .., ..]);
            let mut found: Vec<Peak> = Vec::new();
            for row in 0..height {
                for column in 0..width {
                    let value = plane[[row, column]];
                    if !(value >= threshold) {
                        continue; // also skips NaN, which torch would not report either
                    }
                    let mut pooled = f32::NEG_INFINITY;
                    for dy in -1i32..=1 {
                        for dx in -1i32..=1 {
                            let y = row as i32 + dy;
                            let x = column as i32 + dx;
                            if y < 0 || x < 0 || y >= height as i32 || x >= width as i32 {
                                continue; // outside the plane counts as -inf, as padding does
                            }
                            let neighbour = plane[[y as usize, x as usize]];
                            if neighbour > pooled {
                                pooled = neighbour;
                            }
                        }
                    }
                    if value >= pooled - 1e-6 {
                        found.push([column as f32, row as f32, value]);
                    }
                }
            }
            found.sort_by(|a, b| {
                b[2].total_cmp(&a[2]) // score, descending
                    .then(a[1].total_cmp(&b[1])) // then row
                    .then(a[0].total_cmp(&b[0])) // then column
            });
            found.truncate(max_peaks_per_class);
            per_sample.push(found);
        }
        out.push(per_sample);
    }
    out
}

/// Read a raw little-endian f32 tensor in C order.
pub fn read_f32(path: &Path, shape: [usize; 4]) -> Array4<f32> {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    let expected = shape.iter().product::<usize>();
    assert_eq!(values.len(), expected, "{}: expected {expected} floats", path.display());
    Array4::from_shape_vec(shape, values).expect("shape matches the element count")
}
