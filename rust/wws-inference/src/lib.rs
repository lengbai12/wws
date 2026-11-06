//! Runs the trained Operations minimap model in Rust.
//!
//! The model itself is trained in Python (`tools/replay/train_minimap_heatmap.py`) and
//! exported to ONNX by `tools/replay/export_onnx.py`. This crate is the other end of
//! that handover: what a live agent would call once per frame to turn a minimap crop
//! into per-class heatmap logits, plus the peaks the Python reference decodes from them.
//!
//! The contract with the corpus is fixed by the manifest, and this crate does not
//! invent anything on top of it:
//!
//! * input is the prepared float crop stack, `(N, 3, 361, 360)`, already divided by 255
//!   and averaged down - that preparation is `prepare_images()` on the Python side and
//!   would be ported here when the frame source exists;
//! * `heatmaps` are LOGITS, so sigmoid is applied before decoding;
//! * `rings` are LOGITS too, one per range-ring label, thresholded at 0.5.

use ndarray::{Array2, Array4, ArrayView4};
use ort::session::Session;
use ort::value::Tensor;

pub use wws_perception::{Peak, decode};

/// A loaded model, ready to run.
///
/// ONNX Runtime sessions are cheap to reuse and expensive to build, so a live agent
/// holds one for the whole session.
pub struct Model {
    session: Session,
    pub grid: usize,
}

impl Model {
    /// Load an ONNX graph. `grid` is the manifest's output grid (176 for this corpus).
    pub fn load(path: &str, grid: usize) -> ort::Result<Self> {
        let session = Session::builder()?.commit_from_file(path)?;
        Ok(Self { session, grid })
    }

    /// One forward pass: `(heatmap_logits, ring_logits)`.
    pub fn logits(&mut self, input: ArrayView4<'_, f32>) -> ort::Result<(Array4<f32>, Array2<f32>)> {
        let tensor = Tensor::from_array(input.to_owned())?;
        let outputs = self.session.run(ort::inputs!["image" => tensor])?;
        let heatmaps = outputs["heatmaps"].try_extract_array::<f32>()?.to_owned();
        let rings = outputs["rings"].try_extract_array::<f32>()?.to_owned();
        Ok((heatmaps.into_dimensionality().unwrap(), rings.into_dimensionality().unwrap()))
    }

    /// Logits, sigmoid, then the reference decode: what a frame's perception step needs.
    pub fn peaks(
        &mut self,
        input: ArrayView4<'_, f32>,
        threshold: f32,
        max_peaks_per_class: usize,
    ) -> ort::Result<Vec<Vec<Vec<Peak>>>> {
        let (heatmaps, _rings) = self.logits(input)?;
        let probabilities = heatmaps.mapv(|value| 1.0 / (1.0 + (-value).exp()));
        Ok(decode(probabilities.view(), threshold, max_peaks_per_class))
    }
}
