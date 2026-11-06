//! Parity gate: our Rust decode() must find exactly the peaks the Python reference finds.
//!
//! The fixture is produced by `tools/replay/dump_decode_fixture.py`, which runs the
//! Python decode() over real corpus targets and over a synthetic soft field, and writes
//! both the input bytes and the reference's own peaks. Comparing them is the only honest
//! way to claim the port is faithful: "it runs and looks reasonable" says nothing about
//! whether a peak at the threshold or on the border is reported the same way.
//!
//! Peak order is canonical on both sides (score desc, row, column), so this compares
//! which peaks were found and their exact scores, not the order a sort happened to pick.

use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::PathBuf;
use wws_perception::{decode, read_f32};

#[derive(Deserialize)]
struct Case {
    shape: [usize; 4],
    file: String,
    peaks: Vec<Vec<Vec<[f32; 3]>>>,
}

#[derive(Deserialize)]
struct Fixture {
    grid: usize,
    threshold: f32,
    max_peaks_per_class: usize,
    cases: BTreeMap<String, Case>,
}

#[test]
fn decode_matches_the_python_reference() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/decode");
    let text = std::fs::read_to_string(dir.join("peaks.json")).expect("fixture peaks.json");
    let fixture: Fixture = serde_json::from_str(&text).expect("fixture parses");

    let mut matched_cases = 0;
    let mut matched_peaks = 0;
    for (name, case) in &fixture.cases {
        assert_eq!(case.shape[2], fixture.grid, "{name}: fixture grid");
        let field = read_f32(&dir.join(&case.file), case.shape);
        let found = decode(field.view(), fixture.threshold, fixture.max_peaks_per_class);

        assert_eq!(found.len(), case.peaks.len(), "{name}: sample count");
        for (sample, (got, expected)) in found.iter().zip(case.peaks.iter()).enumerate() {
            assert_eq!(got.len(), expected.len(), "{name} sample {sample}: channel count");
            for (channel, (got, expected)) in got.iter().zip(expected.iter()).enumerate() {
                assert_eq!(
                    got.len(),
                    expected.len(),
                    "{name} sample {sample} channel {channel}: peak count"
                );
                for (index, (got, expected)) in got.iter().zip(expected.iter()).enumerate() {
                    assert_eq!(
                        got, expected,
                        "{name} sample {sample} channel {channel} peak {index}"
                    );
                }
                matched_peaks += got.len();
            }
        }
        matched_cases += 1;
    }

    assert_eq!(matched_cases, 2, "expected the targets and soft cases");
    println!("decode matches the reference: {matched_cases} cases, {matched_peaks} peaks");
}
