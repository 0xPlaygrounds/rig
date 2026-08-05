//! Structural guard: every streaming triage site runs on the single-policy
//! driver.
//!
//! The `run_wire_stream`/`run_wire_buffered` driver (and its factored
//! `triage_frame` helper) in `providers/internal/adapter.rs` is the ONLY place
//! allowed to decide what happens to `WireEvent::Unknown` / `WireEvent::Corrupt`
//! frames. The websocket divergence fixed on this branch is the standing proof
//! that hand-copied triage tables drift; this test makes reintroducing one a CI
//! failure.
//!
//! Mechanism: non-test provider source may *classify* (produce a `WireEvent`)
//! but never *triage* it — and triage requires matching the `Unknown`/`Corrupt`
//! variants. So any mention of `WireEvent::Unknown` or `WireEvent::Corrupt`
//! outside the driver (`adapter.rs`), the classify layer (`wire.rs`), and test
//! code is a restated policy table.

#![allow(clippy::expect_used)]

use std::path::PathBuf;

/// File basenames where the policy table and classifiers legitimately name the
/// triage variants.
const ALLOWED_FILES: &[&str] = &["adapter.rs", "wire.rs"];

/// Directories that hold test harness code rather than shipped policy.
const SKIPPED_DIRS: &[&str] = &["tests", "test_utils", "fixtures", "target"];

/// No provider restates the driver's Unknown/Corrupt policy table.
#[test]
fn every_triage_site_runs_on_the_single_policy_driver() {
    // rig-core/tests -> workspace crates/ directory, so the guard also covers
    // the out-of-core adapter crates (bedrock, candle, gemini-grpc).
    let crates_dir: PathBuf = [env!("CARGO_MANIFEST_DIR"), ".."].iter().collect();

    let mut violations = Vec::new();
    let mut pending = vec![crates_dir];

    while let Some(dir) = pending.pop() {
        let entries = std::fs::read_dir(&dir).expect("workspace directory should be readable");
        for entry in entries {
            let entry = entry.expect("directory entry should be readable");
            let path = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy().into_owned();

            if path.is_dir() {
                if !SKIPPED_DIRS.contains(&name.as_str()) {
                    pending.push(path);
                }
                continue;
            }

            if path.extension().is_none_or(|ext| ext != "rs")
                || ALLOWED_FILES.contains(&name.as_str())
            {
                continue;
            }

            let source = std::fs::read_to_string(&path).expect("source file should be readable");
            // Inline unit-test modules may pattern-match classification
            // results; only the code before the test module ships.
            let shipped = source
                .split("#[cfg(test)]")
                .next()
                .expect("split always yields at least one part");

            for (index, line) in shipped.lines().enumerate() {
                if line.contains("WireEvent::Unknown") || line.contains("WireEvent::Corrupt") {
                    violations.push(format!("{}:{}: {}", path.display(), index + 1, line.trim()));
                }
            }
        }
    }

    assert!(
        violations.is_empty(),
        "Unknown/Corrupt triage restated outside the driver (adapter.rs) and \
         classify layer (wire.rs) — route it through run_wire_stream / \
         run_wire_buffered / triage_frame instead:\n{}",
        violations.join("\n")
    );
}
