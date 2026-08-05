//! Workspace registry for the wire-conformance corpus: every streaming wire
//! family must carry a `streaming_conformance_suite!` invocation somewhere in
//! the workspace's test sources, or CI fails (#2258, Part III (c) —
//! permanently closes the fixtures-covered-a-subset gap).

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use rig_core::test_utils::streaming_conformance::WIRE_FAMILIES;

/// Collect every `.rs` file under `dir`, recursively.
fn rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            rs_files(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

/// The `provider: "…"` families named by `streaming_conformance_suite!`
/// invocations in `text`.
fn invoked_families(text: &str) -> Vec<String> {
    let needle = concat!("streaming_conformance_suite", "!");
    let mut families = Vec::new();
    for (index, _) in text.match_indices(needle) {
        // The `provider:` field is the invocation's first token; search a
        // bounded window so an unrelated later occurrence is not captured.
        let window = text
            .get(index..)
            .map(|rest| rest.chars().take(400).collect::<String>())
            .unwrap_or_default();
        let Some(provider_at) = window.find("provider:") else {
            continue;
        };
        let Some(after) = window.get(provider_at..) else {
            continue;
        };
        let mut quoted = after.splitn(3, '"');
        let _ = quoted.next();
        if let Some(family) = quoted.next() {
            families.push(family.to_string());
        }
    }
    families
}

#[test]
fn all_wire_families_have_conformance_suites() {
    // The `rig` facade is the workspace-root package, so its manifest dir is
    // the workspace root.
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    rs_files(&root.join("tests"), &mut files);
    if let Ok(crates) = fs::read_dir(root.join("crates")) {
        for entry in crates.flatten() {
            rs_files(&entry.path().join("tests"), &mut files);
        }
    }

    let mut covered = BTreeSet::new();
    for file in &files {
        let Ok(text) = fs::read_to_string(file) else {
            continue;
        };
        covered.extend(invoked_families(&text));
    }

    let missing: Vec<&&str> = WIRE_FAMILIES
        .iter()
        .filter(|family| !covered.contains(**family))
        .collect();
    assert!(
        missing.is_empty(),
        "wire families without a streaming_conformance_suite! invocation: {missing:?} \
         (covered: {covered:?})",
    );

    // A family name outside the canonical list is a typo or an unregistered
    // family; both must be fixed at the WIRE_FAMILIES source of truth.
    let unknown: Vec<&String> = covered
        .iter()
        .filter(|family| !WIRE_FAMILIES.contains(&family.as_str()))
        .collect();
    assert!(
        unknown.is_empty(),
        "suite invocations name families missing from WIRE_FAMILIES: {unknown:?}",
    );
}
