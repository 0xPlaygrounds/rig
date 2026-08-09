//! Tripwire binding nightly.yaml's `pull_request.paths` filter to the
//! integration suites it exists to cover.
//!
//! The filter decides which PRs get the full Docker-backed lane before merge,
//! and its glob list must mirror `tests/integrations/` by hand: when a new
//! suite lands (say `tests/integrations/milvus.rs` covering
//! `crates/rig-milvus`) and the glob is forgotten, every later PR touching
//! only that crate merges with zero runtime coverage of the code it changed —
//! the exact failure mode the filter's own header comment says it prevents,
//! and (before this test) a seam with no tripwire, unlike its siblings: the
//! streaming-conformance registry pins ci.yaml's steps and the cassette
//! partition guard pins the suites/directories bijection.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

/// Suites in `tests/integrations/` that deliberately have no
/// `crates/rig-<name>/**` glob, with the reason the omission is sound.
const GLOB_EXEMPT_SUITES: &[(&str, &str)] = &[(
    "bedrock",
    "gated behind the `bedrock` feature, which ci.yaml's PR-gate sweep enables on every PR — \
     the fast lane already executes this suite, so a full-lane trigger would add nothing",
)];

/// Path entries whose presence is load-bearing for reasons documented in
/// nightly.yaml itself; deleting any of them silently reopens a coverage
/// hole a past review closed.
const REQUIRED_PATHS: &[&str] = &[
    // The suites' own sources.
    "tests/integrations/**",
    "tests/integrations.rs",
    // Companion-crate dependency versions are `workspace = true` references
    // resolved in the root manifest and lockfile; without these, dependency
    // bumps merge with zero integration coverage.
    "Cargo.toml",
    "Cargo.lock",
    // The facade feature-forwarding guard: its failure class is invisible to
    // every `--all-features` build on the PR gate.
    "tests/tool_facade_features.rs",
    // Editing the lane must run the lane.
    ".github/workflows/nightly.yaml",
];

fn nightly_pull_request_paths() -> BTreeSet<String> {
    let workflow = Path::new(env!("CARGO_MANIFEST_DIR")).join(".github/workflows/nightly.yaml");
    let text = fs::read_to_string(&workflow).expect("nightly.yaml should be readable");
    let doc: serde_yaml::Value =
        serde_yaml::from_str(&text).expect("nightly.yaml should parse as YAML");
    // YAML 1.1 resolves a bare `on` key to a boolean in some parsers; accept
    // either representation rather than depending on serde_yaml's choice.
    let triggers = doc
        .get("on")
        .or_else(|| doc.get(serde_yaml::Value::Bool(true)))
        .expect("nightly.yaml should have an on: block");
    triggers
        .get("pull_request")
        .and_then(|pull_request| pull_request.get("paths"))
        .and_then(serde_yaml::Value::as_sequence)
        .expect("nightly.yaml should filter pull_request by paths")
        .iter()
        .map(|path| {
            path.as_str()
                .expect("paths entries should be strings")
                .to_string()
        })
        .collect()
}

fn integration_suites() -> BTreeSet<String> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/integrations");
    fs::read_dir(&dir)
        .expect("tests/integrations should be readable")
        .map(|entry| {
            let entry = entry.expect("tests/integrations entry should be readable");
            let path = entry.path();
            if path.is_dir() {
                path.file_name()
            } else {
                path.file_stem()
            }
            .expect("suite entry should have a name")
            .to_string_lossy()
            .into_owned()
        })
        .collect()
}

#[test]
fn nightly_paths_cover_every_integration_suite() {
    let paths = nightly_pull_request_paths();
    let suites = integration_suites();
    let exempt: BTreeSet<&str> = GLOB_EXEMPT_SUITES.iter().map(|(name, _)| *name).collect();

    let mut failures = Vec::new();

    for (name, reason) in GLOB_EXEMPT_SUITES {
        assert!(
            !reason.is_empty(),
            "exemption for {name} needs a reason the missing glob is sound"
        );
        if !suites.contains(*name) {
            failures.push(format!(
                "GLOB_EXEMPT_SUITES names {name:?}, but tests/integrations/ has no such suite — \
                 delete the stale exemption"
            ));
        }
    }

    for suite in &suites {
        if exempt.contains(suite.as_str()) {
            continue;
        }
        let glob = format!("crates/rig-{suite}/**");
        if !paths.contains(&glob) {
            failures.push(format!(
                "tests/integrations/{suite} has no {glob:?} entry in nightly.yaml's \
                 pull_request paths, so a PR touching only crates/rig-{suite}/ merges with \
                 zero runtime coverage of the code it changed — add the glob (or an exemption \
                 here, with a reason)"
            ));
        }
    }

    // The mirror must hold in both directions: a glob whose suite is gone is
    // dead weight that triggers 20+ minutes of full lane for nothing, and
    // its presence suggests coverage that does not exist.
    for path in &paths {
        if let Some(name) = path
            .strip_prefix("crates/rig-")
            .and_then(|rest| rest.strip_suffix("/**"))
            && !suites.contains(name)
        {
            failures.push(format!(
                "nightly.yaml's paths list {path:?}, but tests/integrations/ has no {name} \
                 suite — the glob triggers the full lane while covering nothing"
            ));
        }
    }

    for required in REQUIRED_PATHS {
        if !paths.contains(*required) {
            failures.push(format!(
                "nightly.yaml's pull_request paths no longer include {required:?} — see \
                 REQUIRED_PATHS in {} for why removing it reopens a closed coverage hole",
                file!(),
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "nightly.yaml path filter drifted from tests/integrations/:\n{}",
        failures.join("\n")
    );
}
