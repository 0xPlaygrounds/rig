//! Safety checks for committed cassette fixtures.
//!
//! Scope is secrets, and only secrets: every committed cassette must be
//! scanned — by exactly one test binary, so the cost is paid once — for
//! credentials that should never have been recorded.
//!
//! Fixture *ownership* is no longer checked here. `cargo xtask
//! check-cassette-provenance` owns it and checks strictly more: it reconciles
//! the declaration manifest against the committed fixtures and against the
//! scenario literals discovered in the provider suites, so a fixture nobody
//! declared, a declaration with no fixture and a scenario with neither all
//! fail there rather than only the two of those a test-time YAML/AST diff
//! could see.
//!
//! The provider list is read from that same manifest,
//! `crates/rig-cassette/fixtures/scenarios.json`, rather than duplicated as a
//! constant here. A second copy of the registry is a second place to forget,
//! and forgetting it *here* means a provider's cassettes are scanned for
//! secrets by nobody.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

use syn::{Expr, ExprLit, Lit};

const CASSETTE_ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/cassettes");

/// Crate-relative path of the declaration manifest: the single registry of
/// provider suites.
const MANIFEST: &str = "fixtures/scenarios.json";

/// The provider suites declared in the manifest, in declaration order.
///
/// Read straight out of the JSON with `serde_json` rather than through
/// `rig_test_support::provenance`: this module needs one field off the top of
/// each provider entry and nothing else, and the strict manifest parser
/// behind `provenance` is deliberately crate-private. The strict parse still
/// happens — in the xtask guard and in every test that resolves a scenario's
/// provenance — so a malformed manifest fails loudly elsewhere; what matters
/// here is only that the set of provider directories comes from the registry
/// instead of from a hand-maintained copy of it.
fn registered_providers() -> Vec<String> {
    let path = repo_path(MANIFEST);
    let contents = fs::read_to_string(&path)
        .expect("crates/rig-cassette/fixtures/scenarios.json should be readable");
    let manifest: serde_json::Value = serde_json::from_str(&contents)
        .expect("crates/rig-cassette/fixtures/scenarios.json should be valid JSON");
    let providers = manifest
        .get("providers")
        .and_then(serde_json::Value::as_array)
        .expect("crates/rig-cassette/fixtures/scenarios.json should have a `providers` array")
        .iter()
        .map(|entry| {
            entry
                .get("provider")
                .and_then(serde_json::Value::as_str)
                .expect("each `providers` entry should carry a string `provider` name")
                .to_owned()
        })
        .collect::<Vec<_>>();

    // An empty registry would make the partition below vacuous for the
    // per-provider half while still failing the directory half, so it is
    // rejected outright: the failure should name the cause, not 18 unrelated
    // "unregistered directory" lines.
    assert!(
        !providers.is_empty(),
        "{} declares no providers, so no cassette directory is registered for scanning",
        display_repo_path(&path)
    );

    providers
}

#[test]
fn cassettes_do_not_contain_obvious_secrets() {
    let root = Path::new(CASSETTE_ROOT);
    if !root.exists() {
        return;
    }

    // Each provider binary scans only its own `crates/rig-cassette/fixtures/cassettes/<provider>`
    // directory. This module compiles into every provider test binary, and
    // the scan (YAML parse + scrub + re-serialize + base64 decode + several
    // regex families per file) is expensive — when every binary scanned the
    // whole tree, CI ran the identical full-tree scan once per binary, and
    // that duplication alone was the single largest execution cost in the PR
    // gate's test sweep (~16s × 16 binaries per run).
    //
    // Scoping is safe because the partition below is asserted, in every
    // binary, before anything is skipped:
    //
    //   * every top-level entry under `crates/rig-cassette/fixtures/cassettes` must be a directory
    //     named after a provider declared in `fixtures/scenarios.json` — a
    //     stray file or an undeclared provider directory fails everywhere
    //     rather than silently escaping the scan;
    //   * every declared provider's `tests/<provider>.rs` must include this
    //     module — so each declared directory is provably scanned by exactly
    //     the binary that owns it, and declaring a provider without wiring
    //     the scan into its binary fails everywhere too;
    //   * every declared provider name must be a valid crate identifier —
    //     `env!("CARGO_CRATE_NAME")` mangles hyphens to underscores, so a
    //     hyphenated provider would resolve `own_dir` to a path that never
    //     exists and skip its own scan without a single failure.
    let mut failures = Vec::new();

    let providers = registered_providers();
    let registered: BTreeSet<&str> = providers.iter().map(String::as_str).collect();
    for entry in fs::read_dir(root).expect("cassette root should be readable") {
        let entry = entry.expect("cassette root entry should be readable");
        let name = entry.file_name();
        let name = name.to_string_lossy().into_owned();
        if !entry.path().is_dir() {
            failures.push(format!(
                "crates/rig-cassette/fixtures/cassettes/{name} is not a provider directory; loose files under the \
                 cassette root are scanned by no binary"
            ));
        } else if !registered.contains(name.as_str()) {
            failures.push(format!(
                "crates/rig-cassette/fixtures/cassettes/{name} has no entry in \
                 crates/rig-cassette/fixtures/scenarios.json, so no test binary scans it for \
                 secrets — declare the provider there"
            ));
        }
    }
    for provider in &providers {
        if !provider
            .chars()
            .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_')
        {
            failures.push(format!(
                "provider {provider:?} is not equal to its test binary's CARGO_CRATE_NAME (hyphens \
                 and other non-identifier characters are mangled), so its cassette directory would \
                 be scanned by no binary — rename the provider or its directory"
            ));
        }
        let binary_source = repo_path(&format!("tests/{provider}.rs"));
        if !binary_compiles_cassette_scan(&binary_source) {
            failures.push(format!(
                "tests/{provider}.rs does not include common/cassette_safety.rs as an \
                 unconditional `mod`, so crates/rig-cassette/fixtures/cassettes/{provider} is \
                 scanned for secrets by no binary"
            ));
        }
    }

    let own_dir = root.join(env!("CARGO_CRATE_NAME"));
    if own_dir.is_dir() {
        scan_dir(&own_dir, &mut failures);
    }

    assert!(
        failures.is_empty(),
        "cassette secret scan failed:\n{}",
        failures.join("\n")
    );
}

fn scan_dir(dir: &Path, failures: &mut Vec<String>) {
    for entry in fs::read_dir(dir).expect("cassette directory should be readable") {
        let entry = entry.expect("cassette directory entry should be readable");
        let path = entry.path();

        if path.is_dir() {
            scan_dir(&path, failures);
            continue;
        }

        if path.extension().and_then(|ext| ext.to_str()) != Some("yaml") {
            continue;
        }

        let contents = fs::read_to_string(&path).expect("cassette should be readable as UTF-8");
        failures.extend(crate::cassettes::cassette_safety_failures(&path, &contents));
    }
}

/// Structural, not substring: the guarded claim is "this binary *compiles*
/// the secret scan", so the check must parse the source and find an actual
/// `#[path = ".../common/cassette_safety.rs"] mod …` item with no `#[cfg]`
/// attached. A raw `contents.contains(...)` would stay satisfied by a
/// commented-out include or by a cfg-gated one — a false green on the safety
/// net itself, the same paper-claim failure mode the streaming-conformance
/// registry's CI-step check guards against.
fn binary_compiles_cassette_scan(source: &Path) -> bool {
    let Ok(contents) = fs::read_to_string(source) else {
        return false;
    };
    let Ok(syntax) = syn::parse_file(&contents) else {
        return false;
    };
    syntax.items.iter().any(|item| {
        let syn::Item::Mod(module) = item else {
            return false;
        };
        let cfg_gated = module
            .attrs
            .iter()
            .any(|attr| attr.path().is_ident("cfg") || attr.path().is_ident("cfg_attr"));
        let includes_scan = module.attrs.iter().any(|attr| {
            attr.path().is_ident("path")
                && matches!(
                    &attr.meta,
                    syn::Meta::NameValue(name_value) if matches!(
                        &name_value.value,
                        Expr::Lit(ExprLit { lit: Lit::Str(path), .. })
                            if path.value().ends_with("common/cassette_safety.rs")
                    )
                )
        });
        includes_scan && !cfg_gated
    })
}

fn repo_path(path: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(path)
}

fn display_repo_path(path: &Path) -> String {
    path.strip_prefix(env!("CARGO_MANIFEST_DIR"))
        .unwrap_or(path)
        .display()
        .to_string()
}
