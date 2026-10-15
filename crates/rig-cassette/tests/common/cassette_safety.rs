//! Safety checks for committed cassette fixtures.
//!
//! Suites are discovered from the tree, so these conventions are the
//! invariant:
//!
//! * a provider is a `tests/providers/<provider>/` directory with a
//!   `tests/<provider>.rs` binary, and its cassettes live under
//!   `fixtures/cassettes/<provider>/`;
//! * a cassette wrapper is any function named `with_*cassette*`, and its first
//!   argument is the scenario (a string literal or `CassetteSpec::new("...")`);
//! * a wrapper is called only from its own provider's directory, so the
//!   directory a call sits in names the provider its scenario belongs to.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

use syn::{Expr, ExprLit, Lit};

const CASSETTE_ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/cassettes");

const PROVIDER_SOURCE_ROOT: &str = "tests/providers";

/// Every provider with a `tests/providers/<provider>/` directory and a
/// `tests/<provider>.rs` binary. A directory without a binary is reported by
/// `collect_expected_cassette_paths` when it calls cassette wrappers.
fn discovered_providers() -> BTreeSet<String> {
    provider_source_dirs()
        .into_iter()
        .map(|(provider, _)| provider)
        .filter(|provider| repo_path(&format!("tests/{provider}.rs")).is_file())
        .collect()
}

/// `(name, path)` for every directory directly under `tests/providers/`, sorted.
fn provider_source_dirs() -> Vec<(String, PathBuf)> {
    let mut dirs = Vec::new();
    for entry in fs::read_dir(repo_path(PROVIDER_SOURCE_ROOT))
        .expect("provider source root should be readable")
    {
        let entry = entry.expect("provider source entry should be readable");
        let path = entry.path();
        if path.is_dir() {
            dirs.push((entry.file_name().to_string_lossy().into_owned(), path));
        }
    }
    dirs.sort();
    dirs
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
    //   * every top-level entry under the cassette root must be a directory
    //     named after a discovered provider, so a stray file or a cassette
    //     directory no binary owns fails everywhere rather than silently
    //     escaping the scan;
    //   * every provider's `tests/<provider>.rs` must include this module, so
    //     each cassette directory is provably scanned by exactly the binary
    //     that owns it;
    //   * every provider name must be a valid crate identifier, because
    //     `env!("CARGO_CRATE_NAME")` mangles hyphens to underscores, so a
    //     hyphenated provider would resolve `own_dir` to a path that never
    //     exists and skip its own scan without a single failure.
    let mut failures = Vec::new();

    let discovered = discovered_providers();
    for entry in fs::read_dir(root).expect("cassette root should be readable") {
        let entry = entry.expect("cassette root entry should be readable");
        let name = entry.file_name();
        let name = name.to_string_lossy().into_owned();
        if !entry.path().is_dir() {
            failures.push(format!(
                "crates/rig-cassette/fixtures/cassettes/{name} is not a provider directory; loose files under the \
                 cassette root are scanned by no binary"
            ));
        } else if !discovered.contains(&name) {
            failures.push(format!(
                "crates/rig-cassette/fixtures/cassettes/{name} has no tests/providers/{name}/ directory \
                 with a tests/{name}.rs binary, so no test binary scans it for secrets: add both, \
                 or delete the cassette directory"
            ));
        }
    }
    for provider in &discovered {
        if !provider
            .chars()
            .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_')
        {
            failures.push(format!(
                "provider {provider:?} is not equal to its test binary's CARGO_CRATE_NAME (hyphens and \
                 other non-identifier characters are mangled), so its cassette directory would \
                 be scanned by no binary: rename the provider or its directory"
            ));
        }
        let binary_source = repo_path(&format!("tests/{provider}.rs"));
        if !binary_compiles_cassette_scan(&binary_source) {
            failures.push(format!(
                "tests/{provider}.rs does not include common/cassette_safety.rs as an unconditional \
                 `mod`, so crates/rig-cassette/fixtures/cassettes/{provider} is scanned for secrets by no binary"
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

#[test]
fn cassette_files_match_registered_scenarios() {
    let root = Path::new(CASSETTE_ROOT);
    let actual = collect_yaml_files(root);
    let (expected, mut failures) = collect_expected_cassette_paths();

    let missing = expected
        .difference(&actual)
        .cloned()
        .collect::<BTreeSet<_>>();
    let orphaned = actual
        .difference(&expected)
        .cloned()
        .collect::<BTreeSet<_>>();

    if !missing.is_empty() {
        failures.push(format!(
            "missing cassette file(s) for registered scenario(s):\n{}",
            format_path_list(&missing)
        ));
    }

    if !orphaned.is_empty() {
        failures.push(format!(
            "orphaned cassette file(s) without registered scenario(s):\n{}",
            format_path_list(&orphaned)
        ));
    }

    assert!(
        failures.is_empty(),
        "cassette scenario/file check failed:\n{}",
        failures.join("\n\n")
    );
}

/// A committed reply that is an account failure (a refused credential, a
/// spent quota, a rate limit, an empty balance) must be the declared subject
/// of the cell that recorded it: declared on its `CassetteSpec`, or by the
/// wrapper that presents the rejected credential. The recorder refuses an
/// undeclared one at record time; this keeps the committed corpus to the
/// same rule. Each binary checks its own provider's fixtures.
#[test]
fn account_failures_are_declared_by_their_cells() {
    let own = env!("CARGO_CRATE_NAME");
    let source_dir = repo_path(&format!("{PROVIDER_SOURCE_ROOT}/{own}"));
    if !source_dir.is_dir() {
        return;
    }
    let mut failures = Vec::new();

    // Every helper definition, by name: the file it is in and what it
    // declares. Same-named helpers in different modules stay apart.
    let mut definitions =
        std::collections::BTreeMap::<String, Vec<(PathBuf, BTreeSet<String>)>>::new();
    for dir in [format!("tests/providers/{own}"), "tests/common".to_owned()] {
        let dir = repo_path(&dir);
        if !dir.exists() {
            continue;
        }
        for source in collect_rust_files(&dir) {
            let contents = fs::read_to_string(&source).expect("test source should be readable");
            match rig_test_support::scenario_registry::declaring_functions(&contents) {
                Ok(functions) => {
                    for (function, kinds) in functions {
                        definitions
                            .entry(function)
                            .or_default()
                            .push((source.clone(), kinds.into_iter().collect()));
                    }
                }
                Err(error) => failures.push(format!("{}: {error}", display_repo_path(&source))),
            }
        }
    }

    let mut declared = std::collections::BTreeMap::<PathBuf, BTreeSet<String>>::new();
    for source in collect_rust_files(&source_dir) {
        let sites = match cassette_sites_in_file(&source) {
            Ok(sites) => sites,
            Err(error) => {
                failures.push(error);
                continue;
            }
        };
        for site in sites {
            let candidates = definitions
                .get(&site.wrapper)
                .map(Vec::as_slice)
                .unwrap_or_default();
            let wrapper_kinds = match rig_test_support::scenario_registry::wrapper_declarations(
                candidates, &source,
            ) {
                Ok(kinds) => kinds,
                Err(files) => {
                    failures.push(format!(
                        "{}: `{}` is defined in {} with different account-failure \
                             declarations; rename one so the call names one helper",
                        display_repo_path(&source),
                        site.wrapper,
                        files
                            .iter()
                            .map(|file| display_repo_path(file))
                            .collect::<BTreeSet<_>>()
                            .into_iter()
                            .collect::<Vec<_>>()
                            .join(", ")
                    ));
                    BTreeSet::new()
                }
            };
            let kinds = declared
                .entry(crate::cassettes::cassette_path(own, &site.scenario))
                .or_default();
            kinds.extend(site.declared);
            kinds.extend(wrapper_kinds);
        }
    }

    for fixture in collect_yaml_files(&Path::new(CASSETTE_ROOT).join(own)) {
        let contents = fs::read_to_string(&fixture).expect("cassette should be readable");
        for found in crate::cassettes::cassette_account_failures(&contents) {
            let kind = format!("{:?}", found.failure);
            if !declared
                .get(&fixture)
                .is_some_and(|kinds| kinds.contains(&kind))
            {
                failures.push(format!(
                    "{} interaction {} ({}, status {}) is an undeclared {kind} failure",
                    display_repo_path(&fixture),
                    found.index,
                    found.request,
                    found.status
                ));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "account failures without a declaring cell:\n{}",
        failures.join("\n")
    );
}

/// Fixtures recorded before the recorder refused stored state: each stores a
/// Responses response it never deletes. The cleanup pass removes that state;
/// re-recording one of them must send `store: false` and drop its line.
const STORED_STATE_GRANDFATHERED: &str = include_str!("stored_state_grandfathered.txt");

/// A committed Responses fixture on a storing provider sends `store: false`
/// or deletes what it stored in the same cassette, unless it predates the
/// rule and is grandfathered. A grandfathered line that no longer stores
/// state, or names no fixture, is stale and must go.
#[test]
fn stored_responses_are_deleted_or_grandfathered() {
    let own = env!("CARGO_CRATE_NAME");
    let grandfathered: BTreeSet<&str> = STORED_STATE_GRANDFATHERED
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();
    let root = Path::new(CASSETTE_ROOT);
    let mut failures = Vec::new();
    let mut storing = BTreeSet::new();
    for fixture in collect_yaml_files(&root.join(own)) {
        let contents = fs::read_to_string(&fixture).expect("cassette should be readable");
        let stored = crate::cassettes::cassette_stored_state(own, &contents);
        let relative = fixture
            .strip_prefix(root)
            .expect("fixture under the cassette root")
            .display()
            .to_string();
        if stored.is_empty() {
            continue;
        }
        storing.insert(relative.clone());
        if !grandfathered.contains(relative.as_str()) {
            failures.push(format!(
                "{relative} stores responses it never deletes ({}): send `store: false` or \
                 delete them in the same cassette",
                stored.join(", ")
            ));
        }
    }
    for line in grandfathered
        .iter()
        .filter(|line| line.split('/').next() == Some(own))
    {
        if !storing.contains(*line) {
            failures.push(format!(
                "stored_state_grandfathered.txt lists {line}, which no longer stores state: \
                 remove the line"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "stored Responses state check failed:\n{}",
        failures.join("\n")
    );
}

/// No cassette test compares a volatile key (one the recorder normalizes)
/// exactly outside replay: such a comparison passes in CI and fails only on
/// the next recording. Each binary checks its own provider's sources and the
/// shared ones.
#[test]
fn volatile_keys_are_not_compared_exactly_outside_replay() {
    let own = env!("CARGO_CRATE_NAME");
    let mut failures = Vec::new();
    for dir in [format!("tests/providers/{own}"), "tests/common".to_owned()] {
        let dir = repo_path(&dir);
        if !dir.exists() {
            continue;
        }
        for source in collect_rust_files(&dir) {
            let contents = fs::read_to_string(&source).expect("test source should be readable");
            match rig_test_support::comparison_guard::exact_volatile_comparisons(
                &contents,
                crate::cassettes::volatile_json_keys(),
            ) {
                Ok(found) => failures.extend(
                    found
                        .into_iter()
                        .map(|finding| format!("{}: {finding}", display_repo_path(&source))),
                ),
                Err(error) => failures.push(format!("{}: {error}", display_repo_path(&source))),
            }
        }
    }
    assert!(
        failures.is_empty(),
        "volatile keys compared exactly outside replay:\n{}",
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

        if let Some(stem) = path
            .file_name()
            .and_then(|name| name.to_str())
            .and_then(|name| name.strip_suffix(".requests.json"))
        {
            // A request snapshot describes its fixture's recorded requests.
            if !path.with_file_name(format!("{stem}.yaml")).is_file() {
                failures.push(format!(
                    "{} is a request snapshot with no fixture beside it; run \
                     `cargo xtask cassette snapshots`",
                    display_repo_path(&path)
                ));
            }
            let contents = fs::read_to_string(&path).expect("snapshot should be readable as UTF-8");
            failures.extend(crate::cassettes::artifact_safety_failures(&path, &contents));
            continue;
        }

        if path.extension().and_then(|ext| ext.to_str()) != Some("yaml") {
            continue;
        }

        let contents = fs::read_to_string(&path).expect("cassette should be readable as UTF-8");
        failures.extend(crate::cassettes::cassette_safety_failures(&path, &contents));
    }
}

fn collect_yaml_files(root: &Path) -> BTreeSet<PathBuf> {
    let mut files = BTreeSet::new();
    if root.exists() {
        collect_yaml_files_in_dir(root, &mut files);
    }
    files
}

fn collect_yaml_files_in_dir(dir: &Path, files: &mut BTreeSet<PathBuf>) {
    for entry in fs::read_dir(dir).expect("cassette directory should be readable") {
        let entry = entry.expect("cassette directory entry should be readable");
        let path = entry.path();

        if path.is_dir() {
            collect_yaml_files_in_dir(&path, files);
            continue;
        }

        if path.extension().and_then(|ext| ext.to_str()) == Some("yaml") {
            files.insert(path);
        }
    }
}

fn collect_expected_cassette_paths() -> (BTreeSet<PathBuf>, Vec<String>) {
    let mut expected = BTreeSet::new();
    let mut failures = Vec::new();
    let providers = discovered_providers();

    for (provider, source_dir) in provider_source_dirs() {
        let has_binary = providers.contains(&provider);
        for source_file in collect_rust_files(&source_dir) {
            let sites = match cassette_sites_in_file(&source_file) {
                Ok(sites) => sites,
                Err(error) => {
                    failures.push(error);
                    continue;
                }
            };
            if !has_binary && !sites.is_empty() {
                failures.push(format!(
                    "{} calls cassette wrappers but tests/providers/{provider}/ has no \
                     tests/{provider}.rs binary, so its scenarios are checked by nothing",
                    display_repo_path(&source_file)
                ));
                continue;
            }
            for site in sites {
                if let Some(other) = foreign_provider(&site.wrapper, &provider, &providers) {
                    failures.push(format!(
                        "{} calls {}, which names provider {other:?}, from {provider:?}'s \
                         directory; a wrapper is called only from its own provider's \
                         tests/providers/<provider>/ directory",
                        display_repo_path(&source_file),
                        site.wrapper
                    ));
                }
                expected.insert(crate::cassettes::cassette_path(&provider, &site.scenario));
            }
        }
    }

    (expected, failures)
}

/// The provider a wrapper name names, when it is not `provider` itself:
/// `with_openrouter_openai_cassette` names openrouter, while
/// `with_local_reasoning_content_cassette` names none.
fn foreign_provider<'p>(
    wrapper: &str,
    provider: &str,
    providers: &'p BTreeSet<String>,
) -> Option<&'p str> {
    let segments: Vec<&str> = wrapper.split('_').collect();
    if segments.contains(&provider) {
        return None;
    }
    providers
        .iter()
        .map(String::as_str)
        .find(|candidate| segments.contains(candidate))
}

fn collect_rust_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    collect_rust_files_in_dir(dir, &mut files);
    files.sort();
    files
}

fn collect_rust_files_in_dir(dir: &Path, files: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("cassette source directory should be readable") {
        let entry = entry.expect("cassette source directory entry should be readable");
        let path = entry.path();

        if path.is_dir() {
            collect_rust_files_in_dir(&path, files);
            continue;
        }

        if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
            files.push(path);
        }
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

fn cassette_sites_in_file(
    path: &Path,
) -> Result<Vec<rig_test_support::scenario_registry::ScenarioSite>, String> {
    let source = fs::read_to_string(path)
        .map_err(|error| format!("{} should be readable: {error}", display_repo_path(path)))?;
    rig_test_support::scenario_registry::cassette_scenario_sites(&source)
        .map_err(|error| format!("{}: {error}", display_repo_path(path)))
}

fn repo_path(path: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(path)
}

fn format_path_list(paths: &BTreeSet<PathBuf>) -> String {
    paths
        .iter()
        .map(|path| format!("- {}", display_repo_path(path)))
        .collect::<Vec<_>>()
        .join("\n")
}

fn display_repo_path(path: &Path) -> String {
    path.strip_prefix(env!("CARGO_MANIFEST_DIR"))
        .unwrap_or(path)
        .display()
        .to_string()
}
