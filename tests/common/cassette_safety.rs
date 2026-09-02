//! Safety checks for committed cassette fixtures.
//!
//! There is no registry of providers or wrappers. Suites are discovered from
//! the tree, and the conventions the discovery relies on are the invariant:
//!
//! * a provider is a directory `tests/providers/<provider>/` that has a test
//!   binary `tests/<provider>.rs`, and its cassettes live under
//!   `tests/cassettes/<provider>/`;
//! * a cassette wrapper is any function whose name starts with `with_` and
//!   contains `cassette`; its first argument is the scenario (a string
//!   literal or `CassetteSpec::new("...")`);
//! * a wrapper is called only from inside its own provider's directory, so
//!   the directory a call sits in is the provider its scenario belongs to.
//!
//! A wrapper that names another provider in its identifier but is called
//! from a different provider's directory is rejected, because the scenario
//! would otherwise be registered under the wrong provider and surface as a
//! confusing missing/orphaned pair.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

use syn::visit::{self, Visit};
use syn::{Expr, ExprCall, ExprLit, ItemFn, Lit};

const CASSETTE_ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/cassettes");
const PROVIDER_SOURCE_ROOT: &str = "tests/providers";

struct ProviderCassetteSuite {
    provider: String,
    source_dir: PathBuf,
}

/// Every provider directory under `tests/providers/` that has a `tests/<provider>.rs`
/// binary. Directories without a binary are reported by the caller if they
/// contain cassette wrapper calls (see `collect_expected_cassette_paths`).
fn discovered_suites() -> Vec<ProviderCassetteSuite> {
    provider_source_dirs()
        .into_iter()
        .filter(|(provider, _)| repo_path(&format!("tests/{provider}.rs")).is_file())
        .map(|(provider, source_dir)| ProviderCassetteSuite {
            provider,
            source_dir,
        })
        .collect()
}

/// `(name, path)` for every directory directly under `tests/providers/`, sorted.
fn provider_source_dirs() -> Vec<(String, PathBuf)> {
    let root = repo_path(PROVIDER_SOURCE_ROOT);
    let mut dirs = Vec::new();
    for entry in fs::read_dir(&root).expect("tests/providers should be readable") {
        let entry = entry.expect("tests/providers entry should be readable");
        let path = entry.path();
        if path.is_dir() {
            dirs.push((entry.file_name().to_string_lossy().into_owned(), path));
        }
    }
    dirs.sort();
    dirs
}

/// The naming convention a cassette wrapper follows. Discovery is by call
/// site, not by definition, because some providers define their wrappers
/// through macros.
fn is_cassette_wrapper_name(name: &str) -> bool {
    name.starts_with("with_") && name.contains("cassette")
}

#[test]
fn cassettes_do_not_contain_obvious_secrets() {
    let root = Path::new(CASSETTE_ROOT);
    if !root.exists() {
        return;
    }

    // Each provider binary scans only its own `tests/cassettes/<provider>`
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
    //   * every top-level entry under `tests/cassettes` must be a directory
    //     named after a discovered suite (a `tests/providers/<provider>/`
    //     directory with a `tests/<provider>.rs` binary) — a stray file or a
    //     cassette directory no binary owns fails everywhere rather than
    //     silently escaping the scan;
    //   * every discovered suite's `tests/<provider>.rs` must include this
    //     module — so each cassette directory is provably scanned by exactly
    //     the binary that owns it, and adding a provider without wiring the
    //     scan into its binary fails everywhere too;
    //   * every provider name must be a valid crate identifier —
    //     `env!("CARGO_CRATE_NAME")` mangles hyphens to underscores, so a
    //     hyphenated provider would resolve `own_dir` to a path that never
    //     exists and skip its own scan without a single failure.
    let mut failures = Vec::new();

    let suites = discovered_suites();
    let discovered: BTreeSet<&str> = suites.iter().map(|suite| suite.provider.as_str()).collect();
    for entry in fs::read_dir(root).expect("cassette root should be readable") {
        let entry = entry.expect("cassette root entry should be readable");
        let name = entry.file_name();
        let name = name.to_string_lossy().into_owned();
        if !entry.path().is_dir() {
            failures.push(format!(
                "tests/cassettes/{name} is not a provider directory; loose files under the \
                 cassette root are scanned by no binary"
            ));
        } else if !discovered.contains(name.as_str()) {
            failures.push(format!(
                "tests/cassettes/{name} has no tests/providers/{name}/ directory with a \
                 tests/{name}.rs binary, so no test binary scans it for secrets — add both, \
                 or delete the cassette directory"
            ));
        }
    }
    for suite in &suites {
        if !suite
            .provider
            .chars()
            .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_')
        {
            failures.push(format!(
                "provider {:?} is not equal to its test binary's CARGO_CRATE_NAME (hyphens and \
                 other non-identifier characters are mangled), so its cassette directory would \
                 be scanned by no binary — rename the provider or its directory",
                suite.provider
            ));
        }
        // A provider with no cassette directory has nothing to scan; the
        // moment `tests/cassettes/<provider>` appears, its binary must compile
        // this module or every binary fails.
        if !root.join(&suite.provider).is_dir() {
            continue;
        }
        let binary_source = repo_path(&format!("tests/{}.rs", suite.provider));
        if !binary_compiles_cassette_scan(&binary_source) {
            failures.push(format!(
                "tests/{}.rs does not include common/cassette_safety.rs as an unconditional \
                 `mod`, so tests/cassettes/{} is scanned for secrets by no binary",
                suite.provider, suite.provider
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

    let suites = discovered_suites();
    let providers: BTreeSet<&str> = suites.iter().map(|suite| suite.provider.as_str()).collect();

    for suite in &suites {
        for source_file in collect_rust_files(&suite.source_dir) {
            match cassette_scenarios_in_file(&source_file, &suite.provider, &providers) {
                Ok(scenarios) => {
                    for scenario in scenarios {
                        expected
                            .insert(crate::cassettes::cassette_path(&suite.provider, &scenario));
                    }
                }
                Err(error) => failures.push(error),
            }
        }
    }

    // A provider directory with no binary is scanned by nothing: any cassette
    // wrapper call in it would register scenarios nowhere.
    for (provider, source_dir) in provider_source_dirs() {
        if providers.contains(provider.as_str()) {
            continue;
        }
        for source_file in collect_rust_files(&source_dir) {
            if let Ok(scenarios) = cassette_scenarios_in_file(&source_file, &provider, &providers)
                && !scenarios.is_empty()
            {
                failures.push(format!(
                    "{} calls cassette wrappers but tests/providers/{provider}/ has no \
                     tests/{provider}.rs binary, so its scenarios are checked by nothing",
                    display_repo_path(&source_file)
                ));
            }
        }
    }

    (expected, failures)
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

fn cassette_scenarios_in_file(
    path: &Path,
    provider: &str,
    providers: &BTreeSet<&str>,
) -> Result<Vec<String>, String> {
    let contents = fs::read_to_string(path)
        .map_err(|error| format!("{} should be readable: {error}", display_repo_path(path)))?;
    let syntax = syn::parse_file(&contents)
        .map_err(|error| format!("{} should parse as Rust: {error}", display_repo_path(path)))?;
    let mut visitor = CassetteScenarioVisitor {
        path,
        provider,
        providers,
        scenarios: Vec::new(),
        failures: Vec::new(),
    };
    visitor.visit_file(&syntax);

    if visitor.failures.is_empty() {
        Ok(visitor.scenarios)
    } else {
        Err(visitor.failures.join("\n"))
    }
}

struct CassetteScenarioVisitor<'a> {
    path: &'a Path,
    provider: &'a str,
    providers: &'a BTreeSet<&'a str>,
    scenarios: Vec<String>,
    failures: Vec<String>,
}

impl<'ast, 'a> Visit<'ast> for CassetteScenarioVisitor<'a> {
    fn visit_item_fn(&mut self, node: &'ast ItemFn) {
        // A `#[ignore]`d test documents that its cassette isn't recorded yet
        // (e.g. no provider API key available to record with); don't require
        // a file for scenarios it references.
        if node.attrs.iter().any(|attr| attr.path().is_ident("ignore")) {
            return;
        }
        // A wrapper's own body forwards its scenario to a base wrapper; that is
        // a definition, not a call site, and its argument is a variable.
        if is_cassette_wrapper_name(&node.sig.ident.to_string()) {
            return;
        }

        visit::visit_item_fn(self, node);
    }

    fn visit_expr_call(&mut self, node: &'ast ExprCall) {
        if let Some(wrapper_name) = cassette_wrapper_name(node)
            && is_cassette_wrapper_name(&wrapper_name)
        {
            if let Some(other) =
                foreign_provider_in_wrapper(&wrapper_name, self.provider, self.providers)
            {
                self.failures.push(format!(
                    "{} calls {wrapper_name}, which names provider {other:?}, from provider \
                     {:?}'s directory; a wrapper is called only from its own provider's \
                     tests/providers/<provider>/ directory",
                    display_repo_path(self.path),
                    self.provider
                ));
            }
            match node.args.first() {
                Some(expr) => match cassette_scenario_value(expr) {
                    Some(scenario) => self.scenarios.push(scenario),
                    None => self.failures.push(format!(
                        "{} calls {wrapper_name} without a string-literal cassette scenario",
                        display_repo_path(self.path)
                    )),
                },
                _ => self.failures.push(format!(
                    "{} calls {wrapper_name} without a string-literal cassette scenario",
                    display_repo_path(self.path)
                )),
            }
        }

        visit::visit_expr_call(self, node);
    }
}

/// The provider a wrapper name points at, when it is not the directory's own.
/// Segments of the identifier are compared against the discovered provider
/// names: `with_openrouter_openai_cassette` in openrouter's directory names its
/// own provider and passes; `with_openai_cassette` in openrouter's directory
/// does not. Names that mention no provider (`with_local_reasoning_content_cassette`)
/// pass anywhere.
fn foreign_provider_in_wrapper<'p>(
    wrapper_name: &str,
    provider: &str,
    providers: &BTreeSet<&'p str>,
) -> Option<&'p str> {
    let segments: Vec<&str> = wrapper_name.split('_').collect();
    if segments.contains(&provider) {
        return None;
    }
    providers
        .iter()
        .copied()
        .find(|candidate| segments.contains(candidate))
}

fn cassette_scenario_value(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Lit(ExprLit {
            lit: Lit::Str(scenario),
            ..
        }) => Some(scenario.value()),
        Expr::Call(call) if is_cassette_spec_new(call) => call.args.first().and_then(|expr| {
            let Expr::Lit(ExprLit {
                lit: Lit::Str(scenario),
                ..
            }) = expr
            else {
                return None;
            };

            Some(scenario.value())
        }),
        Expr::MethodCall(method_call) => cassette_scenario_value(&method_call.receiver),
        Expr::Paren(paren) => cassette_scenario_value(&paren.expr),
        _ => None,
    }
}

fn is_cassette_spec_new(call: &ExprCall) -> bool {
    let Expr::Path(path) = call.func.as_ref() else {
        return false;
    };

    let mut segments = path.path.segments.iter().rev();
    matches!(
        (segments.next(), segments.next()),
        (Some(method), Some(receiver))
            if method.ident == "new" && receiver.ident == "CassetteSpec"
    )
}

fn cassette_wrapper_name(node: &ExprCall) -> Option<String> {
    let Expr::Path(path) = node.func.as_ref() else {
        return None;
    };

    path.path
        .segments
        .last()
        .map(|segment| segment.ident.to_string())
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
