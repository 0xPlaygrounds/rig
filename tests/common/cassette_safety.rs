//! Safety checks for committed cassette fixtures.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

use syn::visit::{self, Visit};
use syn::{Expr, ExprCall, ExprLit, Lit};

const CASSETTE_ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/cassettes");

struct ProviderCassetteSuite {
    provider: &'static str,
    source_dir: &'static str,
    wrapper_names: &'static [&'static str],
}

const PROVIDER_CASSETTE_SUITES: &[ProviderCassetteSuite] = &[
    ProviderCassetteSuite {
        provider: "openai",
        source_dir: "tests/providers/openai/cassette",
        wrapper_names: &[
            "with_openai_cassette",
            "with_openai_completions_cassette",
            "with_openai_cassette_result",
            "with_openai_completions_cassette_result",
        ],
    },
    ProviderCassetteSuite {
        provider: "anthropic",
        source_dir: "tests/providers/anthropic/cassette",
        wrapper_names: &[
            "with_anthropic_cassette",
            "with_anthropic_cassette_result",
            "with_anthropic_files_cassette",
        ],
    },
    ProviderCassetteSuite {
        provider: "gemini",
        source_dir: "tests/providers/gemini/cassette",
        wrapper_names: &["with_gemini_cassette", "with_gemini_interactions_cassette"],
    },
];

#[test]
fn cassettes_do_not_contain_obvious_secrets() {
    let root = Path::new(CASSETTE_ROOT);
    if !root.exists() {
        return;
    }

    let mut failures = Vec::new();
    scan_dir(root, &mut failures);

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

    for suite in PROVIDER_CASSETTE_SUITES {
        let source_dir = repo_path(suite.source_dir);
        if !source_dir.exists() {
            failures.push(format!(
                "cassette source directory does not exist: {}",
                display_repo_path(&source_dir)
            ));
            continue;
        }

        for source_file in collect_rust_files(&source_dir) {
            match cassette_scenarios_in_file(&source_file, suite.wrapper_names) {
                Ok(scenarios) => {
                    for scenario in scenarios {
                        expected.insert(crate::cassettes::cassette_path(suite.provider, &scenario));
                    }
                }
                Err(error) => failures.push(error),
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

fn cassette_scenarios_in_file(
    path: &Path,
    wrapper_names: &[&'static str],
) -> Result<Vec<String>, String> {
    let contents = fs::read_to_string(path)
        .map_err(|error| format!("{} should be readable: {error}", display_repo_path(path)))?;
    let syntax = syn::parse_file(&contents)
        .map_err(|error| format!("{} should parse as Rust: {error}", display_repo_path(path)))?;
    let mut visitor = CassetteScenarioVisitor {
        path,
        wrapper_names,
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
    wrapper_names: &'a [&'static str],
    scenarios: Vec<String>,
    failures: Vec<String>,
}

impl<'ast, 'a> Visit<'ast> for CassetteScenarioVisitor<'a> {
    fn visit_expr_call(&mut self, node: &'ast ExprCall) {
        if let Some(wrapper_name) = cassette_wrapper_name(node)
            && self.wrapper_names.contains(&wrapper_name.as_str())
        {
            match node.args.first() {
                Some(Expr::Lit(ExprLit {
                    lit: Lit::Str(scenario),
                    ..
                })) => self.scenarios.push(scenario.value()),
                _ => self.failures.push(format!(
                    "{} calls {wrapper_name} without a string-literal cassette scenario",
                    display_repo_path(self.path)
                )),
            }
        }

        visit::visit_expr_call(self, node);
    }
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
