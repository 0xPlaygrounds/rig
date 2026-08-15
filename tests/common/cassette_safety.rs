//! Safety checks for committed cassette fixtures.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

use syn::visit::{self, Visit};
use syn::{Expr, ExprCall, ExprLit, ItemFn, Lit};

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
            "with_openai_cassette_bogus_key",
            "with_openai_completions_cassette",
            "with_openai_cassette_result",
            "with_openai_completions_cassette_result",
            "with_openai_vllm_cassette",
            "with_local_reasoning_content_cassette",
        ],
    },
    ProviderCassetteSuite {
        provider: "chatgpt",
        source_dir: "tests/providers/chatgpt/cassette",
        wrapper_names: &[
            "with_chatgpt_cassette",
            "with_chatgpt_cassette_default_instructions",
            "with_chatgpt_noninteractive_oauth_cassette",
        ],
    },
    ProviderCassetteSuite {
        provider: "copilot",
        source_dir: "tests/providers/copilot",
        wrapper_names: &[
            "with_copilot_cassette",
            "with_copilot_cassette_result",
            "with_copilot_noninteractive_oauth_cassette",
        ],
    },
    ProviderCassetteSuite {
        provider: "anthropic",
        source_dir: "tests/providers/anthropic/cassette",
        wrapper_names: &[
            "with_anthropic_cassette",
            "with_anthropic_cassette_result",
            "with_anthropic_cassette_bogus_key",
            "with_anthropic_files_cassette",
            "with_anthropic_gateway_cassette",
            "with_anthropic_stop_sequence_cassette",
            "with_anthropic_empty_stop_cassette",
        ],
    },
    ProviderCassetteSuite {
        provider: "bedrock",
        source_dir: "tests/providers/bedrock/cassette",
        wrapper_names: &["with_bedrock_cassette"],
    },
    ProviderCassetteSuite {
        provider: "doubleword",
        source_dir: "tests/providers/doubleword/cassette",
        wrapper_names: &[
            "with_doubleword_cassette",
            "with_doubleword_cassette_result",
        ],
    },
    ProviderCassetteSuite {
        provider: "cohere",
        source_dir: "tests/providers/cohere/cassette",
        wrapper_names: &["with_cohere_cassette"],
    },
    ProviderCassetteSuite {
        provider: "venice",
        source_dir: "tests/providers/venice/cassette",
        wrapper_names: &[
            "with_venice_cassette",
            "with_venice_cassette_result",
            "with_venice_direct_cassette",
        ],
    },
    ProviderCassetteSuite {
        provider: "gemini",
        source_dir: "tests/providers/gemini/cassette",
        wrapper_names: &[
            "with_gemini_cassette",
            "with_gemini_cassette_bogus_key",
            "with_gemini_code_execution_cassette",
            "with_gemini_interactions_cassette",
            "with_gemini_stream_terminal_cassette",
            "with_gemini_thought_text_cassette",
        ],
    },
    ProviderCassetteSuite {
        provider: "ollama",
        source_dir: "tests/providers/ollama/cassette",
        wrapper_names: &["with_ollama_cassette"],
    },
    ProviderCassetteSuite {
        provider: "llamafile",
        source_dir: "tests/providers/llamafile/cassette",
        wrapper_names: &["with_llamafile_cassette"],
    },
    ProviderCassetteSuite {
        provider: "xai",
        source_dir: "tests/providers/xai",
        wrapper_names: &[
            "with_xai_cassette",
            "with_xai_cassette_bogus_key",
            "with_xai_cassette_result",
        ],
    },
    ProviderCassetteSuite {
        provider: "openrouter",
        source_dir: "tests/providers/openrouter/cassette",
        wrapper_names: &[
            "with_openrouter_cassette",
            "with_openrouter_cassette_result",
            "with_openrouter_openai_cassette",
        ],
    },
    ProviderCassetteSuite {
        provider: "deepseek",
        source_dir: "tests/providers/deepseek",
        wrapper_names: &["with_deepseek_cassette", "with_deepseek_cassette_result"],
    },
    ProviderCassetteSuite {
        provider: "groq",
        source_dir: "tests/providers/groq",
        wrapper_names: &[
            "with_groq_cassette_result",
            "with_groq_cassette_bogus_key_result",
        ],
    },
    ProviderCassetteSuite {
        provider: "mistral",
        source_dir: "tests/providers/mistral",
        wrapper_names: &["with_mistral_cassette_result"],
    },
    ProviderCassetteSuite {
        provider: "perplexity",
        source_dir: "tests/providers/perplexity/cassette",
        wrapper_names: &["with_perplexity_cassette"],
    },
    ProviderCassetteSuite {
        provider: "mistralrs",
        source_dir: "tests/providers/mistralrs/cassette",
        wrapper_names: &[
            "with_mistralrs_cassette",
            "with_mistralrs_completions_cassette",
            "with_mistralrs_raw_cassette",
        ],
    },
];

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
    //     named after a suite registered in `PROVIDER_CASSETTE_SUITES` — a
    //     stray file or an unregistered provider directory fails everywhere
    //     rather than silently escaping the scan;
    //   * every registered suite's `tests/<provider>.rs` must include this
    //     module — so each registered directory is provably scanned by
    //     exactly the binary that owns it, and adding a suite without wiring
    //     the scan into its binary fails everywhere too;
    //   * every registered provider name must be a valid crate identifier —
    //     `env!("CARGO_CRATE_NAME")` mangles hyphens to underscores, so a
    //     hyphenated provider would resolve `own_dir` to a path that never
    //     exists and skip its own scan without a single failure.
    let mut failures = Vec::new();

    let registered: BTreeSet<&str> = PROVIDER_CASSETTE_SUITES
        .iter()
        .map(|suite| suite.provider)
        .collect();
    for entry in fs::read_dir(root).expect("cassette root should be readable") {
        let entry = entry.expect("cassette root entry should be readable");
        let name = entry.file_name();
        let name = name.to_string_lossy().into_owned();
        if !entry.path().is_dir() {
            failures.push(format!(
                "tests/cassettes/{name} is not a provider directory; loose files under the \
                 cassette root are scanned by no binary"
            ));
        } else if !registered.contains(name.as_str()) {
            failures.push(format!(
                "tests/cassettes/{name} has no PROVIDER_CASSETTE_SUITES entry, so no test \
                 binary scans it for secrets — register it in \
                 tests/common/cassette_safety.rs"
            ));
        }
    }
    for suite in PROVIDER_CASSETTE_SUITES {
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
    fn visit_item_fn(&mut self, node: &'ast ItemFn) {
        // A `#[ignore]`d test documents that its cassette isn't recorded yet
        // (e.g. no provider API key available to record with); don't require
        // a file for scenarios it references.
        if node.attrs.iter().any(|attr| attr.path().is_ident("ignore")) {
            return;
        }

        visit::visit_item_fn(self, node);
    }

    fn visit_expr_call(&mut self, node: &'ast ExprCall) {
        if let Some(wrapper_name) = cassette_wrapper_name(node)
            && self.wrapper_names.contains(&wrapper_name.as_str())
        {
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
