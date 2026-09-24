//! Safety checks for committed cassette fixtures.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

use syn::{Expr, ExprLit, Lit};

const CASSETTE_ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/cassettes");

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
            "with_openai_corpus_retrieval_cassette",
            "with_openai_corpus_output_cassette",
            "with_openai_corpus_host_cassette",
            "with_openai_corpus_delta_cassette",
            "with_openai_corpus_breadth_cassette",
            "with_openai_lifecycle_cassette",
            "with_openai_prompt_caching_cassette",
            "with_openai_completions_prompt_caching_cassette",
            "with_openai_turn_metadata_cassette",
            "with_openai_cassette_bogus_key",
            "with_openai_completions_cassette",
            "with_openai_cassette_result",
            "with_openai_completions_cassette_result",
            "with_openai_vllm_cassette",
            "with_local_reasoning_content_cassette",
            "with_openai_refusal_cassette",
            "with_openai_max_tokens_cassette",
            "with_openai_image_params_cassette",
            "with_openai_truncation_cassette",
            "with_openai_chat_stream_logprobs_cassette_result",
            "with_openai_tool_truncation_cassette_result",
            "with_openai_tool_lifecycle_cassette_result",
            "with_openai_terminal_metadata_cassette_result",
            "with_openai_history_roundtrip_cassette_result",
            "with_openai_transcription_cassette",
            "with_openai_audio_cassette",
            "with_openai_websocket_cassette",
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
            "with_anthropic_lifecycle_cassette",
            "with_anthropic_turn_metadata_cassette",
            "with_anthropic_cassette_result",
            "with_anthropic_cassette_bogus_key",
            "with_anthropic_files_cassette",
            "with_anthropic_gateway_cassette",
            "with_anthropic_stop_sequence_cassette",
            "with_anthropic_empty_stop_cassette",
            "with_anthropic_reasoning_usage_cassette",
            "with_anthropic_corpus_request_shape_cassette",
            "with_anthropic_corpus_hooks_cassette",
            "with_anthropic_corpus_serving_cassette",
            "with_anthropic_corpus_outcome_cassette",
            "with_anthropic_corpus_endings_cassette",
            "with_anthropic_corpus_output_cassette",
            "with_anthropic_corpus_host_cassette",
            "with_anthropic_corpus_memory_cassette",
            "with_anthropic_corpus_shaping_cassette",
            "with_anthropic_corpus_oracle_cassette",
            "with_anthropic_corpus_causal_cassette",
            "with_anthropic_corpus_layers_cassette",
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
            "with_doubleword_prompt_caching_cassette",
            "with_doubleword_cassette",
            "with_doubleword_bogus_key_cassette",
            "with_doubleword_cassette_result",
            "with_doubleword_embedding_cassette",
        ],
    },
    ProviderCassetteSuite {
        provider: "cohere",
        source_dir: "tests/providers/cohere/cassette",
        wrapper_names: &[
            "with_cohere_cassette",
            "with_cohere_prompt_caching_cassette",
        ],
    },
    ProviderCassetteSuite {
        provider: "venice",
        source_dir: "tests/providers/venice/cassette",
        wrapper_names: &[
            "with_venice_prompt_caching_cassette",
            "with_venice_cassette",
            "with_venice_cassette_result",
            "with_venice_direct_cassette",
        ],
    },
    ProviderCassetteSuite {
        provider: "gemini",
        source_dir: "tests/providers/gemini/cassette",
        wrapper_names: &[
            "with_gemini_prompt_caching_cassette",
            "with_gemini_cassette",
            "with_gemini_corpus_retrieval_cassette",
            "with_gemini_corpus_delta_cassette",
            "with_gemini_corpus_breadth_cassette",
            "with_gemini_lifecycle_cassette",
            "with_gemini_turn_metadata_cassette",
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
        provider: "llamacpp",
        source_dir: "tests/providers/llamacpp/cassette",
        wrapper_names: &[
            "with_llamacpp_cassette",
            "with_llamacpp_cassette_result",
            "with_llamacpp_bare_openai_cassette",
            "with_llamacpp_embeddings_cassette",
            "with_llamacpp_vision_cassette",
            "with_llamacpp_small_context_cassette",
            "with_llamacpp_no_jinja_cassette",
            "with_llamacpp_rerank_cassette",
            "with_llamacpp_pooling_none_cassette",
            "with_llamacpp_causal_embeddings_cassette",
            "with_llamacpp_competent_cassette",
            "with_llamacpp_llama_family_cassette",
            "with_llamacpp_mistral_family_cassette",
            "with_llamacpp_gemma_family_cassette",
            "with_llamacpp_prompt_caching_cassette",
            "with_llamacpp_large_vision_cassette",
            "with_llamacpp_raw_http_cassette",
            "with_llamacpp_api_key_cassette",
            "with_llamacpp_missing_api_key_cassette",
        ],
    },
    ProviderCassetteSuite {
        provider: "xai",
        source_dir: "tests/providers/xai",
        wrapper_names: &[
            "with_xai_prompt_caching_cassette",
            "with_xai_cassette",
            "with_xai_cassette_bogus_key",
            "with_xai_cassette_result",
        ],
    },
    ProviderCassetteSuite {
        provider: "openrouter",
        source_dir: "tests/providers/openrouter/cassette",
        wrapper_names: &[
            "with_openrouter_prompt_caching_cassette",
            "with_openrouter_cassette",
            "with_openrouter_cassette_result",
            "with_openrouter_cassette_bogus_key_result",
            "with_openrouter_openai_cassette",
            "with_openrouter_refusal_cassette",
            "with_openrouter_usage_cassette",
            "with_openrouter_stream_logprobs_cassette_result",
            "with_openrouter_tool_truncation_cassette_result",
            "with_openrouter_tool_lifecycle_cassette_result",
            "with_openrouter_terminal_metadata_cassette_result",
            "with_openrouter_history_roundtrip_cassette_result",
            "with_openrouter_reasoning_tool_order_cassette_result",
        ],
    },
    ProviderCassetteSuite {
        provider: "deepseek",
        source_dir: "tests/providers/deepseek",
        wrapper_names: &[
            "with_deepseek_prompt_caching_cassette",
            "with_deepseek_cassette",
            "with_deepseek_cassette_result",
            "with_deepseek_cassette_bogus_key_result",
            "with_deepseek_truncation_cassette_result",
            "with_deepseek_block_order_cassette_result",
            "with_deepseek_wire_shape_cassette_result",
            "with_deepseek_followup_hunt_cassette_result",
            "with_deepseek_stream_logprobs_cassette_result",
        ],
    },
    ProviderCassetteSuite {
        provider: "groq",
        source_dir: "tests/providers/groq",
        wrapper_names: &[
            "with_groq_prompt_caching_cassette",
            "with_groq_cassette_result",
            "with_groq_cassette_bogus_key_result",
        ],
    },
    ProviderCassetteSuite {
        provider: "mistral",
        source_dir: "tests/providers/mistral",
        wrapper_names: &[
            "with_mistral_embedding_cassette",
            "with_mistral_prompt_caching_cassette",
            "with_mistral_cassette_result",
            "with_mistral_multimodal_cassette",
            "with_mistral_cassette_bogus_key_result",
            "with_mistral_capability_cassette",
            "with_mistral_terminal_metadata_cassette_result",
            "with_mistral_tool_truncation_cassette_result",
            "with_mistral_tool_lifecycle_cassette_result",
            "with_mistral_history_roundtrip_cassette_result",
            "with_mistral_request_shape_cassette_result",
            "with_mistral_logprobs_rejection_cassette_result",
        ],
    },
    ProviderCassetteSuite {
        provider: "perplexity",
        source_dir: "tests/providers/perplexity/cassette",
        wrapper_names: &[
            "with_perplexity_cassette",
            "with_perplexity_prompt_caching_cassette",
        ],
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
                "crates/rig-cassette/fixtures/cassettes/{name} is not a provider directory; loose files under the \
                 cassette root are scanned by no binary"
            ));
        } else if !registered.contains(name.as_str()) {
            failures.push(format!(
                "crates/rig-cassette/fixtures/cassettes/{name} has no PROVIDER_CASSETTE_SUITES entry, so no test \
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
                 `mod`, so crates/rig-cassette/fixtures/cassettes/{} is scanned for secrets by no binary",
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

/// A committed reply that is an account failure (a refused credential, a
/// spent quota, a rate limit, an empty balance) must be the declared subject
/// of the cell that recorded it: declared on its `CassetteSpec`, or by the
/// wrapper that presents the rejected credential. The recorder refuses an
/// undeclared one at record time; this keeps the committed corpus to the
/// same rule. Each binary checks its own provider's fixtures.
#[test]
fn account_failures_are_declared_by_their_cells() {
    let own = env!("CARGO_CRATE_NAME");
    let Some(suite) = PROVIDER_CASSETTE_SUITES
        .iter()
        .find(|suite| suite.provider == own)
    else {
        return;
    };
    let mut failures = Vec::new();

    let mut declaring = std::collections::BTreeMap::<String, Vec<String>>::new();
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
                        let entry = declaring.entry(function).or_default();
                        for kind in kinds {
                            if !entry.contains(&kind) {
                                entry.push(kind);
                            }
                        }
                    }
                }
                Err(error) => failures.push(format!("{}: {error}", display_repo_path(&source))),
            }
        }
    }

    let mut declared = std::collections::BTreeMap::<PathBuf, BTreeSet<String>>::new();
    for source in collect_rust_files(&repo_path(suite.source_dir)) {
        let contents = fs::read_to_string(&source).expect("test source should be readable");
        let sites = match rig_test_support::scenario_registry::cassette_scenario_sites(
            &contents,
            suite.wrapper_names,
        ) {
            Ok(sites) => sites,
            Err(error) => {
                failures.push(format!("{}: {error}", display_repo_path(&source)));
                continue;
            }
        };
        for site in sites {
            let kinds = declared
                .entry(crate::cassettes::cassette_path(
                    suite.provider,
                    &site.scenario,
                ))
                .or_default();
            kinds.extend(site.declared);
            kinds.extend(declaring.get(&site.wrapper).into_iter().flatten().cloned());
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
    let source = fs::read_to_string(path)
        .map_err(|error| format!("{} should be readable: {error}", display_repo_path(path)))?;
    rig_test_support::scenario_registry::cassette_scenarios(&source, wrapper_names)
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
