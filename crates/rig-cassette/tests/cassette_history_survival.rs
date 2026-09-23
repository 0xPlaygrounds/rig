//! Corpus guard: what a provider delivered must come back, and every tool
//! call must be answered. A provider hands Rig opaque fields only it can
//! interpret (thinking signatures, encrypted or redacted reasoning,
//! reasoning item ids, tool-call ids); Rig normalizes them into history and
//! serializes them again on the next turn, and the provider either needs
//! them back or rejects the request. The per-scenario tests assert their own
//! turn; this target compares what turn N delivered with what turn N+1 sent
//! over every committed cassette, at zero provider cost, with the rule in
//! `test-support/rig-test-support/src/history_survival.rs` so the recorded
//! round-trip cells apply the identical rule to fresh exchanges.
//!
//! Four checks:
//!
//! 1. [`delivered_opaque_fields_reach_the_next_request`]: each continuation
//!    request carries every opaque value the previous response delivered.
//! 2. [`every_recorded_request_pairs_tool_calls_with_results`]: no request
//!    leaves a tool call unanswered or a result unmatched, including the
//!    request after a fault.
//! 3. [`every_native_request_pairs_tool_calls_with_results`]: the same
//!    pairing rule over the normalized `chat_history` of every completion
//!    request in the effect goldens, which include the requests after
//!    cancellations, invalid arguments and provider faults.
//! 4. [`every_provider_and_content_kind_is_examined`]: the first checks
//!    actually looked at each provider, and each content kind the corpus can
//!    show was seen somewhere. A kind no cassette carries is a coverage gap
//!    to record, not a silent pass.

#![allow(clippy::expect_used, clippy::panic, clippy::indexing_slicing)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::Value;

use rig_test_support::history_survival::{
    Dialect, TOKEN_KINDS, Token, continues, legacy_tokens, lost_tokens,
    unpaired_normalized_tool_calls, unpaired_tool_calls,
};

/// Scenarios whose continuation legitimately lacks a delivered value.
///
/// `(cassette path suffix, token kind, reason)`. The reason must cite the
/// provider behavior; an entry that stops matching a real loss is reported
/// as stale so exemptions cannot outlive the behavior they excuse.
const SURVIVAL_EXEMPT: &[(&str, &str, &str)] = &[(
    "anthropic/response_identity_edge/repaired_invalid_call_keeps_call_identity.yaml",
    "tool_call_id",
    "the cell's repair hook renames the call from `sum_values` to `add` by design, \
     so the id returns on a call whose name no longer anchors it",
)];

/// Scenarios whose recorded requests deliberately carry unpaired calls.
const PAIRING_EXEMPT: &[(&str, &str)] = &[];

/// Content kinds no committed cassette can show for a provider, with the
/// reason. Absent entries are findings: a provider that could carry the kind
/// but has no recording of it is a coverage gap.
const KIND_COVERAGE_EXEMPT: &[(&str, &str, &str)] = &[];

#[derive(Deserialize)]
struct RecordedInteraction {
    when: RecordedRequest,
    then: RecordedResponse,
}

#[derive(Deserialize)]
struct RecordedRequest {
    #[serde(default)]
    path: String,
    #[serde(default)]
    body: Option<String>,
}

#[derive(Deserialize)]
struct RecordedResponse {
    #[serde(default)]
    status: u16,
    #[serde(default)]
    body: Option<String>,
}

struct Exchange {
    dialect: Dialect,
    request: Value,
    status: u16,
    response: String,
}

fn exchanges(contents: &str) -> Vec<Exchange> {
    serde_yaml::Deserializer::from_str(contents)
        .filter_map(|document| RecordedInteraction::deserialize(document).ok())
        .filter_map(|interaction| {
            let request = serde_json::from_str::<Value>(&interaction.when.body?).ok()?;
            Some(Exchange {
                dialect: Dialect::from_path(&interaction.when.path),
                request,
                status: interaction.then.status,
                response: interaction.then.body.unwrap_or_default(),
            })
        })
        .collect()
}

fn cassette_root() -> PathBuf {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/cassettes");
    assert!(root.is_dir(), "cassette root missing: {}", root.display());
    root
}

fn cassette_files(dir: &Path, found: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries {
        let path = entry.expect("cassette directory entry").path();
        if path.is_dir() {
            cassette_files(&path, found);
        } else if path
            .extension()
            .is_some_and(|extension| extension == "yaml")
        {
            found.push(path);
        }
    }
}

fn all_cassettes(root: &Path) -> Vec<(String, Vec<Exchange>)> {
    let mut files = Vec::new();
    cassette_files(root, &mut files);
    files.sort();
    assert!(!files.is_empty(), "no cassettes under {}", root.display());
    files
        .into_iter()
        .map(|file| {
            let scenario = file
                .strip_prefix(root)
                .expect("cassette under root")
                .to_string_lossy()
                .replace('\\', "/");
            let contents = std::fs::read_to_string(&file).expect("cassette readable");
            (scenario, exchanges(&contents))
        })
        .collect()
}

fn provider_of(scenario: &str) -> &str {
    scenario.split('/').next().unwrap_or(scenario)
}

/// Per-provider counts of what the checks examined.
#[derive(Default)]
struct Census {
    continuation_pairs: usize,
    requests_paired: usize,
    delivered: BTreeMap<&'static str, usize>,
}

fn census_report(census: &BTreeMap<String, Census>) -> String {
    let mut lines = vec![format!(
        "{:<12} {:>6} {:>8} {}",
        "provider", "pairs", "requests", "delivered kinds"
    )];
    for (provider, entry) in census {
        let kinds = entry
            .delivered
            .iter()
            .map(|(kind, count)| format!("{kind}={count}"))
            .collect::<Vec<_>>()
            .join(" ");
        lines.push(format!(
            "{:<12} {:>6} {:>8} {kinds}",
            provider, entry.continuation_pairs, entry.requests_paired
        ));
    }
    lines.join("\n")
}

/// Continuation pairs across the corpus: `(scenario, pair index, response
/// of the earlier exchange, request of the later one)`.
fn continuation_pairs(
    cassettes: &[(String, Vec<Exchange>)],
) -> Vec<(&str, usize, &Exchange, &Exchange)> {
    let mut pairs = Vec::new();
    for (scenario, exchanges) in cassettes {
        for (index, window) in exchanges.windows(2).enumerate() {
            let (earlier, later) = (&window[0], &window[1]);
            if earlier.dialect == Dialect::Unmodeled
                || earlier.dialect != later.dialect
                || earlier.status != 200
                || !continues(earlier.dialect, &earlier.request, &later.request)
            {
                continue;
            }
            pairs.push((scenario.as_str(), index, earlier, later));
        }
    }
    pairs
}

#[test]
fn delivered_opaque_fields_reach_the_next_request() {
    for (suffix, kind, reason) in SURVIVAL_EXEMPT {
        assert!(
            !reason.trim().is_empty(),
            "SURVIVAL_EXEMPT `{suffix}` ({kind}) needs a reason"
        );
    }
    let root = cassette_root();
    let cassettes = all_cassettes(&root);
    let mut losses: Vec<String> = Vec::new();
    let mut used: BTreeSet<(&str, &str)> = BTreeSet::new();
    let mut compared = 0usize;
    let mut legacy = 0usize;
    let mut legacy_by_scenario: BTreeMap<String, usize> = BTreeMap::new();
    let mut legacy_absent: Vec<String> = Vec::new();

    for (scenario, index, earlier, later) in continuation_pairs(&cassettes) {
        compared += 1;
        let placeholders = legacy_tokens(earlier.dialect, &earlier.response);
        legacy += placeholders.len();
        if !placeholders.is_empty() {
            *legacy_by_scenario.entry(scenario.to_string()).or_default() += placeholders.len();
        }
        // A placeholder cannot prove its slot, but its absence is still worth
        // reporting.
        let carried = rig_test_support::history_survival::string_values(&later.request);
        legacy_absent.extend(
            placeholders
                .iter()
                .filter(|token| !carried.contains(&token.value))
                .map(|token| {
                    format!(
                        "{scenario} exchange {index}: {} {}",
                        token.kind, token.value
                    )
                }),
        );
        // Only a gateway relays several families under one wire; elsewhere a
        // model change is no family switch and every value is judged.
        let switched = scenario.starts_with("openrouter/")
            && rig_test_support::history_survival::switches_reasoning_family(
                &earlier.request,
                &earlier.response,
                &later.request,
            );
        let lost: Vec<Token> = lost_tokens(earlier.dialect, &earlier.response, &later.request)
            .into_iter()
            // A switch to another model family carries tool ids, not reasoning.
            .filter(|token| !switched || token.kind == "tool_call_id")
            .collect();
        for token in lost {
            if let Some((suffix, kind, _)) = SURVIVAL_EXEMPT
                .iter()
                .find(|(suffix, kind, _)| scenario.ends_with(suffix) && *kind == token.kind)
            {
                used.insert((suffix, kind));
                continue;
            }
            losses.push(format!(
                "{scenario} exchange {index}: response delivered {} {:?}, the next request does not carry it in its slot",
                token.kind, token.value
            ));
        }
    }

    let stale: Vec<String> = SURVIVAL_EXEMPT
        .iter()
        .filter(|(suffix, kind, _)| !used.contains(&(suffix, kind)))
        .map(|(suffix, kind, _)| format!("{suffix} ({kind})"))
        .collect();

    assert!(compared > 0, "no continuation pairs found in the corpus");
    // Legacy placeholders cannot prove a slot; they are counted, not judged.
    eprintln!("continuation pairs: {compared}; legacy placeholder values not judged: {legacy}");
    let mut legacy_by_provider: BTreeMap<&str, usize> = BTreeMap::new();
    for (scenario, count) in &legacy_by_scenario {
        let provider = scenario.split('/').next().unwrap_or_default();
        *legacy_by_provider.entry(provider).or_default() += count;
    }
    eprintln!("legacy placeholder values by provider: {legacy_by_provider:?}");
    for (scenario, count) in &legacy_by_scenario {
        eprintln!("  {scenario}: {count}");
    }
    eprintln!(
        "legacy placeholder values absent from the next request (reported, not judged): {}\n{}",
        legacy_absent.len(),
        legacy_absent.join("\n")
    );
    assert!(
        losses.is_empty() && stale.is_empty(),
        "opaque content lost between turns ({} pairs compared):\n{}\n\nstale exemptions:\n{}",
        compared,
        losses.join("\n"),
        stale.join("\n")
    );
}

#[test]
fn every_recorded_request_pairs_tool_calls_with_results() {
    for (suffix, reason) in PAIRING_EXEMPT {
        assert!(
            !reason.trim().is_empty(),
            "PAIRING_EXEMPT `{suffix}` needs a reason"
        );
    }
    let root = cassette_root();
    let cassettes = all_cassettes(&root);
    let mut failures = Vec::new();
    let mut used = BTreeSet::new();
    let mut checked = 0usize;

    for (scenario, exchanges) in &cassettes {
        for (index, exchange) in exchanges.iter().enumerate() {
            if exchange.dialect == Dialect::Unmodeled {
                continue;
            }
            checked += 1;
            let unpaired = unpaired_tool_calls(exchange.dialect, &exchange.request);
            if unpaired.is_empty() {
                continue;
            }
            if let Some((suffix, _)) = PAIRING_EXEMPT
                .iter()
                .find(|(suffix, _)| scenario.ends_with(suffix))
            {
                used.insert(*suffix);
                continue;
            }
            failures.push(format!(
                "{scenario} exchange {index}: {}",
                unpaired
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join("; ")
            ));
        }
    }

    let stale: Vec<&str> = PAIRING_EXEMPT
        .iter()
        .filter(|(suffix, _)| !used.contains(suffix))
        .map(|(suffix, _)| *suffix)
        .collect();

    assert!(checked > 0, "no modeled requests found in the corpus");
    assert!(
        failures.is_empty() && stale.is_empty(),
        "unpaired tool calls in recorded requests ({checked} requests checked):\n{}\n\nstale exemptions:\n{}",
        failures.join("\n"),
        stale.join("\n")
    );
}

fn effect_logs(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries {
            let path = entry.expect("effect directory entry").path();
            if path.is_dir() {
                stack.push(path);
            } else if path.to_string_lossy().ends_with(".effects.json") {
                files.push(path);
            }
        }
    }
    files.sort();
    files
}

#[test]
fn every_native_request_pairs_tool_calls_with_results() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/effects");
    let files = effect_logs(&root);
    assert!(!files.is_empty(), "no effect logs under {}", root.display());
    let mut failures = Vec::new();
    let mut requests = 0usize;
    let mut with_tools = 0usize;
    for file in &files {
        let log: Value =
            serde_json::from_str(&std::fs::read_to_string(file).expect("log readable"))
                .expect("effect log parses");
        let Some(records) = log.get("records").and_then(Value::as_array) else {
            continue;
        };
        for (index, record) in records.iter().enumerate() {
            let Some(history) = record
                .pointer("/kind/request/chat_history")
                .and_then(Value::as_array)
            else {
                continue;
            };
            requests += 1;
            if history
                .iter()
                .any(|message| message.to_string().contains("\"toolcall\""))
            {
                with_tools += 1;
            }
            let unpaired = unpaired_normalized_tool_calls(history);
            if !unpaired.is_empty() {
                failures.push(format!(
                    "{} record {index}: {}",
                    file.strip_prefix(&root).unwrap_or(file).display(),
                    unpaired
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join("; ")
                ));
            }
        }
    }
    assert!(with_tools > 0, "no native request carried a tool call");
    assert!(
        failures.is_empty(),
        "unpaired tool calls in native requests ({requests} requests, {with_tools} with tools):\n{}",
        failures.join("\n")
    );
    eprintln!("native requests checked: {requests}, with tool calls: {with_tools}");
}

#[test]
fn every_provider_and_content_kind_is_examined() {
    for (provider, kind, reason) in KIND_COVERAGE_EXEMPT {
        assert!(
            !reason.trim().is_empty(),
            "KIND_COVERAGE_EXEMPT `{provider}` ({kind}) needs a reason"
        );
        assert!(
            TOKEN_KINDS.contains(kind),
            "KIND_COVERAGE_EXEMPT `{provider}` names unknown kind {kind}"
        );
    }
    let root = cassette_root();
    let cassettes = all_cassettes(&root);
    let mut census: BTreeMap<String, Census> = BTreeMap::new();

    for (scenario, exchanges) in &cassettes {
        let entry = census.entry(provider_of(scenario).to_owned()).or_default();
        entry.requests_paired += exchanges
            .iter()
            .filter(|exchange| exchange.dialect != Dialect::Unmodeled)
            .count();
    }
    for (scenario, _, earlier, _) in continuation_pairs(&cassettes) {
        let entry = census.entry(provider_of(scenario).to_owned()).or_default();
        entry.continuation_pairs += 1;
        for token in
            rig_test_support::history_survival::response_tokens(earlier.dialect, &earlier.response)
        {
            *entry.delivered.entry(token.kind).or_default() += 1;
        }
    }

    let report = census_report(&census);
    eprintln!("{report}");
    if let Some(path) = std::env::var_os("RIG_HISTORY_SURVIVAL_CENSUS") {
        std::fs::write(&path, format!("{report}\n")).expect("census report written");
    }

    let mut findings = Vec::new();
    for (provider, entry) in &census {
        if entry.continuation_pairs == 0 {
            findings.push(format!(
                "{provider}: no continuation pair was compared; the survival check never looked at it"
            ));
        }
    }
    for kind in TOKEN_KINDS {
        let seen = census
            .values()
            .any(|entry| entry.delivered.contains_key(kind));
        if !seen {
            findings.push(format!(
                "{kind}: no committed cassette delivers this kind, so the survival rule for it is untested by the corpus"
            ));
        }
    }
    assert!(
        findings.is_empty(),
        "survival coverage census:\n{report}\n\nfindings:\n{}",
        findings.join("\n")
    );
}
