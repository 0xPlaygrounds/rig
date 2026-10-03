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
//! Five checks:
//!
//! 1. [`delivered_opaque_fields_reach_the_next_request`]: each continuation
//!    request carries every opaque value the previous response delivered.
//! 2. [`same_model_replay_sends_the_recorded_output_items`]: a continuation
//!    on the same model sends each output item of the previous reply back
//!    equal to the recorded item, as JSON values, in the reply's order. A
//!    message-shaped wire sends the projection its rebuild makes of the
//!    reply's message instead.
//! 3. [`every_recorded_request_pairs_tool_calls_with_results`]: no request
//!    leaves a tool call unanswered or a result unmatched, including the
//!    request after a fault.
//! 4. [`every_native_request_pairs_tool_calls_with_results`]: the same
//!    pairing rule over the normalized `chat_history` of every completion
//!    request in the effect goldens, which include the requests after
//!    cancellations, invalid arguments and provider faults.
//! 5. [`every_provider_and_content_kind_is_examined`]: the first checks
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
const SURVIVAL_EXEMPT: &[(&str, &str, &str)] = &[
    (
        "anthropic/corpus_shaping/active_tools_none_second_turn.yaml",
        "tool_call_id",
        "the request lets the model call no tools, so the core sends the history's calls and results as text (rule R4): the recorded call items are not replayed by design",
    ),
    (
        "anthropic/corpus_shaping/tool_choice_none_on_committed_output.yaml",
        "tool_call_id",
        "the request lets the model call no tools, so the core sends the history's calls and results as text (rule R4): the recorded call items are not replayed by design",
    ),
    (
        "deepseek/corpus_matrix/shaping_active_tools_none_second_turn.yaml",
        "tool_call_id",
        "the request lets the model call no tools, so the core sends the history's calls and results as text (rule R4): the recorded call items are not replayed by design",
    ),
    (
        "deepseek/corpus_matrix/shaping_tool_choice_none_on_committed_output.yaml",
        "tool_call_id",
        "the request lets the model call no tools, so the core sends the history's calls and results as text (rule R4): the recorded call items are not replayed by design",
    ),
    (
        "doubleword/corpus_matrix/shaping_active_tools_none_second_turn.yaml",
        "tool_call_id",
        "the request lets the model call no tools, so the core sends the history's calls and results as text (rule R4): the recorded call items are not replayed by design",
    ),
    (
        "doubleword/corpus_matrix/shaping_tool_choice_none_on_committed_output.yaml",
        "tool_call_id",
        "the request lets the model call no tools, so the core sends the history's calls and results as text (rule R4): the recorded call items are not replayed by design",
    ),
    (
        "gemini/corpus_matrix/shaping_active_tools_none_second_turn.yaml",
        "thought_signature",
        "the request lets the model call no tools, so the core sends the history's calls and results as text (rule R4): the recorded call items are not replayed by design",
    ),
    (
        "gemini/corpus_matrix/shaping_active_tools_none_second_turn.yaml",
        "tool_call_id",
        "the request lets the model call no tools, so the core sends the history's calls and results as text (rule R4): the recorded call items are not replayed by design",
    ),
    (
        "openai/corpus_matrix_chat/shaping_active_tools_none_second_turn.yaml",
        "tool_call_id",
        "the request lets the model call no tools, so the core sends the history's calls and results as text (rule R4): the recorded call items are not replayed by design",
    ),
    (
        "openai/corpus_matrix_chat/shaping_tool_choice_none_on_committed_output.yaml",
        "tool_call_id",
        "the request lets the model call no tools, so the core sends the history's calls and results as text (rule R4): the recorded call items are not replayed by design",
    ),
    (
        "openai/corpus_matrix_responses/shaping_active_tools_none_second_turn.yaml",
        "encrypted_content",
        "the request lets the model call no tools, so the core sends the history's calls and results as text (rule R4): the recorded call items are not replayed by design",
    ),
    (
        "openai/corpus_matrix_responses/shaping_active_tools_none_second_turn.yaml",
        "reasoning_id",
        "the request lets the model call no tools, so the core sends the history's calls and results as text (rule R4): the recorded call items are not replayed by design",
    ),
    (
        "openai/corpus_matrix_responses/shaping_active_tools_none_second_turn.yaml",
        "tool_call_id",
        "the request lets the model call no tools, so the core sends the history's calls and results as text (rule R4): the recorded call items are not replayed by design",
    ),
    (
        "openai/corpus_matrix_responses/shaping_tool_choice_none_on_committed_output.yaml",
        "encrypted_content",
        "the request lets the model call no tools, so the core sends the history's calls and results as text (rule R4): the recorded call items are not replayed by design",
    ),
    (
        "openai/corpus_matrix_responses/shaping_tool_choice_none_on_committed_output.yaml",
        "reasoning_id",
        "the request lets the model call no tools, so the core sends the history's calls and results as text (rule R4): the recorded call items are not replayed by design",
    ),
    (
        "openai/corpus_matrix_responses/shaping_tool_choice_none_on_committed_output.yaml",
        "tool_call_id",
        "the request lets the model call no tools, so the core sends the history's calls and results as text (rule R4): the recorded call items are not replayed by design",
    ),
    (
        "openai/streaming_grammar/three_turn_tool_session.yaml",
        "encrypted_content",
        "the request lets the model call no tools, so the core sends the history's calls and results as text (rule R4): the recorded call items are not replayed by design",
    ),
    (
        "openai/streaming_grammar/three_turn_tool_session.yaml",
        "reasoning_id",
        "the request lets the model call no tools, so the core sends the history's calls and results as text (rule R4): the recorded call items are not replayed by design",
    ),
    (
        "openai/streaming_grammar/three_turn_tool_session.yaml",
        "tool_call_id",
        "the request lets the model call no tools, so the core sends the history's calls and results as text (rule R4): the recorded call items are not replayed by design",
    ),
    (
        "openai/streaming_grammar/tool_then_followup_text.yaml",
        "tool_call_id",
        "the request lets the model call no tools, so the core sends the history's calls and results as text (rule R4): the recorded call items are not replayed by design",
    ),
    (
        "venice/corpus_matrix/shaping_active_tools_none_second_turn.yaml",
        "tool_call_id",
        "the request lets the model call no tools, so the core sends the history's calls and results as text (rule R4): the recorded call items are not replayed by design",
    ),
    (
        "venice/corpus_matrix/shaping_tool_choice_none_on_committed_output.yaml",
        "tool_call_id",
        "the request lets the model call no tools, so the core sends the history's calls and results as text (rule R4): the recorded call items are not replayed by design",
    ),
    (
        "anthropic/response_identity_edge/repaired_invalid_call_keeps_call_identity.yaml",
        "tool_call_id",
        "the cell's repair hook renames the call from `sum_values` to `add` by design, \
         so the id returns on a call whose name no longer anchors it",
    ),
    (
        "xai/prompt_caching/streaming_probe.yaml",
        "encrypted_content",
        "the streaming cache probe rebuilds each answer from its text deltas rather than \
         from rig's decoded choice, so the history it sends holds no reasoning item",
    ),
    (
        "xai/prompt_caching/streaming_probe.yaml",
        "reasoning_id",
        "the same text-only rebuild of each streamed answer",
    ),
    (
        "gemini/agent_run_recovery/repair_renames_tool_call_and_executes_it.yaml",
        "thought_signature",
        "the repair hook renames the call, and an edited block replays from its canonical \
         fields, so the signature its provider item carried is not sent",
    ),
    (
        "gemini/agent_run_streamed/streamed_repair_continues_the_same_stream.yaml",
        "thought_signature",
        "the same rename by a repair hook, on a streamed turn",
    ),
    (
        "gemini/auto_caching/support_chat_100_current_turn.yaml",
        "thought_signature",
        "the run opts into ThoughtReplay::CurrentTurn, which stops re-sending a turn's \
         signatures once the next user message arrives (Gemini accepts that and stops \
         billing the turn's thoughts again); the test asserts that dropped signatures \
         are the only change to a finished turn",
    ),
    (
        "gemini/auto_caching/support_chat_100_streamed.yaml",
        "thought_signature",
        "the same opt-in current-turn thought replay, streamed",
    ),
    (
        "gemini/auto_caching/support_chat_100_resume.yaml",
        "thought_signature",
        "the same opt-in current-turn thought replay, across a checkpoint and resume",
    ),
    (
        "gemini/auto_caching/support_chat_100_compaction.yaml",
        "thought_signature",
        "the same opt-in current-turn thought replay, with a compaction at turn 51",
    ),
];

/// Scenarios whose same-model continuation legitimately sends an output item
/// other than as recorded. `(cassette path suffix, reason)`, reported as
/// stale once it stops matching a real difference.
const VERBATIM_EXEMPT: &[(&str, &str)] = &[
    (
        "anthropic/corpus_shaping/active_tools_none_second_turn.yaml",
        "the request lets the model call no tools, so the core sends the history's calls and results as text (rule R4): the recorded call items are not replayed by design",
    ),
    (
        "anthropic/corpus_shaping/tool_choice_none_on_committed_output.yaml",
        "the request lets the model call no tools, so the core sends the history's calls and results as text (rule R4): the recorded call items are not replayed by design",
    ),
    (
        "deepseek/corpus_matrix/shaping_active_tools_none_second_turn.yaml",
        "the request lets the model call no tools, so the core sends the history's calls and results as text (rule R4): the recorded call items are not replayed by design",
    ),
    (
        "deepseek/corpus_matrix/shaping_tool_choice_none_on_committed_output.yaml",
        "the request lets the model call no tools, so the core sends the history's calls and results as text (rule R4): the recorded call items are not replayed by design",
    ),
    (
        "doubleword/corpus_matrix/shaping_active_tools_none_second_turn.yaml",
        "the request lets the model call no tools, so the core sends the history's calls and results as text (rule R4): the recorded call items are not replayed by design",
    ),
    (
        "doubleword/corpus_matrix/shaping_tool_choice_none_on_committed_output.yaml",
        "the request lets the model call no tools, so the core sends the history's calls and results as text (rule R4): the recorded call items are not replayed by design",
    ),
    (
        "gemini/corpus_matrix/shaping_active_tools_none_second_turn.yaml",
        "the request lets the model call no tools, so the core sends the history's calls and results as text (rule R4): the recorded call items are not replayed by design",
    ),
    (
        "openai/corpus_matrix_chat/shaping_active_tools_none_second_turn.yaml",
        "the request lets the model call no tools, so the core sends the history's calls and results as text (rule R4): the recorded call items are not replayed by design",
    ),
    (
        "openai/corpus_matrix_chat/shaping_tool_choice_none_on_committed_output.yaml",
        "the request lets the model call no tools, so the core sends the history's calls and results as text (rule R4): the recorded call items are not replayed by design",
    ),
    (
        "openai/corpus_matrix_responses/shaping_active_tools_none_second_turn.yaml",
        "the request lets the model call no tools, so the core sends the history's calls and results as text (rule R4): the recorded call items are not replayed by design",
    ),
    (
        "openai/corpus_matrix_responses/shaping_tool_choice_none_on_committed_output.yaml",
        "the request lets the model call no tools, so the core sends the history's calls and results as text (rule R4): the recorded call items are not replayed by design",
    ),
    (
        "openai/streaming_grammar/three_turn_tool_session.yaml",
        "the request lets the model call no tools, so the core sends the history's calls and results as text (rule R4): the recorded call items are not replayed by design",
    ),
    (
        "openai/streaming_grammar/tool_then_followup_text.yaml",
        "the request lets the model call no tools, so the core sends the history's calls and results as text (rule R4): the recorded call items are not replayed by design",
    ),
    (
        "venice/corpus_matrix/shaping_active_tools_none_second_turn.yaml",
        "the request lets the model call no tools, so the core sends the history's calls and results as text (rule R4): the recorded call items are not replayed by design",
    ),
    (
        "venice/corpus_matrix/shaping_tool_choice_none_on_committed_output.yaml",
        "the request lets the model call no tools, so the core sends the history's calls and results as text (rule R4): the recorded call items are not replayed by design",
    ),
    (
        "anthropic/response_identity_edge/repaired_invalid_call_keeps_call_identity.yaml",
        "the repair hook renames the call, and an edited block replays from its canonical fields",
    ),
    (
        "gemini/agent_run_recovery/repair_renames_tool_call_and_executes_it.yaml",
        "the repair hook renames the call, and an edited block replays from its canonical fields",
    ),
    (
        "gemini/agent_run_streamed/streamed_repair_continues_the_same_stream.yaml",
        "the same rename by a repair hook, on a streamed turn",
    ),
    (
        "gemini/auto_caching/support_chat_100_compaction.yaml",
        "the run opts into ThoughtReplay::CurrentTurn, which drops a finished turn's signatures",
    ),
    (
        "gemini/auto_caching/support_chat_100_current_turn.yaml",
        "the same opt-in current-turn thought replay",
    ),
    (
        "gemini/auto_caching/support_chat_100_resume.yaml",
        "the same opt-in current-turn thought replay, across a checkpoint and resume",
    ),
    (
        "gemini/reasoning_tool_roundtrip/nonstreaming.yaml",
        "Gemini can split one thought across parts; both modes merge consecutive thought parts \
         into one block so a stream and a whole reply fold alike, and the block replays as one part",
    ),
    (
        "openai/prompt_caching/responses_streaming_probe.yaml",
        "the cache probe rebuilds each answer from its text, so the history holds no item ids",
    ),
    (
        "xai/prompt_caching/streaming_probe.yaml",
        "the cache probe rebuilds each answer from its text, so the history holds no reasoning",
    ),
];

/// Scenarios whose recorded requests deliberately carry unpaired calls.
const PAIRING_EXEMPT: &[(&str, &str)] = &[];

/// Content kinds no committed cassette can show for a provider, with the
/// reason. Absent entries are findings: a provider that could carry the kind
/// but has no recording of it is a coverage gap.
const KIND_COVERAGE_EXEMPT: &[(&str, &str, &str)] = &[];

/// Providers no committed cassette continues a conversation for, with the
/// reason. An entry is stale once the provider has a continuation pair.
const CONTINUATION_EXEMPT: &[(&str, &str)] = &[];

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
    path: String,
    request: Value,
    status: u16,
    response: String,
}

/// Every recorded exchange with a JSON request body, in order. A Gemini
/// request that reads an explicit cache this cassette created is spliced with
/// the body that created it (`cache_prefix::splice_cached_content`), so the
/// checks see the whole conversation it stands for rather than the tail it
/// sent.
fn exchanges(contents: &str) -> Vec<Exchange> {
    let mut caches: std::collections::HashMap<String, Value> = std::collections::HashMap::new();
    serde_yaml::Deserializer::from_str(contents)
        .filter_map(|document| RecordedInteraction::deserialize(document).ok())
        .filter_map(|interaction| {
            let request = serde_json::from_str::<Value>(&interaction.when.body?).ok()?;
            if interaction.when.path.ends_with("/cachedContents")
                && let Some(name) = interaction
                    .then
                    .body
                    .as_deref()
                    .and_then(|body| serde_json::from_str::<Value>(body).ok())
                    .and_then(|reply| reply.get("name")?.as_str().map(str::to_owned))
            {
                caches.insert(name, request.clone());
            }
            let request = request
                .get("cachedContent")
                .and_then(Value::as_str)
                .and_then(|name| caches.get(name))
                .and_then(|cache| {
                    rig_test_support::cache_prefix::splice_cached_content(&request, cache)
                })
                .unwrap_or(request);
            Some(Exchange {
                dialect: Dialect::from_path(&interaction.when.path),
                path: interaction.when.path,
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

/// The model an exchange addressed: the request's `model`, or the model
/// segment of a Gemini or Bedrock path.
fn model_of(exchange: &Exchange) -> Option<&str> {
    if let Some(model) = exchange.request.get("model").and_then(Value::as_str) {
        return Some(model.strip_prefix("models/").unwrap_or(model));
    }
    let path = exchange.path.as_str();
    if let Some((_, rest)) = path.split_once("/models/") {
        return rest.split(':').next();
    }
    path.split_once("/model/")
        .and_then(|(_, rest)| rest.split('/').next())
}

/// Whether `later` continues on the model that answered `earlier`. A
/// continuation on another model replays canonical fields only, which the
/// session cells assert on their own.
fn same_model(earlier: &Exchange, later: &Exchange) -> bool {
    model_of(earlier) == model_of(later)
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
        if !same_model(earlier, later) {
            continue;
        }
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

/// The output items of a whole reply, or of a Responses stream's finished
/// items, as the provider sent them. `None` for a stream whose items arrive
/// as deltas.
fn output_items(dialect: Dialect, body: &str) -> Option<Vec<Value>> {
    let documents = rig_test_support::history_survival::response_documents(body);
    if dialect == Dialect::OpenAiResponses && documents.len() > 1 {
        return Some(
            documents
                .into_iter()
                .filter(|document| document["type"] == "response.output_item.done")
                .map(|document| document["item"].clone())
                .collect(),
        );
    }
    let [reply] = documents.as_slice() else {
        return None;
    };
    let items = match dialect {
        Dialect::AnthropicMessages => reply.get("content")?.clone(),
        Dialect::OpenAiResponses => reply.get("output")?.clone(),
        Dialect::GeminiGenerateContent => reply.pointer("/candidates/0/content/parts")?.clone(),
        Dialect::GeminiInteractions => reply.get("steps")?.clone(),
        Dialect::BedrockConverse => reply.pointer("/output/message/content")?.clone(),
        Dialect::ChatCompletions => {
            Value::Array(vec![reply.pointer("/choices/0/message")?.clone()])
        }
        Dialect::Unmodeled => return None,
    };
    match items {
        Value::Array(items) => Some(items),
        _ => None,
    }
}

/// The history items a request sends, in order: the content of each model
/// turn for wires whose turns hold content arrays, the turns themselves
/// otherwise.
fn replayed_items(dialect: Dialect, request: &Value) -> Vec<Value> {
    let Some(turns) = dialect
        .conversation_field()
        .and_then(|field| request.get(field))
        .and_then(Value::as_array)
    else {
        return Vec::new();
    };
    let content = match dialect {
        Dialect::AnthropicMessages | Dialect::BedrockConverse => Some(("assistant", "content")),
        Dialect::GeminiGenerateContent => Some(("model", "parts")),
        _ => None,
    };
    match content {
        Some((role, field)) => turns
            .iter()
            .filter(|turn| turn["role"] == role)
            .filter_map(|turn| turn.get(field).and_then(Value::as_array))
            .flatten()
            .cloned()
            .collect(),
        None => turns.clone(),
    }
}

/// The item a same-model replay sends for the reply item `item`. An
/// item-shaped wire sends its items as they came. A message-shaped wire
/// (Chat) rebuilds the message from its blocks, as pi's
/// `openai-completions` does, so the item is the projection of the reply's
/// message that rebuild sends: never the whole message.
fn as_replayed(scenario: &str, dialect: Dialect, item: Value) -> Value {
    match dialect {
        Dialect::ChatCompletions => chat_projection(provider_of(scenario), &item),
        Dialect::AnthropicMessages
        | Dialect::GeminiGenerateContent
        | Dialect::GeminiInteractions
        | Dialect::OpenAiResponses
        | Dialect::BedrockConverse
        | Dialect::Unmodeled => item,
    }
}

/// A call's arguments as the rebuild sends them: the object they state,
/// as JSON text when `text`.
fn canonical_arguments(arguments: Option<&Value>, text: bool) -> Value {
    let object = match arguments {
        Some(Value::String(raw)) => match serde_json::from_str::<Value>(raw) {
            Ok(Value::String(inner)) => serde_json::from_str(&inner).unwrap_or_default(),
            Ok(value) => value,
            // A cut call states what pi's tolerant parse reads from it.
            Err(_) => rig_core::json_utils::parse_partial_object(raw)
                .map(Value::Object)
                .unwrap_or_default(),
        },
        Some(value) => value.clone(),
        None => Value::Null,
    };
    let object = if object.is_object() {
        object
    } else {
        Value::Object(Default::default())
    };
    if text {
        Value::String(object.to_string())
    } else {
        object
    }
}

/// `call` as the rebuild sends it: its item with `arguments` canonical, and
/// without the stream `index` Chat calls carry.
fn replayed_call(call: &Value, text: bool, keep_index: bool) -> Value {
    let mut call = call.clone();
    if let Some(fields) = call.as_object_mut()
        && !keep_index
    {
        fields.shift_remove("index");
    }
    if call.get("type").and_then(Value::as_str) != Some("custom") {
        let arguments = canonical_arguments(call.pointer("/function/arguments"), text);
        call["function"]["arguments"] = arguments;
    }
    call
}

/// The assistant message a Chat dialect's rebuild sends for a reply's
/// `message`: its text as `content`, its reasoning under the first field
/// that carries it, `reasoning_details` and an answer's audio id, and each
/// call. DeepSeek takes `content` and `reasoning_content` on every
/// assistant turn, Mistral `content`.
fn chat_projection(provider: &str, message: &Value) -> Value {
    let mut out = serde_json::Map::new();
    out.insert("role".to_owned(), "assistant".into());
    let text = ["content", "refusal"].iter().find_map(|key| {
        message
            .get(*key)
            .and_then(Value::as_str)
            .filter(|text| !text.is_empty())
    });
    if let Some(text) = text.filter(|text| !text.trim().is_empty()) {
        out.insert("content".to_owned(), text.into());
    }
    let reasoning = ["reasoning_content", "reasoning", "reasoning_text"]
        .iter()
        .find_map(|key| {
            let text = message.get(*key).and_then(Value::as_str)?;
            (!text.is_empty()).then_some((*key, text))
        });
    if let Some((field, text)) = reasoning.filter(|(_, text)| !text.trim().is_empty()) {
        out.insert(field.to_owned(), text.into());
    }
    if let Some(details) = message.get("reasoning_details").filter(|details| {
        details
            .as_array()
            .is_some_and(|details| !details.is_empty())
    }) {
        out.insert("reasoning_details".to_owned(), details.clone());
    }
    if let Some(id) = message.pointer("/audio/id") {
        out.insert("audio".to_owned(), serde_json::json!({ "id": id }));
    }
    let calls: Vec<Value> = message
        .get("tool_calls")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|call| replayed_call(call, true, false))
        .collect();
    if !calls.is_empty() {
        out.insert("tool_calls".to_owned(), calls.into());
    }
    // A message with no content and no calls is not sent, as pi skips it.
    if !["content", "audio", "tool_calls"]
        .iter()
        .any(|key| out.contains_key(*key))
    {
        return Value::Null;
    }
    if provider == "deepseek" {
        out.entry("reasoning_content").or_insert_with(|| "".into());
    }
    if provider == "deepseek" || provider == "mistral" {
        out.entry("content").or_insert_with(|| "".into());
    }
    Value::Object(out)
}

#[test]
fn same_model_replay_sends_the_recorded_output_items() {
    for (suffix, reason) in VERBATIM_EXEMPT {
        assert!(
            !reason.trim().is_empty(),
            "VERBATIM_EXEMPT `{suffix}` needs a reason"
        );
    }
    let root = cassette_root();
    let cassettes = all_cassettes(&root);
    let mut failures = Vec::new();
    let mut used = BTreeSet::new();
    let mut compared = 0usize;
    for (scenario, index, earlier, later) in continuation_pairs(&cassettes) {
        if !same_model(earlier, later) {
            continue;
        }
        let Some(items) = output_items(earlier.dialect, &earlier.response) else {
            continue;
        };
        let items: Vec<Value> = items
            .into_iter()
            .map(|item| as_replayed(scenario, earlier.dialect, item))
            .filter(|item| !item.is_null())
            .collect();
        compared += 1;
        let sent: Vec<Value> = replayed_items(later.dialect, &later.request);
        let mut from = 0;
        let missing: Vec<&Value> = items
            .iter()
            .filter(
                |item| match sent.iter().skip(from).position(|sent| sent == *item) {
                    Some(at) => {
                        from += at + 1;
                        false
                    }
                    None => true,
                },
            )
            .collect();
        if missing.is_empty() {
            continue;
        }
        if let Some((suffix, _)) = VERBATIM_EXEMPT
            .iter()
            .find(|(suffix, _)| scenario.ends_with(suffix))
        {
            used.insert(*suffix);
            continue;
        }
        failures.push(format!(
            "{scenario} exchange {index}: not sent as recorded, in order: {}",
            missing
                .iter()
                .map(|item| item.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    let stale: Vec<&str> = VERBATIM_EXEMPT
        .iter()
        .filter(|(suffix, _)| !used.contains(suffix))
        .map(|(suffix, _)| *suffix)
        .collect();
    assert!(compared > 0, "no same-model continuation pairs found");
    eprintln!("same-model replies compared item by item: {compared}");
    assert!(
        failures.is_empty() && stale.is_empty(),
        "same-model replay changed recorded output items ({compared} replies compared):\n{}\n\nstale exemptions:\n{}",
        failures.join("\n"),
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
        let exempt = CONTINUATION_EXEMPT
            .iter()
            .any(|(exempt, _)| exempt == provider);
        if entry.continuation_pairs == 0 && !exempt {
            findings.push(format!(
                "{provider}: no continuation pair was compared; the survival check never looked at it"
            ));
        }
        if entry.continuation_pairs > 0 && exempt {
            findings.push(format!(
                "{provider}: CONTINUATION_EXEMPT is stale, the corpus now continues its conversations"
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
