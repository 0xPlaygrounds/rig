//! Matrix for raw provider response capture on the blocking path:
//! `CompletionResponse::raw` beside the normalized fields.
//!
//! # The feature
//!
//! Capture is always on. Every response `completion` returns carries `raw`:
//! exactly what `raw_completion` would have returned — the response as
//! `anthropic::completion::CompletionResponse` parsed it — serialized with
//! `serde_json::to_value`. `raw` is `Value::Null` only on a
//! `CompletionResponse` built by hand, with no provider response behind it;
//! `Value::Null` never means "not requested". This matrix pins three properties
//! against live recordings: presence and lossless typed round-trip, a
//! provider-specific field the normalized response provably lacks
//! (`stop_sequence`), and that `raw` and the normalized fields tell one story
//! (re-normalizing `raw` reproduces them) — then repeats the round trip on the
//! two turn shapes where a lossy `raw` would show first: an extended-thinking
//! turn (signed `thinking` block) and a forced tool call (`tool_use` block).
//!
//! # Matrix
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `raw_round_trips_into_provider_type` | plain text request | `raw` populated; deserializes into the Anthropic type and re-serializes equal; wire fields equal the fixture's | recorded |
//! | 2 | `raw_exposes_stop_sequence` | `stop_sequences: ["alpha"]` request | `raw["stop_sequence"] == "alpha"`; normalized response has no such field | recorded |
//! | 3 | `normalized_fields_match_raw_renormalized` | plain text request | `CompletionResponse::deserialize(raw).normalize("anthropic")` reproduces `identity()`, `finish_reason()`, `model`, `usage`, `choice` | recorded |
//! | 4 | `raw_exposes_thinking_block_and_signature` | extended thinking (`thinking.enabled`, budget 1024) | `raw` round-trips; `raw["content"]` carries a `type: "thinking"` block with `thinking` + `signature` and `usage.output_tokens_details.thinking_tokens`, verbatim; the normalized response re-spells both (`type: "reasoning"`, `reasoning_tokens`) and has no `"thinking"` key at all | recorded |
//! | 5 | `raw_exposes_tool_use_block` | forced tool call (`tool_choice: required`, one tool) | `raw` round-trips; `raw["content"]` carries a `type: "tool_use"` block with an `input` *object*, `raw["stop_reason"] == "tool_use"` verbatim; normalized `finish_reason() == ToolCalls`, `type: "toolcall"`, `function.arguments` — no `"tool_use"` spelling anywhere | recorded |
//!
//! Every recorded cell re-derives its premise from its own fixture after the
//! wrapper returns: the recorded body names a `msg_…` id, the response carries
//! a `request-id` header, and the recorded stop reason is the one the cell is
//! about. Cell 2 reuses the `stop_sequences: ["alpha"]` request shape from
//! `empty_stop_sequence_matrix.rs`, where a one-word reply matches the
//! sequence and Anthropic reports it back on `stop_sequence`. Cell 3 is not
//! cell 1 restated: cell 1 proves `raw` is lossless against the *provider*
//! type; cell 3 proves rig's own normalization of that value agrees with the
//! normalized response delivered beside it — the single-response form of the
//! parity contract `raw_completion_parity_matrix.rs` records across two
//! exchanges. Cells 4 and 5 take the round trip off the text-only path: cell 4
//! reuses the `thinking.enabled` request from `reasoning_usage_matrix.rs`
//! (the wire shape with a `signature`, where a lossy re-serialization would
//! show first) and premise-asserts the fixture body actually carries a
//! `thinking` block; cell 5 reuses the `weather_tool` + `tool_choice` pattern
//! from `empty_stop_sequence_matrix.rs` so the turn is a `tool_use` terminal
//! by construction, and premise-asserts the fixture body carries the
//! `tool_use` block.

use rig::completion::{
    CompletionModel as _, CompletionResponse as RigCompletionResponse, FinishReason,
    NormalizeCompletionResponse, ResponseIdentity, ToolDefinition, Usage,
};
use rig::message::{AssistantContent, ReasoningContent, ToolChoice};
use rig::prelude::*;
use rig::providers::anthropic;
use rig::providers::anthropic::completion::{CompletionResponse, Content};
use serde::Deserialize;
use serde_json::{Value, json};

use super::super::support::{
    assert_ids_match_recording, recorded_request_id_headers, recorded_response_body,
    with_anthropic_cassette,
};

const ANTHROPIC_PROVIDER: &str = "anthropic";
const PROMPT: &str = "Reply with exactly: raw capture probe";
/// From `empty_stop_sequence_matrix.rs`: one word, so the `alpha` sequence
/// matches and Anthropic names it on `stop_sequence`.
const IMMEDIATE_PROMPT: &str = "Reply with exactly this one word and nothing else: alpha";
/// From `reasoning_usage_matrix.rs`: costs a few thinking tokens without a
/// long answer.
const THINKING_PROMPT: &str = "What is 17 * 23? Reply with just the number.";
/// A question the forced tool answers, so the `tool_use` block's `input`
/// carries a real `city`.
const TOOL_PROMPT: &str = "What is the weather in Paris right now?";

const ROUND_TRIP_SCENARIO: &str = "raw_capture_matrix/raw_round_trips_into_provider_type";
const STOP_SEQUENCE_SCENARIO: &str = "raw_capture_matrix/raw_exposes_stop_sequence";
const RENORMALIZED_SCENARIO: &str = "raw_capture_matrix/normalized_fields_match_raw_renormalized";
const THINKING_SCENARIO: &str = "raw_capture_matrix/raw_exposes_thinking_block_and_signature";
const TOOL_USE_SCENARIO: &str = "raw_capture_matrix/raw_exposes_tool_use_block";

type AnthropicModel = anthropic::CompletionModel;

fn probe_request(model: &AnthropicModel) -> rig::completion::CompletionRequest {
    model.completion_request(PROMPT).max_tokens(32).build()
}

/// From `reasoning_usage_matrix.rs`: extended thinking on, minimum budget,
/// `max_tokens` above it as Anthropic requires.
fn thinking_request(model: &AnthropicModel) -> rig::completion::CompletionRequest {
    model
        .completion_request(THINKING_PROMPT)
        .max_tokens(2048)
        .additional_params(json!({ "thinking": { "type": "enabled", "budget_tokens": 1024 } }))
        .build()
}

/// From `empty_stop_sequence_matrix.rs`.
fn weather_tool() -> ToolDefinition {
    ToolDefinition {
        name: "get_weather".to_string(),
        description: "Get the current weather for a city.".to_string(),
        parameters: json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"]
        }),
    }
}

/// `tool_choice: required` (Anthropic `any`) so the turn is a `tool_use`
/// terminal by construction, not by the model's mood.
fn tool_request(model: &AnthropicModel) -> rig::completion::CompletionRequest {
    model
        .completion_request(TOOL_PROMPT)
        .max_tokens(256)
        .tool(weather_tool())
        .tool_choice(ToolChoice::Required)
        .build()
}

/// Every string a JSON value contains — object keys and string leaves — so a
/// cell can prove a wire spelling (`"thinking"`, `"tool_use"`) is absent from
/// the normalized response *anywhere*, not just at one path.
fn collect_strings<'a>(value: &'a Value, out: &mut Vec<&'a str>) {
    match value {
        Value::String(text) => out.push(text.as_str()),
        Value::Array(items) => items.iter().for_each(|item| collect_strings(item, out)),
        Value::Object(fields) => {
            for (key, field) in fields {
                out.push(key.as_str());
                collect_strings(field, out);
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) => {}
    }
}

fn contains_string(value: &Value, needle: &str) -> bool {
    let mut strings = Vec::new();
    collect_strings(value, &mut strings);
    strings.contains(&needle)
}

/// What a cell observed on the normalized response, kept for the assertions
/// that run after the wrapper returns.
#[derive(Debug, Clone, PartialEq)]
struct Observed {
    identity: ResponseIdentity,
    finish_reason: Option<FinishReason>,
    model: Option<String>,
    usage: Usage,
    choice: Vec<AssistantContent>,
    text: String,
    raw: Value,
    /// The normalized response itself, serialized — for asserting what it
    /// does *not* carry.
    normalized: Value,
}

impl Observed {
    fn from_response(response: &RigCompletionResponse) -> Self {
        let text = response
            .choice
            .iter()
            .filter_map(|content| match content {
                AssistantContent::Text(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");
        Self {
            identity: response.identity(),
            finish_reason: response.finish_reason(),
            model: response.model.clone(),
            usage: response.usage,
            choice: response.choice.to_vec(),
            text,
            raw: response.raw.clone(),
            normalized: serde_json::to_value(response).expect("normalized response serializes"),
        }
    }
}

type ObservedSink = std::sync::Arc<std::sync::Mutex<Option<Observed>>>;

/// The body of cells 1 and 3: one probe completion, its normalized view kept
/// for the assertions that run after the wrapper has written the fixture.
async fn probe_body(client: anthropic::Client, sink: ObservedSink) {
    request_body(
        client,
        anthropic::completion::CLAUDE_HAIKU_4_5,
        probe_request,
        sink,
    )
    .await;
}

/// The body of cells 4 and 5: one completion of the request the cell
/// describes, on the model the cell names, its normalized view kept for the
/// assertions that run after the wrapper has written the fixture.
async fn request_body(
    client: anthropic::Client,
    model_name: &str,
    build: impl FnOnce(&AnthropicModel) -> rig::completion::CompletionRequest,
    sink: ObservedSink,
) {
    let model = client.completion_model(model_name);
    let response = model
        .completion(build(&model))
        .await
        .expect("completion should succeed");
    *sink.lock().expect("sink") = Some(Observed::from_response(&response));
}

fn take_observed(sink: &ObservedSink) -> Observed {
    let observed = sink.lock().expect("sink").take();
    observed.expect("the cell body ran")
}

/// Pin the normalized fields to the fixture the cell recorded: the wire body
/// (`id`, `model`, `stop_reason`, `usage`, text) and the `request-id` header.
fn assert_matches_fixture(scenario: &str, observed: &Observed) {
    let body = assert_identity_matches_fixture(scenario, observed);
    assert_eq!(body["stop_reason"], "end_turn", "{scenario}: premise");
    assert_eq!(observed.finish_reason, Some(FinishReason::Stop));
    let recorded_text = body["content"]
        .as_array()
        .expect("content array")
        .iter()
        .filter_map(|block| block["text"].as_str())
        .collect::<Vec<_>>()
        .join("");
    assert_eq!(observed.text, recorded_text);
}

/// The stop-reason-agnostic half of [`assert_matches_fixture`]: identity
/// (`msg_…` id, `request-id` header), model, and usage totals are the
/// fixture's. Returns the recorded body for the cell's own premise checks.
fn assert_identity_matches_fixture(scenario: &str, observed: &Observed) -> Value {
    let body = recorded_response_body(scenario);
    assert_ids_match_recording(
        std::slice::from_ref(&observed.identity.message_id),
        &[body["id"].as_str().map(str::to_string)],
        scenario,
    );
    let request_ids = recorded_request_id_headers(scenario);
    assert_eq!(request_ids.len(), 1, "{scenario}: one recorded interaction");
    assert!(
        request_ids[0].is_some(),
        "{scenario}: premise — the recorded response carries a `request-id` header"
    );
    assert_ids_match_recording(
        std::slice::from_ref(&observed.identity.provider_request_id),
        &request_ids,
        scenario,
    );
    assert_eq!(observed.identity.response_id, None);
    assert_eq!(observed.model.as_deref(), body["model"].as_str());
    assert_eq!(
        observed.usage.input_tokens,
        body["usage"]["input_tokens"]
            .as_u64()
            .expect("input_tokens")
    );
    assert_eq!(
        observed.usage.output_tokens,
        body["usage"]["output_tokens"]
            .as_u64()
            .expect("output_tokens")
    );
    body
}

/// Cells 4 and 5 share this: `raw` is populated, and reads back into the
/// provider type without loss.
fn assert_raw_round_trips(raw: &Value) -> CompletionResponse {
    assert!(
        !raw.is_null(),
        "every response `completion` returns carries `raw`"
    );
    let typed = CompletionResponse::deserialize(raw)
        .expect("`raw` is the serialized anthropic::completion::CompletionResponse");
    assert_eq!(
        serde_json::to_value(&typed).expect("re-serialize"),
        *raw,
        "raw ↔ typed must round-trip without loss"
    );
    typed
}

// ---------------------------------------------------------------------------
// 1: typed round trip
// ---------------------------------------------------------------------------

#[tokio::test]
async fn raw_round_trips_into_provider_type() {
    let sink = ObservedSink::default();
    with_anthropic_cassette("raw_capture_matrix/raw_round_trips_into_provider_type", {
        let sink = sink.clone();
        move |client| probe_body(client, sink)
    })
    .await;
    let observed = take_observed(&sink);
    let raw = &observed.raw;
    assert!(
        !raw.is_null(),
        "every response `completion` returns carries `raw`"
    );

    // Typed access is recoverable, and lossless: the provider type reads its
    // own serialization back and re-serializes to the identical value.
    let typed = CompletionResponse::deserialize(raw)
        .expect("`raw` is the serialized anthropic::completion::CompletionResponse");
    assert_eq!(
        serde_json::to_value(&typed).expect("re-serialize"),
        *raw,
        "raw ↔ typed must round-trip without loss"
    );

    // `raw` is the value `raw_completion` would have returned — the wire as
    // rig's type parsed it — so its wire-derived fields equal the recorded
    // body's, and its transport id is the header the request driver stamped.
    let body = recorded_response_body(ROUND_TRIP_SCENARIO);
    assert_ids_match_recording(
        &[raw["id"].as_str().map(str::to_string)],
        &[body["id"].as_str().map(str::to_string)],
        ROUND_TRIP_SCENARIO,
    );
    assert_eq!(raw["model"], body["model"]);
    assert_eq!(raw["stop_reason"], body["stop_reason"]);
    assert_eq!(raw["role"], body["role"]);
    assert_eq!(raw["usage"]["input_tokens"], body["usage"]["input_tokens"]);
    assert_eq!(
        raw["usage"]["output_tokens"],
        body["usage"]["output_tokens"]
    );
    assert_eq!(
        raw["content"][0]["text"], body["content"][0]["text"],
        "the parsed text block is the recorded one"
    );
    assert_ids_match_recording(
        &[raw["provider_request_id"].as_str().map(str::to_string)],
        &recorded_request_id_headers(ROUND_TRIP_SCENARIO),
        ROUND_TRIP_SCENARIO,
    );
    // And the normalized view beside it reports what the fixture recorded.
    assert_matches_fixture(ROUND_TRIP_SCENARIO, &observed);
}

// ---------------------------------------------------------------------------
// 2: a field rig does not normalize
// ---------------------------------------------------------------------------

#[tokio::test]
async fn raw_exposes_stop_sequence() {
    let sink: ObservedSink = Default::default();
    let observed = sink.clone();
    with_anthropic_cassette(
        "raw_capture_matrix/raw_exposes_stop_sequence",
        move |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_HAIKU_4_5);
            let response = model
                .completion(
                    model
                        .completion_request(IMMEDIATE_PROMPT)
                        .max_tokens(32)
                        .additional_params(json!({ "stop_sequences": ["alpha"] }))
                        .build(),
                )
                .await
                .expect("stop-sequence completion should succeed");
            *observed.lock().expect("sink") = Some(Observed::from_response(&response));
        },
    )
    .await;
    let observed = sink
        .lock()
        .expect("sink")
        .clone()
        .expect("the cell body ran");

    // Premise, from the fixture: the recorded turn stopped on the sequence and
    // named it — the shape `empty_stop_sequence_matrix.rs` established.
    let body = recorded_response_body(STOP_SEQUENCE_SCENARIO);
    assert_eq!(
        body["stop_reason"], "stop_sequence",
        "premise: the recorded turn stopped on a sequence"
    );
    assert_eq!(
        body["stop_sequence"], "alpha",
        "premise: the recorded turn names the sequence it stopped on"
    );

    // The normalized `CompletionResponse` has no `stop_sequence` field —
    // rig folds the stop into `FinishReason::Stop` and the sequence itself is
    // not part of the normalized vocabulary. Its serialized form proves it.
    let raw = &observed.raw;
    assert!(
        !raw.is_null(),
        "every response `completion` returns carries `raw`"
    );
    assert_eq!(observed.finish_reason, Some(FinishReason::Stop));
    let normalized_keys: Vec<String> = observed
        .normalized
        .as_object()
        .expect("the normalized response serializes as an object")
        .keys()
        .cloned()
        .collect();
    assert!(
        !normalized_keys.iter().any(|key| key == "stop_sequence"),
        "the normalized response has no `stop_sequence` field ({normalized_keys:?}) — \
         `raw` is the only way to read it"
    );

    // …and `raw` carries it, verbatim from the wire.
    assert_eq!(raw["stop_sequence"], "alpha");
    assert_eq!(raw["stop_reason"], "stop_sequence");
    let typed = CompletionResponse::deserialize(raw).expect("typed access");
    assert_eq!(typed.stop_sequence.as_deref(), Some("alpha"));
}

// ---------------------------------------------------------------------------
// 3: raw and the normalized fields tell one story
// ---------------------------------------------------------------------------

/// The normalized response and `raw` describe the same exchange: reading
/// `raw` back into the provider type and running rig's own
/// `NormalizeCompletionResponse` over it reproduces every normalized field
/// delivered beside it — identity, finish reason, model, usage, and the
/// choice — and each of those is what the fixture recorded.
#[tokio::test]
async fn normalized_fields_match_raw_renormalized() {
    let sink = ObservedSink::default();
    with_anthropic_cassette(
        "raw_capture_matrix/normalized_fields_match_raw_renormalized",
        {
            let sink = sink.clone();
            move |client| probe_body(client, sink)
        },
    )
    .await;
    let observed = take_observed(&sink);
    let raw = &observed.raw;
    assert!(
        !raw.is_null(),
        "every response `completion` returns carries `raw`"
    );

    let renormalized: RigCompletionResponse = CompletionResponse::deserialize(raw)
        .expect("`raw` is the serialized anthropic::completion::CompletionResponse")
        .normalize(ANTHROPIC_PROVIDER)
        .expect("the provider type re-normalizes");
    assert_eq!(
        renormalized.identity(),
        observed.identity,
        "identity (message id, transport id) survives raw → typed → normalize"
    );
    assert_eq!(renormalized.finish_reason(), observed.finish_reason);
    assert_eq!(renormalized.model, observed.model);
    assert_eq!(renormalized.usage, observed.usage);
    assert_eq!(
        renormalized.choice.to_vec(),
        observed.choice,
        "the choice rig derives from `raw` is the choice it delivered"
    );

    // …and none of that is vacuous: the normalized fields are the fixture's.
    assert_matches_fixture(RENORMALIZED_SCENARIO, &observed);
}

// ---------------------------------------------------------------------------
// 4: an extended-thinking turn — the wire shape with a signature
// ---------------------------------------------------------------------------

/// `raw` on a thinking turn is still lossless against the provider type, and
/// carries the wire's own spelling of the reasoning: a `content[]` block of
/// `type: "thinking"` with `thinking` text and a `signature`, plus
/// `usage.output_tokens_details.thinking_tokens`. The normalized response
/// re-spells the block as `type: "reasoning"` / `content[].type: "text"` and
/// folds the bucket into `usage.reasoning_tokens` — nowhere in it does the
/// string `"thinking"` appear.
#[tokio::test]
async fn raw_exposes_thinking_block_and_signature() {
    let sink = ObservedSink::default();
    with_anthropic_cassette(
        "raw_capture_matrix/raw_exposes_thinking_block_and_signature",
        {
            let sink = sink.clone();
            move |client| {
                request_body(
                    client,
                    anthropic::completion::CLAUDE_SONNET_4_6,
                    thinking_request,
                    sink,
                )
            }
        },
    )
    .await;
    let observed = take_observed(&sink);
    let raw = &observed.raw;
    let typed = assert_raw_round_trips(raw);

    // Premise, from the fixture: the recorded body carries a `thinking` block
    // with a signature, and the usage breakdown says thinking happened.
    let body = assert_identity_matches_fixture(THINKING_SCENARIO, &observed);
    let recorded_blocks = body["content"].as_array().expect("content array");
    let recorded_thinking = recorded_blocks
        .iter()
        .find(|block| block["type"] == "thinking")
        .expect("premise: the recorded body carries a `thinking` content block");
    let recorded_thinking_text = recorded_thinking["thinking"]
        .as_str()
        .expect("premise: the recorded thinking block has `thinking` text");
    let recorded_signature = recorded_thinking["signature"].as_str().map(str::to_string);
    assert!(
        recorded_signature
            .as_deref()
            .is_some_and(|sig| !sig.is_empty()),
        "premise: the recorded thinking block carries a non-empty `signature`"
    );
    let recorded_thinking_tokens = body["usage"]["output_tokens_details"]["thinking_tokens"]
        .as_u64()
        .expect("premise: the recorded usage carries `output_tokens_details.thinking_tokens`");
    assert!(
        recorded_thinking_tokens > 0,
        "premise: the recorded turn actually spent thinking tokens"
    );
    assert_eq!(body["stop_reason"], "end_turn", "premise");
    assert_eq!(observed.finish_reason, Some(FinishReason::Stop));

    // `raw` carries the wire's own spelling of the reasoning, verbatim.
    let raw_blocks = raw["content"].as_array().expect("raw content array");
    let raw_thinking = raw_blocks
        .iter()
        .find(|block| block["type"] == "thinking")
        .expect("`raw` carries the `thinking` content block");
    assert_eq!(raw_thinking["thinking"], recorded_thinking_text);
    assert_ids_match_recording(
        &[raw_thinking["signature"].as_str().map(str::to_string)],
        std::slice::from_ref(&recorded_signature),
        THINKING_SCENARIO,
    );
    assert!(
        raw_thinking["signature"]
            .as_str()
            .is_some_and(|sig| !sig.is_empty()),
        "`raw` carries the signature"
    );
    assert_eq!(
        raw["usage"]["output_tokens_details"]["thinking_tokens"],
        json!(recorded_thinking_tokens),
        "`raw` carries the provider's usage breakdown, verbatim"
    );
    // …and typed access reads it as the provider's own variant.
    let typed_thinking = typed
        .content
        .iter()
        .find_map(|block| match block {
            Content::Thinking {
                thinking,
                signature,
            } => Some((thinking.as_str(), signature.as_deref())),
            _ => None,
        })
        .expect("typed `raw` carries `Content::Thinking`");
    assert_eq!(typed_thinking.0, recorded_thinking_text);
    assert!(typed_thinking.1.is_some_and(|sig| !sig.is_empty()));

    // The normalized response carries the same reasoning — but re-spelled.
    // Its choice has a `Reasoning` block whose text and signature are the
    // wire's; its usage folds the bucket into `reasoning_tokens`; and the
    // wire spelling `"thinking"` — block type, text key, usage bucket — occurs
    // nowhere in it. Only `raw` speaks it.
    let normalized_reasoning = observed
        .choice
        .iter()
        .find_map(|content| match content {
            AssistantContent::Reasoning(reasoning) => Some(reasoning),
            _ => None,
        })
        .expect("the normalized choice carries the reasoning");
    assert_eq!(
        normalized_reasoning.content,
        vec![ReasoningContent::Text {
            text: recorded_thinking_text.to_string(),
            signature: raw_thinking["signature"].as_str().map(str::to_string),
        }],
        "the normalized reasoning is the wire's text and signature, re-spelled"
    );
    assert_eq!(observed.usage.reasoning_tokens, recorded_thinking_tokens);
    let mut normalized_without_raw = observed.normalized.clone();
    normalized_without_raw
        .as_object_mut()
        .expect("the normalized response serializes as an object")
        .remove("raw")
        .expect("the normalized response carries `raw`");
    assert!(
        !contains_string(&normalized_without_raw, "thinking"),
        "the normalized response never spells `thinking` — `raw` is the only way to read \
         the wire's block type, text key and usage bucket: {normalized_without_raw}"
    );
    assert!(
        !contains_string(&normalized_without_raw, "thinking_tokens"),
        "the normalized usage says `reasoning_tokens`, not `thinking_tokens`"
    );
    assert!(
        contains_string(&normalized_without_raw, "reasoning"),
        "the normalized response spells the block `reasoning`"
    );
}

// ---------------------------------------------------------------------------
// 5: a forced tool call — the wire's tool-call representation
// ---------------------------------------------------------------------------

/// `raw` on a `tool_use` turn is still lossless against the provider type, and
/// carries the wire's own tool-call representation: a `content[]` block of
/// `type: "tool_use"` with `id`, `name` and an `input` *object*, under
/// `stop_reason: "tool_use"`. The normalized response reports
/// `FinishReason::ToolCalls` and a `type: "toolcall"` block with
/// `function.arguments` — the string `"tool_use"` occurs nowhere in it.
#[tokio::test]
async fn raw_exposes_tool_use_block() {
    let sink = ObservedSink::default();
    with_anthropic_cassette("raw_capture_matrix/raw_exposes_tool_use_block", {
        let sink = sink.clone();
        move |client| {
            request_body(
                client,
                anthropic::completion::CLAUDE_HAIKU_4_5,
                tool_request,
                sink,
            )
        }
    })
    .await;
    let observed = take_observed(&sink);
    let raw = &observed.raw;
    let typed = assert_raw_round_trips(raw);

    // Premise, from the fixture: the recorded turn is a `tool_use` terminal
    // whose body carries a `tool_use` block for the forced tool, with an
    // `input` object naming the city.
    let body = assert_identity_matches_fixture(TOOL_USE_SCENARIO, &observed);
    assert_eq!(
        body["stop_reason"], "tool_use",
        "premise: the recorded turn stopped to call a tool"
    );
    let recorded_blocks = body["content"].as_array().expect("content array");
    let recorded_tool_use = recorded_blocks
        .iter()
        .find(|block| block["type"] == "tool_use")
        .expect("premise: the recorded body carries a `tool_use` content block");
    assert_eq!(recorded_tool_use["name"], "get_weather", "premise");
    let recorded_input = recorded_tool_use["input"]
        .as_object()
        .expect("premise: the recorded `tool_use.input` is a JSON object");
    assert!(
        recorded_input["city"].is_string(),
        "premise: the recorded `input` names a `city`: {recorded_input:?}"
    );
    let recorded_tool_id = recorded_tool_use["id"].as_str().map(str::to_string);
    assert!(
        recorded_tool_id
            .as_deref()
            .is_some_and(|id| id.starts_with("toolu_")),
        "premise: the recorded `tool_use` block has a `toolu_…` id"
    );

    // Normalized: the provider's stop reason maps onto `ToolCalls`; the
    // spelling `tool_use` is only on `raw`.
    assert_eq!(observed.finish_reason, Some(FinishReason::ToolCalls));
    assert_eq!(raw["stop_reason"], "tool_use");
    assert_eq!(typed.stop_reason.as_deref(), Some("tool_use"));

    // `raw` carries the wire's tool-call block, verbatim: `input` is an
    // object (Anthropic sends structured JSON, not an arguments string).
    let raw_blocks = raw["content"].as_array().expect("raw content array");
    let raw_tool_use = raw_blocks
        .iter()
        .find(|block| block["type"] == "tool_use")
        .expect("`raw` carries the `tool_use` content block");
    assert_eq!(raw_tool_use["name"], "get_weather");
    assert!(
        raw_tool_use["input"].is_object(),
        "the wire's `input` is a JSON object: {}",
        raw_tool_use["input"]
    );
    assert_eq!(raw_tool_use["input"], recorded_tool_use["input"]);
    assert_ids_match_recording(
        &[raw_tool_use["id"].as_str().map(str::to_string)],
        std::slice::from_ref(&recorded_tool_id),
        TOOL_USE_SCENARIO,
    );
    let (typed_id, typed_name, typed_input) = typed
        .content
        .iter()
        .find_map(|block| match block {
            Content::ToolUse { id, name, input } => Some((id.as_str(), name.as_str(), input)),
            _ => None,
        })
        .expect("typed `raw` carries `Content::ToolUse`");
    assert_eq!(typed_name, "get_weather");
    assert_eq!(*typed_input, recorded_tool_use["input"]);

    // The normalized choice carries the same call, re-spelled: `toolcall`
    // with `function.name` / `function.arguments`, its provider id the wire's
    // — and the string `"tool_use"` occurs nowhere in the normalized response.
    let normalized_call = observed
        .choice
        .iter()
        .find_map(|content| match content {
            AssistantContent::ToolCall(call) => Some(call),
            _ => None,
        })
        .expect("the normalized choice carries the tool call");
    assert_eq!(normalized_call.function.name, "get_weather");
    assert_eq!(
        normalized_call.function.arguments,
        recorded_tool_use["input"]
    );
    assert_eq!(
        normalized_call
            .provider
            .as_ref()
            .map(|provider| provider.call_id.as_str()),
        Some(typed_id),
        "the normalized call's provider id is the wire's `tool_use.id`"
    );
    let mut normalized_without_raw = observed.normalized.clone();
    normalized_without_raw
        .as_object_mut()
        .expect("the normalized response serializes as an object")
        .remove("raw")
        .expect("the normalized response carries `raw`");
    assert!(
        !contains_string(&normalized_without_raw, "tool_use"),
        "the normalized response never spells `tool_use` — `raw` is the only way to read \
         the wire's block type and stop reason: {normalized_without_raw}"
    );
    assert!(
        contains_string(&normalized_without_raw, "toolcall"),
        "the normalized response spells the block `toolcall`"
    );
}
