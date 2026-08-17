//! Matrix for raw provider response capture on the blocking path:
//! `CompletionResponse::raw` beside the normalized fields.
//!
//! # The feature
//!
//! Capture is always on. Every response `completion` returns carries `raw`:
//! exactly what `raw_completion` would have returned — the response as
//! `anthropic::completion::CompletionResponse` parsed it — serialized with
//! `serde_json::to_value`. `raw` is `Option` only because a
//! `CompletionResponse` built by hand has no provider response behind it;
//! `None` never means "not requested". This matrix pins three properties
//! against live recordings: presence and lossless typed round-trip, a
//! provider-specific field the normalized response provably lacks
//! (`stop_sequence`), and that `raw` and the normalized fields tell one story
//! (re-normalizing `raw` reproduces them).
//!
//! # Matrix
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `raw_round_trips_into_provider_type` | plain text request | `raw` populated; deserializes into the Anthropic type and re-serializes equal; wire fields equal the fixture's | recorded |
//! | 2 | `raw_exposes_stop_sequence` | `stop_sequences: ["alpha"]` request | `raw["stop_sequence"] == "alpha"`; normalized response has no such field | recorded |
//! | 3 | `normalized_fields_match_raw_renormalized` | plain text request | `CompletionResponse::deserialize(raw).normalize("anthropic")` reproduces `identity()`, `finish_reason()`, `model`, `usage`, `choice` | recorded |
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
//! exchanges.

use rig::completion::{
    CompletionModel as _, CompletionResponse as RigCompletionResponse, FinishReason,
    NormalizeCompletionResponse, ResponseIdentity, Usage,
};
use rig::message::AssistantContent;
use rig::prelude::*;
use rig::providers::anthropic;
use rig::providers::anthropic::completion::CompletionResponse;
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

const ROUND_TRIP_SCENARIO: &str = "raw_capture_matrix/raw_round_trips_into_provider_type";
const STOP_SEQUENCE_SCENARIO: &str = "raw_capture_matrix/raw_exposes_stop_sequence";
const RENORMALIZED_SCENARIO: &str = "raw_capture_matrix/normalized_fields_match_raw_renormalized";

type AnthropicModel = anthropic::completion::CompletionModel;

fn probe_request(model: &AnthropicModel) -> rig::completion::CompletionRequest {
    model.completion_request(PROMPT).max_tokens(32).build()
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
    raw: Option<Value>,
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
            raw: response.raw.as_deref().cloned(),
            normalized: serde_json::to_value(response).expect("normalized response serializes"),
        }
    }
}

type ObservedSink = std::sync::Arc<std::sync::Mutex<Option<Observed>>>;

/// The body of cells 1 and 3: one probe completion, its normalized view kept
/// for the assertions that run after the wrapper has written the fixture.
async fn probe_body(client: anthropic::Client, sink: ObservedSink) {
    let model = client.completion_model(anthropic::completion::CLAUDE_HAIKU_4_5);
    let response = model
        .completion(probe_request(&model))
        .await
        .expect("probe completion should succeed");
    *sink.lock().expect("sink") = Some(Observed::from_response(&response));
}

fn take_observed(sink: &ObservedSink) -> Observed {
    let observed = sink.lock().expect("sink").take();
    observed.expect("the cell body ran")
}

/// Pin the normalized fields to the fixture the cell recorded: the wire body
/// (`id`, `model`, `stop_reason`, `usage`, text) and the `request-id` header.
fn assert_matches_fixture(scenario: &str, observed: &Observed) {
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
    assert_eq!(body["stop_reason"], "end_turn", "{scenario}: premise");
    assert_eq!(observed.finish_reason, Some(FinishReason::Stop));
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
    let recorded_text = body["content"]
        .as_array()
        .expect("content array")
        .iter()
        .filter_map(|block| block["text"].as_str())
        .collect::<Vec<_>>()
        .join("");
    assert_eq!(observed.text, recorded_text);
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
    let raw = observed
        .raw
        .as_ref()
        .expect("every response `completion` returns carries `raw`");

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
    let raw = observed
        .raw
        .as_ref()
        .expect("every response `completion` returns carries `raw`");
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
    let raw = observed
        .raw
        .as_ref()
        .expect("every response `completion` returns carries `raw`");

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
