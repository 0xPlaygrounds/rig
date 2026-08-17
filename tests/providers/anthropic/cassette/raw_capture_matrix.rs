//! Matrix for opt-in raw provider response capture on the blocking path:
//! `CompletionRequest::capture_raw_response` → `CompletionResponse::raw`.
//!
//! # The feature
//!
//! `raw` is populated only when the request opted in, and its value is exactly
//! what `raw_completion` would have returned — the response as
//! `anthropic::completion::CompletionResponse` parsed it — serialized with
//! `serde_json::to_value`. It is local policy: nothing about the flag reaches
//! the wire. This matrix pins all four properties against live recordings:
//! absence by default, presence and typed round-trip when on, a
//! provider-specific field the normalized response provably lacks
//! (`stop_sequence`), and the on-wire request invariant.
//!
//! # Matrix
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `flag_off_raw_is_none` | default request | `raw.is_none()` | recorded |
//! | 2 | `flag_on_raw_round_trips` | `capture_raw_response(true)` | `raw` deserializes into the Anthropic type and re-serializes equal | recorded |
//! | 3 | `flag_on_exposes_stop_sequence` | `stop_sequences: ["alpha"]` request | `raw["stop_sequence"] == "alpha"`; normalized response has no such field | recorded |
//! | 4 | `request_invariant_off_vs_on` | recorded request bodies of 1 and 2 | byte-identical | recorded (derived from cells 1 and 2) |
//! | 5 | `normalized_fields_identical_off_vs_on` | identity / finish_reason / model / usage / choice of 1 vs 2 | equal, each pinned to its fixture | recorded (derived from cells 1 and 2) |
//!
//! Cells 4 and 5 make no request of their own: they are the cross-fixture
//! comparisons of cells 1 and 2, so they read those two fixtures directly and
//! fail loudly if either fixture is missing. Cells 1
//! and 2 send the *same* prompt precisely so that comparison means something.
//! Cell 3 reuses the `stop_sequences: ["alpha"]` request shape from
//! `empty_stop_sequence_matrix.rs`, where a one-word reply matches the
//! sequence and Anthropic reports it back on `stop_sequence`.

use rig::completion::{
    CompletionModel as _, CompletionResponse as RigCompletionResponse, FinishReason,
    ResponseIdentity, Usage,
};
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

const OFF_SCENARIO: &str = "raw_capture_matrix/flag_off_raw_is_none";
const ON_SCENARIO: &str = "raw_capture_matrix/flag_on_raw_round_trips";
const STOP_SEQUENCE_SCENARIO: &str = "raw_capture_matrix/flag_on_exposes_stop_sequence";

type AnthropicModel = anthropic::completion::CompletionModel;

fn probe_request(model: &AnthropicModel, capture: bool) -> rig::completion::CompletionRequest {
    model
        .completion_request(PROMPT)
        .max_tokens(32)
        .capture_raw_response(capture)
        .build()
}

/// What a cell observed on the normalized response, kept for cross-cell
/// comparison after the wrapper returns.
#[derive(Debug, Clone, PartialEq)]
struct Observed {
    identity: ResponseIdentity,
    finish_reason: Option<FinishReason>,
    model: Option<String>,
    usage: Usage,
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
                rig::message::AssistantContent::Text(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");
        Self {
            identity: response.identity(),
            finish_reason: response.finish_reason(),
            model: response.model.clone(),
            usage: response.usage,
            text,
            raw: response.raw.as_deref().cloned(),
            normalized: serde_json::to_value(response).expect("normalized response serializes"),
        }
    }
}

type ObservedSink = std::sync::Arc<std::sync::Mutex<Option<Observed>>>;

/// The body of cells 1 and 2: one probe completion, its normalized view kept
/// for the assertions that run after the wrapper has written the fixture.
async fn probe_body(client: anthropic::Client, capture: bool, sink: ObservedSink) {
    let model = client.completion_model(anthropic::completion::CLAUDE_HAIKU_4_5);
    let response = model
        .completion(probe_request(&model, capture))
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
// 1: default off
// ---------------------------------------------------------------------------

#[tokio::test]
async fn flag_off_raw_is_none() {
    let sink = ObservedSink::default();
    with_anthropic_cassette("raw_capture_matrix/flag_off_raw_is_none", {
        let sink = sink.clone();
        move |client| probe_body(client, false, sink)
    })
    .await;
    let observed = take_observed(&sink);
    assert!(
        observed.raw.is_none(),
        "capture is opt-in: a request that did not ask must not pay for it"
    );
    assert_matches_fixture(OFF_SCENARIO, &observed);
}

// ---------------------------------------------------------------------------
// 2: on → typed round trip
// ---------------------------------------------------------------------------

#[tokio::test]
async fn flag_on_raw_round_trips() {
    let sink = ObservedSink::default();
    with_anthropic_cassette("raw_capture_matrix/flag_on_raw_round_trips", {
        let sink = sink.clone();
        move |client| probe_body(client, true, sink)
    })
    .await;
    let observed = take_observed(&sink);
    let raw = observed
        .raw
        .as_ref()
        .expect("the request opted in, so `raw` must be populated");

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
    let body = recorded_response_body(ON_SCENARIO);
    assert_ids_match_recording(
        &[raw["id"].as_str().map(str::to_string)],
        &[body["id"].as_str().map(str::to_string)],
        ON_SCENARIO,
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
        &recorded_request_id_headers(ON_SCENARIO),
        ON_SCENARIO,
    );
    // And the normalized view beside it is unchanged by capture.
    assert_matches_fixture(ON_SCENARIO, &observed);
}

// ---------------------------------------------------------------------------
// 3: a field rig does not normalize
// ---------------------------------------------------------------------------

#[tokio::test]
async fn flag_on_exposes_stop_sequence() {
    let sink: ObservedSink = Default::default();
    let observed = sink.clone();
    with_anthropic_cassette(
        "raw_capture_matrix/flag_on_exposes_stop_sequence",
        move |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_HAIKU_4_5);
            let response = model
                .completion(
                    model
                        .completion_request(IMMEDIATE_PROMPT)
                        .max_tokens(32)
                        .additional_params(json!({ "stop_sequences": ["alpha"] }))
                        .capture_raw_response(true)
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
    let raw = observed.raw.as_ref().expect("the request opted in");
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
// 4–5: cross-fixture invariants between the off and on cells
// ---------------------------------------------------------------------------

/// The flag never reaches the provider: the request bodies the off and on
/// cells recorded are byte-identical.
#[test]
fn request_invariant_off_vs_on() {
    let off = crate::cassettes::recorded_interaction_bodies(ANTHROPIC_PROVIDER, OFF_SCENARIO);
    let on = crate::cassettes::recorded_interaction_bodies(ANTHROPIC_PROVIDER, ON_SCENARIO);
    assert_eq!(off.len(), 1, "{OFF_SCENARIO}: one recorded interaction");
    assert_eq!(on.len(), 1, "{ON_SCENARIO}: one recorded interaction");
    assert_eq!(
        off[0].0, on[0].0,
        "the recorded request bodies must be byte-identical: capture is local policy"
    );
    // The same, structurally, so a mismatch is readable.
    assert_eq!(
        crate::cassettes::recorded_json_request(ANTHROPIC_PROVIDER, OFF_SCENARIO),
        crate::cassettes::recorded_json_request(ANTHROPIC_PROVIDER, ON_SCENARIO),
    );
    // And neither body mentions the flag by name in any spelling.
    for (request, _) in off.iter().chain(on.iter()) {
        assert!(
            !request.contains("capture_raw") && !request.contains("captureRaw"),
            "the flag is `#[serde(skip)]`; it must never serialize: {request}"
        );
    }
}

/// Capture changes nothing about the normalized response: every normalized
/// field the off and on cells report is the field their own fixture recorded,
/// and the two recordings agree wherever the wire is deterministic.
#[test]
fn normalized_fields_identical_off_vs_on() {
    let off = recorded_response_body(OFF_SCENARIO);
    let on = recorded_response_body(ON_SCENARIO);

    // Deterministic across two identical prompts: model, stop reason, role,
    // input token count. (Message ids and request ids are per-exchange by
    // nature; the recorded text is compared too — for this pinned "reply
    // exactly" prompt the model complies on both recordings.)
    assert_eq!(off["model"], on["model"]);
    assert_eq!(off["stop_reason"], on["stop_reason"]);
    assert_eq!(off["role"], on["role"]);
    assert_eq!(off["usage"]["input_tokens"], on["usage"]["input_tokens"]);
    assert_eq!(off["content"], on["content"]);

    // Both fixtures carry the identity the normalized route reports.
    for scenario in [OFF_SCENARIO, ON_SCENARIO] {
        let ids = recorded_request_id_headers(scenario);
        assert_eq!(ids.len(), 1);
        assert!(ids[0].is_some(), "{scenario}: `request-id` header recorded");
        let body = recorded_response_body(scenario);
        assert!(
            body["id"].as_str().is_some_and(|id| id.starts_with("msg_")),
            "{scenario}: message id recorded"
        );
    }
}
