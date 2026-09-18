//! Matrix for raw response capture on both Copilot blocking routes
//! ([`CompletionResponse::raw`](rig::completion::CompletionResponse::raw)).
//!
//! # The feature
//!
//! Capture is always on. Every completion the driver returns carries `raw`:
//! the route's own reply *document*, set from the reply's bytes
//! (`driver::call` does `serde_json::from_slice(&body)`), untagged. So it is
//! not a round trip through whatever type the decoder parsed — it is the body
//! Copilot sent, and it reads back into the type that route owns
//! ([`openai::CompletionResponse`] on chat completions,
//! [`responses_api::CompletionResponse`] on the Responses route). Nothing
//! about it is sent to Copilot. `raw == Value::Null` means only that a
//! `CompletionResponse` was built by hand without a provider response behind
//! it, which no cell here can produce.
//!
//! That the document is the body, and not the parse, is what lets a caller
//! reach the fields rig's shared types do not model: Copilot's chat route
//! sends a `copilot_usage` block that the shared chat-completions type has no
//! field for, and it reaches the caller through `raw` for exactly that
//! reason. Cell 1 pins it, so the capability stays stated rather than
//! rediscovered.
//!
//! Which route a turn took is a fact about the wire rather than about `raw`,
//! so each typed-access cell asserts it on the bound wire itself.
//!
//! Provider-only fields per route: the chat route's `system_fingerprint` and
//! `copilot_usage`; the Responses route's `object`/`status`.
//!
//! # Matrix
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `chat_raw_round_trips_provider_type` | chat route, typed access | the wire is `CopilotWire::Chat`; `raw` is untagged, reads back as `openai::CompletionResponse`, and carries the recorded reply's own keys including the unmodelled `copilot_usage` | unrecorded (no COPILOT credentials in this environment) |
//! | 2 | `chat_raw_exposes_system_fingerprint` | chat route, provider-only field | `raw.system_fingerprint` equals the fixture body's | unrecorded (no COPILOT credentials in this environment) |
//! | 3 | `chat_normalized_fields_equal_raw_renormalized` | chat route, normalized view | every normalized field equals the provider-native field it was mapped from, on `raw` and on the fixture body | unrecorded (no COPILOT credentials in this environment) |
//! | 4 | `responses_raw_round_trips_provider_type` | responses route, typed access | the wire is `CopilotWire::Responses`; `raw` is untagged, reads back as `responses_api::CompletionResponse`, and carries the recorded reply's own keys | unrecorded (no COPILOT credentials in this environment) |
//! | 5 | `responses_raw_exposes_envelope` | responses route, provider-only fields | `raw.object`/`raw.status` equal the fixture body's | unrecorded (no COPILOT credentials in this environment) |
//! | 6 | `responses_normalized_fields_equal_raw_renormalized` | responses route, normalized view | every normalized field equals the provider-native field it was mapped from, on `raw` and on the fixture body | unrecorded (no COPILOT credentials in this environment) |
//!
//! Every cell is unrecorded: none of `GITHUB_COPILOT_API_KEY`,
//! `COPILOT_API_KEY`, `COPILOT_GITHUB_ACCESS_TOKEN`/`GITHUB_TOKEN` nor a Copilot
//! OAuth cache was present when this matrix was written, and a fixture is
//! never fabricated. To record: export `GITHUB_COPILOT_API_KEY` (the harness
//! placeholders it on disk), remove the `#[ignore]` attributes, flip the table
//! to `recorded`, then run
//! `RIG_PROVIDER_TEST_MODE=record cargo test -p rig --all-features --test copilot copilot::raw_capture_matrix -- --nocapture --test-threads=1`
//! and review `tests/cassettes/copilot/raw_capture_matrix/`.

use rig::completion::{CompletionModel as _, FinishReason};
use rig::driver::Bound;
use rig::providers::copilot;
use rig::providers::copilot::wire::CopilotWire;
use rig::providers::openai;
use rig::providers::openai::responses_api;
use rig::providers::openai::wire::OpenAiWire;
use serde::Deserialize;
use serde_json::Value;

use crate::cassettes::{CassetteMode, recorded_json_turn};
use crate::copilot::with_copilot_cassette;
use crate::support::{Observed, assert_wire_value_matches, assistant_text, normalized_without_raw};

const COPILOT_PROVIDER: &str = "copilot";
const CHAT_MODEL: &str = copilot::GPT_4O;
const RESPONSES_MODEL: &str = copilot::GPT_5_3_CODEX;
const PROMPT: &str = "Reply with exactly the single word: pong";

fn request(model: &Bound<CopilotWire>) -> rig::completion::CompletionRequest {
    model.completion_request(PROMPT).max_tokens(64).build()
}

/// Chat-route premise: the recorded body is a chat-completions response with
/// usage and a `system_fingerprint`.
fn assert_recorded_chat_body(body: &Value, scenario: &str) {
    assert!(
        body.get("choices").is_some_and(Value::is_array),
        "{scenario}: the recorded body must be a chat-completions response"
    );
    assert!(
        body.get("usage").is_some_and(Value::is_object),
        "{scenario}: the recorded body must report usage"
    );
    assert!(
        body.get("system_fingerprint")
            .and_then(Value::as_str)
            .is_some(),
        "{scenario}: the recorded body must carry `system_fingerprint` — without \
         it this cell cannot prove raw exposes a provider-only field"
    );
}

/// Responses-route premise: the recorded body is a completed Responses
/// envelope with usage.
fn assert_recorded_responses_body(body: &Value, scenario: &str) {
    assert_eq!(
        body.get("object").and_then(Value::as_str),
        Some("response"),
        "{scenario}: the recorded body must be a Responses envelope"
    );
    assert_eq!(
        body.get("status").and_then(Value::as_str),
        Some("completed"),
        "{scenario}: the recorded turn must be completed"
    );
    assert!(
        body.pointer("/usage/total_tokens").is_some(),
        "{scenario}: the recorded body must report usage"
    );
}

/// The scrubber placeholders generated ids, so a replay compares them exactly
/// while a live recording can only require that both sides carry one.
fn assert_id_matches(live: Option<&str>, recorded: Option<&str>, what: &str) {
    match CassetteMode::current() {
        CassetteMode::Replay => assert_eq!(
            live, recorded,
            "{what}: the replayed id must equal the recorded one"
        ),
        CassetteMode::Record => assert!(
            live.is_some() && recorded.is_some(),
            "{what}: both the live value and the recording must carry it"
        ),
    }
}

/// `raw` is the reply document: it carries the top-level keys the recorded
/// reply carried, whatever subset of them rig's wire type models. Scrubbing
/// rewrites values and never keys, so the key set compares exactly in both
/// cassette modes.
fn assert_is_reply_document(raw: &Value, body: &Value, scenario: &str) {
    let keys = |value: &Value| -> Vec<String> {
        let mut keys: Vec<String> = value
            .as_object()
            .unwrap_or_else(|| panic!("{scenario}: a reply document must be a JSON object"))
            .keys()
            .cloned()
            .collect();
        keys.sort();
        keys
    };
    assert_eq!(
        keys(raw),
        keys(body),
        "{scenario}: raw must carry the reply document's own top-level keys"
    );
}

/// The normalized finish reason the chat route's `finish_reason` maps to.
fn finish_reason_of(reason: &str) -> FinishReason {
    match reason {
        "stop" => FinishReason::Stop,
        "length" => FinishReason::Length,
        "tool_calls" => FinishReason::ToolCalls,
        other => panic!("unexpected recorded finish reason {other:?}"),
    }
}

// ===========================================================================
// Chat-completions route
// ===========================================================================

#[tokio::test]
#[ignore = "unrecorded (no COPILOT credentials in this environment)"]
async fn chat_raw_round_trips_provider_type() {
    let scenario = "raw_capture_matrix/chat_raw_round_trips_provider_type";
    let captured = Observed::default();
    let sink = captured.clone();
    with_copilot_cassette(
        "raw_capture_matrix/chat_raw_round_trips_provider_type",
        |client| async move {
            let model = client.completion(CHAT_MODEL);
            assert!(
                matches!(model.wire.wire, OpenAiWire::Chat(_)),
                "premise: gpt-4o routes through chat completions"
            );
            let response = model
                .completion(request(&model))
                .await
                .expect("completion should succeed");

            let raw = &response.raw;
            assert!(
                raw.get("api").is_none(),
                "raw is the route's own reply document, not a rig-tagged envelope"
            );
            openai::CompletionResponse::deserialize(raw)
                .expect("raw must read back as the chat route's own response type");
            assert!(
                raw.get("copilot_usage").is_some(),
                "raw carries Copilot's `copilot_usage` block, which the shared \
                 chat-completions type does not model: it reaches the caller \
                 because raw is the reply document"
            );
            assert_eq!(response.provider, COPILOT_PROVIDER);
            assert!(!response.choice.is_empty());
            sink.put(response.raw);
        },
    )
    .await;

    let raw = captured.take();
    let (_, body) = recorded_json_turn(COPILOT_PROVIDER, scenario);
    assert_recorded_chat_body(&body, scenario);
    openai::CompletionResponse::deserialize(&body)
        .expect("recorded body must be a chat-completions response");
    assert_is_reply_document(&raw, &body, scenario);
    assert_eq!(
        raw.get("copilot_usage"),
        body.get("copilot_usage"),
        "raw must carry the recorded reply's `copilot_usage` block verbatim"
    );
}

#[tokio::test]
#[ignore = "unrecorded (no COPILOT credentials in this environment)"]
async fn chat_raw_exposes_system_fingerprint() {
    let scenario = "raw_capture_matrix/chat_raw_exposes_system_fingerprint";
    let captured = Observed::default();
    let sink = captured.clone();
    with_copilot_cassette(
        "raw_capture_matrix/chat_raw_exposes_system_fingerprint",
        |client| async move {
            let model = client.completion(CHAT_MODEL);
            let response = model
                .completion(request(&model))
                .await
                .expect("completion should succeed");
            let normalized = normalized_without_raw(response.clone());
            assert!(
                normalized.get("system_fingerprint").is_none(),
                "normalized CompletionResponse must not grow a `system_fingerprint` field"
            );
            let raw = response.raw;
            sink.put(raw);
        },
    )
    .await;

    let raw = captured.take();
    let (_, body) = recorded_json_turn(COPILOT_PROVIDER, scenario);
    assert_recorded_chat_body(&body, scenario);
    // `fp_…` fingerprints are placeholdered on disk like generated ids.
    assert_wire_value_matches(&raw, &body, "system_fingerprint");
    assert_eq!(raw["model"], body["model"]);
    let typed = openai::CompletionResponse::deserialize(&raw)
        .expect("raw must read back as the chat route's own response type");
    assert!(
        typed.system_fingerprint.is_some(),
        "the typed raw carries the fingerprint the wire sent"
    );
}

/// `raw` and the normalized response are two views of one reply, produced by
/// one decoder: so every normalized field must equal the provider-native field
/// it was mapped from — read off `raw` through the route's own response type,
/// and again off the recorded wire body.
#[tokio::test]
#[ignore = "unrecorded (no COPILOT credentials in this environment)"]
async fn chat_normalized_fields_equal_raw_renormalized() {
    let scenario = "raw_capture_matrix/chat_normalized_fields_equal_raw_renormalized";
    let captured = Observed::default();
    let sink = captured.clone();
    with_copilot_cassette(
        "raw_capture_matrix/chat_normalized_fields_equal_raw_renormalized",
        |client| async move {
            let model = client.completion(CHAT_MODEL);
            let response = model
                .completion(request(&model))
                .await
                .expect("completion should succeed");

            let reply = openai::CompletionResponse::deserialize(&response.raw)
                .expect("raw must read back as the chat route's own response type");
            let usage = reply.usage.as_ref().expect("the turn must report usage");
            let completion_tokens = usage
                .completion_tokens
                .expect("Copilot's chat route reports completion tokens");
            let choice = reply.choices.first().expect("the turn must carry a choice");

            assert_eq!(response.provider, COPILOT_PROVIDER);
            assert_eq!(
                response.model.as_deref(),
                Some(reply.model.as_str()),
                "model"
            );
            assert_eq!(
                response.response_id.as_deref(),
                Some(reply.id.as_str()),
                "response id"
            );
            assert_eq!(
                response.usage.input_tokens,
                Some(usage.prompt_tokens as u64),
                "input tokens"
            );
            assert_eq!(
                response.usage.output_tokens,
                Some(completion_tokens as u64),
                "output tokens"
            );
            assert_eq!(
                response.usage.total_tokens,
                Some(usage.total_tokens as u64),
                "total tokens"
            );
            assert_eq!(
                response.finish_reason(),
                Some(finish_reason_of(&choice.finish_reason)),
                "finish reason"
            );
            assert!(!response.choice.is_empty());

            sink.put(response);
        },
    )
    .await;

    let response = captured.take();
    let (_, body) = recorded_json_turn(COPILOT_PROVIDER, scenario);
    assert_recorded_chat_body(&body, scenario);
    assert_id_matches(
        response.response_id.as_deref(),
        body["id"].as_str(),
        "response id",
    );
    assert_eq!(response.model.as_deref(), body["model"].as_str(), "model");
    assert_eq!(
        response.usage.input_tokens,
        body["usage"]["prompt_tokens"].as_u64(),
        "input tokens"
    );
    assert_eq!(
        response.usage.output_tokens,
        body["usage"]["completion_tokens"].as_u64(),
        "output tokens"
    );
    assert_eq!(
        response.usage.total_tokens,
        body["usage"]["total_tokens"].as_u64(),
        "total tokens"
    );
    assert_eq!(
        assistant_text(&response.choice),
        body["choices"][0]["message"]["content"]
            .as_str()
            .expect("the recorded choice must carry text"),
        "choice text"
    );
    assert_eq!(
        response.finish_reason(),
        Some(finish_reason_of(
            body["choices"][0]["finish_reason"]
                .as_str()
                .expect("the recorded choice must carry a finish reason")
        )),
        "finish reason"
    );
}

// ===========================================================================
// Responses route
// ===========================================================================

#[tokio::test]
#[ignore = "unrecorded (no COPILOT credentials in this environment)"]
async fn responses_raw_round_trips_provider_type() {
    let scenario = "raw_capture_matrix/responses_raw_round_trips_provider_type";
    let captured = Observed::default();
    let sink = captured.clone();
    with_copilot_cassette(
        "raw_capture_matrix/responses_raw_round_trips_provider_type",
        |client| async move {
            let model = client.completion(RESPONSES_MODEL);
            assert!(
                matches!(model.wire.wire, OpenAiWire::Responses(_)),
                "premise: the codex model routes through the Responses API"
            );
            let response = model
                .completion(request(&model))
                .await
                .expect("completion should succeed");

            let raw = &response.raw;
            assert!(
                raw.get("api").is_none(),
                "raw is the route's own reply document, not a rig-tagged envelope"
            );
            let typed = responses_api::CompletionResponse::deserialize(raw)
                .expect("raw must read back as the Responses route's own response type");
            assert!(
                typed.provider_request_id.is_none(),
                "the transport request id is a reply header, so the document never carries it"
            );
            assert_eq!(response.provider, COPILOT_PROVIDER);
            assert!(!response.choice.is_empty());
            sink.put(response.raw);
        },
    )
    .await;

    let raw = captured.take();
    let (_, body) = recorded_json_turn(COPILOT_PROVIDER, scenario);
    assert_recorded_responses_body(&body, scenario);
    responses_api::CompletionResponse::deserialize(&body)
        .expect("recorded body must be a Responses envelope");
    assert_is_reply_document(&raw, &body, scenario);
}

#[tokio::test]
#[ignore = "unrecorded (no COPILOT credentials in this environment)"]
async fn responses_raw_exposes_envelope() {
    let scenario = "raw_capture_matrix/responses_raw_exposes_envelope";
    let captured = Observed::default();
    let sink = captured.clone();
    with_copilot_cassette(
        "raw_capture_matrix/responses_raw_exposes_envelope",
        |client| async move {
            let model = client.completion(RESPONSES_MODEL);
            let response = model
                .completion(request(&model))
                .await
                .expect("completion should succeed");
            let normalized = normalized_without_raw(response.clone());
            for field in ["object", "status", "created_at"] {
                assert!(
                    normalized.get(field).is_none(),
                    "normalized CompletionResponse must not grow a `{field}` field"
                );
            }
            let raw = response.raw;
            sink.put(raw);
        },
    )
    .await;

    let raw = captured.take();
    let (_, body) = recorded_json_turn(COPILOT_PROVIDER, scenario);
    assert_recorded_responses_body(&body, scenario);
    for field in ["object", "status", "model"] {
        assert_eq!(
            raw.get(field),
            body.get(field),
            "raw.{field} must equal the recorded wire value"
        );
    }
    assert_wire_value_matches(&raw, &body, "created_at");
    let typed = responses_api::CompletionResponse::deserialize(&raw)
        .expect("raw must read back as the Responses route's own response type");
    assert_eq!(typed.status, responses_api::ResponseStatus::Completed);
}

/// The Responses-route twin of `chat_normalized_fields_equal_raw_renormalized`.
#[tokio::test]
#[ignore = "unrecorded (no COPILOT credentials in this environment)"]
async fn responses_normalized_fields_equal_raw_renormalized() {
    let scenario = "raw_capture_matrix/responses_normalized_fields_equal_raw_renormalized";
    let captured = Observed::default();
    let sink = captured.clone();
    with_copilot_cassette(
        "raw_capture_matrix/responses_normalized_fields_equal_raw_renormalized",
        |client| async move {
            let model = client.completion(RESPONSES_MODEL);
            let response = model
                .completion(request(&model))
                .await
                .expect("completion should succeed");

            let reply = responses_api::CompletionResponse::deserialize(&response.raw)
                .expect("raw must read back as the Responses route's own response type");
            let usage = reply.usage.as_ref().expect("the turn must report usage");

            assert_eq!(response.provider, COPILOT_PROVIDER);
            assert_eq!(
                response.model.as_deref(),
                Some(reply.model.as_str()),
                "model"
            );
            assert_eq!(
                response.response_id.as_deref(),
                Some(reply.id.as_str()),
                "response id"
            );
            assert_eq!(
                response.usage.input_tokens,
                Some(usage.input_tokens),
                "input tokens"
            );
            assert_eq!(
                response.usage.output_tokens,
                Some(usage.output_tokens),
                "output tokens"
            );
            assert_eq!(
                response.usage.total_tokens,
                Some(usage.total_tokens),
                "total tokens"
            );
            assert_eq!(
                reply.status,
                responses_api::ResponseStatus::Completed,
                "the turn must be completed"
            );
            assert_eq!(
                response.finish_reason(),
                Some(FinishReason::Stop),
                "a completed Responses turn stops"
            );
            assert!(!response.choice.is_empty());

            sink.put(response);
        },
    )
    .await;

    let response = captured.take();
    let (_, body) = recorded_json_turn(COPILOT_PROVIDER, scenario);
    assert_recorded_responses_body(&body, scenario);
    assert_id_matches(
        response.response_id.as_deref(),
        body["id"].as_str(),
        "response id",
    );
    assert_eq!(response.model.as_deref(), body["model"].as_str(), "model");
    assert_eq!(
        response.usage.input_tokens,
        body["usage"]["input_tokens"].as_u64(),
        "input tokens"
    );
    assert_eq!(
        response.usage.output_tokens,
        body["usage"]["output_tokens"].as_u64(),
        "output tokens"
    );
    assert_eq!(
        response.usage.total_tokens,
        body["usage"]["total_tokens"].as_u64(),
        "total tokens"
    );
}
