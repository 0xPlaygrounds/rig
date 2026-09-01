//! Matrix for raw terminal-record capture on both Copilot streaming routes
//! ([`StreamFinal::raw`](rig::streaming::StreamFinal::raw)).
//!
//! # The feature
//!
//! Capture is always on. The terminal record of every stream the seam yields
//! carries `raw`: the value
//! [`CompletionModel::raw_stream`](rig::providers::copilot::CompletionModel::raw_stream)
//! would have yielded as its `FinalResponse` — the route-tagged
//! [`CopilotStreamingResponse`](rig::providers::copilot::CopilotStreamingResponse)
//! (`{"api":"chat", …}` wrapping the chat-completions terminal record,
//! `{"api":"responses", …}` wrapping the Responses one) — serialized with
//! `serde_json::to_value`. It is the terminal record only, and nothing about it
//! is sent to Copilot. `raw == Value::Null` means only that a `StreamFinal` was
//! built by hand without a provider terminal behind it, which no cell here can
//! produce.
//!
//! Terminal-only fields per route: on the chat route the shared terminal type
//! accumulates unknown top-level chunk fields under `additional_params`, which
//! is where Copilot's own `copilot_usage` block (with `total_nano_aiu`) and
//! the `system_fingerprint` land — neither has a home on the normalized
//! [`StreamFinal`](rig::streaming::StreamFinal); on the Responses route the
//! terminal `status`.
//!
//! # Matrix
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `chat_stream_raw_terminal_round_trips_provider_type` | chat route, typed access | `CopilotStreamingResponse::deserialize(&*raw)` is `Chat(_)` and re-serializes equal | unrecorded (no COPILOT credentials in this environment) |
//! | 2 | `chat_stream_raw_exposes_copilot_usage` | chat route, terminal-only fields | `raw.additional_params.copilot_usage` equals the terminal frame's; usage equals the frame's | unrecorded (no COPILOT credentials in this environment) |
//! | 3 | `responses_stream_raw_terminal_round_trips_provider_type` | responses route, typed access | `CopilotStreamingResponse::deserialize(&*raw)` is `Responses(_)` and re-serializes equal | unrecorded (no COPILOT credentials in this environment) |
//! | 4 | `responses_stream_raw_exposes_terminal_status` | responses route, terminal-only field | `raw.status == "completed"` as the recorded `response.completed` frame says | unrecorded (no COPILOT credentials in this environment) |
//!
//! Every cell is unrecorded: none of `GITHUB_COPILOT_API_KEY`,
//! `COPILOT_API_KEY`, `COPILOT_GITHUB_ACCESS_TOKEN`/`GITHUB_TOKEN` nor a Copilot
//! OAuth cache was present when this matrix was written, and a fixture is
//! never fabricated. To record: export `GITHUB_COPILOT_API_KEY`, remove the
//! `#[ignore]` attributes, flip the table to `recorded`, then run
//! `RIG_PROVIDER_TEST_MODE=record cargo test -p rig --all-features --test copilot copilot::raw_stream_capture_matrix -- --nocapture --test-threads=1`
//! and review `tests/cassettes/copilot/raw_stream_capture_matrix/`.

use futures::StreamExt;
use rig::completion::CompletionModel as _;
use rig::prelude::*;
use rig::providers::copilot::{self, CopilotStreamingResponse};
use rig::providers::openai::responses_api;
use rig::streaming::{StreamFinal, StreamedAssistantContent};
use serde::Deserialize;
use serde_json::Value;

use crate::cassettes::{CassetteMode, recorded_interaction_bodies, recorded_sse_json_frames};
use crate::copilot::with_copilot_cassette;

const COPILOT_PROVIDER: &str = "copilot";
const CHAT_MODEL: &str = copilot::GPT_4O;
const RESPONSES_MODEL: &str = copilot::GPT_5_3_CODEX;
const PROMPT: &str = "Reply with exactly the single word: pong";

fn request(model: &copilot::CompletionModel) -> rig::completion::CompletionRequest {
    model.completion_request(PROMPT).max_tokens(64).build()
}

/// Every cell records exactly one interaction; a scenario with more has
/// drifted from the matrix's premise.
fn assert_single_interaction(scenario: &str) {
    assert_eq!(
        recorded_interaction_bodies(COPILOT_PROVIDER, scenario).len(),
        1,
        "{scenario}: the scenario must record exactly one interaction"
    );
}

async fn terminal_of(mut stream: rig::streaming::StreamingCompletionResponse) -> StreamFinal {
    let mut finals = Vec::new();
    while let Some(item) = stream.next().await {
        if let StreamedAssistantContent::Final(record) = item.expect("stream item should be ok") {
            finals.push(record);
        }
    }
    assert_eq!(
        finals.len(),
        1,
        "stream should yield exactly one terminal record"
    );
    finals.remove(0)
}

/// Chat-route premise: the recorded SSE stream's last frame carries `usage`
/// (and Copilot's `copilot_usage`). Returns `(all frames, terminal frame)`.
fn recorded_chat_frames(scenario: &str) -> (Vec<Value>, Value) {
    assert_single_interaction(scenario);
    let frames = recorded_sse_json_frames(COPILOT_PROVIDER, scenario);
    let terminal = frames
        .iter()
        .rev()
        .find(|frame| frame.get("usage").is_some_and(Value::is_object))
        .cloned()
        .unwrap_or_else(|| {
            panic!("{scenario}: the recorded stream must carry a usage-bearing frame")
        });
    assert!(
        terminal.get("copilot_usage").is_some_and(Value::is_object),
        "{scenario}: the usage frame must carry Copilot's `copilot_usage` block — \
         without it this cell cannot prove raw exposes a provider-only field"
    );
    (frames, terminal)
}

/// Responses-route premise: the recorded SSE stream ends with a
/// `response.completed` frame carrying usage. Returns its `response`.
fn recorded_responses_terminal(scenario: &str) -> Value {
    assert_single_interaction(scenario);
    let frames = recorded_sse_json_frames(COPILOT_PROVIDER, scenario);
    let terminal = frames
        .iter()
        .rev()
        .find(|frame| frame.get("type").and_then(Value::as_str) == Some("response.completed"))
        .unwrap_or_else(|| {
            panic!("{scenario}: the recorded stream must end with response.completed")
        });
    let response = terminal["response"].clone();
    assert!(
        response.pointer("/usage/total_tokens").is_some(),
        "{scenario}: the terminal frame must report usage"
    );
    response
}

// ===========================================================================
// Chat-completions route
// ===========================================================================

#[tokio::test]
#[ignore = "unrecorded (no COPILOT credentials in this environment)"]
async fn chat_stream_raw_terminal_round_trips_provider_type() {
    let scenario = "raw_stream_capture_matrix/chat_stream_raw_terminal_round_trips_provider_type";
    let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = std::sync::Arc::clone(&captured);
    with_copilot_cassette(
        "raw_stream_capture_matrix/chat_stream_raw_terminal_round_trips_provider_type",
        |client| async move {
            let model = client.completion_model(CHAT_MODEL);
            let terminal = terminal_of(
                model
                    .stream(request(&model))
                    .await
                    .expect("stream should start"),
            )
            .await;
            let raw = &terminal.raw;
            assert_eq!(raw["api"], "chat", "the route tag rides along on raw");
            let typed = CopilotStreamingResponse::deserialize(raw)
                .expect("raw must deserialize into CopilotStreamingResponse");
            let CopilotStreamingResponse::Chat(chat) = &typed else {
                panic!("chat-route raw must read back as the Chat variant");
            };
            assert_eq!(
                serde_json::to_value(&typed).expect("terminal type should serialize"),
                *raw,
                "CopilotStreamingResponse must round-trip through its own serde"
            );
            assert_eq!(chat.usage.prompt_tokens as u64, terminal.usage.input_tokens);
            assert_eq!(
                chat.usage.completion_tokens.map(|tokens| tokens as u64),
                Some(terminal.usage.output_tokens)
            );
            assert_eq!(chat.provider_request_id, terminal.provider_request_id);
            *sink.lock().expect("capture mutex") = Some(raw.clone());
        },
    )
    .await;

    let (_, terminal_frame) = recorded_chat_frames(scenario);
    let raw = captured
        .lock()
        .expect("capture mutex")
        .take()
        .expect("the test body must have captured raw");
    assert_eq!(
        raw["usage"]["prompt_tokens"], terminal_frame["usage"]["prompt_tokens"],
        "raw usage must be the terminal frame's usage"
    );
}

#[tokio::test]
#[ignore = "unrecorded (no COPILOT credentials in this environment)"]
async fn chat_stream_raw_exposes_copilot_usage() {
    let scenario = "raw_stream_capture_matrix/chat_stream_raw_exposes_copilot_usage";
    let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = std::sync::Arc::clone(&captured);
    with_copilot_cassette(
        "raw_stream_capture_matrix/chat_stream_raw_exposes_copilot_usage",
        |client| async move {
            let model = client.completion_model(CHAT_MODEL);
            let terminal = terminal_of(
                model
                    .stream(request(&model))
                    .await
                    .expect("stream should start"),
            )
            .await;
            let mut without_raw = terminal.clone();
            without_raw.raw = Value::Null;
            let normalized =
                serde_json::to_value(&without_raw).expect("StreamFinal should serialize");
            for field in ["copilot_usage", "system_fingerprint", "additional_params"] {
                assert!(
                    normalized.get(field).is_none(),
                    "normalized StreamFinal must not grow a `{field}` field"
                );
            }
            let raw = terminal.raw;
            *sink.lock().expect("capture mutex") = Some(raw);
        },
    )
    .await;

    let raw = captured
        .lock()
        .expect("capture mutex")
        .take()
        .expect("the test body must have captured raw");
    let (frames, terminal_frame) = recorded_chat_frames(scenario);
    let params = raw
        .get("additional_params")
        .expect("raw terminal must carry the accumulated chunk envelope under additional_params");
    assert_eq!(
        params.get("copilot_usage"),
        terminal_frame.get("copilot_usage"),
        "raw.additional_params.copilot_usage must equal the recorded terminal frame's block"
    );
    assert_eq!(raw["usage"], terminal_frame["usage"]);
    // `fp_…` fingerprints are placeholdered on disk; only a replay compares
    // them exactly, a live recording checks the field is there.
    let recorded_fingerprint = frames
        .iter()
        .find_map(|frame| frame.get("system_fingerprint"))
        .cloned()
        .unwrap_or_else(|| panic!("{scenario}: recorded chunks must carry system_fingerprint"));
    match CassetteMode::current() {
        CassetteMode::Replay => {
            assert_eq!(
                params.get("system_fingerprint"),
                Some(&recorded_fingerprint)
            );
        }
        CassetteMode::Record => assert!(
            params
                .get("system_fingerprint")
                .is_some_and(Value::is_string),
            "raw.additional_params.system_fingerprint must carry the chunk fingerprint"
        ),
    }
    let CopilotStreamingResponse::Chat(typed) =
        CopilotStreamingResponse::deserialize(&raw).expect("raw must deserialize")
    else {
        panic!("chat-route raw must read back as the Chat variant");
    };
    let typed_params = typed
        .additional_params
        .expect("typed terminal must carry additional_params");
    assert_eq!(
        typed_params
            .get("copilot_usage")
            .and_then(|usage| usage.get("total_nano_aiu")),
        terminal_frame.pointer("/copilot_usage/total_nano_aiu")
    );
}

// ===========================================================================
// Responses route
// ===========================================================================

#[tokio::test]
#[ignore = "unrecorded (no COPILOT credentials in this environment)"]
async fn responses_stream_raw_terminal_round_trips_provider_type() {
    let scenario =
        "raw_stream_capture_matrix/responses_stream_raw_terminal_round_trips_provider_type";
    let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = std::sync::Arc::clone(&captured);
    with_copilot_cassette(
        "raw_stream_capture_matrix/responses_stream_raw_terminal_round_trips_provider_type",
        |client| async move {
            let model = client.completion_model(RESPONSES_MODEL);
            let terminal = terminal_of(
                model
                    .stream(request(&model))
                    .await
                    .expect("stream should start"),
            )
            .await;
            let raw = &terminal.raw;
            assert_eq!(raw["api"], "responses", "the route tag rides along on raw");
            let typed = CopilotStreamingResponse::deserialize(raw)
                .expect("raw must deserialize into CopilotStreamingResponse");
            let CopilotStreamingResponse::Responses(responses) = &typed else {
                panic!("responses-route raw must read back as the Responses variant");
            };
            assert_eq!(
                serde_json::to_value(&typed).expect("terminal type should serialize"),
                *raw,
                "CopilotStreamingResponse must round-trip through its own serde"
            );
            assert_eq!(responses.usage.total_tokens, terminal.usage.total_tokens);
            assert_eq!(responses.response_id, terminal.response_id);
            assert_eq!(responses.message_id, terminal.message_id);
            assert_eq!(responses.provider_request_id, terminal.provider_request_id);
            *sink.lock().expect("capture mutex") = Some(raw.clone());
        },
    )
    .await;

    let terminal = recorded_responses_terminal(scenario);
    let raw = captured
        .lock()
        .expect("capture mutex")
        .take()
        .expect("the test body must have captured raw");
    assert_eq!(
        raw["usage"]["total_tokens"],
        terminal["usage"]["total_tokens"]
    );
}

#[tokio::test]
#[ignore = "unrecorded (no COPILOT credentials in this environment)"]
async fn responses_stream_raw_exposes_terminal_status() {
    let scenario = "raw_stream_capture_matrix/responses_stream_raw_exposes_terminal_status";
    let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = std::sync::Arc::clone(&captured);
    with_copilot_cassette(
        "raw_stream_capture_matrix/responses_stream_raw_exposes_terminal_status",
        |client| async move {
            let model = client.completion_model(RESPONSES_MODEL);
            let terminal = terminal_of(
                model
                    .stream(request(&model))
                    .await
                    .expect("stream should start"),
            )
            .await;
            let mut without_raw = terminal.clone();
            without_raw.raw = Value::Null;
            let normalized =
                serde_json::to_value(&without_raw).expect("StreamFinal should serialize");
            assert!(normalized.get("status").is_none());
            let raw = terminal.raw;
            *sink.lock().expect("capture mutex") = Some(raw);
        },
    )
    .await;

    let raw = captured
        .lock()
        .expect("capture mutex")
        .take()
        .expect("the test body must have captured raw");
    let terminal = recorded_responses_terminal(scenario);
    assert_eq!(terminal["status"], Value::String("completed".to_string()));
    assert_eq!(raw["status"], terminal["status"]);
    assert_eq!(raw["usage"], terminal["usage"]);
    let CopilotStreamingResponse::Responses(typed) =
        CopilotStreamingResponse::deserialize(&raw).expect("raw must deserialize")
    else {
        panic!("responses-route raw must read back as the Responses variant");
    };
    assert_eq!(typed.status, Some(responses_api::ResponseStatus::Completed));
}
