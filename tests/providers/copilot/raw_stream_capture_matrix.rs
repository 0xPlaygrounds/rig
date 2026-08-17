//! Matrix for opt-in raw terminal-record capture on both Copilot streaming
//! routes (`CompletionRequest::capture_raw_response` → `StreamFinal::raw`).
//!
//! # The feature
//!
//! `StreamFinal::raw` is the value
//! [`CompletionModel::raw_stream`](rig::providers::copilot::CompletionModel::raw_stream)
//! would have yielded as its `FinalResponse` — the route-tagged
//! [`CopilotStreamingResponse`](rig::providers::copilot::CopilotStreamingResponse)
//! (`{"api":"chat", …}` wrapping the chat-completions terminal record,
//! `{"api":"responses", …}` wrapping the Responses one) — serialized with
//! `serde_json::to_value`. It is the terminal record only, populated only when
//! the request opted in, and never on the wire.
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
//! | 1 | `chat_stream_capture_off_raw_is_none` | chat route, flag off | terminal `raw == None` | unrecorded (no COPILOT credentials in this environment) |
//! | 2 | `chat_stream_capture_on_terminal_round_trips_provider_type` | chat route, flag on | `CopilotStreamingResponse::deserialize(&*raw)` is `Chat(_)` and re-serializes equal | unrecorded (no COPILOT credentials in this environment) |
//! | 3 | `chat_stream_capture_on_exposes_copilot_usage` | chat route, terminal-only fields | `raw.additional_params.copilot_usage` equals the terminal frame's; usage equals the frame's | unrecorded (no COPILOT credentials in this environment) |
//! | 4 | `chat_stream_request_invariant_off_vs_on` | chat route, on-wire request | flag-off and flag-on request bodies byte-identical | unrecorded (no COPILOT credentials in this environment) |
//! | 5 | `responses_stream_capture_off_raw_is_none` | responses route, flag off | terminal `raw == None` | unrecorded (no COPILOT credentials in this environment) |
//! | 6 | `responses_stream_capture_on_terminal_round_trips_provider_type` | responses route, flag on | `CopilotStreamingResponse::deserialize(&*raw)` is `Responses(_)` and re-serializes equal | unrecorded (no COPILOT credentials in this environment) |
//! | 7 | `responses_stream_capture_on_exposes_terminal_status` | responses route, terminal-only field | `raw.status == "completed"` as the recorded `response.completed` frame says | unrecorded (no COPILOT credentials in this environment) |
//! | 8 | `responses_stream_request_invariant_off_vs_on` | responses route, on-wire request | flag-off and flag-on request bodies byte-identical | unrecorded (no COPILOT credentials in this environment) |
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

fn request(
    model: &copilot::CompletionModel,
    capture_raw: bool,
) -> rig::completion::CompletionRequest {
    model
        .completion_request(PROMPT)
        .max_tokens(64)
        .capture_raw_response(capture_raw)
        .build()
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

fn recorded_request_bodies(scenario: &str) -> Vec<String> {
    recorded_interaction_bodies(COPILOT_PROVIDER, scenario)
        .into_iter()
        .map(|(request, _)| request)
        .collect()
}

fn assert_two_identical_requests(scenario: &str, model: &str) {
    let requests = recorded_request_bodies(scenario);
    assert_eq!(
        requests.len(),
        2,
        "{scenario}: expected the off and on requests"
    );
    assert_eq!(
        requests[0], requests[1],
        "the flag-on streaming request body must be byte-identical to the \
         flag-off one — capture_raw_response must never reach Copilot"
    );
    assert!(!requests[0].contains("capture_raw"));
    let body: Value = serde_json::from_str(&requests[0]).expect("recorded request should be JSON");
    assert_eq!(body["stream"], Value::Bool(true));
    assert_eq!(body["model"], model);
}

// ===========================================================================
// Chat-completions route
// ===========================================================================

#[tokio::test]
#[ignore = "unrecorded (no COPILOT credentials in this environment)"]
async fn chat_stream_capture_off_raw_is_none() {
    let scenario = "raw_stream_capture_matrix/chat_stream_capture_off_raw_is_none";
    with_copilot_cassette(
        "raw_stream_capture_matrix/chat_stream_capture_off_raw_is_none",
        |client| async move {
            let model = client.completion_model(CHAT_MODEL);
            let request = request(&model, false);
            assert!(!request.capture_raw_response, "premise: default is off");
            let terminal =
                terminal_of(model.stream(request).await.expect("stream should start")).await;
            assert!(
                terminal.raw.is_none(),
                "terminal raw must stay None when capture was not requested, got {:?}",
                terminal.raw
            );
            assert!(terminal.usage.total_tokens > 0, "usage is unaffected");
            assert_eq!(terminal.provider, COPILOT_PROVIDER);
        },
    )
    .await;

    recorded_chat_frames(scenario);
}

#[tokio::test]
#[ignore = "unrecorded (no COPILOT credentials in this environment)"]
async fn chat_stream_capture_on_terminal_round_trips_provider_type() {
    let scenario =
        "raw_stream_capture_matrix/chat_stream_capture_on_terminal_round_trips_provider_type";
    let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = std::sync::Arc::clone(&captured);
    with_copilot_cassette(
        "raw_stream_capture_matrix/chat_stream_capture_on_terminal_round_trips_provider_type",
        |client| async move {
            let model = client.completion_model(CHAT_MODEL);
            let terminal = terminal_of(
                model
                    .stream(request(&model, true))
                    .await
                    .expect("stream should start"),
            )
            .await;
            let raw = terminal
                .raw
                .as_deref()
                .expect("terminal raw must be populated when capture was requested");
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
async fn chat_stream_capture_on_exposes_copilot_usage() {
    let scenario = "raw_stream_capture_matrix/chat_stream_capture_on_exposes_copilot_usage";
    let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = std::sync::Arc::clone(&captured);
    with_copilot_cassette(
        "raw_stream_capture_matrix/chat_stream_capture_on_exposes_copilot_usage",
        |client| async move {
            let model = client.completion_model(CHAT_MODEL);
            let terminal = terminal_of(
                model
                    .stream(request(&model, true))
                    .await
                    .expect("stream should start"),
            )
            .await;
            let mut without_raw = terminal.clone();
            without_raw.raw = None;
            let normalized =
                serde_json::to_value(&without_raw).expect("StreamFinal should serialize");
            for field in ["copilot_usage", "system_fingerprint", "additional_params"] {
                assert!(
                    normalized.get(field).is_none(),
                    "normalized StreamFinal must not grow a `{field}` field"
                );
            }
            let raw = terminal
                .raw
                .as_deref()
                .expect("terminal raw must be populated when capture was requested")
                .clone();
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

/// One scenario, two interactions in wire order — off then on.
#[tokio::test]
#[ignore = "unrecorded (no COPILOT credentials in this environment)"]
async fn chat_stream_request_invariant_off_vs_on() {
    let scenario = "raw_stream_capture_matrix/chat_stream_request_invariant_off_vs_on";
    with_copilot_cassette(
        "raw_stream_capture_matrix/chat_stream_request_invariant_off_vs_on",
        |client| async move {
            let model = client.completion_model(CHAT_MODEL);
            let off = terminal_of(
                model
                    .stream(request(&model, false))
                    .await
                    .expect("flag-off stream should start"),
            )
            .await;
            let on = terminal_of(
                model
                    .stream(request(&model, true))
                    .await
                    .expect("flag-on stream should start"),
            )
            .await;
            assert!(off.raw.is_none());
            assert!(on.raw.is_some());
        },
    )
    .await;

    assert_two_identical_requests(scenario, CHAT_MODEL);
}

// ===========================================================================
// Responses route
// ===========================================================================

#[tokio::test]
#[ignore = "unrecorded (no COPILOT credentials in this environment)"]
async fn responses_stream_capture_off_raw_is_none() {
    let scenario = "raw_stream_capture_matrix/responses_stream_capture_off_raw_is_none";
    with_copilot_cassette(
        "raw_stream_capture_matrix/responses_stream_capture_off_raw_is_none",
        |client| async move {
            let model = client.completion_model(RESPONSES_MODEL);
            let request = request(&model, false);
            assert!(!request.capture_raw_response, "premise: default is off");
            let terminal =
                terminal_of(model.stream(request).await.expect("stream should start")).await;
            assert!(
                terminal.raw.is_none(),
                "terminal raw must stay None when capture was not requested, got {:?}",
                terminal.raw
            );
            assert!(terminal.usage.total_tokens > 0, "usage is unaffected");
        },
    )
    .await;

    recorded_responses_terminal(scenario);
}

#[tokio::test]
#[ignore = "unrecorded (no COPILOT credentials in this environment)"]
async fn responses_stream_capture_on_terminal_round_trips_provider_type() {
    let scenario =
        "raw_stream_capture_matrix/responses_stream_capture_on_terminal_round_trips_provider_type";
    let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = std::sync::Arc::clone(&captured);
    with_copilot_cassette(
        "raw_stream_capture_matrix/responses_stream_capture_on_terminal_round_trips_provider_type",
        |client| async move {
            let model = client.completion_model(RESPONSES_MODEL);
            let terminal = terminal_of(
                model
                    .stream(request(&model, true))
                    .await
                    .expect("stream should start"),
            )
            .await;
            let raw = terminal
                .raw
                .as_deref()
                .expect("terminal raw must be populated when capture was requested");
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
async fn responses_stream_capture_on_exposes_terminal_status() {
    let scenario = "raw_stream_capture_matrix/responses_stream_capture_on_exposes_terminal_status";
    let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = std::sync::Arc::clone(&captured);
    with_copilot_cassette(
        "raw_stream_capture_matrix/responses_stream_capture_on_exposes_terminal_status",
        |client| async move {
            let model = client.completion_model(RESPONSES_MODEL);
            let terminal = terminal_of(
                model
                    .stream(request(&model, true))
                    .await
                    .expect("stream should start"),
            )
            .await;
            let mut without_raw = terminal.clone();
            without_raw.raw = None;
            let normalized =
                serde_json::to_value(&without_raw).expect("StreamFinal should serialize");
            assert!(normalized.get("status").is_none());
            let raw = terminal
                .raw
                .as_deref()
                .expect("terminal raw must be populated when capture was requested")
                .clone();
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

/// One scenario, two interactions in wire order — off then on.
#[tokio::test]
#[ignore = "unrecorded (no COPILOT credentials in this environment)"]
async fn responses_stream_request_invariant_off_vs_on() {
    let scenario = "raw_stream_capture_matrix/responses_stream_request_invariant_off_vs_on";
    with_copilot_cassette(
        "raw_stream_capture_matrix/responses_stream_request_invariant_off_vs_on",
        |client| async move {
            let model = client.completion_model(RESPONSES_MODEL);
            let off = terminal_of(
                model
                    .stream(request(&model, false))
                    .await
                    .expect("flag-off stream should start"),
            )
            .await;
            let on = terminal_of(
                model
                    .stream(request(&model, true))
                    .await
                    .expect("flag-on stream should start"),
            )
            .await;
            assert!(off.raw.is_none());
            assert!(on.raw.is_some());
        },
    )
    .await;

    assert_two_identical_requests(scenario, RESPONSES_MODEL);
}
