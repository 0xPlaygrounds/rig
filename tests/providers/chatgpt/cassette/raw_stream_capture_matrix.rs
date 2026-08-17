//! Matrix for opt-in raw terminal-record capture on ChatGPT's streaming
//! `/responses` path (`CompletionRequest::capture_raw_response` →
//! `StreamFinal::raw`).
//!
//! # The feature
//!
//! `StreamFinal::raw` is the value
//! [`ResponsesCompletionModel::raw_stream`](rig::providers::chatgpt::ResponsesCompletionModel::raw_stream)
//! would have yielded as its `FinalResponse` — the Responses API's
//! [`StreamingCompletionResponse`](rig::providers::openai::responses_api::streaming::StreamingCompletionResponse):
//! the terminal `response.completed` event's usage, status, ids and model —
//! serialized with `serde_json::to_value`. It is the terminal record only,
//! populated only when the request opted in, and never on the wire.
//!
//! The terminal record spells the provider's `status` (`completed`), which
//! the normalized [`StreamFinal`](rig::streaming::StreamFinal) folds into a
//! finish reason and does not carry; cell 3 reads it back through `raw`.
//!
//! # Matrix
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `stream_capture_off_raw_is_none` | flag off (default) | terminal `raw == None` | unrecorded (no CHATGPT credentials in this environment) |
//! | 2 | `stream_capture_on_terminal_round_trips_provider_type` | flag on | `responses_api::streaming::StreamingCompletionResponse::deserialize(&*raw)` re-serializes equal | unrecorded (no CHATGPT credentials in this environment) |
//! | 3 | `stream_capture_on_exposes_terminal_status` | terminal-only field | `raw.status == "completed"` as the recorded `response.completed` frame says; usage equals the frame's | unrecorded (no CHATGPT credentials in this environment) |
//! | 4 | `stream_request_invariant_off_vs_on` | on-wire request | flag-off and flag-on request bodies byte-identical | unrecorded (no CHATGPT credentials in this environment) |
//!
//! Every cell is unrecorded: neither `CHATGPT_ACCESS_TOKEN`/`CHATGPT_ACCOUNT_ID`
//! nor a usable ChatGPT OAuth cache was present when this matrix was written,
//! and a fixture is never fabricated. To record: export the two variables,
//! remove the `#[ignore]` attributes, flip the table to `recorded`, then run
//! `RIG_PROVIDER_TEST_MODE=record cargo test -p rig --all-features --test chatgpt chatgpt::cassette::raw_stream_capture_matrix -- --nocapture --test-threads=1`
//! and review `tests/cassettes/chatgpt/raw_stream_capture_matrix/`.

use futures::StreamExt;
use rig::completion::CompletionModel as _;
use rig::prelude::*;
use rig::providers::chatgpt;
use rig::providers::openai::responses_api;
use rig::streaming::{StreamFinal, StreamedAssistantContent};
use serde::Deserialize;
use serde_json::Value;

use super::super::support::with_chatgpt_cassette;
use crate::cassettes::{recorded_interaction_bodies, recorded_sse_json_frames};

const CHATGPT_PROVIDER: &str = "chatgpt";
const MODEL: &str = chatgpt::GPT_5_4;
const PROMPT: &str = "Reply with exactly the single word: pong";

fn request(
    model: &chatgpt::ResponsesCompletionModel,
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

/// The premise every streaming cell rests on: the recorded SSE stream ends
/// with a `response.completed` frame carrying usage. Returns its `response`.
fn recorded_terminal_response(scenario: &str) -> Value {
    let frames = recorded_sse_json_frames(CHATGPT_PROVIDER, scenario);
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
        "{scenario}: the terminal frame must report usage — without it the \
         terminal record carries no usage and this cell proves nothing"
    );
    response
}

fn recorded_request_bodies(scenario: &str) -> Vec<String> {
    recorded_interaction_bodies(CHATGPT_PROVIDER, scenario)
        .into_iter()
        .map(|(request, _)| request)
        .collect()
}

// ---------------------------------------------------------------------------
// 1: off → None
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "unrecorded (no CHATGPT credentials in this environment)"]
async fn stream_capture_off_raw_is_none() {
    let scenario = "raw_stream_capture_matrix/stream_capture_off_raw_is_none";
    with_chatgpt_cassette(
        "raw_stream_capture_matrix/stream_capture_off_raw_is_none",
        |client| async move {
            let model = client.completion_model(MODEL);
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
            assert_eq!(terminal.provider, CHATGPT_PROVIDER);
        },
    )
    .await;

    recorded_terminal_response(scenario);
}

// ---------------------------------------------------------------------------
// 2: on → raw is the raw_stream FinalResponse, serialized
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "unrecorded (no CHATGPT credentials in this environment)"]
async fn stream_capture_on_terminal_round_trips_provider_type() {
    let scenario = "raw_stream_capture_matrix/stream_capture_on_terminal_round_trips_provider_type";
    let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = std::sync::Arc::clone(&captured);
    with_chatgpt_cassette(
        "raw_stream_capture_matrix/stream_capture_on_terminal_round_trips_provider_type",
        |client| async move {
            let model = client.completion_model(MODEL);
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
            let typed = responses_api::streaming::StreamingCompletionResponse::deserialize(raw)
                .expect("raw must deserialize into the Responses terminal type");
            assert_eq!(
                serde_json::to_value(&typed).expect("terminal type should serialize"),
                *raw,
                "responses_api::streaming::StreamingCompletionResponse must round-trip"
            );

            assert_eq!(typed.usage.total_tokens, terminal.usage.total_tokens);
            assert_eq!(typed.usage.input_tokens, terminal.usage.input_tokens);
            assert_eq!(typed.usage.output_tokens, terminal.usage.output_tokens);
            assert_eq!(typed.response_id, terminal.response_id);
            assert_eq!(typed.message_id, terminal.message_id);
            assert_eq!(typed.model, terminal.model);
            *sink.lock().expect("capture mutex") = Some(raw.clone());
        },
    )
    .await;

    let terminal = recorded_terminal_response(scenario);
    let raw = captured
        .lock()
        .expect("capture mutex")
        .take()
        .expect("the test body must have captured raw");
    assert_eq!(
        raw["usage"]["total_tokens"], terminal["usage"]["total_tokens"],
        "raw usage must be the terminal frame's usage"
    );
}

// ---------------------------------------------------------------------------
// 3: terminal-only field
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "unrecorded (no CHATGPT credentials in this environment)"]
async fn stream_capture_on_exposes_terminal_status() {
    let scenario = "raw_stream_capture_matrix/stream_capture_on_exposes_terminal_status";
    let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = std::sync::Arc::clone(&captured);
    with_chatgpt_cassette(
        "raw_stream_capture_matrix/stream_capture_on_exposes_terminal_status",
        |client| async move {
            let model = client.completion_model(MODEL);
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
            assert!(
                normalized.get("status").is_none(),
                "normalized StreamFinal must not grow a `status` field"
            );

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
    let terminal = recorded_terminal_response(scenario);
    assert_eq!(
        terminal["status"],
        Value::String("completed".to_string()),
        "{scenario}: premise — the recorded terminal frame is completed"
    );
    assert_eq!(raw["status"], terminal["status"]);
    assert_eq!(raw["usage"], terminal["usage"]);
    let typed = responses_api::streaming::StreamingCompletionResponse::deserialize(&raw)
        .expect("raw must deserialize");
    assert_eq!(typed.status, Some(responses_api::ResponseStatus::Completed));
}

// ---------------------------------------------------------------------------
// 4: the flag never reaches the provider
// ---------------------------------------------------------------------------

/// One scenario, two interactions in wire order — off then on.
#[tokio::test]
#[ignore = "unrecorded (no CHATGPT credentials in this environment)"]
async fn stream_request_invariant_off_vs_on() {
    let scenario = "raw_stream_capture_matrix/stream_request_invariant_off_vs_on";
    with_chatgpt_cassette(
        "raw_stream_capture_matrix/stream_request_invariant_off_vs_on",
        |client| async move {
            let model = client.completion_model(MODEL);
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
            assert_eq!(off.provider, on.provider);
        },
    )
    .await;

    let requests = recorded_request_bodies(scenario);
    assert_eq!(
        requests.len(),
        2,
        "{scenario}: the scenario must record exactly the off and on requests"
    );
    assert_eq!(
        requests[0], requests[1],
        "the flag-on streaming request body must be byte-identical to the \
         flag-off one — capture_raw_response must never reach ChatGPT"
    );
    assert!(!requests[0].contains("capture_raw"));
    let body: Value = serde_json::from_str(&requests[0]).expect("recorded request should be JSON");
    assert_eq!(body["stream"], Value::Bool(true));
    assert_eq!(body["model"], MODEL);
}
