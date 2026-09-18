//! Matrix for raw terminal-record capture on ChatGPT's streaming `/responses`
//! path ([`StreamFinal::raw`](rig::streaming::StreamFinal::raw)).
//!
//! # The feature
//!
//! Capture is always on. The terminal record of every stream the seam yields
//! carries `raw`: the value
//! the Responses API stream adapter produces as its native terminal — the Responses API's
//! [`StreamingCompletionResponse`](rig::providers::openai::responses_api::streaming::StreamingCompletionResponse):
//! the terminal `response.completed` event's usage, status, ids and model —
//! serialized with `serde_json::to_value`. It is the terminal record only, and
//! nothing about it is sent to ChatGPT. `raw == Value::Null` means only that a
//! `StreamFinal` was built by hand without a provider terminal behind it,
//! which no cell here can produce.
//!
//! The terminal record spells the provider's `status` (`completed`), which
//! the normalized [`StreamFinal`](rig::streaming::StreamFinal) folds into a
//! finish reason and does not carry; cell 2 reads it back through `raw`.
//!
//! # Matrix
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `stream_raw_terminal_round_trips_provider_type` | typed access | `responses_api::streaming::StreamingCompletionResponse::deserialize(&*raw)` re-serializes equal | unrecorded (no CHATGPT credentials in this environment) |
//! | 2 | `stream_raw_exposes_terminal_status` | terminal-only field | `raw.status == "completed"` as the recorded `response.completed` frame says; usage equals the frame's | unrecorded (no CHATGPT credentials in this environment) |
//!
//! Every cell is unrecorded: neither `CHATGPT_ACCESS_TOKEN`/`CHATGPT_ACCOUNT_ID`
//! nor a usable ChatGPT OAuth cache was present when this matrix was written,
//! and a fixture is never fabricated. To record: export the two variables,
//! remove the `#[ignore]` attributes, flip the table to `recorded`, then run
//! `RIG_PROVIDER_TEST_MODE=record cargo test -p rig --all-features --test chatgpt chatgpt::cassette::raw_stream_capture_matrix -- --nocapture --test-threads=1`
//! and review `tests/cassettes/chatgpt/raw_stream_capture_matrix/`.

use rig::completion::CompletionModel as _;
use rig::driver::Bound;
use rig::providers::chatgpt;
use rig::providers::openai::responses_api;
use rig::providers::openai::wire::OpenAiWire;
use serde::Deserialize;
use serde_json::Value;

use super::super::support::with_chatgpt_cassette;
use crate::cassettes::{recorded_interaction_bodies, recorded_sse_json_frames};
use crate::support::{Observed, collect_sole_terminal};

const CHATGPT_PROVIDER: &str = "chatgpt";
const MODEL: &str = chatgpt::GPT_5_4;
const PROMPT: &str = "Reply with exactly the single word: pong";

type ChatGptModel = Bound<OpenAiWire>;

fn request(model: &ChatGptModel) -> rig::completion::CompletionRequest {
    model.completion_request(PROMPT).max_tokens(64).build()
}

/// The premise every streaming cell rests on: the scenario recorded exactly
/// one interaction whose SSE stream ends with a `response.completed` frame
/// carrying usage. Returns its `response`.
fn recorded_terminal_response(scenario: &str) -> Value {
    assert_eq!(
        recorded_interaction_bodies(CHATGPT_PROVIDER, scenario).len(),
        1,
        "{scenario}: the scenario must record exactly one interaction"
    );
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

// ---------------------------------------------------------------------------
// 1: raw is the provider-native terminal record, serialized
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "unrecorded (no CHATGPT credentials in this environment)"]
async fn stream_raw_terminal_round_trips_provider_type() {
    let scenario = "raw_stream_capture_matrix/stream_raw_terminal_round_trips_provider_type";
    let captured = Observed::default();
    let sink = captured.clone();
    with_chatgpt_cassette(
        "raw_stream_capture_matrix/stream_raw_terminal_round_trips_provider_type",
        |client| async move {
            let model = client.completion(MODEL);
            let terminal = collect_sole_terminal(
                model
                    .stream(request(&model))
                    .await
                    .expect("stream should start"),
            )
            .await;

            let raw = &terminal.raw;
            let typed = responses_api::streaming::StreamingCompletionResponse::deserialize(raw)
                .expect("raw must deserialize into the Responses terminal type");
            assert_eq!(
                serde_json::to_value(&typed).expect("terminal type should serialize"),
                *raw,
                "responses_api::streaming::StreamingCompletionResponse must round-trip"
            );

            let typed_usage = typed
                .usage
                .as_ref()
                .expect("the Responses terminal carries usage");
            assert_eq!(Some(typed_usage.total_tokens), terminal.usage.total_tokens);
            assert_eq!(Some(typed_usage.input_tokens), terminal.usage.input_tokens);
            assert_eq!(
                Some(typed_usage.output_tokens),
                terminal.usage.output_tokens
            );
            assert_eq!(typed.response_id, terminal.response_id);
            assert_eq!(typed.message_id, terminal.message_id);
            assert_eq!(typed.model, terminal.model);
            sink.put(raw.clone());
        },
    )
    .await;

    let terminal = recorded_terminal_response(scenario);
    let raw = captured.take();
    assert_eq!(
        raw["usage"]["total_tokens"], terminal["usage"]["total_tokens"],
        "raw usage must be the terminal frame's usage"
    );
}

// ---------------------------------------------------------------------------
// 2: terminal-only field
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "unrecorded (no CHATGPT credentials in this environment)"]
async fn stream_raw_exposes_terminal_status() {
    let scenario = "raw_stream_capture_matrix/stream_raw_exposes_terminal_status";
    let captured = Observed::default();
    let sink = captured.clone();
    with_chatgpt_cassette(
        "raw_stream_capture_matrix/stream_raw_exposes_terminal_status",
        |client| async move {
            let model = client.completion(MODEL);
            let terminal = collect_sole_terminal(
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
            assert!(
                normalized.get("status").is_none(),
                "normalized StreamFinal must not grow a `status` field"
            );

            let raw = terminal.raw;
            sink.put(raw);
        },
    )
    .await;

    let raw = captured.take();
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
