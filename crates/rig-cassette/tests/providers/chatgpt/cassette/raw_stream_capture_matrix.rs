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
//! and review `crates/rig-cassette/fixtures/cassettes/chatgpt/raw_stream_capture_matrix/`.

use rig::completion::CompletionModel as _;
use rig::driver::Bound;
use rig::providers::chatgpt;
use rig::providers::openai::responses_api;
use rig::providers::openai::wire::OpenAiWire;
use serde_json::Value;

use super::super::support::with_chatgpt_cassette;
use crate::cassettes::{recorded_interaction_bodies, recorded_sse_json_frames};
use crate::raw_capture::{
    assert_normalized_lacks, capture_sole_terminal, responses, stream_normalized_without_raw,
};
use crate::support::Observed;

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
///
/// Stays local: the usage a Responses stream reports hangs under a typed
/// event's `response` envelope, not on the frame itself, so neither
/// [`crate::raw_capture::chat::recorded_sole_usage_frame`]'s rule nor
/// [`crate::raw_capture::chat::recorded_agreeing_usage_frames`]' describes
/// this wire.
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
            capture_sole_terminal(client.completion(MODEL), request, sink)
                .await
                .expect("stream should start");
        },
    )
    .await;

    let terminal = captured.take();
    let raw = &terminal.raw;
    // The Responses stream's terminal record is its own type, so this is the
    // Responses contract's round trip, not `chat::assert_terminal_round_trips`,
    // which speaks the chat-completions terminal.
    let typed = responses::assert_terminal_round_trips(&terminal);
    // The shared round trip compares the whole normalized accounting, which a
    // terminal carrying no usage at all would satisfy by both sides being
    // empty; this cell's premise is that the record has usage.
    assert!(
        typed.usage.is_some(),
        "the Responses terminal carries usage"
    );

    let recorded = recorded_terminal_response(scenario);
    assert_eq!(
        raw["usage"]["total_tokens"], recorded["usage"]["total_tokens"],
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
            capture_sole_terminal(client.completion(MODEL), request, sink)
                .await
                .expect("stream should start");
        },
    )
    .await;

    let terminal = captured.take();
    assert_normalized_lacks(&stream_normalized_without_raw(&terminal), &["status"]);

    let raw = &terminal.raw;
    let recorded = recorded_terminal_response(scenario);
    assert_eq!(
        recorded["status"],
        Value::String("completed".to_string()),
        "{scenario}: premise — the recorded terminal frame is completed"
    );
    assert_eq!(raw["status"], recorded["status"]);
    assert_eq!(raw["usage"], recorded["usage"]);
    let typed = responses::assert_terminal_round_trips(&terminal);
    assert_eq!(typed.status, Some(responses_api::ResponseStatus::Completed));
}
