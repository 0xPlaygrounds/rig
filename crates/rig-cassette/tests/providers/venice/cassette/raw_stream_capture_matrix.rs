//! Raw provider response capture on Venice's streaming chat-completions path.
//!
//! Every stream's terminal [`rig::completion::CompletionResponse::raw`] carries the decoder's
//! own terminal record: for Venice the shared chat-completions record, a JSON object with the
//! keys `usage`, `finish_reason`, `response_id`, `model`, `logprobs` and `additional_params`.
//! Capture is always on, and a `Value::Null` only means a terminal built by hand. Venice stamps
//! the request's `cost` on the terminal frame alone, and the record keeps it beside the
//! `object` tag the frames repeat. `stream_raw_exposes_terminal_cost` pins that `raw.cost.usd`
//! and the usage's cost equal the recorded terminal frame's, and that no earlier frame carried
//! a cost.
//!
//! The cell first checks that usage appears only on the stream's last data frame, so a
//! recording that stopped reporting usage fails loudly. Venice contracts no request-id header,
//! so the terminal's `provider_request_id` is `None`.

use rig::completion::CompletionRequest;
use serde_json::json;

use super::super::DEFAULT_MODEL;
use super::super::support::with_venice_cassette_result;
use crate::raw_capture::{capture_text_and_terminal, chat};
use crate::support::Observed;

const PROVIDER: &str = "venice";
const PROMPT: &str = "Reply with the single word: pong";

fn request() -> CompletionRequest {
    CompletionRequest::new(PROMPT)
        .max_tokens(16)
        .additional_params(json!({"venice_parameters": {"disable_thinking": true}}))
}

// ================================================================
// 1. raw is the terminal record
// ================================================================

// ================================================================
// 2. A terminal-only field the normalized record lacks
// ================================================================

#[tokio::test]
async fn stream_raw_exposes_terminal_cost() {
    const SCENARIO: &str = "raw_stream_capture_matrix/stream_raw_exposes_terminal_cost";
    let sink = Observed::default();
    with_venice_cassette_result(
        "raw_stream_capture_matrix/stream_raw_exposes_terminal_cost",
        |client| {
            capture_text_and_terminal(client.completion(DEFAULT_MODEL), request(), sink.clone())
        },
    )
    .await
    .expect("stream_raw_exposes_terminal_cost should replay from its cassette");

    let (_, terminal) = sink.take();
    let frame = chat::recorded_sole_usage_frame(PROVIDER, SCENARIO);
    let recorded_cost = frame["cost"]["usd"]
        .as_f64()
        .expect("Venice stamps the cost on the terminal frame");
    // Terminal-only: no earlier frame carried a cost.
    let frames = crate::cassettes::recorded_sse_json_frames(PROVIDER, SCENARIO);
    assert_eq!(
        frames
            .iter()
            .filter(|frame| frame.get("cost").is_some())
            .count(),
        1,
        "exactly the terminal frame carries cost"
    );

    // The rebuilt document states the cost where a unary body does, and
    // carries the unary tag.
    let raw = &terminal.raw;
    assert_eq!(raw["cost"]["usd"], json!(recorded_cost));
    assert_eq!(raw["object"], json!("chat.completion"));
    // The normalized terminal reports it as the usage's cost.
    assert_eq!(
        terminal.usage.cost.map(|cost| cost.total),
        Some(recorded_cost)
    );
}
