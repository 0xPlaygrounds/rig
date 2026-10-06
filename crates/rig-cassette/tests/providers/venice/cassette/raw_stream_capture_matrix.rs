//! Raw provider response capture on Venice's streaming chat-completions
//! path.
//!
//! **The feature.** Every stream's terminal
//! [`rig::completion::CompletionResponse::raw`] carries the decoder's own
//! terminal record: for Venice the shared chat-completions record, a JSON
//! object with the keys `usage`, `finish_reason`, `response_id`, `model`,
//! `logprobs` and `additional_params`. It is that record rather than the
//! socket's bytes. Capture is always on: there is no flag to request it,
//! nothing about it reaches the wire, and a `Value::Null` only ever means a
//! terminal built by hand with no provider record behind it. It is the
//! terminal record only, never the stream's frames. Venice stamps the
//! request's `cost` on the terminal frame alone, and the terminal's
//! accumulated `additional_params` keeps it, alongside the `object` tag the
//! frames repeat; neither has a slot on the normalized terminal, so both are
//! pinned here as reachable only through `raw`.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 2 | `stream_raw_exposes_terminal_cost` | terminal-only field | `raw.additional_params.cost.usd` equals the recorded terminal frame's, and no earlier frame carried a cost | recorded |
//!
//! Every cell is recorded. The premise every cell re-derives from its own
//! fixture is that usage appears on exactly one frame — the stream's last
//! data frame — so the raw terminal record's usage is knowable from the bytes
//! and a recording whose stream stopped reporting usage fails loudly instead
//! of covering nothing. Venice contracts no request-id header, so the
//! terminal's `provider_request_id` is `None` — pinned as the documented
//! outcome.

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
    // The normalized terminal has no slot for the cost.
    let normalized = crate::support::normalized_without_raw(terminal.clone());
    crate::raw_capture::assert_normalized_lacks(&normalized, &["cost"]);
}
