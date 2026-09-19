//! Raw provider response capture on Venice's streaming chat-completions
//! path.
//!
//! **The feature.** Every stream's terminal
//! [`rig::streaming::StreamFinal::raw`] carries the decoder's own terminal
//! record — for Venice the shared chat-completions
//! [`StreamingCompletionResponse`], whose accounting is [`ChatUsage`] —
//! serialized. That is a serialization rather than the socket's bytes, so the
//! round trip through the record's own type is exact. Capture is
//! always on: there is no flag to request it, nothing about it reaches the
//! wire, and a `Value::Null` only ever means a terminal built by hand with no
//! provider record behind it. It is the terminal record only, never the
//! stream's frames. Venice stamps the request's `cost` on the terminal frame
//! alone, and the terminal's accumulated `additional_params` keeps it,
//! alongside the `object` tag the frames repeat; neither has a slot on the
//! normalized terminal, so both are pinned here as reachable only through
//! `raw`.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `stream_raw_round_trips_terminal_type` | typed round trip | terminal `raw` deserializes into `StreamingCompletionResponse<ChatUsage>` and re-serializes equal; its accounting normalizes to the terminal's usage; the normalized terminal reproduces the recorded terminal frame | recorded |
//! | 2 | `stream_raw_exposes_terminal_cost` | terminal-only field | `raw.additional_params.cost.usd` equals the recorded terminal frame's, and no earlier frame carried a cost | recorded |
//!
//! Every cell is recorded. The premise every cell re-derives from its own
//! fixture is that usage appears on exactly one frame — the stream's last
//! data frame — so the raw terminal record's usage is knowable from the bytes
//! and a recording whose stream stopped reporting usage fails loudly instead
//! of covering nothing. Venice contracts no request-id header, so the
//! terminal's `provider_request_id` is `None` — pinned as the documented
//! outcome.

use rig::completion::{CompletionModel, CompletionRequest};
use rig::providers::venice::VeniceParameters;
use serde_json::json;

use super::super::DEFAULT_MODEL;
use super::super::support::with_venice_cassette_result;
use crate::raw_capture::{assert_no_request_id, capture_text_and_terminal, chat};
use crate::support::Observed;

const PROVIDER: &str = "venice";
const PROMPT: &str = "Reply with the single word: pong";

fn request(model: &(impl CompletionModel + Clone)) -> CompletionRequest {
    model
        .completion_request(PROMPT)
        .max_tokens(16)
        .additional_params(
            VeniceParameters::new()
                .disable_thinking(true)
                .into_additional_params(),
        )
        .build()
}

// ================================================================
// 1. raw round-trips the terminal type
// ================================================================

#[tokio::test]
async fn stream_raw_round_trips_terminal_type() {
    const SCENARIO: &str = "raw_stream_capture_matrix/stream_raw_round_trips_terminal_type";
    let sink = Observed::default();
    with_venice_cassette_result(
        "raw_stream_capture_matrix/stream_raw_round_trips_terminal_type",
        |client| capture_text_and_terminal(client.completion(DEFAULT_MODEL), request, sink.clone()),
    )
    .await
    .expect("stream_raw_round_trips_terminal_type should replay from its cassette");

    let (text, terminal) = sink.take();
    assert!(!text.is_empty());
    // The captured document is the decoder's terminal record serialized, so
    // it round-trips exactly and its native fields are the normalized ones —
    // including that the transport id is stamped on the normalized terminal
    // rather than on the native record.
    chat::assert_terminal_round_trips(&terminal);

    let frame = chat::recorded_sole_usage_frame(PROVIDER, SCENARIO);
    chat::assert_terminal_reproduces_frame(&terminal, PROVIDER, &frame, "the recorded frame");
    // Venice contracts no id header, so `None` is the documented outcome.
    assert_no_request_id(terminal.provider_request_id.as_deref(), "Venice");
    let request_body = crate::cassettes::recorded_json_request(PROVIDER, SCENARIO);
    assert_eq!(request_body["stream"], json!(true));
}

// ================================================================
// 2. A terminal-only field the normalized record lacks
// ================================================================

#[tokio::test]
async fn stream_raw_exposes_terminal_cost() {
    const SCENARIO: &str = "raw_stream_capture_matrix/stream_raw_exposes_terminal_cost";
    let sink = Observed::default();
    with_venice_cassette_result(
        "raw_stream_capture_matrix/stream_raw_exposes_terminal_cost",
        |client| capture_text_and_terminal(client.completion(DEFAULT_MODEL), request, sink.clone()),
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

    let raw = &terminal.raw;
    assert_eq!(
        raw["additional_params"]["cost"]["usd"],
        json!(recorded_cost)
    );
    assert_eq!(
        raw["additional_params"]["object"],
        json!("chat.completion.chunk")
    );
    // The normalized terminal has no slot for either.
    let normalized = crate::raw_capture::stream_normalized_without_raw(&terminal);
    crate::raw_capture::assert_normalized_lacks(&normalized, &["cost", "additional_params"]);
}
