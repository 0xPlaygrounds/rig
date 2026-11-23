//! Raw provider response capture on OpenRouter's streaming chat-completions path.
//!
//! Every stream's terminal [`rig::completion::CompletionResponse::raw`] carries the record the
//! chat decoder reassembled from the reply's frames: a JSON object with the keys `usage`,
//! `finish_reason`, `response_id`, `model`, `logprobs` and `additional_params`. Capture is
//! always on, and a `Value::Null` only means a terminal built by hand. A gateway reports things
//! the normalized terminal has no slot for, and the cell pins that `raw.usage.cost` and the
//! routed `raw.additional_params.provider` equal the recorded terminal frame's.
//!
//! The cell first checks, through [`chat::recorded_sole_usage_frame`], that usage appears only
//! on the stream's last data frame, so a recording that stopped reporting usage fails loudly.
//! OpenRouter contracts no request-id header, so the terminal's `provider_request_id` is
//! `None`.

use rig::completion::CompletionRequest;
use serde_json::json;

use super::super::DEFAULT_MODEL;
use super::super::support::with_openrouter_cassette_result;
use crate::raw_capture::{capture_text_and_terminal, chat};
use crate::support::Observed;

const PROVIDER: &str = "openrouter";
const PROMPT: &str = "Reply with the single word: pong";

fn request() -> CompletionRequest {
    CompletionRequest::new(PROMPT).max_tokens(16)
}

// ================================================================
// 1. raw is the terminal record
// ================================================================

// ================================================================
// 2. Terminal-only fields the normalized record lacks
// ================================================================

#[tokio::test]
async fn stream_raw_exposes_terminal_cost_and_provider() {
    const SCENARIO: &str =
        "raw_stream_capture_matrix/stream_raw_exposes_terminal_cost_and_provider";
    let sink = Observed::default();
    with_openrouter_cassette_result(
        "raw_stream_capture_matrix/stream_raw_exposes_terminal_cost_and_provider",
        |client| {
            capture_text_and_terminal(client.completion(DEFAULT_MODEL), request(), sink.clone())
        },
    )
    .await
    .expect("stream_raw_exposes_terminal_cost_and_provider should replay from its cassette");
    let (text, terminal) = sink.take();
    assert!(!text.is_empty());

    let frame = chat::recorded_sole_usage_frame(PROVIDER, SCENARIO);
    let recorded_cost = frame["usage"]["cost"]
        .as_f64()
        .expect("OpenRouter's terminal usage reports cost");
    let recorded_provider = frame["provider"]
        .as_str()
        .expect("OpenRouter's frames name the routed provider");

    let raw = &terminal.raw;
    assert_eq!(raw["usage"]["cost"], json!(recorded_cost));
    assert_eq!(raw["provider"], json!(recorded_provider));
    // The normalized terminal's `provider` is rig's descriptor name, not the
    // routed upstream; its usage's cost is the one OpenRouter reported.
    assert_eq!(terminal.provider(), PROVIDER);
    assert_eq!(
        terminal.usage.cost.map(|cost| cost.total),
        Some(recorded_cost)
    );
}
