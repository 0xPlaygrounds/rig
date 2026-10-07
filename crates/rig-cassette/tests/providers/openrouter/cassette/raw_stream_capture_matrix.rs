//! Raw provider response capture on OpenRouter's streaming chat-completions
//! path.
//!
//! **The feature.** Every stream's terminal
//! [`rig::completion::CompletionResponse::raw`] carries the record the chat
//! decoder reassembled from the reply's frames: a JSON object with the keys
//! `usage`, `finish_reason`, `response_id`, `model`, `logprobs` and
//! `additional_params`. Unlike a unary reply's `raw`, this one is that record
//! rather than the socket's bytes. Capture is always on: there is no flag to
//! request it, nothing about it reaches the wire, and a `Value::Null` only
//! ever means a terminal built by hand with no provider record behind it. It
//! is the terminal record only, never the stream's frames.
//!
//! **Why it matters here.** A gateway reports things the normalized terminal
//! has no slot for: the terminal's accumulated `additional_params` carries
//! the routed `provider` the frames repeat. OpenRouter's terminal usage also
//! carries the turn's `cost`, which the normalized usage reports as well. The record's `usage` is the provider's usage
//! object, extra fields included, so both reach a caller through `raw`. That
//! is the capability the cell pins: `raw.usage.cost` and
//! `raw.additional_params.provider` equal the recorded terminal frame's.
//!
//! The premise the cell re-derives from its fixture is that usage appears on
//! exactly one frame, the stream's last data frame, so the raw terminal
//! record's usage is knowable from the bytes and a recording whose stream
//! stopped reporting usage fails loudly instead of covering nothing. That is [`chat::recorded_sole_usage_frame`], the
//! single-frame rule rather than the agreeing-closing-frames rule the
//! dialects that repeat their accounting need. OpenRouter contracts no
//! request-id header, so the terminal's `provider_request_id` is `None` —
//! pinned as the documented outcome.

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
