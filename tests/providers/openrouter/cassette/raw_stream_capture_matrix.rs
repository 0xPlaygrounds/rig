//! Raw provider response capture on OpenRouter's streaming chat-completions
//! path.
//!
//! **The feature.** Every stream's terminal
//! [`rig::streaming::StreamFinal::raw`] carries the record the chat decoder
//! reassembled from the reply's frames — the shared chat terminal
//! (`openai::wire::StreamingCompletionResponse`), serialized. Unlike a unary
//! reply's `raw`, this one is a serialization of that record rather than the
//! socket's bytes, so reading it back through the same type is exact. Capture is always on: there is no flag to request it,
//! nothing about it reaches the wire, and a `Value::Null` only ever means a
//! terminal built by hand with no provider record behind it. It is the terminal
//! record only, never the stream's frames.
//!
//! **Why it matters here.** A gateway reports things the normalized terminal
//! has no slot for: OpenRouter's terminal usage carries the turn's `cost`, and
//! the terminal's accumulated `additional_params` carries the routed
//! `provider` the frames repeat. `ChatUsage` flattens a dialect's extra usage
//! fields, so both reach a caller through `raw` — which is the capability
//! these two cells pin.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `stream_raw_reads_back_as_terminal_type` | typed round trip | terminal `raw` deserializes into the shared chat terminal and re-serializes equal, its `usage` maps to the normalized usage, and the normalized terminal reproduces the recorded terminal frame | recorded |
//! | 2 | `stream_raw_exposes_terminal_cost_and_provider` | terminal-only fields | `raw.usage.cost` and `raw.additional_params.provider` equal the recorded terminal frame's | recorded |
//!
//! The scenario literals — and therefore the fixture filenames — keep the
//! names they were recorded under; the cell names describe what the cells now
//! assert.
//!
//! Every cell is recorded. The premise every cell re-derives from its own
//! fixture is that usage appears on exactly one frame — the stream's last data
//! frame — so the raw terminal record's usage is knowable from the bytes and a
//! recording whose stream stopped reporting usage fails loudly instead of
//! covering nothing. That is [`chat::recorded_sole_usage_frame`], the
//! single-frame rule rather than the agreeing-closing-frames rule the
//! dialects that repeat their accounting need. OpenRouter contracts no
//! request-id header, so the terminal's `provider_request_id` is `None` —
//! pinned as the documented outcome.

use rig::completion::{CompletionModel, CompletionRequest};
use serde_json::json;

use super::super::DEFAULT_MODEL;
use super::super::support::with_openrouter_cassette_result;
use crate::raw_capture::{assert_no_request_id, capture_text_and_terminal, chat};
use crate::support::Observed;

const PROVIDER: &str = "openrouter";
const PROMPT: &str = "Reply with the single word: pong";

fn request(model: &(impl CompletionModel + Clone)) -> CompletionRequest {
    model.completion_request(PROMPT).max_tokens(16).build()
}

// ================================================================
// 1. raw reads back as the terminal type
// ================================================================

#[tokio::test]
async fn stream_raw_reads_back_as_terminal_type() {
    const SCENARIO: &str = "raw_stream_capture_matrix/stream_raw_round_trips_terminal_type";
    let sink = Observed::default();
    with_openrouter_cassette_result(
        "raw_stream_capture_matrix/stream_raw_round_trips_terminal_type",
        |client| capture_text_and_terminal(client.completion(DEFAULT_MODEL), request, sink.clone()),
    )
    .await
    .expect("stream_raw_round_trips_terminal_type should replay from its cassette");
    let (text, terminal) = sink.take();
    assert!(!text.is_empty());

    // The streamed terminal's `raw` is the decoder's own record serialized
    // (`emit_terminal` builds it with `serde_json::to_value`), not socket
    // bytes — so the round trip back through the same type is exact, and the
    // accounting that comes back with it is what the normalized usage was
    // mapped from: `ChatUsage` flattens the dialect's extra usage fields.
    chat::assert_terminal_round_trips(&terminal);

    let frame = chat::recorded_sole_usage_frame(PROVIDER, SCENARIO);
    chat::assert_terminal_reproduces_frame(&terminal, PROVIDER, &frame, "the recorded frame");
    // The one claim the shared frame contract leaves to the cell: this
    // dialect names no id header, so the normalized terminal reports `None`
    // — and the native record inside `raw` carries none either, which the
    // round trip above pinned.
    assert_no_request_id(terminal.provider_request_id.as_deref(), "OpenRouter");
    let request_body = crate::cassettes::recorded_json_request(PROVIDER, SCENARIO);
    assert_eq!(request_body["stream"], json!(true));
}

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
        |client| capture_text_and_terminal(client.completion(DEFAULT_MODEL), request, sink.clone()),
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
    assert_eq!(
        raw["additional_params"]["provider"],
        json!(recorded_provider)
    );
    // The normalized terminal has no slot for either: its `provider` is rig's
    // descriptor name, not the routed upstream.
    assert_eq!(terminal.provider, PROVIDER);
    let normalized_usage = serde_json::to_value(terminal.usage).expect("usage serializes");
    assert!(
        normalized_usage.get("cost").is_none(),
        "the normalized usage has no cost slot: {normalized_usage}"
    );
}
