//! Raw provider response capture on Perplexity's streaming chat-completions
//! path.
//!
//! **The feature.** Every stream's terminal
//! [`rig::streaming::StreamFinal::raw`] carries the value the model's inherent
//! `raw_stream` yielded as its terminal record — for Perplexity the shared
//! chat-completions terminal [`StreamingCompletionResponse`] over the shared
//! [`openai::Usage`] — serialized. Capture is always on: there is no flag to
//! request it, nothing about it reaches the wire, and a `Value::Null` only ever
//! means a terminal built by hand with no provider record behind it. It is the
//! terminal record only, never the stream's frames. Perplexity reports usage on
//! *every* frame and the last frame's counts are the terminal's; the terminal's
//! accumulated `additional_params` carries the `object` tag the frames repeat.
//! Neither the tag nor the raw usage block has a slot on the normalized
//! terminal.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `stream_raw_round_trips_terminal_type` | typed round trip | terminal `raw` deserializes into `StreamingCompletionResponse<openai::Usage>` and re-serializes equal; the normalized terminal reproduces the recorded last frame | recorded |
//! | 2 | `stream_raw_exposes_terminal_usage_and_object` | terminal-only field | `raw.usage` counts equal the recorded last frame's; `raw.additional_params.object` equals the frames' tag | recorded |
//!
//! Every cell is recorded. The premise every cell re-derives from its own
//! fixture is that the recorded stream's last data frame carries usage and a
//! finish reason — so the raw terminal record's usage is knowable from the
//! bytes and a recording whose stream stopped reporting usage fails loudly
//! instead of covering nothing. Perplexity contracts no request-id header,
//! so the terminal's `provider_request_id` is `None` — pinned as the
//! documented outcome.

use rig::completion::{CompletionModel, CompletionRequest};
use rig::prelude::*;
use rig::providers::openai::completion::streaming::StreamingCompletionResponse;
use rig::providers::{openai, perplexity};
use rig::streaming::StreamFinal;
use serde::Deserialize;
use serde_json::{Value, json};

use super::super::support::{assert_matches_recorded_token, with_perplexity_cassette};
use crate::support::collect_text_and_terminal;

type PerplexityTerminal = StreamingCompletionResponse<openai::Usage>;

const PROVIDER: &str = "perplexity";
const MODEL: &str = perplexity::SONAR;
const PROMPT: &str = "Reply with the single word: pong";

fn request(model: &perplexity::CompletionModel) -> CompletionRequest {
    model.completion_request(PROMPT).max_tokens(16).build()
}

/// The recorded stream's last data frame: it carries the finish reason and
/// the usage the terminal record reports (Perplexity repeats usage on every
/// frame; the last frame's counts win).
fn recorded_terminal_frame(scenario: &str) -> Value {
    let frames = crate::cassettes::recorded_sse_json_frames(PROVIDER, scenario);
    let terminal = frames
        .last()
        .expect("the recorded stream must carry at least one frame")
        .clone();
    assert!(
        terminal["usage"].is_object(),
        "the last frame must carry usage: {terminal}"
    );
    assert!(
        terminal["choices"][0]["finish_reason"].is_string(),
        "the last frame must carry the finish reason: {terminal}"
    );
    terminal
}

fn assert_terminal_reproduces_frame(terminal: &StreamFinal, frame: &Value) {
    assert_eq!(terminal.provider, PROVIDER, "provider");
    assert_matches_recorded_token(
        terminal.response_id.as_deref(),
        frame["id"].as_str(),
        "response id",
    );
    assert_eq!(terminal.model.as_deref(), frame["model"].as_str(), "model");
    assert_eq!(
        terminal.usage.input_tokens,
        frame["usage"]["prompt_tokens"]
            .as_u64()
            .expect("prompt_tokens"),
        "input tokens"
    );
    assert_eq!(
        terminal.usage.output_tokens,
        frame["usage"]["completion_tokens"]
            .as_u64()
            .expect("completion_tokens"),
        "output tokens"
    );
    assert_eq!(
        terminal.usage.total_tokens,
        frame["usage"]["total_tokens"]
            .as_u64()
            .expect("total_tokens"),
        "total tokens"
    );
    assert_eq!(
        terminal.provider_request_id, None,
        "Perplexity contracts no id header"
    );
}

// ================================================================
// 1. raw round-trips the terminal type
// ================================================================

#[tokio::test]
async fn stream_raw_round_trips_terminal_type() {
    const SCENARIO: &str = "raw_stream_capture_matrix/stream_raw_round_trips_terminal_type";
    let observed = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = observed.clone();
    with_perplexity_cassette(
        "raw_stream_capture_matrix/stream_raw_round_trips_terminal_type",
        |client| async move {
            let model = client.completion_model(MODEL);
            let stream = model
                .stream(request(&model))
                .await
                .expect("the stream should open");
            let (text, terminal) = collect_text_and_terminal(stream).await;
            let terminal = terminal.expect("stream should end with a terminal record");
            assert!(!text.is_empty());
            let raw = &terminal.raw;
            let typed = PerplexityTerminal::deserialize(raw)
                .expect("raw is the chat-completions terminal over the shared OpenAI usage");
            assert_eq!(
                serde_json::to_value(&typed).expect("typed serializes"),
                *raw,
                "the captured value is the typed terminal serialized, nothing more"
            );
            assert_eq!(typed.response_id, terminal.response_id);
            assert_eq!(typed.finish_reason, terminal.finish_reason);
            assert_eq!(typed.provider_request_id, terminal.provider_request_id);
            *sink.lock().expect("observation lock") = Some(terminal);
        },
    )
    .await;

    let terminal = observed
        .lock()
        .expect("observation lock")
        .take()
        .expect("the cell should observe a terminal record");
    let frame = recorded_terminal_frame(SCENARIO);
    assert_terminal_reproduces_frame(&terminal, &frame);
    let request_body = crate::cassettes::recorded_json_request(PROVIDER, SCENARIO);
    assert_eq!(request_body["stream"], json!(true));
}

// ================================================================
// 2. Terminal-only fields the normalized record lacks
// ================================================================

#[tokio::test]
async fn stream_raw_exposes_terminal_usage_and_object() {
    const SCENARIO: &str = "raw_stream_capture_matrix/stream_raw_exposes_terminal_usage_and_object";
    let observed = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = observed.clone();
    with_perplexity_cassette(
        "raw_stream_capture_matrix/stream_raw_exposes_terminal_usage_and_object",
        |client| async move {
            let model = client.completion_model(MODEL);
            let stream = model
                .stream(request(&model))
                .await
                .expect("the stream should open");
            let (_, terminal) = collect_text_and_terminal(stream).await;
            *sink.lock().expect("observation lock") =
                Some(terminal.expect("stream should end with a terminal record"));
        },
    )
    .await;

    let terminal = observed
        .lock()
        .expect("observation lock")
        .take()
        .expect("the cell should observe a terminal record");
    let frame = recorded_terminal_frame(SCENARIO);
    let recorded_object = frame["object"]
        .as_str()
        .expect("Perplexity tags every chunk with an object");
    let recorded_usage = &frame["usage"];

    let raw = &terminal.raw;
    assert_eq!(
        raw["usage"]["prompt_tokens"],
        recorded_usage["prompt_tokens"]
    );
    assert_eq!(
        raw["usage"]["completion_tokens"],
        recorded_usage["completion_tokens"]
    );
    assert_eq!(raw["usage"]["total_tokens"], recorded_usage["total_tokens"]);
    assert_eq!(raw["additional_params"]["object"], json!(recorded_object));
    // The normalized terminal has no slot for the tag; and the raw usage is
    // the shared *type* serialized, so Perplexity's `cost` /
    // `search_context_size` usage extras — unmodeled — are not in it either.
    let normalized = serde_json::to_value(&terminal).expect("terminal serializes");
    assert!(normalized.get("object").is_none() && normalized.get("additional_params").is_none());
    assert!(
        recorded_usage.get("cost").is_some(),
        "the recorded usage carries Perplexity's cost block: {recorded_usage}"
    );
    assert!(raw["usage"].get("cost").is_none());
}
