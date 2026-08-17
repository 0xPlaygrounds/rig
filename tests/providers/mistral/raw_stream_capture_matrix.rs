//! Raw provider response capture on Mistral's streaming chat-completions
//! path.
//!
//! **The feature.** Every stream's terminal
//! [`rig::streaming::StreamFinal::raw`] carries the value the model's inherent
//! `raw_stream` yielded as its terminal record — for Mistral the shared
//! chat-completions terminal [`StreamingCompletionResponse`] over Mistral's own
//! [`mistral::Usage`] — serialized. Capture is always on: there is no flag to
//! request it, nothing about it reaches the wire, and a `Value::Null` only ever
//! means a terminal built by hand with no provider record behind it. It is the
//! terminal record only, never the stream's frames. Mistral's terminal usage
//! carries the capacity tier (`usage.service_tier`) that the normalized `Usage`
//! has no slot for, so it is the terminal-only field pinned here.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `stream_raw_round_trips_terminal_type` | typed round trip | terminal `raw` deserializes into `StreamingCompletionResponse<mistral::Usage>` and re-serializes equal; the normalized terminal reproduces the recorded terminal frame and `mistral-correlation-id` header | recorded |
//! | 2 | `stream_raw_exposes_terminal_service_tier` | terminal-only field | `raw.usage.service_tier` equals the recorded terminal frame's | recorded |
//!
//! Every cell is recorded. The premise every cell re-derives from its own
//! fixture is that usage appears on exactly one frame — Mistral's finish
//! frame, the stream's last data frame, which also carries the final content
//! delta — so the raw terminal record's usage is knowable from the bytes and
//! a recording whose stream stopped reporting usage fails loudly instead of
//! covering nothing.

use rig::completion::{CompletionModel, CompletionRequest};
use rig::prelude::*;
use rig::providers::mistral;
use rig::providers::openai::completion::streaming::StreamingCompletionResponse;
use rig::streaming::StreamFinal;
use serde::Deserialize;
use serde_json::{Value, json};

use super::DEFAULT_MODEL;
use super::support::{
    assert_matches_recorded_token, recorded_response_headers, with_mistral_cassette_result,
};
use crate::support::collect_text_and_terminal;

type MistralTerminal = StreamingCompletionResponse<mistral::Usage>;

const PROVIDER: &str = "mistral";
const PROMPT: &str = "Reply with the single word: pong";

fn request(model: &mistral::CompletionModel) -> CompletionRequest {
    model.completion_request(PROMPT).max_tokens(16).build()
}

/// The single recorded frame that carries usage — Mistral's finish frame,
/// which also carries the last content delta and the `finish_reason` — after
/// checking that it is the stream's last data frame.
fn recorded_terminal_frame(scenario: &str) -> Value {
    let frames = crate::cassettes::recorded_sse_json_frames(PROVIDER, scenario);
    let mut with_usage = frames
        .iter()
        .enumerate()
        .filter(|(_, frame)| !frame["usage"].is_null());
    let (index, terminal) = with_usage
        .next()
        .expect("the recorded stream must carry usage on its terminal frame");
    assert!(
        with_usage.next().is_none(),
        "usage must be reported on exactly one (terminal) frame"
    );
    assert_eq!(
        index + 1,
        frames.len(),
        "the usage-bearing frame must be the stream's last data frame"
    );
    assert!(
        terminal["choices"][0]["finish_reason"].is_string(),
        "Mistral's terminal frame carries the finish reason: {terminal}"
    );
    terminal.clone()
}

fn recorded_request_id(scenario: &str) -> Option<String> {
    recorded_response_headers(scenario)[0]
        .iter()
        .find(|(name, _)| name == "mistral-correlation-id")
        .map(|(_, value)| value.clone())
}

fn assert_terminal_reproduces_frame(
    terminal: &StreamFinal,
    frame: &Value,
    request_id: Option<&str>,
) {
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
    assert!(
        request_id.is_some(),
        "the recorded SSE response must carry mistral-correlation-id"
    );
    assert_matches_recorded_token(
        terminal.provider_request_id.as_deref(),
        request_id,
        "request id",
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
    with_mistral_cassette_result(
        "raw_stream_capture_matrix/stream_raw_round_trips_terminal_type",
        |client| async move {
            let model = client.completion_model(DEFAULT_MODEL);
            let stream = model.stream(request(&model)).await?;
            let (text, terminal) = collect_text_and_terminal(stream).await;
            let terminal = terminal.expect("stream should end with a terminal record");
            assert!(!text.is_empty());
            let raw = &terminal.raw;
            let typed = MistralTerminal::deserialize(raw)
                .expect("raw is the chat-completions terminal over Mistral usage");
            assert_eq!(
                serde_json::to_value(&typed).expect("typed serializes"),
                *raw,
                "the captured value is the typed terminal serialized, nothing more"
            );
            assert_eq!(typed.response_id, terminal.response_id);
            assert_eq!(typed.finish_reason, terminal.finish_reason);
            assert_eq!(typed.provider_request_id, terminal.provider_request_id);
            *sink.lock().expect("observation lock") = Some(terminal);
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("stream_raw_round_trips_terminal_type should replay from its cassette");

    let terminal = observed
        .lock()
        .expect("observation lock")
        .take()
        .expect("the cell should observe a terminal record");
    let frame = recorded_terminal_frame(SCENARIO);
    assert_terminal_reproduces_frame(&terminal, &frame, recorded_request_id(SCENARIO).as_deref());
    let request_body = crate::cassettes::recorded_json_request(PROVIDER, SCENARIO);
    assert_eq!(request_body["stream"], json!(true));
}

// ================================================================
// 2. A terminal-only field the normalized record lacks
// ================================================================

#[tokio::test]
async fn stream_raw_exposes_terminal_service_tier() {
    const SCENARIO: &str = "raw_stream_capture_matrix/stream_raw_exposes_terminal_service_tier";
    let observed = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = observed.clone();
    with_mistral_cassette_result(
        "raw_stream_capture_matrix/stream_raw_exposes_terminal_service_tier",
        |client| async move {
            let model = client.completion_model(DEFAULT_MODEL);
            let stream = model.stream(request(&model)).await?;
            let (_, terminal) = collect_text_and_terminal(stream).await;
            *sink.lock().expect("observation lock") =
                Some(terminal.expect("stream should end with a terminal record"));
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("stream_raw_exposes_terminal_service_tier should replay from its cassette");

    let terminal = observed
        .lock()
        .expect("observation lock")
        .take()
        .expect("the cell should observe a terminal record");
    let frame = recorded_terminal_frame(SCENARIO);
    let recorded_tier = frame["usage"]["service_tier"]
        .as_str()
        .expect("Mistral's terminal usage reports service_tier");
    let recorded_prompt_tokens = frame["usage"]["prompt_tokens"]
        .as_u64()
        .expect("terminal usage reports prompt_tokens");

    let raw = &terminal.raw;
    assert_eq!(raw["usage"]["service_tier"], json!(recorded_tier));
    assert_eq!(raw["usage"]["prompt_tokens"], json!(recorded_prompt_tokens));
    // The normalized terminal has no slot for the tier.
    let normalized_usage = serde_json::to_value(terminal.usage).expect("usage serializes");
    assert!(
        normalized_usage.get("service_tier").is_none(),
        "the normalized usage has no tier slot: {normalized_usage}"
    );
}
