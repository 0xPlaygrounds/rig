//! Raw provider response capture on DeepSeek's streaming chat-completions
//! path.
//!
//! **The feature.** Every stream's terminal [`rig::streaming::StreamFinal::raw`]
//! carries the value the model's inherent `raw_stream` yielded as its
//! terminal record — for DeepSeek the shared chat-completions terminal
//! [`StreamingCompletionResponse`] parameterized over DeepSeek's own
//! [`deepseek::Usage`] — serialized. Capture is always on: there is no flag
//! to request it, nothing about it reaches the wire, and a `None` only ever
//! means a terminal built by hand with no provider record behind it. It is
//! the terminal record only, never the stream's frames. DeepSeek's terminal
//! usage carries the `prompt_cache_miss_tokens` count that the normalized
//! `Usage` has no slot for, so it is the natural terminal-only field to pin
//! here.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `stream_raw_round_trips_terminal_type` | typed round trip | terminal `raw` deserializes into `StreamingCompletionResponse<deepseek::Usage>` and re-serializes equal; the normalized terminal reproduces the recorded terminal frame | recorded |
//! | 2 | `stream_raw_exposes_terminal_cache_miss_tokens` | terminal-only field | `raw.usage.prompt_cache_miss_tokens` equals the recorded terminal frame's | recorded |
//!
//! Every cell is recorded. The premise every cell re-derives from its own
//! fixture is that the recorded SSE stream carries usage on exactly one frame
//! — the terminal one — so the raw terminal record's usage is knowable from
//! the bytes and a recording whose stream stopped reporting usage fails
//! loudly instead of covering nothing.

use rig::completion::{CompletionModel, CompletionRequest};
use rig::prelude::*;
use rig::providers::deepseek;
use rig::providers::openai::completion::streaming::StreamingCompletionResponse;
use rig::streaming::StreamFinal;
use serde::Deserialize;
use serde_json::{Value, json};

use super::support::{assert_matches_recorded_token, with_deepseek_cassette_result};
use crate::support::collect_text_and_terminal;

type DeepSeekTerminal = StreamingCompletionResponse<deepseek::Usage>;

const PROVIDER: &str = "deepseek";
const MODEL: &str = deepseek::DEEPSEEK_V4_FLASH;
const PROMPT: &str = "Reply with the single word: pong";

fn request(model: &deepseek::CompletionModel) -> CompletionRequest {
    model
        .completion_request(PROMPT)
        .additional_params(json!({ "thinking": { "type": "disabled" } }))
        .max_tokens(16)
        .build()
}

/// The single recorded frame that carries usage: the terminal frame.
fn recorded_terminal_frame(scenario: &str) -> Value {
    let frames = crate::cassettes::recorded_sse_json_frames(PROVIDER, scenario);
    let mut with_usage = frames.into_iter().filter(|frame| !frame["usage"].is_null());
    let terminal = with_usage
        .next()
        .expect("the recorded stream must carry usage on its terminal frame");
    assert!(
        with_usage.next().is_none(),
        "usage must be reported on exactly one (terminal) frame"
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
        "DeepSeek contracts no id header"
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
    with_deepseek_cassette_result(
        "raw_stream_capture_matrix/stream_raw_round_trips_terminal_type",
        |client| async move {
            let model = client.completion_model(MODEL);
            let stream = model.stream(request(&model)).await?;
            let (text, terminal) = collect_text_and_terminal(stream).await;
            let terminal = terminal.expect("stream should end with a terminal record");
            assert!(!text.is_empty());
            let raw = terminal
                .raw
                .as_deref()
                .expect("every provider-backed terminal carries raw");
            let typed = DeepSeekTerminal::deserialize(raw)
                .expect("raw is the chat-completions terminal over DeepSeek usage");
            assert_eq!(
                serde_json::to_value(&typed).expect("typed serializes"),
                *raw,
                "the captured value is the typed terminal serialized, nothing more"
            );
            assert_eq!(typed.response_id, terminal.response_id);
            assert_eq!(typed.finish_reason, terminal.finish_reason);
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
    assert_terminal_reproduces_frame(&terminal, &frame);
    let request_body = crate::cassettes::recorded_json_request(PROVIDER, SCENARIO);
    assert_eq!(request_body["stream"], json!(true));
}

// ================================================================
// 2. A terminal-only field the normalized record lacks
// ================================================================

#[tokio::test]
async fn stream_raw_exposes_terminal_cache_miss_tokens() {
    const SCENARIO: &str =
        "raw_stream_capture_matrix/stream_raw_exposes_terminal_cache_miss_tokens";
    let observed = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = observed.clone();
    with_deepseek_cassette_result(
        "raw_stream_capture_matrix/stream_raw_exposes_terminal_cache_miss_tokens",
        |client| async move {
            let model = client.completion_model(MODEL);
            let stream = model.stream(request(&model)).await?;
            let (_, terminal) = collect_text_and_terminal(stream).await;
            *sink.lock().expect("observation lock") =
                Some(terminal.expect("stream should end with a terminal record"));
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("stream_raw_exposes_terminal_cache_miss_tokens should replay from its cassette");

    let terminal = observed
        .lock()
        .expect("observation lock")
        .take()
        .expect("the cell should observe a terminal record");
    let frame = recorded_terminal_frame(SCENARIO);
    let recorded_miss = frame["usage"]["prompt_cache_miss_tokens"]
        .as_u64()
        .expect("DeepSeek's terminal usage reports prompt_cache_miss_tokens");
    let recorded_hit = frame["usage"]["prompt_cache_hit_tokens"]
        .as_u64()
        .expect("DeepSeek's terminal usage reports prompt_cache_hit_tokens");

    let raw = terminal
        .raw
        .as_deref()
        .expect("every provider-backed terminal carries raw");
    assert_eq!(
        raw["usage"]["prompt_cache_miss_tokens"],
        json!(recorded_miss)
    );
    assert_eq!(raw["usage"]["prompt_cache_hit_tokens"], json!(recorded_hit));
    // The normalized terminal keeps the hit count (as cached input) and has
    // no slot for the miss count.
    assert_eq!(terminal.usage.cached_input_tokens, recorded_hit);
    let normalized_usage = serde_json::to_value(terminal.usage).expect("usage serializes");
    assert!(
        normalized_usage.get("prompt_cache_miss_tokens").is_none(),
        "the normalized usage has no miss-count slot: {normalized_usage}"
    );
}
