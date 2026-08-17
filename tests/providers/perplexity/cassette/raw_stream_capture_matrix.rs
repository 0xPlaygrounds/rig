//! Raw provider response capture on Perplexity's streaming chat-completions
//! path.
//!
//! **The feature.** With `CompletionRequest::capture_raw_response` on, the
//! stream's terminal [`rig::streaming::StreamFinal::raw`] carries the value
//! the model's inherent `raw_stream` would have yielded as its terminal
//! record — for Perplexity the shared chat-completions terminal
//! [`StreamingCompletionResponse`] over the shared [`openai::Usage`] —
//! serialized. It is the terminal record only, never the stream's frames.
//! Perplexity reports usage on *every* frame and the last frame's counts are
//! the terminal's; the terminal's accumulated `additional_params` carries the
//! `object` tag the frames repeat. Neither the tag nor the raw usage block
//! has a slot on the normalized terminal.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `stream_capture_off_leaves_raw_none` | flag off (the default) | terminal `raw == None` | recorded |
//! | 2 | `stream_capture_on_round_trips_terminal_type` | flag on | terminal `raw` deserializes into `StreamingCompletionResponse<openai::Usage>` and re-serializes equal | recorded |
//! | 3 | `stream_capture_on_exposes_terminal_usage_and_object` | terminal-only field | `raw.usage` counts equal the recorded last frame's; `raw.additional_params.object` equals the frames' tag | recorded |
//! | 4 | `stream_request_invariant_off_vs_on` | on-wire request | the flag-off and flag-on request bodies are byte-identical | recorded |
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

fn request(model: &perplexity::CompletionModel, capture: bool) -> CompletionRequest {
    model
        .completion_request(PROMPT)
        .max_tokens(16)
        .capture_raw_response(capture)
        .build()
}

/// Perplexity rate-limits back-to-back requests on this key (`429
/// request_rate_limit_exceeded` on the second turn of a two-turn scenario).
/// The pause exists only while recording — replay serves the fixture — and
/// leaves no trace in the fixture.
async fn pause_between_live_turns() {
    if crate::cassettes::CassetteMode::current() == crate::cassettes::CassetteMode::Record {
        tokio::time::sleep(std::time::Duration::from_secs(20)).await;
    }
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

fn assert_terminal_reproduces_frame(terminal: &StreamFinal, frame: &Value, context: &str) {
    assert_eq!(terminal.provider, PROVIDER, "{context}: provider");
    assert_matches_recorded_token(
        terminal.response_id.as_deref(),
        frame["id"].as_str(),
        &format!("{context}: response id"),
    );
    assert_eq!(
        terminal.model.as_deref(),
        frame["model"].as_str(),
        "{context}: model"
    );
    assert_eq!(
        terminal.usage.input_tokens,
        frame["usage"]["prompt_tokens"]
            .as_u64()
            .expect("prompt_tokens"),
        "{context}: input tokens"
    );
    assert_eq!(
        terminal.usage.output_tokens,
        frame["usage"]["completion_tokens"]
            .as_u64()
            .expect("completion_tokens"),
        "{context}: output tokens"
    );
    assert_eq!(
        terminal.usage.total_tokens,
        frame["usage"]["total_tokens"]
            .as_u64()
            .expect("total_tokens"),
        "{context}: total tokens"
    );
    assert_eq!(
        terminal.provider_request_id, None,
        "{context}: Perplexity contracts no id header"
    );
}

// ================================================================
// 1. Off leaves the terminal raw None
// ================================================================

#[tokio::test]
async fn stream_capture_off_leaves_raw_none() {
    const SCENARIO: &str = "raw_stream_capture_matrix/stream_capture_off_leaves_raw_none";
    let observed = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = observed.clone();
    with_perplexity_cassette(
        "raw_stream_capture_matrix/stream_capture_off_leaves_raw_none",
        |client| async move {
            let model = client.completion_model(MODEL);
            let stream = model
                .stream(request(&model, false))
                .await
                .expect("the stream should open");
            let (text, terminal) = collect_text_and_terminal(stream).await;
            let terminal = terminal.expect("stream should end with a terminal record");
            assert!(
                terminal.raw.is_none(),
                "raw must stay None unless asked for"
            );
            assert!(!text.is_empty());
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
    assert_terminal_reproduces_frame(&terminal, &frame, "flag off");
    let request_body = crate::cassettes::recorded_json_request(PROVIDER, SCENARIO);
    assert_eq!(request_body["stream"], json!(true));
    assert!(request_body.get("capture_raw_response").is_none());
}

// ================================================================
// 2. On round-trips the terminal type
// ================================================================

#[tokio::test]
async fn stream_capture_on_round_trips_terminal_type() {
    const SCENARIO: &str = "raw_stream_capture_matrix/stream_capture_on_round_trips_terminal_type";
    let observed = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = observed.clone();
    with_perplexity_cassette(
        "raw_stream_capture_matrix/stream_capture_on_round_trips_terminal_type",
        |client| async move {
            let model = client.completion_model(MODEL);
            let stream = model
                .stream(request(&model, true))
                .await
                .expect("the stream should open");
            let (_, terminal) = collect_text_and_terminal(stream).await;
            let terminal = terminal.expect("stream should end with a terminal record");
            let raw = terminal
                .raw
                .as_deref()
                .expect("raw is captured when asked for");
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
    assert_terminal_reproduces_frame(&terminal, &frame, "flag on");
}

// ================================================================
// 3. Terminal-only fields the normalized record lacks
// ================================================================

#[tokio::test]
async fn stream_capture_on_exposes_terminal_usage_and_object() {
    const SCENARIO: &str =
        "raw_stream_capture_matrix/stream_capture_on_exposes_terminal_usage_and_object";
    let observed = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = observed.clone();
    with_perplexity_cassette(
        "raw_stream_capture_matrix/stream_capture_on_exposes_terminal_usage_and_object",
        |client| async move {
            let model = client.completion_model(MODEL);
            let stream = model
                .stream(request(&model, true))
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

    let raw = terminal.raw.as_deref().expect("raw is captured");
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

// ================================================================
// 4. The flag never reaches the wire
// ================================================================

#[tokio::test]
async fn stream_request_invariant_off_vs_on() {
    const SCENARIO: &str = "raw_stream_capture_matrix/stream_request_invariant_off_vs_on";
    with_perplexity_cassette(
        "raw_stream_capture_matrix/stream_request_invariant_off_vs_on",
        |client| async move {
            let model = client.completion_model(MODEL);
            let (_, off) = collect_text_and_terminal(
                model
                    .stream(request(&model, false))
                    .await
                    .expect("the stream should open"),
            )
            .await;
            pause_between_live_turns().await;
            let (_, on) = collect_text_and_terminal(
                model
                    .stream(request(&model, true))
                    .await
                    .expect("the stream should open"),
            )
            .await;
            assert!(off.expect("off terminal").raw.is_none());
            assert!(on.expect("on terminal").raw.is_some());
        },
    )
    .await;

    let interactions = crate::cassettes::recorded_interaction_bodies(PROVIDER, SCENARIO);
    assert_eq!(interactions.len(), 2, "one flag-off and one flag-on stream");
    assert_eq!(
        interactions[0].0, interactions[1].0,
        "the flag-off and flag-on request bodies must be byte-identical"
    );
    assert!(!interactions[0].0.contains("capture_raw"));
}
