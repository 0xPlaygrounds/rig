//! Raw provider response capture on Venice's streaming chat-completions
//! path.
//!
//! **The feature.** With `CompletionRequest::capture_raw_response` on, the
//! stream's terminal [`rig::streaming::StreamFinal::raw`] carries the value
//! the model's inherent `raw_stream` would have yielded as its terminal
//! record — for Venice the shared chat-completions terminal
//! [`StreamingCompletionResponse`] over the shared [`openai::Usage`] —
//! serialized. It is the terminal record only, never the stream's frames.
//! Venice stamps the request's `cost` on the terminal frame alone, and the
//! terminal's accumulated `additional_params` keeps it, alongside the
//! `object` tag the frames repeat; neither has a slot on the normalized
//! terminal, so both are pinned here as reachable only through `raw`.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `stream_capture_off_leaves_raw_none` | flag off (the default) | terminal `raw == None` | recorded |
//! | 2 | `stream_capture_on_round_trips_terminal_type` | flag on | terminal `raw` deserializes into `StreamingCompletionResponse<openai::Usage>` and re-serializes equal | recorded |
//! | 3 | `stream_capture_on_exposes_terminal_cost` | terminal-only field | `raw.additional_params.cost.usd` equals the recorded terminal frame's, and no earlier frame carried a cost | recorded |
//! | 4 | `stream_request_invariant_off_vs_on` | on-wire request | the flag-off and flag-on request bodies are byte-identical | recorded |
//!
//! Every cell is recorded. The premise every cell re-derives from its own
//! fixture is that usage appears on exactly one frame — the stream's last
//! data frame — so the raw terminal record's usage is knowable from the bytes
//! and a recording whose stream stopped reporting usage fails loudly instead
//! of covering nothing. Venice contracts no request-id header, so the
//! terminal's `provider_request_id` is `None` — pinned as the documented
//! outcome.

use rig::completion::{CompletionModel, CompletionRequest};
use rig::prelude::*;
use rig::providers::openai::completion::streaming::StreamingCompletionResponse;
use rig::providers::venice::completion::VeniceParameters;
use rig::providers::{openai, venice};
use rig::streaming::StreamFinal;
use serde::Deserialize;
use serde_json::{Value, json};

use super::super::DEFAULT_MODEL;
use super::super::support::{assert_matches_recorded_token, with_venice_cassette_result};
use crate::support::collect_text_and_terminal;

type VeniceTerminal = StreamingCompletionResponse<openai::Usage>;

const PROVIDER: &str = "venice";
const PROMPT: &str = "Reply with the single word: pong";

fn request(model: &venice::CompletionModel, capture: bool) -> CompletionRequest {
    model
        .completion_request(PROMPT)
        .max_tokens(16)
        .additional_params(
            VeniceParameters::new()
                .disable_thinking(true)
                .into_additional_params(),
        )
        .capture_raw_response(capture)
        .build()
}

/// The single recorded frame that carries usage — the stream's last data
/// frame.
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
    terminal.clone()
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
        "{context}: Venice contracts no id header"
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
    with_venice_cassette_result(
        "raw_stream_capture_matrix/stream_capture_off_leaves_raw_none",
        |client| async move {
            let model = client.completion_model(DEFAULT_MODEL);
            let stream = model.stream(request(&model, false)).await?;
            let (text, terminal) = collect_text_and_terminal(stream).await;
            let terminal = terminal.expect("stream should end with a terminal record");
            assert!(
                terminal.raw.is_none(),
                "raw must stay None unless asked for"
            );
            assert!(!text.is_empty());
            *sink.lock().expect("observation lock") = Some(terminal);
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("stream_capture_off_leaves_raw_none should replay from its cassette");

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
    with_venice_cassette_result(
        "raw_stream_capture_matrix/stream_capture_on_round_trips_terminal_type",
        |client| async move {
            let model = client.completion_model(DEFAULT_MODEL);
            let stream = model.stream(request(&model, true)).await?;
            let (_, terminal) = collect_text_and_terminal(stream).await;
            let terminal = terminal.expect("stream should end with a terminal record");
            let raw = terminal
                .raw
                .as_deref()
                .expect("raw is captured when asked for");
            let typed = VeniceTerminal::deserialize(raw)
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
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("stream_capture_on_round_trips_terminal_type should replay from its cassette");

    let terminal = observed
        .lock()
        .expect("observation lock")
        .take()
        .expect("the cell should observe a terminal record");
    let frame = recorded_terminal_frame(SCENARIO);
    assert_terminal_reproduces_frame(&terminal, &frame, "flag on");
}

// ================================================================
// 3. A terminal-only field the normalized record lacks
// ================================================================

#[tokio::test]
async fn stream_capture_on_exposes_terminal_cost() {
    const SCENARIO: &str = "raw_stream_capture_matrix/stream_capture_on_exposes_terminal_cost";
    let observed = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = observed.clone();
    with_venice_cassette_result(
        "raw_stream_capture_matrix/stream_capture_on_exposes_terminal_cost",
        |client| async move {
            let model = client.completion_model(DEFAULT_MODEL);
            let stream = model.stream(request(&model, true)).await?;
            let (_, terminal) = collect_text_and_terminal(stream).await;
            *sink.lock().expect("observation lock") =
                Some(terminal.expect("stream should end with a terminal record"));
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("stream_capture_on_exposes_terminal_cost should replay from its cassette");

    let terminal = observed
        .lock()
        .expect("observation lock")
        .take()
        .expect("the cell should observe a terminal record");
    let frame = recorded_terminal_frame(SCENARIO);
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

    let raw = terminal.raw.as_deref().expect("raw is captured");
    assert_eq!(
        raw["additional_params"]["cost"]["usd"],
        json!(recorded_cost)
    );
    assert_eq!(
        raw["additional_params"]["object"],
        json!("chat.completion.chunk")
    );
    // The normalized terminal has no slot for either.
    let normalized = serde_json::to_value(&terminal).expect("terminal serializes");
    assert!(normalized.get("cost").is_none() && normalized.get("additional_params").is_none());
}

// ================================================================
// 4. The flag never reaches the wire
// ================================================================

#[tokio::test]
async fn stream_request_invariant_off_vs_on() {
    const SCENARIO: &str = "raw_stream_capture_matrix/stream_request_invariant_off_vs_on";
    with_venice_cassette_result(
        "raw_stream_capture_matrix/stream_request_invariant_off_vs_on",
        |client| async move {
            let model = client.completion_model(DEFAULT_MODEL);
            let (_, off) =
                collect_text_and_terminal(model.stream(request(&model, false)).await?).await;
            let (_, on) =
                collect_text_and_terminal(model.stream(request(&model, true)).await?).await;
            assert!(off.expect("off terminal").raw.is_none());
            assert!(on.expect("on terminal").raw.is_some());
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("stream_request_invariant_off_vs_on should replay from its cassette");

    let interactions = crate::cassettes::recorded_interaction_bodies(PROVIDER, SCENARIO);
    assert_eq!(interactions.len(), 2, "one flag-off and one flag-on stream");
    assert_eq!(
        interactions[0].0, interactions[1].0,
        "the flag-off and flag-on request bodies must be byte-identical"
    );
    assert!(!interactions[0].0.contains("capture_raw"));
}
