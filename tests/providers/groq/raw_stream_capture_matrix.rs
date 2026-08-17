//! Raw provider response capture on Groq's streaming chat-completions path.
//!
//! **The feature.** With `CompletionRequest::capture_raw_response` on, the
//! stream's terminal [`rig::streaming::StreamFinal::raw`] carries the value
//! the model's inherent `raw_stream` would have yielded as its terminal
//! record — for Groq the shared chat-completions terminal
//! [`StreamingCompletionResponse`] over the shared [`openai::Usage`] —
//! serialized. It is the terminal record only, never the stream's frames.
//! Groq's terminal usage carries its timing accounting (`queue_time`,
//! `prompt_time`, ...) that the normalized `Usage` has no slot for, and the
//! terminal's accumulated `additional_params` carries the `x_groq` envelope
//! and `system_fingerprint` the frames repeat; both are reachable only
//! through `raw`.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `stream_capture_off_leaves_raw_none` | flag off (the default) | terminal `raw == None` | recorded |
//! | 2 | `stream_capture_on_round_trips_terminal_type` | flag on | terminal `raw` deserializes into `StreamingCompletionResponse<openai::Usage>` and re-serializes equal | recorded |
//! | 3 | `stream_capture_on_exposes_terminal_queue_time` | terminal-only field | `raw.usage.queue_time` and `raw.additional_params.x_groq.id` equal the recorded terminal frame's | recorded |
//! | 4 | `stream_request_invariant_off_vs_on` | on-wire request | the flag-off and flag-on request bodies are byte-identical | recorded |
//!
//! Every cell is recorded. The premise every cell re-derives from its own
//! fixture is that usage appears only on the stream's closing frames (Groq
//! sends it on the finish frame and again on a trailing empty-choices frame,
//! with identical counts) and never on a content frame — so the raw terminal
//! record's usage is knowable from the bytes and a recording whose stream
//! stopped reporting usage fails loudly instead of covering nothing.

use rig::completion::{CompletionModel, CompletionRequest};
use rig::prelude::*;
use rig::providers::openai::completion::streaming::StreamingCompletionResponse;
use rig::providers::{groq, openai};
use rig::streaming::StreamFinal;
use serde::Deserialize;
use serde_json::{Value, json};

use super::RAW_CAPTURE_MODEL;
use super::support::{
    assert_matches_recorded_token, recorded_response_headers, with_groq_cassette_result,
};
use crate::support::collect_text_and_terminal;

type GroqTerminal = StreamingCompletionResponse<openai::Usage>;

const PROVIDER: &str = "groq";
const PROMPT: &str = "Reply with the single word: pong";

fn request(model: &groq::CompletionModel, capture: bool) -> CompletionRequest {
    model
        .completion_request(PROMPT)
        .max_tokens(16)
        .capture_raw_response(capture)
        .build()
}

/// The last recorded frame that carries usage — the terminal frame — after
/// checking that every usage-bearing frame agrees on the counts and that no
/// content frame carries usage.
fn recorded_terminal_frame(scenario: &str) -> Value {
    let frames = crate::cassettes::recorded_sse_json_frames(PROVIDER, scenario);
    let with_usage: Vec<&Value> = frames
        .iter()
        .filter(|frame| !frame["usage"].is_null())
        .collect();
    let terminal = (*with_usage
        .last()
        .expect("the recorded stream must carry usage on its terminal frame"))
    .clone();
    for frame in &with_usage {
        assert_eq!(
            frame["usage"], terminal["usage"],
            "every usage-bearing frame reports the same counts"
        );
        assert!(
            frame["choices"][0]["delta"]["content"]
                .as_str()
                .is_none_or(str::is_empty),
            "usage must not ride on a content frame: {frame}"
        );
    }
    terminal
}

fn recorded_request_id(scenario: &str) -> Option<String> {
    recorded_response_headers(scenario)[0]
        .iter()
        .find(|(name, _)| name == "x-request-id")
        .map(|(_, value)| value.clone())
}

fn assert_terminal_reproduces_frame(
    terminal: &StreamFinal,
    frame: &Value,
    request_id: Option<&str>,
    context: &str,
) {
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
    assert!(
        request_id.is_some(),
        "{context}: the recorded SSE response must carry x-request-id"
    );
    assert_matches_recorded_token(
        terminal.provider_request_id.as_deref(),
        request_id,
        &format!("{context}: request id"),
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
    with_groq_cassette_result(
        "raw_stream_capture_matrix/stream_capture_off_leaves_raw_none",
        |client| async move {
            let model = client.completion_model(RAW_CAPTURE_MODEL);
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
    assert_terminal_reproduces_frame(
        &terminal,
        &frame,
        recorded_request_id(SCENARIO).as_deref(),
        "flag off",
    );
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
    with_groq_cassette_result(
        "raw_stream_capture_matrix/stream_capture_on_round_trips_terminal_type",
        |client| async move {
            let model = client.completion_model(RAW_CAPTURE_MODEL);
            let stream = model.stream(request(&model, true)).await?;
            let (_, terminal) = collect_text_and_terminal(stream).await;
            let terminal = terminal.expect("stream should end with a terminal record");
            let raw = terminal
                .raw
                .as_deref()
                .expect("raw is captured when asked for");
            let typed = GroqTerminal::deserialize(raw)
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
    assert_terminal_reproduces_frame(
        &terminal,
        &frame,
        recorded_request_id(SCENARIO).as_deref(),
        "flag on",
    );
}

// ================================================================
// 3. Terminal-only fields the normalized record lacks
// ================================================================

#[tokio::test]
async fn stream_capture_on_exposes_terminal_queue_time() {
    const SCENARIO: &str =
        "raw_stream_capture_matrix/stream_capture_on_exposes_terminal_queue_time";
    let observed = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = observed.clone();
    with_groq_cassette_result(
        "raw_stream_capture_matrix/stream_capture_on_exposes_terminal_queue_time",
        |client| async move {
            let model = client.completion_model(RAW_CAPTURE_MODEL);
            let stream = model.stream(request(&model, true)).await?;
            let (_, terminal) = collect_text_and_terminal(stream).await;
            *sink.lock().expect("observation lock") =
                Some(terminal.expect("stream should end with a terminal record"));
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("stream_capture_on_exposes_terminal_queue_time should replay from its cassette");

    let terminal = observed
        .lock()
        .expect("observation lock")
        .take()
        .expect("the cell should observe a terminal record");
    let frame = recorded_terminal_frame(SCENARIO);
    let recorded_queue_time = frame["usage"]["queue_time"]
        .as_f64()
        .expect("Groq's terminal usage reports queue_time");
    // `x_groq` rides only on the finish frame; the accumulated terminal
    // params keep it, so its id is knowable from the frame that carried it.
    let recorded_x_groq_id = crate::cassettes::recorded_sse_json_frames(PROVIDER, SCENARIO)
        .into_iter()
        .find_map(|frame| frame["x_groq"]["id"].as_str().map(str::to_owned))
        .expect("Groq's closing frames carry an x_groq envelope with an id");

    let raw = terminal.raw.as_deref().expect("raw is captured");
    assert_eq!(raw["usage"]["queue_time"], json!(recorded_queue_time));
    assert_matches_recorded_token(
        raw["additional_params"]["x_groq"]["id"].as_str(),
        Some(recorded_x_groq_id.as_str()),
        "x_groq.id",
    );
    // The normalized terminal has no slot for either.
    let normalized_usage = serde_json::to_value(terminal.usage).expect("usage serializes");
    assert!(
        normalized_usage.get("queue_time").is_none(),
        "the normalized usage has no timing slot: {normalized_usage}"
    );
    let normalized = serde_json::to_value(&terminal).expect("terminal serializes");
    assert!(normalized.get("x_groq").is_none() && normalized.get("additional_params").is_none());
}

// ================================================================
// 4. The flag never reaches the wire
// ================================================================

#[tokio::test]
async fn stream_request_invariant_off_vs_on() {
    const SCENARIO: &str = "raw_stream_capture_matrix/stream_request_invariant_off_vs_on";
    with_groq_cassette_result(
        "raw_stream_capture_matrix/stream_request_invariant_off_vs_on",
        |client| async move {
            let model = client.completion_model(RAW_CAPTURE_MODEL);
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
