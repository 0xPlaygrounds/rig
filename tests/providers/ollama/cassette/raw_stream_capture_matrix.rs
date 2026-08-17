//! Matrix for raw terminal-record capture on Ollama's streaming `/api/chat`
//! path ([`StreamFinal::raw`](rig::streaming::StreamFinal::raw)).
//!
//! # The feature
//!
//! Capture is always on. The terminal record of every stream the provider seam
//! yields carries `raw`: the value the model's inherent
//! [`CompletionModel::raw_stream`](rig::providers::ollama::CompletionModel::raw_stream)
//! would have yielded as its `FinalResponse` — Ollama's terminal NDJSON record
//! as [`ollama::StreamingCompletionResponse`] carries it — serialized with
//! `serde_json::to_value` by `normalize_stream`. It is the terminal record
//! only, never the stream's frames, and nothing about it is sent to the daemon.
//! `raw == Value::Null` means only that a `StreamFinal` was built by hand
//! without a provider terminal behind it, which no cell here can produce.
//!
//! Ollama's stream is newline-delimited JSON, not SSE: every line is a chat
//! record and exactly one — the last — carries `done: true` together with the
//! token counts and the nanosecond timings. Those timings (`total_duration`,
//! `eval_duration`, …) are what cell 2 reads back: the normalized
//! [`StreamFinal`](rig::streaming::StreamFinal) has no field for them.
//!
//! # Matrix
//!
//! Recorded cells re-derive their premise from their own fixture bytes after
//! the cassette wrapper returns: the recorded stream must end with a
//! `done: true` record that reports usage, or the cell fails loudly.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `stream_raw_terminal_round_trips_provider_type` | typed access | `ollama::StreamingCompletionResponse::deserialize(&*raw)` re-serializes equal | recorded |
//! | 2 | `stream_raw_exposes_terminal_durations` | terminal-only fields | `eval_duration`/`total_duration`/`eval_count` in `raw` equal the fixture's `done: true` line | recorded |
//!
//! Every cell is recorded: Ollama runs locally with no credential.
//!
//! Re-record with a local Ollama daemon serving `qwen3:4b`:
//! `RIG_PROVIDER_TEST_MODE=record cargo test -p rig --all-features --test ollama ollama::cassette::raw_stream_capture_matrix -- --nocapture --test-threads=1`

use futures::StreamExt;
use rig::completion::CompletionModel as _;
use rig::prelude::*;
use rig::providers::ollama;
use rig::streaming::{StreamFinal, StreamedAssistantContent};
use serde::Deserialize;
use serde_json::{Value, json};

use super::super::support::with_ollama_cassette;
use crate::cassettes::recorded_interaction_bodies;

const OLLAMA_PROVIDER: &str = "ollama";
const MODEL: &str = "qwen3:4b";
const PROMPT: &str = "Reply with exactly the single word: pong";

fn request(model: &ollama::CompletionModel) -> rig::completion::CompletionRequest {
    model
        .completion_request(PROMPT)
        .max_tokens(64)
        .additional_params(json!({ "think": false }))
        .build()
}

/// Drains a stream and returns the single terminal record it yielded.
async fn terminal_of(mut stream: rig::streaming::StreamingCompletionResponse) -> StreamFinal {
    let mut finals = Vec::new();
    while let Some(item) = stream.next().await {
        if let StreamedAssistantContent::Final(record) = item.expect("stream item should be ok") {
            finals.push(record);
        }
    }
    assert_eq!(
        finals.len(),
        1,
        "stream should yield exactly one terminal record"
    );
    finals.remove(0)
}

/// The premise every streaming cell rests on: the scenario recorded exactly
/// one interaction whose NDJSON body ends with a `done: true` record carrying
/// token counts and timings. Returns that terminal line parsed.
fn recorded_terminal_line(scenario: &str) -> Value {
    let bodies = recorded_interaction_bodies(OLLAMA_PROVIDER, scenario);
    assert_eq!(
        bodies.len(),
        1,
        "{scenario}: the scenario must record exactly one interaction"
    );
    let (_, response) = &bodies[0];
    let last = response
        .lines()
        .map(str::trim)
        .rfind(|line| !line.is_empty())
        .unwrap_or_else(|| panic!("{scenario}: recorded stream body should not be empty"));
    let terminal: Value = serde_json::from_str(last)
        .unwrap_or_else(|err| panic!("{scenario}: terminal NDJSON line should be JSON: {err}"));
    assert_eq!(
        terminal.get("done"),
        Some(&Value::Bool(true)),
        "{scenario}: the recorded stream must end with a `done: true` record"
    );
    for field in [
        "eval_count",
        "prompt_eval_count",
        "total_duration",
        "eval_duration",
    ] {
        assert!(
            terminal.get(field).and_then(Value::as_u64).is_some(),
            "{scenario}: the terminal record must report `{field}` — without it \
             the terminal carries no usage and this cell proves nothing"
        );
    }
    terminal
}

// ---------------------------------------------------------------------------
// 1: raw is the raw_stream FinalResponse, serialized
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stream_raw_terminal_round_trips_provider_type() {
    let scenario = "raw_stream_capture_matrix/stream_raw_terminal_round_trips_provider_type";
    let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = std::sync::Arc::clone(&captured);
    with_ollama_cassette(
        "raw_stream_capture_matrix/stream_raw_terminal_round_trips_provider_type",
        |client| async move {
            let model = client.completion_model(MODEL);
            let stream = model
                .stream(request(&model))
                .await
                .expect("stream should start");
            let terminal = terminal_of(stream).await;

            let raw = &terminal.raw;
            let typed = ollama::StreamingCompletionResponse::deserialize(raw)
                .expect("raw must deserialize into ollama::StreamingCompletionResponse");
            assert_eq!(
                serde_json::to_value(&typed).expect("terminal type should serialize"),
                *raw,
                "ollama::StreamingCompletionResponse must round-trip through its own serde"
            );

            // The typed terminal agrees with the normalized one: raw is the
            // record normalize_stream mapped, not a divergent copy.
            assert_eq!(Some(typed.model.as_str()), terminal.model.as_deref());
            assert_eq!(
                typed.eval_count.unwrap_or_default(),
                terminal.usage.output_tokens,
                "normalized output tokens come from the raw eval_count"
            );
            assert_eq!(
                typed.prompt_eval_count.unwrap_or_default(),
                terminal.usage.input_tokens,
                "normalized input tokens come from the raw prompt_eval_count"
            );
            *sink.lock().expect("capture mutex") = Some(raw.clone());
        },
    )
    .await;

    // Premise: the wire's terminal line is what raw carries.
    let terminal_line = recorded_terminal_line(scenario);
    let raw = captured
        .lock()
        .expect("capture mutex")
        .take()
        .expect("the test body must have captured raw");
    assert_eq!(raw["eval_count"], terminal_line["eval_count"]);
    assert_eq!(raw["prompt_eval_count"], terminal_line["prompt_eval_count"]);
    assert_eq!(raw["done_reason"], terminal_line["done_reason"]);
}

// ---------------------------------------------------------------------------
// 2: terminal-only fields
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stream_raw_exposes_terminal_durations() {
    let scenario = "raw_stream_capture_matrix/stream_raw_exposes_terminal_durations";
    let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = std::sync::Arc::clone(&captured);
    with_ollama_cassette(
        "raw_stream_capture_matrix/stream_raw_exposes_terminal_durations",
        |client| async move {
            let model = client.completion_model(MODEL);
            let stream = model
                .stream(request(&model))
                .await
                .expect("stream should start");
            let terminal = terminal_of(stream).await;

            // The normalized terminal record provably lacks the timings.
            let mut without_raw = terminal.clone();
            without_raw.raw = Value::Null;
            let normalized =
                serde_json::to_value(&without_raw).expect("StreamFinal should serialize");
            for field in ["total_duration", "eval_duration", "load_duration"] {
                assert!(
                    normalized.get(field).is_none(),
                    "normalized StreamFinal must not grow a `{field}` field"
                );
            }

            let raw = terminal.raw.clone();
            *sink.lock().expect("capture mutex") = Some(raw);
        },
    )
    .await;

    let raw = captured
        .lock()
        .expect("capture mutex")
        .take()
        .expect("the test body must have captured raw");
    let terminal_line = recorded_terminal_line(scenario);
    for field in [
        "total_duration",
        "load_duration",
        "prompt_eval_duration",
        "eval_duration",
        "eval_count",
        "prompt_eval_count",
    ] {
        assert_eq!(
            raw.get(field),
            terminal_line.get(field),
            "raw.{field} must equal the recorded terminal record's value"
        );
    }
    let typed = ollama::StreamingCompletionResponse::deserialize(&raw)
        .expect("raw must deserialize into ollama::StreamingCompletionResponse");
    assert_eq!(typed.eval_duration, terminal_line["eval_duration"].as_u64());
    assert_eq!(
        typed.total_duration,
        terminal_line["total_duration"].as_u64()
    );
}
