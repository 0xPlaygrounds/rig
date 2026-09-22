//! Raw provider response capture on DeepSeek's streaming chat-completions
//! path.
//!
//! **The feature.** Every stream's terminal
//! [`rig::streaming::StreamFinal::raw`] carries the decoder's own terminal
//! record — for DeepSeek the shared chat-completions
//! [`StreamingCompletionResponse`] over [`ChatUsage`] — serialized. Capture
//! is always on: there is no flag to request it, nothing about it reaches the
//! wire, and a `Value::Null` only ever means a terminal built by hand with no
//! provider record behind it. It is the terminal record only, never the
//! stream's frames.
//!
//! [`ChatUsage`] flattens the OpenAI-compatible counters and keeps whatever
//! else the dialect sent in a sibling map, so DeepSeek's
//! `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens` split survives into
//! `raw` instead of being dropped at the type boundary. Only half of it is
//! normalized — [`ChatUsage::to_normalized`] reads the hit count into
//! `Usage::cached_input_tokens`, and the miss count has no slot at all —
//! which makes the miss count the natural terminal-only field to pin here.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `stream_raw_round_trips_terminal_type` | typed round trip | terminal `raw` deserializes into `StreamingCompletionResponse<ChatUsage>` and re-serializes equal; the normalized terminal reproduces the recorded terminal frame | recorded |
//! | 2 | `stream_raw_exposes_terminal_cache_miss_tokens` | terminal-only field | the terminal accounting reads back as [`ChatUsage`], whose `extra` carries `prompt_cache_miss_tokens` and whose `to_normalized` keeps only the hit half | recorded |
//! | 3 | `stream_reasoning_raw_round_trips_terminal_type` | reasoning turn | a thinking-mode stream's terminal `raw` round-trips the same way and reproduces the recorded terminal frame; the stream's reasoning deltas reassemble the fixture's `delta.reasoning_content` frames, none of which is on the terminal record | recorded |
//!
//! Every cell is recorded. The premise every cell re-derives from its own
//! fixture is [`chat::recorded_sole_usage_frame`]: the recorded SSE stream
//! carries usage on exactly one frame, and that frame is the stream's last
//! data frame — so the raw terminal record's usage is knowable from the bytes
//! and a recording whose stream stopped reporting usage, or started
//! reporting it somewhere other than the close, fails loudly instead of
//! covering nothing. Cell 3 additionally re-derives that the recorded frames
//! carry `delta.reasoning_content` (and that the recorded request asked for
//! thinking), so a recording that stopped reasoning fails instead of
//! covering nothing.

use rig::message::AssistantContent;

use futures::StreamExt as _;
use rig::completion::{CompletionModel, CompletionRequest};
use rig::message::ReasoningContent;
use rig::providers::deepseek;
use rig::streaming::{Delta, StreamEvent, StreamFinal};
use serde::Deserialize;
use serde_json::json;

use super::support::with_deepseek_cassette_result;
use crate::raw_capture::{assert_no_request_id, capture_terminal, capture_text_and_terminal, chat};
use crate::support::Observed;

const PROVIDER: &str = "deepseek";
const MODEL: &str = deepseek::DEEPSEEK_V4_FLASH;
const PROMPT: &str = "Reply with the single word: pong";
/// A question small enough that a thinking-mode turn reasons briefly and
/// still answers within the budget.
const REASONING_PROMPT: &str = "What is 17 multiplied by 23? Reply with only the number.";
/// A thinking-mode turn spends most of its budget on reasoning tokens before
/// it answers, so the reasoning cell needs real headroom.
const REASONING_BUDGET: u64 = 640;

fn request(model: &(impl CompletionModel + Clone)) -> CompletionRequest {
    model
        .completion_request(PROMPT)
        .additional_params(json!({ "thinking": { "type": "disabled" } }))
        .max_tokens(16)
        .build()
}

/// The thinking-mode request shape the `reasoning_*` modules use.
fn reasoning_request(model: &(impl CompletionModel + Clone)) -> CompletionRequest {
    model
        .completion_request(REASONING_PROMPT)
        .additional_params(json!({ "thinking": { "type": "enabled" } }))
        .max_tokens(REASONING_BUDGET)
        .build()
}

/// What a thinking-mode stream yields: the reasoning text (deltas, superseded
/// by a completed block when the stream restates one), the visible text, and
/// the terminal record.
struct ReasoningStreamObservation {
    reasoning: String,
    text: String,
    terminal: Option<StreamFinal>,
}

/// Stays local: the shared [`capture_text_and_terminal`] keeps the text and
/// the terminal record, and cell 3 is about a third thing the stream carried
/// — the reasoning deltas, and the completed block that supersedes them.
async fn collect_reasoning_text_and_terminal(
    mut stream: rig::streaming::StreamingCompletionResponse,
) -> ReasoningStreamObservation {
    let mut observation = ReasoningStreamObservation {
        reasoning: String::new(),
        text: String::new(),
        terminal: None,
    };
    while let Some(item) = stream.next().await {
        match item.expect("stream item should not be an error") {
            StreamEvent::BlockDelta {
                delta: Delta::Reasoning { text: reasoning },
                ..
            } => {
                observation.reasoning.push_str(&reasoning);
            }
            StreamEvent::BlockEnd {
                block: Some(AssistantContent::Reasoning(reasoning)),
                ..
            } => {
                observation.reasoning = reasoning
                    .content
                    .iter()
                    .filter_map(|content| match content {
                        ReasoningContent::Text { text, .. } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect();
            }
            StreamEvent::BlockDelta {
                delta: Delta::Text { text: chunk },
                ..
            } => observation.text.push_str(&chunk),
            StreamEvent::Final(final_record) => {
                observation.terminal = Some(final_record);
            }
            _ => {}
        }
    }
    observation
}

/// The reasoning the recorded frames spell as `delta.reasoning_content`,
/// concatenated in wire order.
fn recorded_reasoning(scenario: &str) -> String {
    crate::cassettes::recorded_sse_json_frames(PROVIDER, scenario)
        .iter()
        .filter_map(|frame| frame["choices"][0]["delta"]["reasoning_content"].as_str())
        .collect()
}

// ================================================================
// 1. raw round-trips the terminal type
// ================================================================

#[tokio::test]
async fn stream_raw_round_trips_terminal_type() {
    const SCENARIO: &str = "raw_stream_capture_matrix/stream_raw_round_trips_terminal_type";
    let sink = Observed::default();
    with_deepseek_cassette_result(
        "raw_stream_capture_matrix/stream_raw_round_trips_terminal_type",
        |client| capture_text_and_terminal(client.completion(MODEL), request, sink.clone()),
    )
    .await
    .expect("stream_raw_round_trips_terminal_type should replay from its cassette");

    let (text, terminal) = sink.take();
    assert!(!text.is_empty());
    chat::assert_terminal_round_trips(&terminal);

    let frame = chat::recorded_sole_usage_frame(PROVIDER, SCENARIO);
    chat::assert_terminal_reproduces_frame(&terminal, PROVIDER, &frame, "the recorded frame");
    assert_no_request_id(terminal.provider_request_id.as_deref(), PROVIDER);
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
    let sink = Observed::default();
    with_deepseek_cassette_result(
        "raw_stream_capture_matrix/stream_raw_exposes_terminal_cache_miss_tokens",
        |client| capture_terminal(client.completion(MODEL), request, sink.clone()),
    )
    .await
    .expect("stream_raw_exposes_terminal_cache_miss_tokens should replay from its cassette");

    let terminal = sink.take();
    let frame = chat::recorded_sole_usage_frame(PROVIDER, SCENARIO);
    let recorded_miss = frame["usage"]["prompt_cache_miss_tokens"]
        .as_u64()
        .expect("DeepSeek's terminal usage reports prompt_cache_miss_tokens");
    let recorded_hit = frame["usage"]["prompt_cache_hit_tokens"]
        .as_u64()
        .expect("DeepSeek's terminal usage reports prompt_cache_hit_tokens");

    let raw = &terminal.raw;
    // Typed, through the wire's own accounting shape: the OpenAI-compatible
    // counters are modeled and DeepSeek's split rides in `extra`.
    let usage = chat::Terminal::deserialize(raw)
        .expect("raw is the chat-completions terminal over the wire's own usage")
        .usage
        .expect("the recorded terminal carries usage");
    assert_eq!(
        usage.extra.get("prompt_cache_miss_tokens"),
        Some(&json!(recorded_miss))
    );
    assert_eq!(
        usage.extra.get("prompt_cache_hit_tokens"),
        Some(&json!(recorded_hit))
    );
    // The normalized terminal keeps the hit count (as cached input) and has
    // no slot for the miss count: `to_normalized` is where exactly half of
    // the split crosses over.
    assert_eq!(
        usage.to_normalized().cached_input_tokens,
        Some(recorded_hit)
    );
    assert_eq!(terminal.usage.cached_input_tokens, Some(recorded_hit));
    let normalized_usage = serde_json::to_value(terminal.usage).expect("usage serializes");
    assert!(
        normalized_usage.get("prompt_cache_miss_tokens").is_none(),
        "the normalized usage has no miss-count slot: {normalized_usage}"
    );
}

// ================================================================
// 3. A thinking-mode stream: the terminal round-trips, reasoning stays on
//    the frames
// ================================================================

#[tokio::test]
async fn stream_reasoning_raw_round_trips_terminal_type() {
    const SCENARIO: &str =
        "raw_stream_capture_matrix/stream_reasoning_raw_round_trips_terminal_type";
    let observed = Observed::default();
    let sink = observed.clone();
    with_deepseek_cassette_result(
        "raw_stream_capture_matrix/stream_reasoning_raw_round_trips_terminal_type",
        |client| async move {
            let model = client.completion(MODEL);
            let stream = model.stream(reasoning_request(&model)).await?;
            sink.put(collect_reasoning_text_and_terminal(stream).await);
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("stream_reasoning_raw_round_trips_terminal_type should replay from its cassette");

    let observation = observed.take();
    let terminal = observation
        .terminal
        .as_ref()
        .expect("the cell should observe a terminal record");
    chat::assert_terminal_round_trips(terminal);
    let frame = chat::recorded_sole_usage_frame(PROVIDER, SCENARIO);
    chat::assert_terminal_reproduces_frame(terminal, PROVIDER, &frame, "the recorded frame");
    assert_no_request_id(terminal.provider_request_id.as_deref(), PROVIDER);
    let request_body = crate::cassettes::recorded_json_request(PROVIDER, SCENARIO);
    assert_eq!(request_body["stream"], json!(true));
    assert_eq!(request_body["thinking"], json!({ "type": "enabled" }));

    // Premise, from the bytes: the recorded frames carry reasoning deltas,
    // and the stream reassembled exactly them.
    let recorded_reasoning = recorded_reasoning(SCENARIO);
    assert!(
        !recorded_reasoning.trim().is_empty(),
        "a thinking-mode stream carries delta.reasoning_content frames"
    );
    assert_eq!(
        observation.reasoning, recorded_reasoning,
        "the stream's reasoning is the recorded reasoning_content deltas in wire order"
    );
    assert!(!observation.text.is_empty(), "the turn should still answer");
    // The reasoning lives on the frames, not the terminal record: raw is the
    // terminal record only.
    assert!(
        terminal.raw.get("reasoning_content").is_none() && terminal.raw.get("choices").is_none(),
        "the terminal raw carries no frame content: {}",
        terminal.raw
    );
}
