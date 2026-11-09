//! Raw provider response capture on DeepSeek's streaming chat-completions
//! path.
//!
//! Every stream's terminal [`rig::completion::CompletionResponse::raw`]
//! carries the `chat.completion` document the stream's chunks rebuild, the
//! shape a unary DeepSeek reply has. Capture is always on, and a `Value::Null`
//! only means a terminal built by hand. The document's `usage` is the
//! provider's usage object, so DeepSeek's `prompt_cache_miss_tokens`, which
//! has no normalized slot, is the terminal-only field the cells pin.
//!
//! Every cell re-derives [`chat::recorded_sole_usage_frame`] from its fixture:
//! the recorded stream carries usage on exactly one frame, its last data
//! frame, so the rebuilt document's usage is knowable from the bytes. The
//! reasoning cell also re-derives that the frames carry
//! `delta.reasoning_content`, which the document states as the message's
//! `reasoning_content`.

use rig::message::AssistantContent;
use rig::streaming::Item;

use futures::StreamExt as _;
use rig::completion::CompletionRequest;
use rig::providers::deepseek;
use rig::streaming::StreamEvent;
use serde_json::json;

use super::support::with_deepseek_cassette_result;
use crate::raw_capture::{assert_no_request_id, chat};
use crate::support::Observed;

const PROVIDER: &str = "deepseek";
const MODEL: &str = deepseek::DEEPSEEK_V4_FLASH;
/// A question small enough that a thinking-mode turn reasons briefly and
/// still answers within the budget.
const REASONING_PROMPT: &str = "What is 17 multiplied by 23? Reply with only the number.";
/// A thinking-mode turn spends most of its budget on reasoning tokens before
/// it answers, so the reasoning cell needs real headroom.
const REASONING_BUDGET: u64 = 640;

/// The thinking-mode request shape the `reasoning_*` modules use.
fn reasoning_request() -> CompletionRequest {
    CompletionRequest::new(REASONING_PROMPT)
        .additional_params(json!({ "thinking": { "type": "enabled" } }))
        .max_tokens(REASONING_BUDGET)
}

/// What a thinking-mode stream yields: the reasoning text (deltas, superseded
/// by a completed block when the stream restates one), the visible text, and
/// the terminal record.
struct ReasoningStreamObservation {
    reasoning: String,
    text: String,
    terminal: Option<rig::completion::CompletionResponse>,
}

/// Stays local: the shared [`capture_text_and_terminal`] keeps the text and
/// the terminal record, and cell 3 is about a third thing the stream carried
/// — the reasoning deltas, and the completed block that supersedes them.
async fn collect_reasoning_text_and_terminal(
    mut stream: rig::streaming::CompletionStream,
) -> ReasoningStreamObservation {
    let mut observation = ReasoningStreamObservation {
        reasoning: String::new(),
        text: String::new(),
        terminal: None,
    };
    while let Some(item) = stream.next().await {
        match item.expect("stream item should not be an error") {
            Item::Event(StreamEvent::Reasoning {
                text: reasoning, ..
            }) => {
                observation.reasoning.push_str(&reasoning);
            }
            Item::Event(StreamEvent::End {
                content: AssistantContent::Reasoning(reasoning),
                ..
            }) => {
                observation.reasoning = reasoning.text;
            }
            Item::Event(StreamEvent::Text { text: chunk, .. }) => observation.text.push_str(&chunk),
            _ => {}
        }
    }
    observation.terminal = Some(
        stream
            .finish()
            .await
            .expect("the stream should yield a terminal record"),
    );
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
// 1. raw is the rebuilt document
// ================================================================

// ================================================================
// 2. A terminal-only field the normalized record lacks
// ================================================================

// ================================================================
// 3. A thinking-mode stream: raw is the rebuilt document, its reasoning
//    where a unary body has it
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
            let stream = model.stream(reasoning_request())?;
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
    // The document states the reasoning where a unary body does.
    assert_eq!(
        terminal.raw["choices"][0]["message"]["reasoning_content"],
        json!(recorded_reasoning),
        "the rebuilt message carries the reasoning: {}",
        terminal.raw
    );
}
