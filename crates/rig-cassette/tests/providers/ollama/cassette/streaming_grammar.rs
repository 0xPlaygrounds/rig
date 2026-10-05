//! Canonical streaming-grammar coverage for Ollama's native chat wire,
//! asserted through the *normalized* path: the aggregated
//! [`Streamed::finish`](rig::streaming::Streamed::finish) response, the terminal `CompletionResponse`
//! record, usage, and finish reason — real recorded wire traffic, not
//! synthetic chunks.
//!
//! Re-record with (local Ollama daemon with `qwen3:4b` pulled, no key needed):
//! `RIG_PROVIDER_TEST_MODE=record cargo test --test ollama streaming_grammar -- --test-threads=1`
//!
//! Modern Ollama daemons issue tool-call ids (`"id":"call_..."`); rig reads
//! them as the durable id and falls back to a minted `tool-{index}` identity
//! only when a daemon omits them (the id-less mint stays pinned by the
//! synthetic `id_less_parallel_tool_calls_assemble_distinct_on_the_chat_wire`
//! corpus scenario). What these recordings pin against real traffic is the
//! property that matters downstream: parallel calls stay distinct — by
//! daemon id and by structure — and assemble with uncorrupted arguments.

use futures::StreamExt;
use rig::completion::CompletionResponse;
use rig::completion::FinishReason;
use rig::message::{AssistantContent, Reasoning, ToolCall};
use rig::streaming::Item;
use rig::streaming::StreamEvent;

use super::super::support::with_ollama_cassette;
use crate::support::{AlphaSignal, ORDERED_TOOL_STREAM_PREAMBLE, ORDERED_TOOL_STREAM_PROMPT};
use rig::completion::CompletionRequest;

const MODEL: &str = "qwen3:4b";

struct StreamRun {
    text: String,
    reasoning_blocks: Vec<Reasoning>,
    reasoning_delta: String,
    tool_calls: Vec<ToolCall>,
    choice: Vec<AssistantContent>,
    response: Option<CompletionResponse>,
}

async fn drain_stream(mut stream: rig::streaming::CompletionStream) -> StreamRun {
    let mut run = StreamRun {
        text: String::new(),
        reasoning_blocks: Vec::new(),
        reasoning_delta: String::new(),
        tool_calls: Vec::new(),
        choice: vec![AssistantContent::text("")],
        response: None,
    };

    let mut raw_items = Vec::new();
    while let Some(item) = stream.next().await {
        let item = item.expect("stream item should be ok");
        raw_items.push(Ok(item.clone()));
        match item {
            Item::Event(StreamEvent::Text { text, .. }) => run.text.push_str(&text),
            Item::Event(StreamEvent::End {
                content: AssistantContent::Reasoning(reasoning),
                ..
            }) => {
                run.reasoning_blocks.push(reasoning);
            }
            Item::Event(StreamEvent::Reasoning { text, .. }) => {
                run.reasoning_delta.push_str(&text);
            }
            Item::Event(StreamEvent::End {
                content: AssistantContent::ToolCall(tool_call),
                ..
            }) => run.tool_calls.push(tool_call),
            _ => {}
        }
    }
    let response = stream.finish().await.expect("the stream ends");

    run.choice = response.choice.clone();
    // The shared lifecycle validator runs over every recorded turn this
    // suite drains (#2258 C1).
    rig_core::test_utils::streaming_conformance::assert_valid_event_stream(&raw_items, &run.choice);
    run.response = Some(response.clone());
    run
}

fn assert_terminal(run: &StreamRun, expected_finish: FinishReason) {
    let terminal = run
        .response
        .as_ref()
        .expect("aggregated stream should retain the terminal record");
    assert_eq!(
        terminal.finish_reason(),
        Some(expected_finish),
        "unexpected finish reason"
    );
    assert!(
        terminal.usage.total_tokens.is_some_and(|n| n > 0),
        "terminal record should carry non-zero usage, got {:?}",
        terminal.usage
    );
}

/// Thinking and a tool call in ONE stream (thinking is the daemon's default
/// for this model): the reasoning part
/// and the tool call survive aggregation as discrete siblings.
#[tokio::test]
async fn thinking_and_tool_call_in_one_stream() {
    with_ollama_cassette(
        "streaming_grammar/thinking_and_tool_call",
        |client| async move {
            let model = client.completion(MODEL);
            let request = CompletionRequest::new(ORDERED_TOOL_STREAM_PROMPT)
                .preamble(ORDERED_TOOL_STREAM_PREAMBLE.to_string())
                .tool(rig::tool::tool_definition(&AlphaSignal));
            let run = drain_stream(model.stream(request).expect("stream should start")).await;

            assert_terminal(&run, FinishReason::ToolCalls);
            assert!(
                !run.reasoning_delta.is_empty() || !run.reasoning_blocks.is_empty(),
                "a thinking model should surface reasoning on the stream"
            );
            let streamed = run
                .tool_calls
                .iter()
                .find(|call| call.function.name == "lookup_harbor_label")
                .expect("stream should yield the lookup_harbor_label call");
            // Discrete parts: reasoning and the tool call live side-by-side.
            assert!(
                run.choice
                    .iter()
                    .any(|content| matches!(content, AssistantContent::Reasoning(_))),
                "aggregated choice should keep the reasoning part, got {:?}",
                run.choice
            );
            let aggregated = run
                .choice
                .iter()
                .find_map(|content| match content {
                    AssistantContent::ToolCall(call)
                        if call.function.name == "lookup_harbor_label" =>
                    {
                        Some(call)
                    }
                    _ => None,
                })
                .expect("aggregated choice should keep the tool call");
            assert_eq!(aggregated.id, streamed.id, "id should aggregate unchanged");
            // Modern Ollama daemons issue a call id (`"id":"call_..."`);
            // rig records it as the provider id and adopts it as the
            // durable id instead of discarding it.
            let provider = streamed
                .id
                .provider()
                .expect("the daemon-issued call id must be preserved");
            assert_eq!(
                streamed.id.provider().map(|provider| provider.as_str()),
                Some(provider.as_str()),
                "the durable id adopts the daemon's call id"
            );
        },
    )
    .await;
}
