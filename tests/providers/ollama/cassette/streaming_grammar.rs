//! Canonical streaming-grammar coverage for Ollama's native chat wire,
//! asserted through the *normalized* path: the aggregated
//! [`StreamingCompletionResponse::choice`], the terminal [`StreamFinal`]
//! record, usage, and finish reason — real recorded wire traffic, not
//! synthetic chunks.
//!
//! Re-record with (local Ollama daemon with `qwen3:4b` pulled, no key needed):
//! `RIG_PROVIDER_TEST_MODE=record cargo test --test ollama streaming_grammar -- --test-threads=1`
//!
//! Ollama's native wire ships tool calls without ids — its `ToolCall` type has
//! no id field at all — and this provider derives each call's identity from
//! the function name rather than minting a `tool-{index}` identity (the mint
//! is the chat-compat adapter's fallback and is pinned by the synthetic
//! `id_less_parallel_tool_calls_assemble_distinct_on_the_chat_wire` corpus
//! scenario). What these recordings pin against real traffic is the property
//! that matters downstream: parallel calls on an id-less wire stay distinct
//! and assemble with uncorrupted arguments.

use futures::StreamExt;
use rig::OneOrMany;
use rig::completion::{CompletionModel, FinishReason};
use rig::message::{AssistantContent, Reasoning, ToolCall};
use rig::prelude::*;
use rig::streaming::{StreamFinal, StreamedAssistantContent};

use super::super::support::with_ollama_cassette;
use crate::support::{
    AlphaSignal, BetaSignal, ORDERED_TOOL_STREAM_PREAMBLE, ORDERED_TOOL_STREAM_PROMPT,
    TWO_TOOL_STREAM_PREAMBLE,
};

const MODEL: &str = "qwen3:4b";

struct StreamRun {
    text: String,
    reasoning_blocks: Vec<Reasoning>,
    reasoning_delta: String,
    tool_calls: Vec<ToolCall>,
    finals: Vec<StreamFinal>,
    choice: OneOrMany<AssistantContent>,
    response: Option<StreamFinal>,
}

async fn drain_stream(mut stream: rig::streaming::StreamingCompletionResponse) -> StreamRun {
    let mut run = StreamRun {
        text: String::new(),
        reasoning_blocks: Vec::new(),
        reasoning_delta: String::new(),
        tool_calls: Vec::new(),
        finals: Vec::new(),
        choice: OneOrMany::one(AssistantContent::text("")),
        response: None,
    };

    while let Some(item) = stream.next().await {
        match item.expect("stream item should be ok") {
            StreamedAssistantContent::Text(text) => run.text.push_str(&text.text),
            StreamedAssistantContent::Reasoning(reasoning) => run.reasoning_blocks.push(reasoning),
            StreamedAssistantContent::ReasoningDelta { reasoning, .. } => {
                run.reasoning_delta.push_str(&reasoning);
            }
            StreamedAssistantContent::ToolCall { tool_call, .. } => run.tool_calls.push(tool_call),
            StreamedAssistantContent::Final(response) => run.finals.push(response),
            _ => {}
        }
    }

    run.choice = stream.choice.clone();
    run.response = stream.response.clone();
    run
}

fn assert_terminal(run: &StreamRun, expected_finish: FinishReason) {
    assert_eq!(
        run.finals.len(),
        1,
        "stream should yield exactly one terminal record"
    );
    let terminal = run
        .response
        .as_ref()
        .expect("aggregated stream should retain the terminal record");
    assert_eq!(
        terminal.finish_reason.as_ref(),
        Some(&expected_finish),
        "unexpected finish reason"
    );
    assert!(
        terminal.usage.total_tokens > 0,
        "terminal record should carry non-zero usage, got {:?}",
        terminal.usage
    );
}

/// Thinking and a tool call in ONE stream (`think: true`): the reasoning part
/// and the tool call survive aggregation as discrete siblings.
#[tokio::test]
async fn thinking_and_tool_call_in_one_stream() {
    with_ollama_cassette(
        "streaming_grammar/thinking_and_tool_call",
        |client| async move {
            let model = client.completion_model(MODEL);
            let request = model
                .completion_request(ORDERED_TOOL_STREAM_PROMPT)
                .preamble(ORDERED_TOOL_STREAM_PREAMBLE.to_string())
                .tool(rig::tool::tool_definition(&AlphaSignal))
                .additional_params(serde_json::json!({ "think": true }))
                .build();
            let run = drain_stream(model.stream(request).await.expect("stream should start")).await;

            assert_terminal(&run, FinishReason::ToolCalls);
            assert!(
                !run.reasoning_delta.is_empty() || !run.reasoning_blocks.is_empty(),
                "think:true should surface reasoning on the stream"
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
            assert!(
                !streamed.id.is_empty(),
                "id-less wire calls must carry a minted, non-empty grammar id"
            );
        },
    )
    .await;
}

/// Parallel tool calls on an **id-less** wire: both calls keep distinct minted
/// identities and uncorrupted arguments — the 2258 item-0 collapse pin against
/// real traffic.
#[tokio::test]
async fn parallel_id_less_tool_calls_stay_distinct() {
    with_ollama_cassette(
        "streaming_grammar/parallel_tool_calls",
        |client| async move {
            let model = client.completion_model(MODEL);
            let request = model
                .completion_request(
                    "Call `lookup_harbor_label` and `lookup_orchard_label` now, both of them \
                     together in this single reply, before writing any text. Emit the two tool \
                     calls in one turn — do not wait for results between them.",
                )
                .preamble(TWO_TOOL_STREAM_PREAMBLE.to_string())
                .tool(rig::tool::tool_definition(&AlphaSignal))
                .tool(rig::tool::tool_definition(&BetaSignal))
                .additional_params(serde_json::json!({ "think": false }))
                .build();
            let run = drain_stream(model.stream(request).await.expect("stream should start")).await;

            assert_terminal(&run, FinishReason::ToolCalls);
            let aggregated: Vec<&ToolCall> = run
                .choice
                .iter()
                .filter_map(|content| match content {
                    AssistantContent::ToolCall(call) => Some(call),
                    _ => None,
                })
                .collect();
            for name in ["lookup_harbor_label", "lookup_orchard_label"] {
                let streamed = run
                    .tool_calls
                    .iter()
                    .find(|call| call.function.name == name)
                    .unwrap_or_else(|| panic!("stream should yield a {name} call"));
                let aggregated_call = aggregated
                    .iter()
                    .find(|call| call.function.name == name)
                    .unwrap_or_else(|| panic!("aggregated choice should keep the {name} call"));
                assert_eq!(
                    aggregated_call.id, streamed.id,
                    "{name} id should aggregate"
                );
                assert!(
                    !streamed.id.is_empty(),
                    "{name} must carry a minted, non-empty grammar id"
                );
                assert!(
                    streamed.function.arguments.is_object(),
                    "{name} arguments must assemble into an object, got {:?}",
                    streamed.function.arguments
                );
            }
            // The model may emit each tool more than once in one turn; the
            // grammar contract is that every streamed call survives as its
            // own aggregated part — no id-less collapse.
            assert!(run.tool_calls.len() >= 2, "turn should stream both calls");
            assert_eq!(
                aggregated.len(),
                run.tool_calls.len(),
                "every streamed call should survive as its own aggregated part"
            );
            let mut ids: Vec<&str> = run.tool_calls.iter().map(|call| call.id.as_str()).collect();
            ids.sort_unstable();
            let total = ids.len();
            ids.dedup();
            assert_eq!(
                ids.len(),
                total,
                "id-less parallel calls must keep distinct minted identities"
            );
        },
    )
    .await;
}
