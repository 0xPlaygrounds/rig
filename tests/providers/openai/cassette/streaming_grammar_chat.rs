//! Canonical streaming-grammar coverage for the OpenAI **chat-completions**
//! wire (the compat family's canonical wire), asserted through the
//! *normalized* path: the aggregated [`StreamingCompletionResponse::choice`],
//! the terminal [`StreamFinal`] record, usage, IDs, and finish reason — real
//! recorded wire traffic, not synthetic chunks.
//!
//! Re-record with:
//! `RIG_PROVIDER_TEST_MODE=record OPENAI_API_KEY=... cargo test --test openai streaming_grammar_chat -- --test-threads=1`
//!
//! Cassette IDs are scrub placeholders (`call_`/`chatcmpl-` prefixes are
//! preserved); assertions derive expected IDs from the recorded turn and never
//! mint literal IDs.

use futures::StreamExt;
use rig::OneOrMany;
use rig::completion::{CompletionModel, FinishReason};
use rig::message::{AssistantContent, ToolCall};
use rig::prelude::*;
use rig::providers::openai;
use rig::streaming::{StreamFinal, StreamedAssistantContent};
use serde_json::json;

use super::super::support::with_openai_completions_cassette;
use crate::support::{AlphaSignal, BetaSignal, TWO_TOOL_STREAM_PREAMBLE, TWO_TOOL_STREAM_PROMPT};

/// Everything observed while draining a normalized stream, alongside the
/// aggregated stream state itself.
struct StreamRun {
    text: String,
    text_chunks: usize,
    tool_calls: Vec<ToolCall>,
    finals: Vec<StreamFinal>,
    choice: OneOrMany<AssistantContent>,
    response: Option<StreamFinal>,
}

async fn drain_stream(mut stream: rig::streaming::StreamingCompletionResponse) -> StreamRun {
    let mut run = StreamRun {
        text: String::new(),
        text_chunks: 0,
        tool_calls: Vec::new(),
        finals: Vec::new(),
        choice: OneOrMany::one(AssistantContent::text("")),
        response: None,
    };

    let mut raw_items = Vec::new();
    while let Some(item) = stream.next().await {
        let item = item.expect("stream item should be ok");
        raw_items.push(Ok(item.clone()));
        match item {
            StreamedAssistantContent::Text(text) => {
                run.text.push_str(&text.text);
                run.text_chunks += 1;
            }
            StreamedAssistantContent::ToolCall { tool_call, .. } => run.tool_calls.push(tool_call),
            StreamedAssistantContent::Final(response) => run.finals.push(response),
            _ => {}
        }
    }

    run.choice = stream.choice.clone();
    // The shared lifecycle validator runs over every recorded turn this
    // suite drains (#2258 C1).
    rig_core::test_utils::streaming_conformance::assert_valid_event_stream(&raw_items, &run.choice);
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

fn aggregated_tool_calls(choice: &OneOrMany<AssistantContent>) -> Vec<&ToolCall> {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::ToolCall(tool_call) => Some(tool_call),
            _ => None,
        })
        .collect()
}

fn aggregated_text(choice: &OneOrMany<AssistantContent>) -> String {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect()
}

/// Parallel tool calls on the chat wire: both indexed calls survive into the
/// aggregated choice as distinct parts with distinct wire ids and uncorrupted
/// arguments (the item-0 identity contract on the compat family's canonical
/// wire, here with real `call_*` ids).
#[tokio::test]
async fn parallel_tool_calls_stay_distinct() {
    with_openai_completions_cassette(
        "streaming_grammar_chat/parallel_tool_calls",
        |client| async move {
            let model = client.completion_model(openai::GPT_4O);
            let request = model
                .completion_request(TWO_TOOL_STREAM_PROMPT)
                .preamble(TWO_TOOL_STREAM_PREAMBLE.to_string())
                .tool(rig::tool::tool_definition(&AlphaSignal))
                .tool(rig::tool::tool_definition(&BetaSignal))
                .additional_params(json!({ "parallel_tool_calls": true }))
                .build();
            let run = drain_stream(model.stream(request).await.expect("stream should start")).await;

            assert_terminal(&run, FinishReason::ToolCalls);
            let aggregated = aggregated_tool_calls(&run.choice);
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
                // IDs derived from the recorded turn, never minted literally.
                assert_eq!(
                    aggregated_call.id, streamed.id,
                    "{name} id should aggregate"
                );
                assert!(!streamed.id.is_empty(), "{name} should carry a wire id");
                assert!(
                    streamed.function.arguments.is_object(),
                    "{name} arguments must assemble into an object, got {:?}",
                    streamed.function.arguments
                );
            }
            assert_eq!(
                aggregated.len(),
                2,
                "aggregated choice should contain exactly the two parallel calls"
            );
            assert_ne!(
                aggregated[0].id, aggregated[1].id,
                "parallel calls must keep distinct identities"
            );
        },
    )
    .await;
}

/// Tool call and assistant content in the same turn: the aggregated choice
/// keeps the text part and the tool-call part as separate siblings.
#[tokio::test]
async fn tool_call_and_content_in_same_turn() {
    with_openai_completions_cassette(
        "streaming_grammar_chat/tool_call_with_content",
        |client| async move {
            let model = client.completion_model(openai::GPT_4O);
            let request = model
                .completion_request("Look up the harbor label for me.")
                .preamble(
                    "Before every tool call, first narrate what you are about to do in one \
                     short sentence of normal assistant text in the same reply, then emit \
                     the tool call. Never call a tool without narrating first."
                        .to_string(),
                )
                .tool(rig::tool::tool_definition(&AlphaSignal))
                .build();
            let run = drain_stream(model.stream(request).await.expect("stream should start")).await;

            assert_terminal(&run, FinishReason::ToolCalls);
            assert!(
                !run.text.trim().is_empty(),
                "turn should stream assistant text alongside the tool call"
            );
            let aggregated_text = aggregated_text(&run.choice);
            assert_eq!(
                aggregated_text, run.text,
                "aggregated choice should keep the streamed text next to the call"
            );
            let calls = aggregated_tool_calls(&run.choice);
            assert_eq!(
                calls.len(),
                1,
                "aggregated choice should keep exactly the one tool call"
            );
            assert_eq!(calls[0].function.name, "lookup_harbor_label");
            let streamed = run
                .tool_calls
                .first()
                .expect("stream should yield the tool call");
            assert_eq!(
                calls[0].id, streamed.id,
                "ids derive from the recorded turn"
            );
        },
    )
    .await;
}

/// `logprobs`-bearing chunks: the wire attaches per-token fields Rig does not
/// model. Forward-compat contract — unknown fields are ignored, the stream
/// completes normally, and the aggregation is unaffected.
#[tokio::test]
async fn logprobs_chunks_are_forward_compatible() {
    with_openai_completions_cassette(
        "streaming_grammar_chat/logprobs_chunks",
        |client| async move {
            let model = client.completion_model(openai::GPT_4O);
            let request = model
                .completion_request("Reply with one short sentence about tides.")
                .additional_params(json!({ "logprobs": true, "top_logprobs": 2 }))
                .build();
            let run = drain_stream(model.stream(request).await.expect("stream should start")).await;

            assert_terminal(&run, FinishReason::Stop);
            assert!(!run.text.trim().is_empty(), "turn should produce text");
            assert_eq!(
                aggregated_text(&run.choice),
                run.text,
                "logprobs-bearing chunks must not disturb text aggregation"
            );
        },
    )
    .await;
}

/// Long multi-chunk text stream (buffer/ordering soak): many text deltas, all
/// preserved in order into one aggregated text part.
#[tokio::test]
async fn long_text_stream_preserves_order() {
    with_openai_completions_cassette(
        "streaming_grammar_chat/long_text_stream",
        |client| async move {
            let model = client.completion_model(openai::GPT_4O);
            let request = model
                .completion_request(
                    "Write a numbered list of exactly 12 one-line facts about rivers. \
                     Number them 1. through 12.",
                )
                .max_tokens(400)
                .build();
            let run = drain_stream(model.stream(request).await.expect("stream should start")).await;

            assert_terminal(&run, FinishReason::Stop);
            assert!(
                run.text_chunks > 10,
                "soak turn should stream many text chunks, got {}",
                run.text_chunks
            );
            assert_eq!(
                aggregated_text(&run.choice),
                run.text,
                "aggregated text must be exactly the ordered streamed deltas"
            );
            // Ordering probe on the wire content itself: the numbered items
            // must appear in ascending order in the aggregated text.
            let mut last_index = 0;
            for number in 1..=12 {
                let needle = format!("{number}.");
                let position = run.text[last_index..]
                    .find(&needle)
                    .map(|offset| last_index + offset)
                    .unwrap_or_else(|| {
                        panic!(
                            "list item {number} missing or out of order in {:?}",
                            run.text
                        )
                    });
                last_index = position;
            }
        },
    )
    .await;
}
