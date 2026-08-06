//! Canonical streaming-grammar coverage for the Anthropic Messages wire,
//! asserted through the *normalized* path: the aggregated
//! [`StreamingCompletionResponse::choice`], the terminal [`StreamFinal`]
//! record, usage, IDs, and finish reason — real recorded wire traffic, not
//! synthetic chunks.
//!
//! Re-record with:
//! `RIG_PROVIDER_TEST_MODE=record ANTHROPIC_API_KEY=... cargo test --test anthropic streaming_grammar -- --test-threads=1`
//!
//! Cassette IDs are scrub placeholders (`msg_`/`toolu_` prefixes are
//! preserved); assertions derive expected IDs from the recorded turn and never
//! mint literal IDs.

use futures::StreamExt;
use rig::OneOrMany;
use rig::completion::{CompletionModel, FinishReason};
use rig::message::{AssistantContent, Reasoning, ToolCall};
use rig::prelude::*;
use rig::providers::anthropic;
use rig::streaming::{StreamFinal, StreamedAssistantContent};

use super::super::support::with_anthropic_cassette;
use crate::support::{AlphaSignal, BetaSignal, TWO_TOOL_STREAM_PREAMBLE, TWO_TOOL_STREAM_PROMPT};

struct StreamRun {
    text: String,
    reasoning_blocks: Vec<Reasoning>,
    reasoning_delta: String,
    tool_calls: Vec<ToolCall>,
    finals: Vec<StreamFinal>,
    choice: OneOrMany<AssistantContent>,
    response: Option<StreamFinal>,
    message_id: Option<String>,
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
        message_id: None,
    };

    let mut raw_items = Vec::new();
    while let Some(item) = stream.next().await {
        let item = item.expect("stream item should be ok");
        raw_items.push(Ok(item.clone()));
        match item {
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
    // The shared lifecycle validator runs over every recorded turn this
    // suite drains (#2258 C1).
    rig_core::test_utils::streaming_conformance::assert_valid_event_stream(&raw_items, &run.choice);
    run.response = stream.response.clone();
    run.message_id = stream.message_id.clone();
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
    // ID contract: Anthropic names the assistant message (`msg_*`).
    if let Some(message_id) = run.message_id.as_deref() {
        assert!(
            message_id.starts_with("msg_"),
            "message_id should be Anthropic's msg_* id, got {message_id}"
        );
    }
}

/// Multi-block turn with extended thinking: the thinking block and the text
/// block(s) survive aggregation as discrete sibling parts — the block
/// boundaries the wire drew are the boundaries the choice keeps (the item-4
/// TextStart-identity shape on real traffic).
#[tokio::test]
async fn thinking_multi_block_turn_keeps_discrete_parts() {
    with_anthropic_cassette(
        "streaming_grammar/thinking_multi_block_turn",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6);
            let request = model
                .completion_request(
                    "How many positive integers n < 300 are divisible by 8 but not by 12? \
                     Think it through, then answer with just the number.",
                )
                .max_tokens(4096)
                .additional_params(serde_json::json!({
                    "thinking": { "type": "enabled", "budget_tokens": 1536 }
                }))
                .build();
            let run = drain_stream(model.stream(request).await.expect("stream should start")).await;

            assert!(!run.text.trim().is_empty(), "turn should produce text");
            assert_terminal(&run, FinishReason::Stop);
            assert!(
                !run.reasoning_delta.is_empty() || !run.reasoning_blocks.is_empty(),
                "extended thinking should surface reasoning on the stream"
            );

            // Discrete parts: the reasoning block precedes the text block in
            // the aggregated choice, and each keeps its own boundary.
            let parts: Vec<&AssistantContent> = run.choice.iter().collect();
            let reasoning_index = parts
                .iter()
                .position(|content| matches!(content, AssistantContent::Reasoning(_)))
                .expect("aggregated choice should keep the thinking block");
            let text_index = parts
                .iter()
                .position(|content| matches!(content, AssistantContent::Text(_)))
                .expect("aggregated choice should keep the text block");
            assert!(
                reasoning_index < text_index,
                "the thinking block precedes the text block on this wire, got {:?}",
                run.choice
            );

            // The streamed text is exactly the concatenation of the discrete
            // aggregated text parts, in order — block boundaries preserved,
            // nothing merged across or duplicated.
            let text_parts: Vec<&str> = parts
                .iter()
                .filter_map(|content| match content {
                    AssistantContent::Text(text) => Some(text.text.as_str()),
                    _ => None,
                })
                .collect();
            assert_eq!(
                text_parts.concat(),
                run.text,
                "aggregated text parts should be exactly the streamed text"
            );
        },
    )
    .await;
}

/// Parallel tool use in one turn: both `tool_use` blocks survive aggregation
/// as distinct parts with their recorded `toolu_*` ids.
#[tokio::test]
async fn parallel_tool_use_stays_distinct() {
    with_anthropic_cassette("streaming_grammar/parallel_tool_use", |client| async move {
        let model = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6);
        let request = model
            .completion_request(TWO_TOOL_STREAM_PROMPT)
            .preamble(TWO_TOOL_STREAM_PREAMBLE.to_string())
            .tool(rig::tool::tool_definition(&AlphaSignal))
            .tool(rig::tool::tool_definition(&BetaSignal))
            .max_tokens(1024)
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
            // IDs derived from the recorded turn, never minted literally;
            // the wire's prefix survives scrubbing.
            assert_eq!(
                aggregated_call.id, streamed.id,
                "{name} id should aggregate"
            );
            assert!(
                streamed.id.starts_with("toolu_"),
                "{name} should carry the wire's toolu_* id, got {}",
                streamed.id
            );
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
    })
    .await;
}
