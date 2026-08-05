//! Canonical streaming-grammar coverage for the OpenAI Responses API,
//! asserted through the *normalized* path: the aggregated
//! [`StreamingCompletionResponse::choice`], the terminal [`StreamFinal`]
//! record, usage, IDs, and finish reason — real recorded wire traffic, not
//! synthetic chunks.
//!
//! Re-record with:
//! `RIG_PROVIDER_TEST_MODE=record OPENAI_API_KEY=... cargo test --test openai streaming_grammar -- --test-threads=1`
//!
//! Cassette IDs are scrub placeholders (prefixes such as `msg_`/`resp_` are
//! preserved); assertions derive expected IDs from the recorded turn and never
//! mint literal IDs.

use futures::StreamExt;
use rig::OneOrMany;
use rig::completion::{CompletionModel, FinishReason};
use rig::message::{AssistantContent, Message, Reasoning, ReasoningContent, ToolCall};
use rig::prelude::*;
use rig::providers::openai;
use rig::streaming::{StreamFinal, StreamedAssistantContent};
use serde_json::json;

use super::super::support::with_openai_cassette;
use crate::support::{
    ALPHA_SIGNAL_OUTPUT, AlphaSignal, BetaSignal, ORDERED_TOOL_STREAM_PREAMBLE,
    ORDERED_TOOL_STREAM_PROMPT, TWO_TOOL_STREAM_PREAMBLE, TWO_TOOL_STREAM_PROMPT,
};

/// Everything observed while draining a normalized stream, alongside the
/// aggregated stream state itself.
struct StreamRun {
    /// Streamed text deltas, concatenated.
    text: String,
    /// Full reasoning blocks yielded as stream events, in order.
    reasoning_blocks: Vec<Reasoning>,
    /// Concatenated reasoning-delta text.
    reasoning_delta: String,
    /// Complete tool calls yielded as stream events, in order.
    tool_calls: Vec<ToolCall>,
    /// Terminal records yielded by the stream.
    finals: Vec<StreamFinal>,
    /// The aggregated choice built by the normalized stream.
    choice: OneOrMany<AssistantContent>,
    /// The normalized terminal record retained on the stream.
    response: Option<StreamFinal>,
    /// Provider-assigned assistant message ID retained on the stream.
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

    while let Some(item) = stream.next().await {
        match item.expect("stream item should be ok") {
            StreamedAssistantContent::Text(text) => run.text.push_str(&text.text),
            StreamedAssistantContent::Reasoning(reasoning) => {
                run.reasoning_blocks.push(reasoning);
            }
            StreamedAssistantContent::ReasoningDelta { reasoning, .. } => {
                run.reasoning_delta.push_str(&reasoning);
            }
            StreamedAssistantContent::ToolCall { tool_call, .. } => {
                run.tool_calls.push(tool_call);
            }
            StreamedAssistantContent::Final(response) => run.finals.push(response),
            _ => {}
        }
    }

    run.choice = stream.choice.clone();
    run.response = stream.response.clone();
    run.message_id = stream.message_id.clone();
    run
}

/// Text of each reasoning part in the aggregated choice, in order.
fn aggregated_reasoning_parts(choice: &OneOrMany<AssistantContent>) -> Vec<ReasoningContent> {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Reasoning(reasoning) => Some(reasoning.content.clone()),
            _ => None,
        })
        .flatten()
        .collect()
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
    // ID contract: the Responses API names both the response (`resp_`) and the
    // assistant output message (`msg_`); prefixes survive cassette scrubbing.
    let response_id = terminal
        .response_id
        .as_deref()
        .expect("Responses API should report a response-scoped ID");
    assert!(
        response_id.starts_with("resp_"),
        "response_id should be response-scoped, got {response_id}"
    );
    if let Some(message_id) = run.message_id.as_deref() {
        assert!(
            message_id.starts_with("msg_") || message_id.starts_with("rs_"),
            "message_id should be an output-item ID, got {message_id}"
        );
        assert_ne!(
            message_id, response_id,
            "message-scoped and response-scoped IDs must not be conflated"
        );
    }
}

/// Streaming reasoning-summary turn: every emitted summary part must survive
/// into the aggregated choice exactly once — no duplicated reasoning text.
///
/// This is the recording that pins the two `34ee8ba5` P1s (reasoning-summary
/// duplication; same-id multi-part collapse). The assertion states what the
/// wire means; it is not weakened to match the current aggregation.
#[tokio::test]
#[ignore = "known failure on main: fixed in phase 2 — 34ee8ba5 P1s"]
async fn reasoning_summary_stream_aggregates_each_part_once() {
    with_openai_cassette(
        "streaming_grammar/reasoning_summary_stream",
        |client| async move {
            let model = client.completion_model(openai::GPT_5_6);
            let request = model
                // Low-effort turns skip the reasoning item entirely on the
                // wire; a multi-step problem at high effort reliably yields
                // summary parts.
                .completion_request(
                    "How many positive integers n < 1000 are divisible by 7 but not by 5? \
                     Work it out carefully, then answer with just the number.",
                )
                .additional_params(json!({
                    "reasoning": { "effort": "high", "summary": "detailed" }
                }))
                .build();
            let run = drain_stream(model.stream(request).await.expect("stream should start")).await;

            assert!(!run.text.trim().is_empty(), "turn should produce text");
            assert_terminal(&run, FinishReason::Stop);

            // The full reasoning blocks are the provider's authoritative
            // statement of the summary parts for this item.
            assert!(
                !run.reasoning_blocks.is_empty(),
                "summary-enabled turn should yield full reasoning blocks"
            );
            let emitted_parts: Vec<String> = run
                .reasoning_blocks
                .iter()
                .flat_map(|block| block.content.iter())
                .filter_map(|part| match part {
                    ReasoningContent::Text { text, .. } => Some(text.clone()),
                    ReasoningContent::Summary(text) => Some(text.clone()),
                    _ => None,
                })
                .collect();
            assert!(
                !emitted_parts.is_empty(),
                "summary-enabled turn should emit readable reasoning parts"
            );

            let aggregated = aggregated_reasoning_parts(&run.choice);

            // The comparison must be part-by-part, not over concatenated text:
            // on main the delta-built item survives with all parts merged into
            // one string while the full blocks clobber each other (the two
            // 34ee8ba5 P1s), which a concatenated-text check cannot see.
            //
            // Every summary part the wire emitted survives as its own
            // aggregated part, exactly once — the full blocks supersede the
            // delta accumulation, so no delta-built duplicate remains either.
            for part in &emitted_parts {
                let occurrences = aggregated
                    .iter()
                    .filter(|aggregated_part| match aggregated_part {
                        ReasoningContent::Text { text, .. } | ReasoningContent::Summary(text) => {
                            text == part
                        }
                        _ => false,
                    })
                    .count();
                assert_eq!(
                    occurrences, 1,
                    "summary part should survive aggregation exactly once: {part:?}"
                );
            }
            let readable_aggregated = aggregated
                .iter()
                .filter(|part| {
                    matches!(
                        part,
                        ReasoningContent::Text { .. } | ReasoningContent::Summary(_)
                    )
                })
                .count();
            assert_eq!(
                readable_aggregated,
                emitted_parts.len(),
                "aggregated choice should contain exactly the emitted reasoning parts, \
                 with the delta accumulation superseded; aggregated parts: {aggregated:?}"
            );
        },
    )
    .await;
}

/// Multi-part reasoning with encrypted content
/// (`include: ["reasoning.encrypted_content"]`, `store: false`): the
/// aggregated choice must retain both the encrypted payload and every
/// readable summary part (34ee8ba5 P1-2 collapses same-id parts so only the
/// encrypted blob survives).
#[tokio::test]
#[ignore = "known failure on main: fixed in phase 2 — 34ee8ba5 P1s"]
async fn encrypted_reasoning_keeps_summary_parts_and_encrypted_payload() {
    with_openai_cassette(
        "streaming_grammar/encrypted_reasoning_multi_part",
        |client| async move {
            let model = client.completion_model(openai::GPT_5_6);
            let request = model
                // Well-known riddles are answered without reasoning; a
                // multi-step count at high effort reliably yields a reasoning
                // item with summary and encrypted parts.
                .completion_request(
                    "How many positive integers n < 500 are divisible by 3 but not by 4? \
                     Work it out carefully, then answer with just the number.",
                )
                .additional_params(json!({
                    "reasoning": { "effort": "high", "summary": "detailed" },
                    "include": ["reasoning.encrypted_content"],
                    "store": false
                }))
                .build();
            let run = drain_stream(model.stream(request).await.expect("stream should start")).await;

            assert!(!run.text.trim().is_empty(), "turn should produce text");
            assert_terminal(&run, FinishReason::Stop);

            let emitted_parts: Vec<ReasoningContent> = run
                .reasoning_blocks
                .iter()
                .flat_map(|block| block.content.iter().cloned())
                .collect();
            assert!(
                emitted_parts
                    .iter()
                    .any(|part| matches!(part, ReasoningContent::Encrypted(payload) if !payload.is_empty())),
                "include=reasoning.encrypted_content should yield an encrypted reasoning part"
            );

            let aggregated = aggregated_reasoning_parts(&run.choice);
            assert!(
                aggregated
                    .iter()
                    .any(|part| matches!(part, ReasoningContent::Encrypted(payload) if !payload.is_empty())),
                "encrypted reasoning payload should survive into the aggregated choice"
            );
            // Part-by-part, not concatenated text: the same-id fallback on
            // main replaces earlier full blocks with later ones (34ee8ba5
            // P1-2), so each readable part must survive as its own aggregated
            // part alongside the encrypted payload.
            for part in &emitted_parts {
                let (ReasoningContent::Text { text, .. } | ReasoningContent::Summary(text)) = part
                else {
                    continue;
                };
                assert!(
                    aggregated.iter().any(|aggregated_part| match aggregated_part {
                        ReasoningContent::Text { text: aggregated_text, .. }
                        | ReasoningContent::Summary(aggregated_text) => aggregated_text == text,
                        _ => false,
                    }),
                    "readable reasoning part destroyed by the encrypted part: {text:?}"
                );
            }
        },
    )
    .await;
}

/// Parallel tool calls streamed in one turn: both calls must land in the
/// aggregated choice and the terminal record must report `ToolCalls`.
#[tokio::test]
async fn parallel_tool_calls_both_survive_aggregation() {
    with_openai_cassette(
        "streaming_grammar/parallel_tool_calls",
        |client| async move {
            let model = client.completion_model(openai::GPT_5_6);
            let request = model
                .completion_request(TWO_TOOL_STREAM_PROMPT)
                .preamble(TWO_TOOL_STREAM_PREAMBLE.to_string())
                .tool(rig::tool::tool_definition(&AlphaSignal))
                .tool(rig::tool::tool_definition(&BetaSignal))
                .additional_params(json!({
                    "reasoning": { "effort": "low" },
                    "parallel_tool_calls": true
                }))
                .build();
            let run = drain_stream(model.stream(request).await.expect("stream should start")).await;

            assert_terminal(&run, FinishReason::ToolCalls);

            let aggregated_calls: Vec<&ToolCall> = run
                .choice
                .iter()
                .filter_map(|content| match content {
                    AssistantContent::ToolCall(tool_call) => Some(tool_call),
                    _ => None,
                })
                .collect();
            for name in ["lookup_harbor_label", "lookup_orchard_label"] {
                let streamed = run
                    .tool_calls
                    .iter()
                    .find(|call| call.function.name == name)
                    .unwrap_or_else(|| panic!("stream should yield a {name} call"));
                let aggregated = aggregated_calls
                    .iter()
                    .find(|call| call.function.name == name)
                    .unwrap_or_else(|| panic!("aggregated choice should keep the {name} call"));
                // IDs derived from the recorded turn, never minted literally.
                assert_eq!(aggregated.id, streamed.id, "{name} id should aggregate");
                assert_eq!(
                    aggregated.call_id, streamed.call_id,
                    "{name} call_id should aggregate"
                );
            }
            assert_eq!(
                aggregated_calls.len(),
                2,
                "aggregated choice should contain exactly the two parallel calls"
            );
        },
    )
    .await;
}

/// Tool call then follow-up text across turns: turn one ends in `ToolCalls`
/// with the call aggregated; the follow-up turn (fed the tool result) ends in
/// `Stop` with text that uses the result.
#[tokio::test]
async fn tool_call_then_followup_text_across_turns() {
    with_openai_cassette(
        "streaming_grammar/tool_then_followup_text",
        |client| async move {
            let model = client.completion_model(openai::GPT_5_6);
            let request = model
                .completion_request(ORDERED_TOOL_STREAM_PROMPT)
                .preamble(ORDERED_TOOL_STREAM_PREAMBLE.to_string())
                .tool(rig::tool::tool_definition(&AlphaSignal))
                .additional_params(json!({ "reasoning": { "effort": "low" } }))
                .build();
            let first =
                drain_stream(model.stream(request).await.expect("stream should start")).await;

            assert_terminal(&first, FinishReason::ToolCalls);
            let tool_call = first
                .choice
                .iter()
                .find_map(|content| match content {
                    AssistantContent::ToolCall(tool_call)
                        if tool_call.function.name == "lookup_harbor_label" =>
                    {
                        Some(tool_call.clone())
                    }
                    _ => None,
                })
                .expect("aggregated first turn should contain the lookup_harbor_label call");

            let assistant_message = Message::Assistant {
                id: first.message_id.clone(),
                content: OneOrMany::one(AssistantContent::ToolCall(tool_call.clone())),
            };
            let tool_result = Message::tool_result_with_call_id(
                tool_call.id.clone(),
                tool_call.call_id.clone(),
                ALPHA_SIGNAL_OUTPUT,
            );
            let followup_request = model
                .completion_request(
                    "Now reply in one short sentence using the provided tool result. \
                     Do not call any tools.",
                )
                .preamble(ORDERED_TOOL_STREAM_PREAMBLE.to_string())
                .messages(vec![
                    Message::user(ORDERED_TOOL_STREAM_PROMPT),
                    assistant_message,
                    tool_result,
                ])
                .additional_params(json!({ "reasoning": { "effort": "low" } }))
                .build();
            let second = drain_stream(
                model
                    .stream(followup_request)
                    .await
                    .expect("follow-up stream should start"),
            )
            .await;

            assert_terminal(&second, FinishReason::Stop);
            assert!(
                second.tool_calls.is_empty(),
                "follow-up turn should not call tools"
            );
            assert!(
                second.text.contains(ALPHA_SIGNAL_OUTPUT),
                "follow-up text should use the tool result, got {:?}",
                second.text
            );
        },
    )
    .await;
}

/// `response.incomplete` via a small `max_output_tokens` budget: the stream
/// still terminates with a terminal record, the finish reason normalizes to
/// `Length`, and whatever partial output arrived is kept.
#[tokio::test]
async fn incomplete_max_output_tokens_normalizes_to_length() {
    with_openai_cassette(
        "streaming_grammar/incomplete_max_output_tokens",
        |client| async move {
            let model = client.completion_model(openai::GPT_5_6);
            let request = model
                .completion_request("Write a 300-word essay about the history of lighthouses.")
                .max_tokens(32)
                .additional_params(json!({ "reasoning": { "effort": "low" } }))
                .build();
            let run = drain_stream(model.stream(request).await.expect("stream should start")).await;

            assert_terminal(&run, FinishReason::Length);
            // Partial content is kept: whatever the stream surfaced before the
            // cutoff (text and/or reasoning) must match the aggregated choice.
            let aggregated_text: String = run
                .choice
                .iter()
                .filter_map(|content| match content {
                    AssistantContent::Text(text) => Some(text.text.as_str()),
                    _ => None,
                })
                .collect();
            assert_eq!(
                aggregated_text, run.text,
                "aggregated choice should keep exactly the streamed partial text"
            );
        },
    )
    .await;
}
