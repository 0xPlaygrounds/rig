//! Canonical streaming-grammar coverage for Gemini (REST `generateContent`
//! streaming plus the Interactions API), asserted through the *normalized*
//! path: the aggregated [`StreamingCompletionResponse::choice`], the terminal
//! [`StreamFinal`] record, usage, IDs, and finish reason — real recorded wire
//! traffic, not synthetic chunks.
//!
//! Re-record with:
//! `RIG_PROVIDER_TEST_MODE=record GEMINI_API_KEY=... cargo test --test gemini streaming_grammar -- --test-threads=1`
//!
//! Cassette IDs are scrub placeholders; assertions derive expected IDs from
//! the recorded turn and never mint literal IDs.

use futures::StreamExt;
use rig::OneOrMany;
use rig::completion::{CompletionModel, FinishReason};
use rig::message::{
    AssistantContent, Message, Reasoning, ReasoningContent, ToolCall, ToolChoice,
    ToolResultContent, UserContent,
};
use rig::prelude::*;
use rig::providers::gemini;
use rig::providers::gemini::completion::gemini_api_types::{
    AdditionalParameters, GenerationConfig, ThinkingConfig, ThinkingLevel,
};
use rig::providers::gemini::interactions_api;
use rig::streaming::{StreamFinal, StreamedAssistantContent};

use crate::support::{
    ALPHA_SIGNAL_OUTPUT, AlphaSignal, ORDERED_TOOL_STREAM_PREAMBLE, ORDERED_TOOL_STREAM_PROMPT,
};

/// Everything observed while draining a normalized stream, alongside the
/// aggregated stream state itself.
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
    // ID contract: Gemini reports a `responseId` for the response as a whole
    // and no replayable assistant-message ID, so the normalized terminal must
    // populate `response_id` and leave `message_id` empty.
    assert!(
        terminal
            .response_id
            .as_deref()
            .is_some_and(|id| !id.is_empty()),
        "Gemini should surface its responseId as the response-scoped ID"
    );
    assert!(
        terminal.message_id.is_none(),
        "Gemini has no replayable assistant-message ID; got {:?}",
        terminal.message_id
    );
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

fn aggregated_reasoning_text(choice: &OneOrMany<AssistantContent>) -> String {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Reasoning(reasoning) => Some(reasoning.content.iter()),
            _ => None,
        })
        .flatten()
        .filter_map(|part| match part {
            ReasoningContent::Text { text, .. } => Some(text.as_str()),
            ReasoningContent::Summary(text) => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

/// `MAX_TOKENS` truncation via a small `maxOutputTokens` budget: terminal
/// record present, finish reason normalized to `Length`, partial text kept.
#[tokio::test]
async fn max_tokens_truncation_normalizes_to_length() {
    let config = GenerationConfig {
        thinking_config: Some(ThinkingConfig {
            thinking_budget: Some(0),
            thinking_level: None,
            include_thoughts: None,
        }),
        ..Default::default()
    };
    let params = AdditionalParameters::default().with_config(config);
    super::super::support::with_gemini_cassette(
        "streaming_grammar/max_tokens_truncation",
        |client| async move {
            let model = client.completion_model(gemini::completion::GEMINI_2_5_FLASH);
            let request = model
                .completion_request("Write a 200-word story about a lighthouse keeper.")
                .max_tokens(24)
                .additional_params(serde_json::to_value(params).expect("params should serialize"))
                .build();
            let run = drain_stream(model.stream(request).await.expect("stream should start")).await;

            assert_terminal(&run, FinishReason::Length);
            assert!(
                !run.text.is_empty(),
                "truncated stream should keep the partial text it produced"
            );
            assert_eq!(
                aggregated_text(&run.choice),
                run.text,
                "aggregated choice should keep exactly the streamed partial text"
            );
        },
    )
    .await;
}

/// Streaming tool-call turn: the call lands in the aggregated choice with the
/// IDs the stream reported, and the terminal record normalizes to `ToolCalls`.
#[tokio::test]
async fn streaming_tool_call_aggregates_with_tool_calls_finish() {
    super::super::support::with_gemini_cassette(
        "streaming_grammar/streaming_tool_call",
        |client| async move {
            let model = client.completion_model(gemini::completion::GEMINI_2_5_FLASH);
            let request = model
                .completion_request(ORDERED_TOOL_STREAM_PROMPT)
                .preamble(ORDERED_TOOL_STREAM_PREAMBLE.to_string())
                .tool(rig::tool::tool_definition(&AlphaSignal))
                .build();
            let run = drain_stream(model.stream(request).await.expect("stream should start")).await;

            assert_terminal(&run, FinishReason::ToolCalls);
            let streamed = run
                .tool_calls
                .iter()
                .find(|call| call.function.name == "lookup_harbor_label")
                .expect("stream should yield the lookup_harbor_label call");
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
            // IDs derived from the recorded turn, never minted literally.
            assert_eq!(aggregated.id, streamed.id);
            assert_eq!(aggregated.call_id, streamed.call_id);
        },
    )
    .await;
}

/// Thinking output on a thinking-capable model: all thinking text the stream
/// surfaced (deltas and/or full blocks) must survive into the aggregated
/// choice's reasoning content — a signed final thinking chunk must not erase
/// the preceding reasoning text (34ee8ba5 Gemini P1).
///
/// This recording's signed chunk arrives with empty text after two unsigned
/// thought deltas, and the aggregation keeps the delta text, so the assertion
/// holds on main; a recording whose signed chunk carries text would trip the
/// P1 erasure.
#[tokio::test]
async fn thinking_stream_aggregates_all_reasoning_text() {
    let config = GenerationConfig {
        thinking_config: Some(ThinkingConfig {
            thinking_budget: None,
            thinking_level: Some(ThinkingLevel::Medium),
            include_thoughts: Some(true),
        }),
        ..Default::default()
    };
    let params = AdditionalParameters::default().with_config(config);
    super::super::support::with_gemini_cassette(
        "streaming_grammar/thinking_stream",
        |client| async move {
            let model = client.completion_model(gemini::completion::GEMINI_3_FLASH_PREVIEW);
            let request = model
                .completion_request(
                    "In one short sentence, why does ice float on water? Think it through first.",
                )
                .additional_params(
                    serde_json::to_value(params).expect("params should serialize"),
                )
                .build();
            let run = drain_stream(model.stream(request).await.expect("stream should start")).await;

            assert!(!run.text.trim().is_empty(), "turn should produce text");
            assert_terminal(&run, FinishReason::Stop);
            assert!(
                !run.reasoning_delta.is_empty() || !run.reasoning_blocks.is_empty(),
                "include_thoughts should surface reasoning on the stream"
            );

            let aggregated = aggregated_reasoning_text(&run.choice);
            assert!(
                !aggregated.is_empty(),
                "aggregated choice should retain reasoning content"
            );
            // Full blocks supersede their deltas, so the aggregated reasoning
            // must contain at least everything streamed as deltas: a trailing
            // signed block must not erase the earlier thinking text.
            assert!(
                aggregated.contains(run.reasoning_delta.trim()),
                "aggregated reasoning lost streamed thinking text;\nstreamed: {:?}\naggregated: {:?}",
                run.reasoning_delta,
                aggregated
            );
        },
    )
    .await;
}

/// Interactions API turn that stops for a declared client tool
/// (`requires_action`), then completes after the tool result is submitted —
/// one recorded exchange, asserted through the normalized conversion.
#[tokio::test]
async fn interactions_requires_action_roundtrip() {
    super::super::support::with_gemini_interactions_cassette(
        "streaming_grammar/interactions_requires_action",
        |client| async move {
            let model = client.completion_model("gemini-3-flash-preview");
            let tool = rig::tool::tool_definition(&AlphaSignal);

            let raw = model
                .raw_completion(
                    model
                        .completion_request(ORDERED_TOOL_STREAM_PROMPT)
                        .preamble(ORDERED_TOOL_STREAM_PREAMBLE.to_string())
                        .tool(tool)
                        .tool_choice(ToolChoice::Required)
                        .additional_params(
                            serde_json::to_value(interactions_api::AdditionalParameters {
                                store: Some(true),
                                ..Default::default()
                            })
                            .expect("params should serialize"),
                        )
                        .build(),
                )
                .await
                .expect("tool-required interaction should succeed");

            // The wire status transition under test.
            assert!(
                matches!(
                    raw.status,
                    Some(interactions_api::InteractionStatus::RequiresAction)
                ),
                "declared client tool should leave the interaction in requires_action, got {:?}",
                raw.status
            );
            let interaction_id = raw.id.clone();
            assert!(!interaction_id.is_empty(), "expected an interaction id");

            let normalized: rig::completion::CompletionResponse = raw
                .try_into()
                .expect("requires_action interaction should normalize");
            assert_eq!(
                normalized.finish_reason(),
                Some(FinishReason::ToolCalls),
                "requires_action should normalize to a ToolCalls finish"
            );
            assert!(
                normalized.usage.total_tokens > 0,
                "interaction should report usage, got {:?}",
                normalized.usage
            );
            let tool_call = normalized
                .choice
                .iter()
                .find_map(|content| match content {
                    AssistantContent::ToolCall(call)
                        if call.function.name == "lookup_harbor_label" =>
                    {
                        Some(call.clone())
                    }
                    _ => None,
                })
                .expect("normalized choice should carry the client tool call");
            // The call id comes from the recorded turn, never minted literally.
            let call_id = tool_call
                .call_id
                .clone()
                .unwrap_or_else(|| tool_call.id.clone());
            assert!(!call_id.is_empty(), "tool call should carry an id");

            let followup = model
                .completion(
                    model
                        .completion_request(Message::from(UserContent::tool_result_with_call_id(
                            tool_call.function.name,
                            call_id,
                            OneOrMany::one(ToolResultContent::text(ALPHA_SIGNAL_OUTPUT)),
                        )))
                        .additional_params(
                            serde_json::to_value(interactions_api::AdditionalParameters {
                                previous_interaction_id: Some(interaction_id),
                                ..Default::default()
                            })
                            .expect("params should serialize"),
                        )
                        .build(),
                )
                .await
                .expect("tool-result follow-up should succeed");

            assert_eq!(
                followup.finish_reason(),
                Some(FinishReason::Stop),
                "completed follow-up should normalize to Stop"
            );
            let text = aggregated_text(&followup.choice);
            assert!(
                text.contains(ALPHA_SIGNAL_OUTPUT),
                "follow-up should use the tool result, got {text:?}"
            );
        },
    )
    .await;
}
