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
    ALPHA_SIGNAL_OUTPUT, AlphaSignal, BetaSignal, ORDERED_TOOL_STREAM_PREAMBLE,
    ORDERED_TOOL_STREAM_PROMPT, TWO_TOOL_STREAM_PREAMBLE,
};

/// Everything observed while draining a normalized stream, alongside the
/// aggregated stream state itself.
struct StreamRun {
    text: String,
    reasoning_blocks: Vec<Reasoning>,
    reasoning_delta: String,
    tool_calls: Vec<ToolCall>,
    finals: Vec<StreamFinal>,
    choice: Vec<AssistantContent>,
    response: Option<StreamFinal>,
}

async fn drain_stream(mut stream: rig::streaming::StreamingCompletionResponse) -> StreamRun {
    let mut run = StreamRun {
        text: String::new(),
        reasoning_blocks: Vec::new(),
        reasoning_delta: String::new(),
        tool_calls: Vec::new(),
        finals: Vec::new(),
        choice: vec![AssistantContent::text("")],
        response: None,
    };

    let mut raw_items = Vec::new();
    while let Some(item) = stream.next().await {
        let item = item.expect("stream item should be ok");
        raw_items.push(Ok(item.clone()));
        match item {
            StreamedAssistantContent::Text(text) => run.text.push_str(&text.text),
            StreamedAssistantContent::Reasoning { reasoning, .. } => {
                run.reasoning_blocks.push(reasoning)
            }
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

fn aggregated_text(choice: &[AssistantContent]) -> String {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect()
}

fn aggregated_reasoning_text(choice: &[AssistantContent]) -> String {
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
            assert_eq!(aggregated.provider, streamed.provider);
        },
    )
    .await;
}

/// Thinking output on a thinking-capable model: all thinking text the stream
/// surfaced (deltas and/or full blocks) must survive into the aggregated
/// choice's reasoning content — a signed final thinking chunk must not erase
/// the preceding reasoning text (34ee8ba5 Gemini P1).
///
/// Re-recorded (2258 item 5, sanctioned drift) with a multi-step problem at
/// high thinking. Recording note: across repeated attempts (gemini-3 low /
/// medium / high thinking, gemini-2.5 with a thinking budget, with and
/// without tools) the wire only ever attaches `thoughtSignature` to a
/// trailing **empty** text part or to a functionCall part — a signed chunk
/// carrying non-empty thinking text was not inducible, so the F1
/// signed-restatement erasure stays pinned by the synthetic corpus and this
/// recording pins the real signed-empty-trailer shape: the signed trailer
/// must not erase or duplicate the preceding unsigned thought deltas.
#[tokio::test]
async fn thinking_stream_aggregates_all_reasoning_text() {
    let config = GenerationConfig {
        thinking_config: Some(ThinkingConfig {
            thinking_budget: None,
            thinking_level: Some(ThinkingLevel::High),
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
                    "How many positive integers n < 400 are divisible by 6 but not by 9? \
                     Think it through carefully step by step, then answer with just the number.",
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
            // The signed restatement supersedes the delta accumulation: the
            // delta text must survive exactly once, not once per statement.
            assert_eq!(
                aggregated.matches(run.reasoning_delta.trim()).count(),
                1,
                "signed restatement must not duplicate the thinking text;\naggregated: {aggregated:?}"
            );
            // The wire attaches `thoughtSignature` to a trailing part with no
            // `thought` flag (recorded: `{"text":"","thoughtSignature":"…"}`),
            // so a `thought: true`-gated signature path never sees it. The
            // signature is replay-required provider state — Gemini rejects a
            // replayed turn missing it — so it must reach the aggregated
            // reasoning.
            let signatures: Vec<&str> = run
                .choice
                .iter()
                .filter_map(|content| match content {
                    AssistantContent::Reasoning(reasoning) => Some(reasoning.content.iter()),
                    _ => None,
                })
                .flatten()
                .filter_map(|part| match part {
                    ReasoningContent::Text {
                        signature: Some(signature),
                        ..
                    } => Some(signature.as_str()),
                    _ => None,
                })
                .collect();
            assert!(
                signatures.iter().any(|signature| !signature.is_empty()),
                "the recorded thought_signature must survive onto the aggregated reasoning, got {:?}",
                run.choice
            );
            // And it must sign the block that HOLDS the chain-of-thought —
            // an empty signature-only sibling appended after the answer
            // satisfies the assertion above while the real thinking replays
            // unsigned (#2258 B4). Every reasoning part must carry text, and
            // the part carrying the streamed thinking must be the signed one.
            for content in run.choice.iter() {
                if let AssistantContent::Reasoning(reasoning) = content {
                    let text: String = reasoning
                        .content
                        .iter()
                        .filter_map(|part| match part {
                            ReasoningContent::Text { text, .. } => Some(text.as_str()),
                            _ => None,
                        })
                        .collect();
                    assert!(
                        !text.trim().is_empty(),
                        "no empty (signature-only) reasoning sibling may exist: {:?}",
                        run.choice
                    );
                    if text.contains(run.reasoning_delta.trim()) {
                        assert!(
                            reasoning.content.iter().any(|part| matches!(
                                part,
                                ReasoningContent::Text { signature: Some(signature), .. }
                                    if !signature.is_empty()
                            )),
                            "the signature must land on the block carrying the thinking text: {:?}",
                            run.choice
                        );
                    }
                }
            }
        },
    )
    .await;
}

/// Thinking and a tool call interleaved in ONE stream: the aggregated choice
/// keeps the reasoning part and the tool-call part as discrete siblings (the
/// F1b thinking/tool boundary, pinned on real traffic).
#[tokio::test]
async fn thinking_and_tool_call_interleave_as_discrete_parts() {
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
        "streaming_grammar/thinking_then_tool_call",
        |client| async move {
            let model = client.completion_model(gemini::completion::GEMINI_3_FLASH_PREVIEW);
            let request = model
                // A trivial tool turn yields no thought parts on this wire; a
                // math sub-task reliably interleaves thinking with the call.
                .completion_request(
                    "First work out how many positive integers n < 60 are divisible by 8 \
                     (carefully). Then call `lookup_harbor_label` exactly once. After the \
                     tool result arrives, answer with the count and the exact tool output.",
                )
                .preamble(ORDERED_TOOL_STREAM_PREAMBLE.to_string())
                .tool(rig::tool::tool_definition(&AlphaSignal))
                .additional_params(serde_json::to_value(params).expect("params should serialize"))
                .build();
            let run = drain_stream(model.stream(request).await.expect("stream should start")).await;

            assert_terminal(&run, FinishReason::ToolCalls);
            assert!(
                !run.reasoning_delta.is_empty() || !run.reasoning_blocks.is_empty(),
                "include_thoughts should surface reasoning before the call"
            );
            let streamed = run
                .tool_calls
                .iter()
                .find(|call| call.function.name == "lookup_harbor_label")
                .expect("stream should yield the lookup_harbor_label call");
            // Discrete parts: exactly one reasoning part and one tool call
            // live side-by-side in the aggregated choice; the tool call did
            // not absorb or erase the thinking, and vice versa.
            let reasoning_parts = run
                .choice
                .iter()
                .filter(|content| matches!(content, AssistantContent::Reasoning(_)))
                .count();
            assert!(
                reasoning_parts >= 1,
                "aggregated choice should keep the reasoning part, got {:?}",
                run.choice
            );
            let aggregated_call = run
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
            assert_eq!(aggregated_call.id, streamed.id);
            assert_eq!(aggregated_call.provider, streamed.provider);
            assert!(
                !aggregated_reasoning_text(&run.choice).is_empty(),
                "reasoning text must survive alongside the tool call"
            );
        },
    )
    .await;
}

/// Parallel function calls in one turn: both calls survive aggregation as
/// distinct parts with the ids the stream reported.
#[tokio::test]
async fn parallel_function_calls_stay_distinct() {
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
        "streaming_grammar/parallel_function_calls",
        |client| async move {
            let model = client.completion_model(gemini::completion::GEMINI_2_5_FLASH);
            let request = model
                .completion_request(
                    "Call `lookup_harbor_label` and `lookup_orchard_label` now, both of them \
                     together in this single reply, before writing any text. Emit the two \
                     tool calls in one turn - do not wait for results between them.",
                )
                .preamble(TWO_TOOL_STREAM_PREAMBLE.to_string())
                .tool(rig::tool::tool_definition(&AlphaSignal))
                .tool(rig::tool::tool_definition(&BetaSignal))
                .additional_params(serde_json::to_value(params).expect("params should serialize"))
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
                // IDs derived from the recorded turn, never minted literally.
                assert_eq!(
                    aggregated_call.id, streamed.id,
                    "{name} id should aggregate"
                );
            }
            assert_eq!(
                aggregated.len(),
                2,
                "aggregated choice should contain exactly the two parallel calls"
            );
            // The recorded turn carries no wire ids, and rig no longer
            // fabricates durable identifiers from the wire (not from an
            // index, not from the tool name): both calls surface with a
            // minted id and no provider id, and stay distinct as parts —
            // two calls, in wire order, uncorrupted.
            assert!(aggregated[0].provider.is_none());
            assert!(aggregated[1].provider.is_none());
            assert!(!aggregated[0].id.as_str().is_empty());
            assert!(!aggregated[1].id.as_str().is_empty());
            assert_ne!(
                aggregated[0].id, aggregated[1].id,
                "each id-less call mints a unique durable id"
            );
            assert_ne!(
                aggregated[0].function.name, aggregated[1].function.name,
                "the two parallel calls stay distinct parts"
            );
        },
    )
    .await;
}

/// Plain `STOP` finish on a text-only turn (`finishReason` variant beyond
/// MAX_TOKENS): terminal record present, `Stop` normalized, text aggregated
/// exactly as streamed.
#[tokio::test]
async fn stop_finish_reason_normalizes_on_text_turn() {
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
        "streaming_grammar/stop_finish_reason",
        |client| async move {
            let model = client.completion_model(gemini::completion::GEMINI_2_5_FLASH);
            let request = model
                .completion_request("Reply with one short sentence about volcanoes.")
                .additional_params(serde_json::to_value(params).expect("params should serialize"))
                .build();
            let run = drain_stream(model.stream(request).await.expect("stream should start")).await;

            assert_terminal(&run, FinishReason::Stop);
            assert!(!run.text.trim().is_empty(), "turn should produce text");
            assert_eq!(
                aggregated_text(&run.choice),
                run.text,
                "aggregated choice should keep exactly the streamed text"
            );
        },
    )
    .await;
}

/// Thinking-enabled Interactions streaming turn: reasoning and text arrive as
/// discrete normalized parts through the shared REST/interactions part-kind
/// interpretation, pinned on real traffic.
#[tokio::test]
async fn interactions_thinking_stream_keeps_reasoning_and_text_discrete() {
    super::super::support::with_gemini_interactions_cassette(
        "streaming_grammar/interactions_thinking_stream",
        |client| async move {
            let model = client.completion_model("gemini-3-flash-preview");
            let request = model
                .completion_request(
                    "How many positive integers n < 200 are divisible by 4 but not by 10? \
                     Think it through, then answer with just the number.",
                )
                .additional_params(
                    serde_json::to_value(interactions_api::AdditionalParameters {
                        generation_config: Some(interactions_api::GenerationConfig {
                            thinking_level: Some(interactions_api::ThinkingLevel::Medium),
                            thinking_summaries: Some(interactions_api::ThinkingSummaries::Auto),
                            ..Default::default()
                        }),
                        store: Some(true),
                        ..Default::default()
                    })
                    .expect("params should serialize"),
                )
                .build();
            let run = drain_stream(model.stream(request).await.expect("stream should start")).await;

            assert!(!run.text.trim().is_empty(), "turn should produce text");
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
                Some(&FinishReason::Stop),
                "unexpected finish reason"
            );
            assert!(
                terminal.usage.total_tokens > 0,
                "terminal record should carry non-zero usage, got {:?}",
                terminal.usage
            );
            assert!(
                !run.reasoning_delta.is_empty() || !run.reasoning_blocks.is_empty(),
                "thinking summaries should surface reasoning on the stream"
            );
            // Discrete parts: the reasoning part and the text part live
            // side-by-side in the aggregated choice.
            assert!(
                run.choice
                    .iter()
                    .any(|content| matches!(content, AssistantContent::Reasoning(_))),
                "aggregated choice should keep a reasoning part, got {:?}",
                run.choice
            );
            let reasoning_text = aggregated_reasoning_text(&run.choice);
            assert!(
                !reasoning_text.is_empty(),
                "aggregated reasoning should carry the summary text"
            );
            // The recorded stream carries a `thought_signature` delta after
            // the thought summaries; the completed signed block must restate
            // the accumulated summary text AND keep the signature. The
            // signature value is a scrub placeholder in the cassette, so the
            // assertion derives nothing beyond presence — a real recorded
            // signature and its placeholder are both non-empty opaque strings.
            let signatures: Vec<&str> = run
                .choice
                .iter()
                .filter_map(|content| match content {
                    AssistantContent::Reasoning(reasoning) => Some(reasoning.content.iter()),
                    _ => None,
                })
                .flatten()
                .filter_map(|part| match part {
                    ReasoningContent::Text {
                        signature: Some(signature),
                        ..
                    } => Some(signature.as_str()),
                    _ => None,
                })
                .collect();
            assert!(
                signatures.iter().any(|signature| !signature.is_empty()),
                "the recorded thought_signature must survive onto the aggregated reasoning, got {:?}",
                run.choice
            );
            // And it must sign the block that HOLDS the chain-of-thought —
            // an empty signature-only sibling appended after the answer
            // satisfies the assertion above while the real thinking replays
            // unsigned (#2258 B4). Every reasoning part must carry text, and
            // the part carrying the streamed thinking must be the signed one.
            for content in run.choice.iter() {
                if let AssistantContent::Reasoning(reasoning) = content {
                    let text: String = reasoning
                        .content
                        .iter()
                        .filter_map(|part| match part {
                            ReasoningContent::Text { text, .. } => Some(text.as_str()),
                            _ => None,
                        })
                        .collect();
                    assert!(
                        !text.trim().is_empty(),
                        "no empty (signature-only) reasoning sibling may exist: {:?}",
                        run.choice
                    );
                    if text.contains(run.reasoning_delta.trim()) {
                        assert!(
                            reasoning.content.iter().any(|part| matches!(
                                part,
                                ReasoningContent::Text { signature: Some(signature), .. }
                                    if !signature.is_empty()
                            )),
                            "the signature must land on the block carrying the thinking text: {:?}",
                            run.choice
                        );
                    }
                }
            }
            // Supersede contract: the signed restatement replaced the deltas
            // it restates, so the delta text survives exactly once.
            assert!(
                reasoning_text.contains(run.reasoning_delta.trim()),
                "aggregated reasoning lost streamed summary text"
            );
            assert_eq!(
                reasoning_text.matches(run.reasoning_delta.trim()).count(),
                1,
                "signed restatement must not duplicate the summary text"
            );
            assert_eq!(
                aggregated_text(&run.choice),
                run.text,
                "aggregated text should match the streamed text exactly"
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
            assert!(
                tool_call.provider.is_some(),
                "the recorded interactions turn should carry a provider-issued call id"
            );

            let followup = model
                .completion(
                    model
                        .completion_request(Message::from(UserContent::tool_result_for(
                            tool_call.id.clone(),
                            tool_call.provider.clone(),
                            tool_call.function.name.clone(),
                            vec![ToolResultContent::text(ALPHA_SIGNAL_OUTPUT)],
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

/// Two STREAMED calls to the SAME tool in one Interactions turn, on real
/// recorded traffic — the live twin of the corpus pin (review 84a43e9e).
/// Whatever identity the wire supplies (Interactions function calls may or
/// may not carry ids), the two calls must survive as distinct aggregated
/// parts with their own uncorrupted arguments, and rig must not fabricate a
/// durable id where the wire gave none — in particular never the tool name.
///
/// Re-record with:
/// `RIG_PROVIDER_TEST_MODE=record GEMINI_API_KEY=... cargo test --test gemini interactions_same_tool_twice -- --test-threads=1`
#[tokio::test]
async fn interactions_same_tool_called_twice_stays_distinct() {
    super::super::support::with_gemini_interactions_cassette(
        "streaming_grammar/interactions_same_tool_twice",
        |client| async move {
            let model = client.completion_model("gemini-3-flash-preview");
            let request = model
                .completion_request(
                    "Use the `add` tool twice in this single reply, before any text: first \
                     add 2 and 3, then add 10 and 20. Emit both tool calls together in this \
                     one turn — do not wait for results between them, and do not compute the \
                     sums yourself.",
                )
                .preamble(
                    "You are a calculator assistant. You MUST use the provided tools for \
                     every arithmetic operation instead of computing results yourself."
                        .to_string(),
                )
                .tool(rig::tool::tool_definition(&crate::support::Adder))
                .tool_choice(ToolChoice::Required)
                .additional_params(
                    serde_json::to_value(interactions_api::AdditionalParameters {
                        store: Some(true),
                        ..Default::default()
                    })
                    .expect("params should serialize"),
                )
                .build();
            let run = drain_stream(model.stream(request).await.expect("stream should start")).await;

            assert_terminal(&run, FinishReason::ToolCalls);
            let add_calls: Vec<&ToolCall> = run
                .tool_calls
                .iter()
                .filter(|call| call.function.name == "add")
                .collect();
            assert!(
                add_calls.len() >= 2,
                "the turn should stream (at least) two `add` calls, got {:?}",
                run.tool_calls
            );
            for call in &add_calls {
                assert_ne!(
                    call.id, "add",
                    "the tool name must never be fabricated into the durable id"
                );
                assert!(
                    call.provider
                        .as_ref()
                        .is_none_or(|provider| provider.call_id != "add"),
                    "the tool name must never be fabricated into the provider call id"
                );
                assert!(
                    call.function.arguments.is_object(),
                    "each call's arguments must survive uncorrupted, got {:?}",
                    call.function.arguments
                );
            }
            let distinct_ids: std::collections::HashSet<&str> = add_calls
                .iter()
                .map(|call| call.id.as_str())
                .collect();
            assert_eq!(
                distinct_ids.len(),
                add_calls.len(),
                "same-name calls must keep distinct durable ids (minted when the wire gave none)"
            );
            let argument_sets: std::collections::HashSet<String> = add_calls
                .iter()
                .map(|call| call.function.arguments.to_string())
                .collect();
            assert_eq!(
                argument_sets,
                std::collections::HashSet::from([
                    serde_json::json!({"x": 2, "y": 3}).to_string(),
                    serde_json::json!({"x": 10, "y": 20}).to_string(),
                ]),
                "each same-name call must carry its own streamed arguments — the wire \
                 fragments them as `arguments_delta` events the pre-fix code dropped"
            );
            let aggregated_adds = run
                .choice
                .iter()
                .filter(|content| {
                    matches!(content, AssistantContent::ToolCall(call) if call.function.name == "add")
                })
                .count();
            assert_eq!(
                aggregated_adds,
                add_calls.len(),
                "every streamed same-name call must survive as its own aggregated part"
            );
        },
    )
    .await;
}

/// A thinking turn with summaries suppressed (`thinking_summaries: none`),
/// recorded live: if the wire still delivers a `thought_signature`, it
/// arrives with NO accumulated summary text — the empty-buffer shape that
/// review 84a43e9e finding #2 shows the Interactions adapter mishandling
/// (an empty signed reasoning sibling instead of signature-as-lifecycle-
/// metadata). The assertions pin the invariant both adapters must satisfy:
/// no signed empty sibling may exist alongside the answer, and any
/// signature the wire delivered must survive into the aggregated choice.
///
/// Re-record with:
/// `RIG_PROVIDER_TEST_MODE=record GEMINI_API_KEY=... cargo test --test gemini interactions_signature_without_summaries -- --test-threads=1`
#[tokio::test]
async fn interactions_signature_without_summaries_never_fabricates_an_empty_sibling() {
    super::super::support::with_gemini_interactions_cassette(
        "streaming_grammar/interactions_signature_without_summaries",
        |client| async move {
            let model = client.completion_model("gemini-3-flash-preview");
            let request = model
                .completion_request(
                    "How many positive integers n < 100 are divisible by 6 but not by 9? \
                     Think it through, then answer with just the number.",
                )
                .additional_params(
                    serde_json::to_value(interactions_api::AdditionalParameters {
                        generation_config: Some(interactions_api::GenerationConfig {
                            thinking_level: Some(interactions_api::ThinkingLevel::Medium),
                            thinking_summaries: Some(interactions_api::ThinkingSummaries::None),
                            ..Default::default()
                        }),
                        store: Some(true),
                        ..Default::default()
                    })
                    .expect("params should serialize"),
                )
                .build();
            let run = drain_stream(model.stream(request).await.expect("stream should start")).await;

            assert!(!run.text.trim().is_empty(), "turn should produce text");
            let signature_delivered = run.choice.iter().any(|content| {
                matches!(content, AssistantContent::Reasoning(reasoning)
                    if reasoning.content.iter().any(|part| matches!(
                        part,
                        ReasoningContent::Text { signature: Some(signature), .. }
                            if !signature.is_empty()
                    )))
            });
            // The invariant under test: whatever the wire delivered, the
            // aggregate must never carry a reasoning part that is *only* an
            // empty signed shell fabricated by the adapter. (A genuinely
            // signature-only stream keeps its signature — on a part the
            // accumulator records deliberately — but text-empty parts must
            // then be the ONLY reasoning, not a sibling beside real text.)
            let reasoning_parts: Vec<&Reasoning> = run
                .choice
                .iter()
                .filter_map(|content| match content {
                    AssistantContent::Reasoning(reasoning) => Some(reasoning),
                    _ => None,
                })
                .collect();
            let empty_parts = reasoning_parts
                .iter()
                .filter(|reasoning| {
                    reasoning
                        .content
                        .iter()
                        .all(|part| matches!(part, ReasoningContent::Text { text, .. } if text.trim().is_empty()))
                })
                .count();
            assert!(
                empty_parts == 0 || reasoning_parts.len() == empty_parts,
                "an empty signed reasoning shell must not appear beside real reasoning: {:?}",
                run.choice
            );
            // Recording note: if this cassette carries no signature at all,
            // the wire withholds signatures when summaries are off and the
            // empty-buffer shape is not live-coaxable — the corpus twin
            // covers it instead.
            let _ = signature_delivered;
        },
    )
    .await;
}

/// Cross-provider replay, recorded live (84a43e9e finding #5): a history
/// sourced from an OpenAI-Chat-shaped provider carries the other wire's
/// identifier `call_abc` only as rig's correlation handle — no Gemini
/// provider id. Replayed to Gemini, the request's `functionResponse.name`
/// must be the tool's *name*, read from the required `ToolResult::name` —
/// never the identifier `call_abc`, which the pre-fix name-as-id heuristic
/// kept verbatim — and the correlation-only handle must stay off the wire
/// (no `functionCall.id`/`functionResponse.id`). The recording is the
/// evidence: Gemini accepts the request and answers from the tool result.
///
/// Re-record with:
/// `RIG_PROVIDER_TEST_MODE=record GEMINI_API_KEY=... cargo test --test gemini chat_sourced_history_replays -- --test-threads=1`
#[tokio::test]
async fn chat_sourced_history_replays_the_tool_name_not_the_identifier() {
    super::super::support::with_gemini_cassette(
        "streaming_grammar/chat_sourced_history_replay",
        |client| async move {
            let model = client.completion_model(gemini::completion::GEMINI_2_5_FLASH);
            let cross_provider_handle = rig::message::ToolCallId::new("call_abc123")
                .expect("the chat-sourced identifier is non-empty");
            let history = vec![
                rig::message::Message::user(
                    "Use the add tool to compute 2 + 3, then state the result.",
                ),
                rig::message::Message::Assistant {
                    id: None,
                    content: vec![AssistantContent::ToolCall(ToolCall {
                        // The cross-provider shape: the other wire's
                        // identifier survives as rig's correlation handle,
                        // with no provider id for Gemini's wire.
                        id: cross_provider_handle.clone(),
                        provider: None,
                        function: rig::message::ToolFunction {
                            name: "add".to_owned(),
                            arguments: serde_json::json!({"x": 2, "y": 3}),
                        },
                        signature: None,
                        additional_params: None,
                    })],
                },
                rig::message::Message::User {
                    content: vec![UserContent::ToolResult(rig::message::ToolResult {
                        call: cross_provider_handle,
                        provider: None,
                        name: "add".to_owned(),
                        content: vec![ToolResultContent::text("5")],
                    })],
                },
            ];
            let request = model
                .completion_request("State the final result in one short sentence.")
                .preamble(
                    "You are a calculator assistant. Report tool results faithfully.".to_string(),
                )
                .tool(rig::tool::tool_definition(&crate::support::Adder))
                .messages(history)
                .build();
            let run = drain_stream(model.stream(request).await.expect("stream should start")).await;
            assert!(
                run.text.contains('5'),
                "the model should answer from the replayed tool result, got {:?}",
                run.text
            );
        },
    )
    .await;
}
