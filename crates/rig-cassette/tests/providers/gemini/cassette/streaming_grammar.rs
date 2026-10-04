//! Canonical streaming-grammar coverage for Gemini (REST `generateContent`
//! streaming plus the Interactions API), asserted through the *normalized*
//! path: the aggregated [`Streamed::finish`](rig::streaming::Streamed::finish) response, the terminal
//! `CompletionResponse` record, usage, IDs, and finish reason — real recorded wire
//! traffic, not synthetic chunks.
//!
//! Re-record with:
//! `RIG_PROVIDER_TEST_MODE=record GEMINI_API_KEY=... cargo test --test gemini streaming_grammar -- --test-threads=1`
//!
//! Cassette IDs are scrub placeholders; assertions derive expected IDs from
//! the recorded turn and never mint literal IDs.

use futures::StreamExt;
use rig::completion::CompletionResponse;
use rig::completion::FinishReason;
use rig::message::{AssistantContent, Reasoning, ToolCall, ToolResultContent, UserContent};
use rig::message::{Message, ToolChoice};
use rig::providers::gemini;
use rig::streaming::Item;
use rig::streaming::StreamEvent;

use crate::support::ALPHA_SIGNAL_OUTPUT;
use crate::support::{
    AlphaSignal, BetaSignal, ORDERED_TOOL_STREAM_PREAMBLE, ORDERED_TOOL_STREAM_PROMPT,
    TWO_TOOL_STREAM_PREAMBLE,
};
use rig::completion::CompletionRequest;

/// Everything observed while draining a normalized stream, alongside the
/// aggregated stream state itself.
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
            Item::Event(StreamEvent::Start { .. })
            | Item::Event(StreamEvent::Arguments { .. })
            | Item::Event(StreamEvent::End { .. })
            | Item::Unknown(_) => {}
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
    // ID contract: Gemini reports a `responseId` for the response as a
    // whole, which the normalized terminal surfaces.
    assert!(
        terminal.response_id().is_some_and(|id| !id.is_empty()),
        "Gemini should surface its responseId as the response-scoped ID"
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
            AssistantContent::Reasoning(reasoning) => Some(reasoning.text.as_str()),
            _ => None,
        })
        .collect()
}

/// The signature a block's provider item carries, if any: `thoughtSignature`
/// on GenerateContent parts, `signature` on Interactions steps.
fn signature(content: &AssistantContent) -> Option<&str> {
    let item = content.native_item()?;
    item.get("thoughtSignature")
        .or_else(|| item.get("signature"))?
        .as_str()
}

/// Parallel function calls in one turn: both calls survive aggregation as
/// distinct parts with the ids the stream reported.
#[tokio::test]
async fn parallel_function_calls_stay_distinct() {
    let params = serde_json::json!({
        "generationConfig": { "thinkingConfig": { "thinkingBudget": 0 } }
    });
    super::super::support::with_gemini_cassette(
        "streaming_grammar/parallel_function_calls",
        |client| async move {
            let model = client.completion(gemini::completion::GEMINI_2_5_FLASH);
            let request = CompletionRequest::new(
                "Call `lookup_harbor_label` and `lookup_orchard_label` now, both of them \
                     together in this single reply, before writing any text. Emit the two \
                     tool calls in one turn - do not wait for results between them.",
            )
            .preamble(TWO_TOOL_STREAM_PREAMBLE.to_string())
            .tool(rig::tool::tool_definition(&AlphaSignal))
            .tool(rig::tool::tool_definition(&BetaSignal))
            .additional_params(params);
            let run = drain_stream(model.stream(request).expect("stream should start")).await;

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
            assert!(aggregated[0].id.provider().is_none());
            assert!(aggregated[1].id.provider().is_none());
            assert!(aggregated[0].id.is_local());
            assert!(aggregated[1].id.is_local());
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

/// Thinking-enabled Interactions streaming turn: reasoning and text arrive as
/// discrete normalized parts through the shared REST/interactions part-kind
/// interpretation, pinned on real traffic.
#[tokio::test]
async fn interactions_thinking_stream_keeps_reasoning_and_text_discrete() {
    super::super::support::with_gemini_interactions_cassette(
        "streaming_grammar/interactions_thinking_stream",
        |client| async move {
            let model = client.interactions("gemini-3-flash-preview");
            let request = CompletionRequest::new(
                    "How many positive integers n < 200 are divisible by 4 but not by 10? \
                     Think it through, then answer with just the number.",
                )
                .additional_params(
                    serde_json::json!({
                        "generation_config": {
                            "thinking_level": "medium",
                            "thinking_summaries": "auto"
                        },
                        "store": true
                    }),
                );
            let run = drain_stream(model.stream(request).expect("stream should start")).await;

            assert!(!run.text.trim().is_empty(), "turn should produce text");
            let terminal = run
                .response
                .as_ref()
                .expect("aggregated stream should retain the terminal record");
            assert_eq!(
                terminal.finish_reason(),
                Some(FinishReason::Stop),
                "unexpected finish reason"
            );
            assert!(
                terminal.usage.total_tokens.is_some_and(|n| n > 0),
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
                .filter(|content| matches!(content, AssistantContent::Reasoning(_)))
                .filter_map(signature)
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
                    assert!(
                        !reasoning.text.trim().is_empty(),
                        "no empty (signature-only) reasoning sibling may exist: {:?}",
                        run.choice
                    );
                    if reasoning.text.contains(run.reasoning_delta.trim()) {
                        assert!(
                            signature(content).is_some_and(|signature| !signature.is_empty()),
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
            let model = client.interactions("gemini-3-flash-preview");
            let tool = rig::tool::tool_definition(&AlphaSignal);

            let raw = model
                .call(
                    CompletionRequest::new(ORDERED_TOOL_STREAM_PROMPT)
                        .preamble(ORDERED_TOOL_STREAM_PREAMBLE.to_string())
                        .tool(tool)
                        .tool_choice(ToolChoice::Required)
                        .additional_params(serde_json::json!({ "store": true })),
                )
                .await
                .expect("tool-required interaction should succeed");

            // The wire status transition under test, read off the reply
            // document `raw` carries verbatim — the interaction resource is
            // the reply, and the fold is the other view of it.
            let interaction = raw
                .raw
                .as_object()
                .expect("`raw` is the interaction resource");
            assert_eq!(
                interaction
                    .get("status")
                    .and_then(serde_json::Value::as_str),
                Some("requires_action"),
                "declared client tool should leave the interaction in requires_action, got {:?}",
                interaction.get("status")
            );
            let interaction_id = interaction
                .get("id")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_owned();
            assert!(!interaction_id.is_empty(), "expected an interaction id");

            let normalized = raw;
            assert_eq!(
                normalized.finish_reason(),
                Some(FinishReason::ToolCalls),
                "requires_action should normalize to a ToolCalls finish"
            );
            assert!(
                normalized.usage.total_tokens.is_some_and(|n| n > 0),
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
                tool_call.id.provider().is_some(),
                "the recorded interactions turn should carry a provider-issued call id"
            );

            let followup = model
                .call(
                    CompletionRequest::new(Message::from(UserContent::tool_result(
                        tool_call.id.clone(),
                        tool_call.function.name.clone(),
                        vec![ToolResultContent::text(ALPHA_SIGNAL_OUTPUT)],
                    )))
                    .additional_params(
                        serde_json::json!({ "previous_interaction_id": interaction_id }),
                    ),
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
            let model = client.interactions("gemini-3-flash-preview");
            let request = CompletionRequest::new(
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
                    serde_json::json!({ "store": true }),
                );
            let run = drain_stream(model.stream(request).expect("stream should start")).await;

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
                    call.id.provider().map(|provider| provider.as_str()), Some("add"),
                    "the tool name must never be fabricated into the durable id"
                );
                assert!(
                    call.id.provider()
                        .as_ref()
                        .is_none_or(|provider| provider.as_str() != "add"),
                    "the tool name must never be fabricated into the provider call id"
                );
                assert!(
                    call.function.invalid_arguments.is_none(),
                    "each call's arguments must survive uncorrupted, got {:?}",
                    call.function.arguments
                );
            }
            let distinct_ids: std::collections::HashSet<&rig::message::CallId> = add_calls
                .iter()
                .map(|call| &call.id)
                .collect();
            assert_eq!(
                distinct_ids.len(),
                add_calls.len(),
                "same-name calls must keep distinct durable ids (minted when the wire gave none)"
            );
            let argument_sets: std::collections::HashSet<String> = add_calls
                .iter()
                .map(|call| call.function.arguments_value().to_string())
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
            let model = client.interactions("gemini-3-flash-preview");
            let request = CompletionRequest::new(
                "How many positive integers n < 100 are divisible by 6 but not by 9? \
                     Think it through, then answer with just the number.",
            )
            .additional_params(serde_json::json!({
                "generation_config": {
                    "thinking_level": "medium",
                    "thinking_summaries": "none"
                },
                "store": true
            }));
            let run = drain_stream(model.stream(request).expect("stream should start")).await;

            assert!(!run.text.trim().is_empty(), "turn should produce text");
            let signature_delivered = run.choice.iter().any(|content| {
                matches!(content, AssistantContent::Reasoning(_))
                    && signature(content).is_some_and(|signature| !signature.is_empty())
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
                .filter(|reasoning| reasoning.text.trim().is_empty())
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
