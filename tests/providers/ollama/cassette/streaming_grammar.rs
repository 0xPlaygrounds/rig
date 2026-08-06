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
    Adder, AlphaSignal, BetaSignal, ORDERED_TOOL_STREAM_PREAMBLE, ORDERED_TOOL_STREAM_PROMPT,
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
            // Ollama's wire carries no tool-call id and rig fabricates none:
            // the durable id is absent (empty), and serializers omit it.
            assert!(
                streamed.id.is_empty(),
                "an id-less wire call must not carry a fabricated durable id"
            );
        },
    )
    .await;
}

/// Parallel tool calls on an **id-less** wire: both calls survive as
/// distinct parts with uncorrupted arguments — the 2258 item-0 collapse pin
/// against real traffic. Stream-side distinctness comes from minted
/// accumulation identities; durably, neither call carries a fabricated id.
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
                    streamed.id.is_empty(),
                    "{name} must not carry a fabricated durable id"
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
            // Distinctness is structural — every streamed call is its own
            // aggregated part (asserted above) — not fabricated: all durable
            // ids are absent, so nothing invented can collide or leak
            // upstream. Cross-referencing streamed and aggregated calls
            // happens by internal correlation ids on the public stream.
            assert!(
                run.tool_calls.iter().all(|call| call.id.is_empty()),
                "no id-less call may carry a fabricated durable id"
            );
        },
    )
    .await;
}

/// Two calls to the SAME tool in one turn, on real recorded traffic — the
/// live twin of the corpus pin (#2258 A2 / review 84a43e9e). Ollama's wire
/// carries no tool-call ids, and rig fabricates none: both calls must
/// survive as distinct aggregated parts with their own uncorrupted
/// arguments and the absent (empty) durable id. The old name-as-id scheme
/// collapsed exactly this shape.
///
/// Re-record with (local Ollama daemon with `qwen3:4b` pulled, no key
/// needed):
/// `RIG_PROVIDER_TEST_MODE=record cargo test --test ollama same_tool_called_twice -- --test-threads=1`
#[tokio::test]
async fn same_tool_called_twice_in_one_turn_stays_distinct() {
    with_ollama_cassette("streaming_grammar/same_tool_twice", |client| async move {
        let model = client.completion_model(MODEL);
        let request = model
            .completion_request(
                "/no_think Use the `add` tool twice in this single reply, before any text: \
                 first add 2 and 3, then add 10 and 20. Emit both tool calls together in \
                 this one turn — do not wait for results between them, and do not compute \
                 the sums yourself.",
            )
            .preamble(
                "You are a calculator assistant. You MUST use the provided tools for every \
                 arithmetic operation instead of computing results yourself."
                    .to_string(),
            )
            .tool(rig::tool::tool_definition(&Adder))
            .additional_params(serde_json::json!({ "think": false }))
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
        // Same name, distinct calls: every call keeps its own arguments and
        // the absent durable id — nothing invented can collide.
        for call in &add_calls {
            assert!(
                call.id.is_empty(),
                "an id-less wire call must not carry a fabricated durable id"
            );
            assert!(
                call.function.arguments.is_object(),
                "each call's arguments must assemble uncorrupted, got {:?}",
                call.function.arguments
            );
        }
        let argument_sets: std::collections::HashSet<String> = add_calls
            .iter()
            .map(|call| call.function.arguments.to_string())
            .collect();
        assert!(
            argument_sets.len() >= 2,
            "the two same-name calls must keep distinct argument payloads, got {argument_sets:?}"
        );
        let aggregated_adds = run
            .choice
            .iter()
            .filter(|content| matches!(content, AssistantContent::ToolCall(call) if call.function.name == "add"))
            .count();
        assert_eq!(
            aggregated_adds,
            add_calls.len(),
            "every streamed same-name call must survive as its own aggregated part"
        );
    })
    .await;
}
