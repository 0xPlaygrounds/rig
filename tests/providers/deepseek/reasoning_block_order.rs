//! Edge matrix for rig#2354: DeepSeek's blocking normalizer appended the
//! reasoning block *after* the tool calls, while its stream emits reasoning
//! first — so the two transports returned differently ordered choices for the
//! same turn.
//!
//! `deepseek.rs`'s `NormalizeCompletionResponse` built `[text?, tool_call…]`
//! and then pushed the reasoning block onto the end. DeepSeek's own stream
//! delivers every `reasoning_content` delta before the first `content` delta
//! and before the tool call (`reasoning_tool_roundtrip/streaming.yaml`:
//! 27 reasoning chunks, then 12 tool-call chunks, then `finish_reason`), and
//! the shared canonical chunk lifecycle fixes that same order — reasoning,
//! then text, then tool events. This matrix is deliberately scoped to
//! DeepSeek's blocking/streaming parity; gateway-specific normalizers such as
//! OpenRouter own separate ordering policies.
//!
//! No data was lost, only reordered; these cells pin that both transports now
//! agree. The blocking enumeration over every (reasoning × text × 0/1/2 tool
//! calls) shape is a unit test in `providers/deepseek.rs`
//! (`deepseek_reasoning_leads_the_choice_on_every_turn_shape`) because the
//! live model cannot be made to produce each shape on demand.

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use rig::completion::{
    Chat, CompletionModel, Message, NormalizeCompletionResponse, ToolDefinition,
};
use rig::message::AssistantContent;
use rig::prelude::*;
use rig::providers::deepseek;
use rig::streaming::StreamingChat;
use serde_json::{Value, json};

use super::support::{collect_raw_stream_outcome, with_deepseek_block_order_cassette_result};
use crate::reasoning::{self, WeatherTool};
use crate::support::collect_stream_observation;

const MODEL: &str = deepseek::DEEPSEEK_V4_FLASH;
/// A reasoner turn spends most of its budget on thinking tokens before it can
/// reach the tool call, so these cells need real headroom.
const REASONER_BUDGET: u64 = 640;

fn thinking_params() -> Value {
    json!({ "thinking": { "type": "enabled" } })
}

fn non_thinking_params() -> Value {
    json!({ "thinking": { "type": "disabled" } })
}

fn weather_tool_definition() -> ToolDefinition {
    ToolDefinition {
        name: "get_weather".to_owned(),
        description: "Get the current weather for a city. Must be called for weather questions."
            .to_owned(),
        parameters: json!({
            "type": "object",
            "properties": {
                "city": { "type": "string", "description": "City name to get weather for" },
            },
            "required": ["city"],
        }),
    }
}

fn air_quality_tool_definition() -> ToolDefinition {
    ToolDefinition {
        name: "get_air_quality".to_owned(),
        description: "Get the current air quality index for a city.".to_owned(),
        parameters: json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"],
        }),
    }
}

fn block_kinds(choice: &[AssistantContent]) -> Vec<&'static str> {
    choice
        .iter()
        .map(|content| match content {
            AssistantContent::Text(_) => "text",
            AssistantContent::ToolCall(_) => "tool_call",
            AssistantContent::Reasoning(_) => "reasoning",
            AssistantContent::Image(_) => "image",
        })
        .collect()
}

/// The invariant both transports must hold: reasoning leads, then text, then
/// tool calls. Stated as first-index comparisons so a turn that happens not to
/// speak (or not to call) still checks what it did produce.
fn assert_reasoning_leads(kinds: &[&str], context: &str) {
    let first = |kind: &str| kinds.iter().position(|found| *found == kind);
    let reasoning = first("reasoning")
        .unwrap_or_else(|| panic!("{context}: premise requires a reasoning block, got {kinds:?}"));
    if let Some(text) = first("text") {
        assert!(
            reasoning < text,
            "{context}: reasoning must precede text, got {kinds:?}"
        );
    }
    if let Some(tool_call) = first("tool_call") {
        assert!(
            reasoning < tool_call,
            "{context}: reasoning must precede the tool call, got {kinds:?}"
        );
    }
}

// ================================================================
// A. Reasoner turn that calls one tool
// ================================================================

#[tokio::test]
async fn blocking_reasoner_tool_turn_leads_with_reasoning() {
    with_deepseek_block_order_cassette_result(
        "reasoning_block_order/blocking_reasoner_tool_turn_leads_with_reasoning",
        |client| async move {
            let model = client.completion_model(MODEL);
            let normalized = model
                .raw_completion(
                    model
                        .completion_request(reasoning::TOOL_USER_PROMPT)
                        .preamble(reasoning::TOOL_SYSTEM_PROMPT.to_owned())
                        .tool(weather_tool_definition())
                        .additional_params(thinking_params())
                        .max_tokens(REASONER_BUDGET)
                        .build(),
                )
                .await?
                .normalize("deepseek")?;

            let kinds = block_kinds(&normalized.choice);
            assert!(
                kinds.contains(&"tool_call"),
                "premise requires a tool call: {kinds:?}"
            );
            assert_reasoning_leads(&kinds, "blocking reasoner tool turn");
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("blocking_reasoner_tool_turn_leads_with_reasoning should replay from its cassette");
}

#[tokio::test]
async fn streaming_reasoner_tool_turn_leads_with_reasoning() {
    with_deepseek_block_order_cassette_result(
        "reasoning_block_order/streaming_reasoner_tool_turn_leads_with_reasoning",
        |client| async move {
            let model = client.completion_model(MODEL);
            let outcome = collect_raw_stream_outcome(
                model
                    .stream(
                        model
                            .completion_request(reasoning::TOOL_USER_PROMPT)
                            .preamble(reasoning::TOOL_SYSTEM_PROMPT.to_owned())
                            .tool(weather_tool_definition())
                            .additional_params(thinking_params())
                            .max_tokens(REASONER_BUDGET)
                            .build(),
                    )
                    .await?,
            )
            .await;

            assert!(
                outcome.order.contains(&"tool_call"),
                "premise requires a tool call: {:?}",
                outcome.order
            );
            assert_reasoning_leads(&outcome.order, "streaming reasoner tool turn");
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("streaming_reasoner_tool_turn_leads_with_reasoning should replay from its cassette");
}

// ================================================================
// B. Reasoner turn that calls two tools
// ================================================================

#[tokio::test]
async fn blocking_reasoner_parallel_tool_turn_leads_with_reasoning() {
    with_deepseek_block_order_cassette_result(
        "reasoning_block_order/blocking_reasoner_parallel_tool_turn_leads_with_reasoning",
        |client| async move {
            let model = client.completion_model(MODEL);
            let normalized = model
                .raw_completion(
                    model
                        .completion_request(
                            "What is the weather AND the air quality in Tokyo? Call both tools in one turn before answering.",
                        )
                        .preamble(reasoning::TOOL_SYSTEM_PROMPT.to_owned())
                        .tool(weather_tool_definition())
                        .tool(air_quality_tool_definition())
                        .additional_params(json!({
                            "thinking": { "type": "enabled" },
                            "parallel_tool_calls": true,
                        }))
                        .max_tokens(REASONER_BUDGET)
                        .build(),
                )
                .await?
                .normalize("deepseek")?;

            let kinds = block_kinds(&normalized.choice);
            assert!(
                kinds.iter().filter(|kind| **kind == "tool_call").count() >= 1,
                "premise requires at least one tool call: {kinds:?}"
            );
            assert_reasoning_leads(&kinds, "blocking reasoner parallel tool turn");
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("blocking_reasoner_parallel_tool_turn_leads_with_reasoning should replay from its cassette");
}

#[tokio::test]
async fn streaming_reasoner_parallel_tool_turn_leads_with_reasoning() {
    with_deepseek_block_order_cassette_result(
        "reasoning_block_order/streaming_reasoner_parallel_tool_turn_leads_with_reasoning",
        |client| async move {
            let model = client.completion_model(MODEL);
            let outcome = collect_raw_stream_outcome(
                model
                    .stream(
                        model
                            .completion_request(
                                "What is the weather AND the air quality in Tokyo? Call both tools in one turn before answering.",
                            )
                            .preamble(reasoning::TOOL_SYSTEM_PROMPT.to_owned())
                            .tool(weather_tool_definition())
                            .tool(air_quality_tool_definition())
                            .additional_params(json!({
                                "thinking": { "type": "enabled" },
                                "parallel_tool_calls": true,
                            }))
                            .max_tokens(REASONER_BUDGET)
                            .build(),
                    )
                    .await?,
            )
            .await;

            assert!(
                outcome.order.contains(&"tool_call"),
                "premise requires a tool call: {:?}",
                outcome.order
            );
            assert_reasoning_leads(&outcome.order, "streaming reasoner parallel tool turn");
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("streaming_reasoner_parallel_tool_turn_leads_with_reasoning should replay from its cassette");
}

// ================================================================
// C. Reasoner turn that only speaks
// ================================================================

#[tokio::test]
async fn blocking_reasoner_text_turn_leads_with_reasoning() {
    with_deepseek_block_order_cassette_result(
        "reasoning_block_order/blocking_reasoner_text_turn_leads_with_reasoning",
        |client| async move {
            let model = client.completion_model(MODEL);
            let normalized = model
                .raw_completion(
                    model
                        .completion_request("Is 91 prime? Answer in one short sentence.")
                        .additional_params(thinking_params())
                        .max_tokens(REASONER_BUDGET)
                        .build(),
                )
                .await?
                .normalize("deepseek")?;

            let kinds = block_kinds(&normalized.choice);
            assert!(
                kinds.contains(&"text"),
                "premise requires spoken text: {kinds:?}"
            );
            assert_reasoning_leads(&kinds, "blocking reasoner text turn");
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("blocking_reasoner_text_turn_leads_with_reasoning should replay from its cassette");
}

#[tokio::test]
async fn streaming_reasoner_text_turn_leads_with_reasoning() {
    with_deepseek_block_order_cassette_result(
        "reasoning_block_order/streaming_reasoner_text_turn_leads_with_reasoning",
        |client| async move {
            let model = client.completion_model(MODEL);
            let outcome = collect_raw_stream_outcome(
                model
                    .stream(
                        model
                            .completion_request("Is 91 prime? Answer in one short sentence.")
                            .additional_params(thinking_params())
                            .max_tokens(REASONER_BUDGET)
                            .build(),
                    )
                    .await?,
            )
            .await;

            assert!(
                outcome.order.contains(&"text"),
                "premise requires spoken text: {:?}",
                outcome.order
            );
            assert_reasoning_leads(&outcome.order, "streaming reasoner text turn");
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("streaming_reasoner_text_turn_leads_with_reasoning should replay from its cassette");
}

// ================================================================
// D. Controls: a non-thinking turn has no reasoning block on either transport
// ================================================================

#[tokio::test]
async fn blocking_non_thinking_tool_turn_has_no_reasoning_block() {
    with_deepseek_block_order_cassette_result(
        "reasoning_block_order/blocking_non_thinking_tool_turn_has_no_reasoning_block",
        |client| async move {
            let model = client.completion_model(MODEL);
            let normalized = model
                .raw_completion(
                    model
                        .completion_request(reasoning::TOOL_USER_PROMPT)
                        .preamble(reasoning::TOOL_SYSTEM_PROMPT.to_owned())
                        .tool(weather_tool_definition())
                        .additional_params(non_thinking_params())
                        .max_tokens(256)
                        .build(),
                )
                .await?
                .normalize("deepseek")?;

            let kinds = block_kinds(&normalized.choice);
            assert!(
                !kinds.contains(&"reasoning"),
                "a non-thinking turn carries no reasoning block: {kinds:?}"
            );
            assert!(kinds.contains(&"tool_call"), "premise: {kinds:?}");
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect(
        "blocking_non_thinking_tool_turn_has_no_reasoning_block should replay from its cassette",
    );
}

#[tokio::test]
async fn streaming_non_thinking_tool_turn_has_no_reasoning_block() {
    with_deepseek_block_order_cassette_result(
        "reasoning_block_order/streaming_non_thinking_tool_turn_has_no_reasoning_block",
        |client| async move {
            let model = client.completion_model(MODEL);
            let outcome = collect_raw_stream_outcome(
                model
                    .stream(
                        model
                            .completion_request(reasoning::TOOL_USER_PROMPT)
                            .preamble(reasoning::TOOL_SYSTEM_PROMPT.to_owned())
                            .tool(weather_tool_definition())
                            .additional_params(non_thinking_params())
                            .max_tokens(256)
                            .build(),
                    )
                    .await?,
            )
            .await;

            assert!(
                !outcome.order.contains(&"reasoning"),
                "a non-thinking stream carries no reasoning block: {:?}",
                outcome.order
            );
            assert!(
                outcome.order.contains(&"tool_call"),
                "premise: {:?}",
                outcome.order
            );
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect(
        "streaming_non_thinking_tool_turn_has_no_reasoning_block should replay from its cassette",
    );
}

// ================================================================
// E. Agent level: the same order reaches persisted history and the stream
// ================================================================

#[tokio::test]
async fn agent_blocking_reasoner_roundtrip_keeps_reasoning_first_in_history() {
    with_deepseek_block_order_cassette_result(
        "reasoning_block_order/agent_blocking_reasoner_roundtrip_keeps_reasoning_first_in_history",
        |client| async move {
            let call_count = Arc::new(AtomicUsize::new(0));
            let agent = client
                .agent(MODEL)
                .preamble(reasoning::TOOL_SYSTEM_PROMPT)
                .tool(WeatherTool::new(call_count.clone()))
                .additional_params(thinking_params())
                .max_tokens(REASONER_BUDGET)
                .default_max_turns(3)
                .build();

            let mut history = Vec::<Message>::new();
            agent
                .chat(reasoning::TOOL_USER_PROMPT, &mut history)
                .await?;

            let assistant_turns = history
                .iter()
                .filter_map(|message| match message {
                    Message::Assistant { content, .. } => Some(block_kinds(content)),
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert!(
                !assistant_turns.is_empty(),
                "the agent should persist at least one assistant turn"
            );
            let reasoning_turns = assistant_turns
                .iter()
                .filter(|kinds| kinds.contains(&"reasoning"))
                .collect::<Vec<_>>();
            assert!(
                !reasoning_turns.is_empty(),
                "premise requires a persisted reasoning block: {assistant_turns:?}"
            );
            for kinds in reasoning_turns {
                assert_reasoning_leads(kinds, "agent history assistant turn");
            }
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("agent_blocking_reasoner_roundtrip_keeps_reasoning_first_in_history should replay from its cassette");
}

#[tokio::test]
async fn agent_streaming_reasoner_roundtrip_streams_reasoning_first() {
    with_deepseek_block_order_cassette_result(
        "reasoning_block_order/agent_streaming_reasoner_roundtrip_streams_reasoning_first",
        |client| async move {
            let call_count = Arc::new(AtomicUsize::new(0));
            let agent = client
                .agent(MODEL)
                .preamble(reasoning::TOOL_SYSTEM_PROMPT)
                .tool(WeatherTool::new(call_count.clone()))
                .additional_params(thinking_params())
                .max_tokens(REASONER_BUDGET)
                .build();

            let mut stream = agent
                .stream_chat(reasoning::TOOL_USER_PROMPT, Vec::<Message>::new())
                .max_turns(3)
                .await;
            let observation = collect_stream_observation(&mut stream).await;

            let reasoning_at = observation
                .events
                .iter()
                .position(|event| *event == "reasoning" || *event == "reasoning_delta");
            let tool_call_at = observation
                .events
                .iter()
                .position(|event| *event == "tool_call");
            assert!(
                reasoning_at.is_some(),
                "premise requires streamed reasoning: {:?}",
                observation.events
            );
            if let (Some(reasoning_at), Some(tool_call_at)) = (reasoning_at, tool_call_at) {
                assert!(
                    reasoning_at < tool_call_at,
                    "streamed reasoning must precede the tool call: {:?}",
                    observation.events
                );
            }
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("agent_streaming_reasoner_roundtrip_streams_reasoning_first should replay from its cassette");
}
