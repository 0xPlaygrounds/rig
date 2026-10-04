//! Edge matrix for rig#2354: DeepSeek's blocking normalization appended the
//! reasoning block *after* the tool calls, while its stream emits reasoning
//! first — so the two transports returned differently ordered choices for the
//! same turn.
//!
//! The blocking mapping built `[text?, tool_call…]` and then pushed the
//! reasoning block onto the end. DeepSeek's own stream delivers every
//! `reasoning_content` delta before the first `content` delta and before the
//! tool call (`reasoning_tool_roundtrip/streaming.yaml`: 27 reasoning chunks,
//! then 12 tool-call chunks, then `finish_reason`), and the shared canonical
//! chunk lifecycle fixes that same order — reasoning, then text, then tool
//! events. The order is now structural rather than agreed: one decoder folds
//! the unary body by synthesizing the events a stream of the same turn would
//! have pushed, so there is no second mapping left to reorder anything. This
//! matrix is deliberately scoped to DeepSeek's blocking/streaming parity;
//! gateway-specific policies such as OpenRouter's own ordering are separate.
//!
//! No data was lost, only reordered; these cells pin that both transports
//! agree. The blocking enumeration over every (reasoning × text × 0/1/2 tool
//! calls) shape is a wire unit test, because the live model cannot be made to
//! produce each shape on demand.

use rig::completion::ToolDefinition;
use rig::providers::deepseek;
use serde_json::json;

use super::support::{collect_raw_stream_outcome, with_deepseek_block_order_cassette_result};
use crate::reasoning::{self};
use rig::completion::CompletionRequest;

const MODEL: &str = deepseek::DEEPSEEK_V4_FLASH;
/// A reasoner turn spends most of its budget on thinking tokens before it can
/// reach the tool call, so these cells need real headroom.
const REASONER_BUDGET: u64 = 640;

fn weather_tool_definition() -> ToolDefinition {
    ToolDefinition {
        name: rig_core::message::ToolName::new("get_weather").expect("tool name"),
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
        name: rig_core::message::ToolName::new("get_air_quality").expect("tool name"),
        description: "Get the current air quality index for a city.".to_owned(),
        parameters: json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"],
        }),
    }
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

// ================================================================
// B. Reasoner turn that calls two tools
// ================================================================

#[tokio::test]
async fn streaming_reasoner_parallel_tool_turn_leads_with_reasoning() {
    with_deepseek_block_order_cassette_result(
        "reasoning_block_order/streaming_reasoner_parallel_tool_turn_leads_with_reasoning",
        |client| async move {
            let model = client.completion(MODEL);
            let outcome = collect_raw_stream_outcome(
                model
                    .stream(CompletionRequest::new(
                                "What is the weather AND the air quality in Tokyo? Call both tools in one turn before answering.",
                            )
                            .preamble(reasoning::TOOL_SYSTEM_PROMPT.to_owned())
                            .tool(weather_tool_definition())
                            .tool(air_quality_tool_definition())
                            .additional_params(json!({
                                "thinking": { "type": "enabled" },
                                "parallel_tool_calls": true,
                            }))
                            .max_tokens(REASONER_BUDGET))
                    ?,
            )
            .await;

            assert_eq!(
                outcome.tool_call_names(),
                vec!["get_weather", "get_air_quality"],
                "parallel premise requires both recorded tool calls in order"
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

// ================================================================
// D. Controls: a non-thinking turn has no reasoning block on either transport
// ================================================================

// ================================================================
// E. Agent level: the same order reaches persisted history and the stream
// ================================================================
