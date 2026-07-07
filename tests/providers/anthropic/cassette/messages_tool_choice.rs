//! Anthropic Messages API `tool_choice` regression tests.
//!
//! Locks down every `tool_choice` shape Rig sends to the Messages API:
//! `required` (mapped to Anthropic's `{"type": "any"}`), `none`, and a
//! specific named tool (`{"type": "tool", "name": ...}`).
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use rig::client::CompletionClient;
use rig::completion::CompletionModel;
use rig::message::{AssistantContent, ToolChoice};
use rig::providers::anthropic;
use rig::tool::Tool;

use super::super::support::with_anthropic_cassette;
use crate::support::{Adder, Subtract, TOOLS_PREAMBLE};

fn tool_call_names(choice: &rig::OneOrMany<AssistantContent>) -> Vec<String> {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::ToolCall(tool_call) => Some(tool_call.function.name.clone()),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn required_maps_to_any_and_forces_tool_use() {
    with_anthropic_cassette(
        "messages_tool_choice/required_maps_to_any_and_forces_tool_use",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6);
            let request = model
                .completion_request("Please greet me.")
                .preamble(TOOLS_PREAMBLE.to_string())
                .max_tokens(1024)
                .tool(rig_core::tool::tool_definition(&Adder))
                .tool_choice(ToolChoice::Required)
                .build();

            let response = model
                .completion(request)
                .await
                .expect("required tool choice completion should succeed");

            let names = tool_call_names(&response.choice);
            assert!(
                !names.is_empty(),
                "tool_choice=required (Anthropic `any`) must force a tool_use even for a \
                 chat prompt, got {:?}",
                response.choice
            );
            assert!(
                names.iter().all(|name| name == Adder::NAME),
                "only the provided tool can be called, saw {names:?}"
            );
            assert_eq!(
                response.raw_response.stop_reason.as_deref(),
                Some("tool_use"),
                "a forced tool_use turn should preserve the tool_use stop reason"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn none_suppresses_tool_use() {
    with_anthropic_cassette(
        "messages_tool_choice/none_suppresses_tool_use",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6);
            // The question must not match the forbidden tool: asking arithmetic
            // with the add tool blocked makes Anthropic return an empty
            // end_turn message instead of answering in text.
            let request = model
                .completion_request("Name the capital of France in one word.")
                .preamble("You are a concise assistant. Answer directly.".to_string())
                .max_tokens(1024)
                .tool(rig_core::tool::tool_definition(&Adder))
                .tool_choice(ToolChoice::None)
                .build();

            let response = model
                .completion(request)
                .await
                .expect("none tool choice completion should succeed");

            let names = tool_call_names(&response.choice);
            assert!(
                names.is_empty(),
                "tool_choice=none must suppress tool_use, saw {names:?}"
            );
            let text: String = response
                .choice
                .iter()
                .filter_map(|content| match content {
                    AssistantContent::Text(text) => Some(text.text.as_str()),
                    _ => None,
                })
                .collect();
            assert!(
                text.to_ascii_lowercase().contains("paris"),
                "model should answer directly without tools, got {text:?}"
            );
            assert_eq!(
                response.raw_response.stop_reason.as_deref(),
                Some("end_turn"),
                "a plain answer should preserve the end_turn stop reason"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn specific_tool_targets_named_tool() {
    with_anthropic_cassette(
        "messages_tool_choice/specific_tool_targets_named_tool",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6);
            let request = model
                .completion_request("Compute 9 minus 4 using a tool.")
                .preamble(TOOLS_PREAMBLE.to_string())
                .max_tokens(1024)
                .tool(rig_core::tool::tool_definition(&Adder))
                .tool(rig_core::tool::tool_definition(&Subtract))
                .tool_choice(ToolChoice::Specific {
                    function_names: vec![Subtract::NAME.to_string()],
                })
                .build();

            let response = model
                .completion(request)
                .await
                .expect("specific tool choice completion should succeed");

            let tool_call = response
                .choice
                .iter()
                .find_map(|content| match content {
                    AssistantContent::ToolCall(tool_call) => Some(tool_call.clone()),
                    _ => None,
                })
                .expect("a specific tool choice must produce a tool_use");
            assert_eq!(
                tool_call.function.name,
                Subtract::NAME,
                "the named tool must be the one called"
            );
            assert_eq!(
                tool_call
                    .function
                    .arguments
                    .get("x")
                    .and_then(|value| value.as_f64()),
                Some(9.0),
                "arguments should reflect the prompt: {:?}",
                tool_call.function.arguments
            );
            assert_eq!(
                tool_call
                    .function
                    .arguments
                    .get("y")
                    .and_then(|value| value.as_f64()),
                Some(4.0),
                "arguments should reflect the prompt: {:?}",
                tool_call.function.arguments
            );
        },
    )
    .await;
}
