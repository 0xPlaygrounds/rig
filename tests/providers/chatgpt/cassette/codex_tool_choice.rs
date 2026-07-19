//! ChatGPT/Codex Responses backend `tool_choice` regression tests.
//!
//! Locks down every `tool_choice` shape the Responses API accepts from Rig:
//! `required` (forced tool use), `none` (suppressed tool use), a single
//! specific function (`{"type": "function", "name": ...}`), and multiple
//! specific functions (`{"type": "allowed_tools", ...}`).
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use rig::agent::tool::Tool;
use rig::client::CompletionClient;
use rig::completion::CompletionModel;
use rig::message::{AssistantContent, ToolChoice};
use rig::providers::chatgpt;

use super::super::support::with_chatgpt_cassette;
use crate::support::{Adder, AlphaSignal, Subtract, TOOLS_PREAMBLE};

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
async fn required_forces_a_tool_call() {
    with_chatgpt_cassette(
        "codex_tool_choice/required_forces_a_tool_call",
        |client| async move {
            let model = client.completion_model(chatgpt::GPT_5_4);
            let request = model
                .completion_request("Please greet me.")
                .preamble(TOOLS_PREAMBLE.to_string())
                .tool(rig::agent::tool::tool_definition(&Adder))
                .tool_choice(ToolChoice::Required)
                .build();

            let response = model
                .completion(request)
                .await
                .expect("required tool choice completion should succeed");

            let names = tool_call_names(&response.choice);
            assert!(
                !names.is_empty(),
                "tool_choice=required must force a tool call even for a chat prompt, got {:?}",
                response.choice
            );
            assert!(
                names.iter().all(|name| name == Adder::NAME),
                "only the provided tool can be called, saw {names:?}"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn none_suppresses_tool_calls() {
    with_chatgpt_cassette(
        "codex_tool_choice/none_suppresses_tool_calls",
        |client| async move {
            let model = client.completion_model(chatgpt::GPT_5_4);
            let request = model
                .completion_request("What is 2 plus 3? Reply with just the number.")
                .preamble(TOOLS_PREAMBLE.to_string())
                .tool(rig::agent::tool::tool_definition(&Adder))
                .tool_choice(ToolChoice::None)
                .build();

            let response = model
                .completion(request)
                .await
                .expect("none tool choice completion should succeed");

            let names = tool_call_names(&response.choice);
            assert!(
                names.is_empty(),
                "tool_choice=none must suppress tool calls, saw {names:?}"
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
                text.contains('5'),
                "model should answer directly without tools, got {text:?}"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn specific_single_function_targets_named_tool() {
    with_chatgpt_cassette(
        "codex_tool_choice/specific_single_function_targets_named_tool",
        |client| async move {
            let model = client.completion_model(chatgpt::GPT_5_4);
            let request = model
                .completion_request("Compute 9 minus 4 using a tool.")
                .preamble(TOOLS_PREAMBLE.to_string())
                .tool(rig::agent::tool::tool_definition(&Adder))
                .tool(rig::agent::tool::tool_definition(&Subtract))
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
                .expect("a specific tool choice must produce a tool call");
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

#[tokio::test]
async fn specific_multiple_functions_use_allowed_tools() {
    with_chatgpt_cassette(
        "codex_tool_choice/specific_multiple_functions_use_allowed_tools",
        |client| async move {
            let model = client.completion_model(chatgpt::GPT_5_4);
            let request = model
                .completion_request("What is 2 plus 3? Use exactly one tool.")
                .preamble(TOOLS_PREAMBLE.to_string())
                .tool(rig::agent::tool::tool_definition(&Adder))
                .tool(rig::agent::tool::tool_definition(&Subtract))
                .tool(rig::agent::tool::tool_definition(&AlphaSignal))
                .tool_choice(ToolChoice::Specific {
                    function_names: vec![Adder::NAME.to_string(), Subtract::NAME.to_string()],
                })
                .build();

            let response = model
                .completion(request)
                .await
                .expect("allowed-tools tool choice completion should succeed");

            let names = tool_call_names(&response.choice);
            assert!(
                !names.is_empty(),
                "allowed_tools in required mode must force a tool call, got {:?}",
                response.choice
            );
            assert!(
                names
                    .iter()
                    .all(|name| name == Adder::NAME || name == Subtract::NAME),
                "only allowed tools may be called, saw {names:?}"
            );
            assert!(
                names.iter().any(|name| name == Adder::NAME),
                "an addition prompt restricted to add/subtract should call add, saw {names:?}"
            );
        },
    )
    .await;
}
