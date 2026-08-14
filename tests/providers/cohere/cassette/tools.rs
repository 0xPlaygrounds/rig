//! Cassette-backed Cohere tool-calling coverage.

use rig::completion::{
    AssistantContent, CompletionModel, FinishReason, Prompt, ToolDefinition, message::ToolChoice,
};
use rig::prelude::*;

use super::super::{
    CASSETTE_MODEL,
    support::{IntegerAdder, IntegerSubtract, with_cohere_cassette},
};
use crate::support::{TOOLS_PREAMBLE, TOOLS_PROMPT, assert_mentions_expected_number};

#[tokio::test]
async fn tool_call_roundtrip() {
    with_cohere_cassette("tools/tool_call_roundtrip", |client| async move {
        let agent = client
            .agent(CASSETTE_MODEL)
            .preamble(TOOLS_PREAMBLE)
            .tool(IntegerAdder)
            .tool(IntegerSubtract)
            .default_max_turns(2)
            .build();

        let response = agent
            .prompt(TOOLS_PROMPT)
            .await
            .expect("tool prompt should succeed");

        assert_mentions_expected_number(&response, -3);
    })
    .await;
}

/// Asserted on a single completion rather than through the agent loop: Cohere
/// applies `REQUIRED` to every turn, so an agent configured this way is forced to
/// keep calling tools and never reaches a final text answer.
#[tokio::test]
async fn required_tool_choice_is_accepted() {
    with_cohere_cassette(
        "tools/required_tool_choice_is_accepted",
        |client| async move {
            let model = client.completion_model(CASSETTE_MODEL);
            let request = model
                .completion_request(TOOLS_PROMPT)
                .preamble(TOOLS_PREAMBLE.to_string())
                .tool(ToolDefinition {
                    name: "subtract".to_string(),
                    description: "Subtract y from x (i.e.: x - y)".to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "x": {"type": "integer", "description": "The number to subtract from"},
                            "y": {"type": "integer", "description": "The number to subtract"}
                        },
                        "required": ["x", "y"]
                    }),
                })
                .tool_choice(ToolChoice::Required)
                .build();

            let response = model
                .completion(request)
                .await
                .expect("required tool choice should be accepted");

            assert_eq!(
                response.finish_reason(),
                Some(FinishReason::ToolCalls),
                "REQUIRED should force a tool call"
            );

            let tool_call = response
                .choice
                .iter()
                .find_map(|content| match content {
                    AssistantContent::ToolCall(tool_call) => Some(tool_call.clone()),
                    _ => None,
                })
                .expect("response should contain a tool call");
            assert_eq!(tool_call.function.name, "subtract");
            assert_eq!(
                tool_call.function.arguments,
                serde_json::json!({"x": 2, "y": 5})
            );
        },
    )
    .await;
}

#[tokio::test]
async fn required_tool_choice_selects_from_multiple_tools() {
    with_cohere_cassette(
        "tools/required_tool_choice_selects_from_multiple_tools",
        |client| async move {
            let model = client.completion_model(CASSETTE_MODEL);
            let request = model
                .completion_request("Use the correct tool to calculate 9 - 4.")
                .tool(rig::tool::tool_definition(&IntegerAdder))
                .tool(rig::tool::tool_definition(&IntegerSubtract))
                .tool_choice(ToolChoice::Required)
                .max_tokens(128)
                .build();

            let response = model
                .completion(request)
                .await
                .expect("REQUIRED with multiple tools should succeed");
            let tool_calls = response
                .choice
                .iter()
                .filter_map(|content| match content {
                    AssistantContent::ToolCall(tool_call) => Some(tool_call),
                    _ => None,
                })
                .collect::<Vec<_>>();

            assert_eq!(response.finish_reason(), Some(FinishReason::ToolCalls));
            assert_eq!(tool_calls.len(), 1, "expected exactly one tool call");
            assert_eq!(tool_calls[0].function.name, "subtract");
            assert_eq!(
                tool_calls[0].function.arguments,
                serde_json::json!({"x": 9, "y": 4})
            );
        },
    )
    .await;
}

#[tokio::test]
async fn none_tool_choice_with_tools_returns_text() {
    with_cohere_cassette(
        "tools/none_tool_choice_with_tools_returns_text",
        |client| async move {
            let model = client.completion_model(CASSETTE_MODEL);
            let request = model
                .completion_request("Calculate 9 - 4. Answer directly without calling a tool.")
                .tool(rig::tool::tool_definition(&IntegerSubtract))
                .tool_choice(ToolChoice::None)
                .max_tokens(32)
                .build();

            let response = model
                .completion(request)
                .await
                .expect("NONE with tools should produce a direct response");

            assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
            assert!(
                response
                    .choice
                    .iter()
                    .any(|content| matches!(content, AssistantContent::Text(text) if !text.text.trim().is_empty())),
                "NONE should produce text"
            );
            assert!(
                response
                    .choice
                    .iter()
                    .all(|content| !matches!(content, AssistantContent::ToolCall(_))),
                "NONE must suppress tool calls"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn none_tool_choice_without_tools_returns_text() {
    with_cohere_cassette(
        "tools/none_tool_choice_without_tools_returns_text",
        |client| async move {
            let model = client.completion_model(CASSETTE_MODEL);
            let request = model
                .completion_request("Reply with the single word ready.")
                .tool_choice(ToolChoice::None)
                .max_tokens(16)
                .build();

            let response = model
                .completion(request)
                .await
                .expect("Cohere permits NONE without a tools parameter");

            assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
            assert!(
                response
                    .choice
                    .iter()
                    .any(|content| matches!(content, AssistantContent::Text(text) if !text.text.trim().is_empty())),
                "NONE without tools should still produce text"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn strict_required_tool_choice_is_accepted() {
    with_cohere_cassette(
        "tools/strict_required_tool_choice_is_accepted",
        |client| async move {
            let model = client.completion_model(CASSETTE_MODEL);
            let request = model
                .completion_request("Use the subtract tool to calculate 11 - 6.")
                .tool(rig::tool::tool_definition(&IntegerSubtract))
                .tool_choice(ToolChoice::Required)
                .additional_params(serde_json::json!({"strict_tools": true}))
                .max_tokens(128)
                .build();

            let response = model
                .completion(request)
                .await
                .expect("strict_tools should compose with REQUIRED");
            let tool_call = response
                .choice
                .iter()
                .find_map(|content| match content {
                    AssistantContent::ToolCall(tool_call) => Some(tool_call),
                    _ => None,
                })
                .expect("REQUIRED should produce a tool call");

            assert_eq!(response.finish_reason(), Some(FinishReason::ToolCalls));
            assert_eq!(tool_call.function.name, "subtract");
            assert_eq!(
                tool_call.function.arguments,
                serde_json::json!({"x": 11, "y": 6})
            );
        },
    )
    .await;
}
