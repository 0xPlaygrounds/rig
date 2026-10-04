//! Cassette-backed Cohere tool-calling coverage.

use rig::completion::{AssistantContent, FinishReason, ToolDefinition, message::ToolChoice};

use super::super::{
    CASSETTE_MODEL,
    support::{IntegerAdder, IntegerSubtract, with_cohere_cassette},
};
use crate::support::{TOOLS_PREAMBLE, TOOLS_PROMPT, assert_mentions_expected_number};
use rig::completion::CompletionRequest;

#[tokio::test]
async fn tool_call_roundtrip() {
    with_cohere_cassette("tools/tool_call_roundtrip", |client| async move {
        let agent = rig::AgentBuilder::new(client.completion(CASSETTE_MODEL))
            .preamble(TOOLS_PREAMBLE)
            .tool(IntegerAdder)
            .tool(IntegerSubtract)
            .default_max_turns(2)
            .build();

        let response = agent
            .prompt(TOOLS_PROMPT)
            .await
            .expect("tool prompt should succeed");

        assert_mentions_expected_number(&response.output(), -3);
    })
    .await;
}

/// Asserted on a single completion rather than through the agent loop: Cohere
/// applies a required choice to every turn, so an agent configured this way is forced to
/// keep calling tools and never reaches a final text answer.
#[tokio::test]
async fn required_tool_choice_is_accepted() {
    with_cohere_cassette(
        "tools/required_tool_choice_is_accepted",
        |client| async move {
            let model = client.completion(CASSETTE_MODEL);
            let request = CompletionRequest::new(TOOLS_PROMPT)
                .preamble(TOOLS_PREAMBLE.to_string())
                .tool(ToolDefinition {
                    name: rig_core::message::ToolName::new("subtract").expect("tool name"),
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
                .tool_choice(ToolChoice::Required);

            let response = model
                .call(request)
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
                tool_call.function.arguments_value(),
                serde_json::json!({"x": 2, "y": 5})
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
            let mut model = client.completion(CASSETTE_MODEL);
            model.wire = model.wire.with_strict_tools();
            let request = CompletionRequest::new("Use the subtract tool to calculate 11 - 6.")
                .tool(rig::tool::tool_definition(&IntegerSubtract))
                .tool_choice(ToolChoice::Required)
                .max_tokens(128);

            let response = model
                .call(request)
                .await
                .expect("strict tools should compose with a required choice");
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
                tool_call.function.arguments_value(),
                serde_json::json!({"x": 11, "y": 6})
            );
        },
    )
    .await;
}
