//! Cassette-backed Cohere tool-calling coverage.
//!
//! Cohere returns tool call arguments as a JSON *string* and expects tool results
//! back as `role: "tool"` messages keyed by `tool_call_id`, so the round trip
//! exercises both halves of the provider's message conversion.

use rig::completion::{
    AssistantContent, CompletionModel, Prompt, ToolDefinition, message::ToolChoice,
};
use rig::prelude::*;
use rig::providers::cohere::completion::FinishReason;

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

/// `tool_choice` has to reach Cohere as the bare string `REQUIRED`. Rig's own
/// `ToolChoice` serializes to a tagged object, which the API rejects with
/// `parameter 'tool_choice' is of type object but should be of type string`.
///
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
                response.raw_response.finish_reason,
                FinishReason::ToolCall,
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
