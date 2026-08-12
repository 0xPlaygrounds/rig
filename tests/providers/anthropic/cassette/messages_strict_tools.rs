//! Anthropic Messages API strict-tool regression tests.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use rig::completion::{CompletionModel, ToolDefinition};
use rig::message::{AssistantContent, ToolChoice};
use rig::prelude::*;
use rig::providers::anthropic;
use serde_json::json;

use super::super::support::with_anthropic_cassette;

#[tokio::test]
async fn strict_tools_opt_in_roundtrip() {
    with_anthropic_cassette(
        "messages_strict_tools/strict_tools_opt_in_roundtrip",
        |client| async move {
            let model = client
                .completion_model(anthropic::completion::CLAUDE_SONNET_4_6)
                .with_strict_tools();
            let request = model
                .completion_request(
                    "Call record_booking exactly once with passengers = 2 and cabin = economy.",
                )
                .preamble("Follow the tool-calling instruction exactly.".to_string())
                .max_tokens(1024)
                .tool_choice(ToolChoice::Required)
                .tool(ToolDefinition {
                    name: "record_booking".to_string(),
                    description: "Record a passenger count and cabin class.".to_string(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "passengers": { "type": "integer" },
                            "cabin": {
                                "type": "string",
                                "enum": ["economy", "business"]
                            }
                        },
                        "required": ["passengers", "cabin"],
                        "additionalProperties": false
                    }),
                })
                .build();

            let response = model
                .completion(request)
                .await
                .expect("strict-tools completion should succeed");

            let tool_call = response
                .choice
                .iter()
                .find_map(|content| match content {
                    AssistantContent::ToolCall(tool_call) => Some(tool_call),
                    _ => None,
                })
                .expect("strict tool call should be produced");
            assert_eq!(tool_call.function.name, "record_booking");
            assert_eq!(tool_call.function.arguments["passengers"], json!(2));
            assert_eq!(tool_call.function.arguments["cabin"], json!("economy"));
        },
    )
    .await;
}
