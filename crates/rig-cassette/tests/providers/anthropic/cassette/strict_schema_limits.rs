//! Live boundary tests for Anthropic's published strict-schema complexity limits.

use rig::completion::ToolDefinition;
use rig::message::{AssistantContent, ToolChoice};
use rig::providers::anthropic;
use rig_test_support::cassette_models::MapWire;
use serde_json::{Value, json};

use super::super::support::with_anthropic_cassette;
use rig::completion::CompletionRequest;

fn assert_single_tool_call(
    response: &rig::completion::CompletionResponse,
    expected_name: &str,
    expected_arguments: &Value,
) {
    let calls = response
        .choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::ToolCall(tool_call) => Some(tool_call),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(calls.len(), 1, "exactly one tool call is expected");
    assert_eq!(calls[0].function.name, expected_name);
    assert_eq!(&calls[0].function.arguments_value(), expected_arguments);
}

#[tokio::test]
async fn sixteen_union_parameters_across_tools_are_accepted() {
    with_anthropic_cassette(
        "strict_schema_limits/sixteen_union_parameters_across_tools_are_accepted",
        |client| async move {
            let model = client
                .completion(anthropic::completion::CLAUDE_SONNET_4_6)
                .map_wire(|wire| wire.with_strict_tools());
            let tools = (0..16)
                .map(|tool_index| ToolDefinition {
                    name: rig_core::message::ToolName::new(format!("union_tool_{tool_index:02}"))
                        .expect("tool name"),
                    description: "A tool with one required nullable parameter.".to_string(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "value": { "type": ["string", "null"] }
                        },
                        "required": ["value"]
                    }),
                })
                .collect::<Vec<_>>();
            let request = CompletionRequest::new("Call union_tool_00 with value = null.")
                .max_tokens(1024)
                .tools(tools)
                .tool_choice(ToolChoice::Specific {
                    function_names: vec![
                        rig_core::message::ToolName::new("union_tool_00").expect("tool name"),
                    ],
                });

            let response = model
                .call(request)
                .await
                .expect("sixteen simple union parameters should be accepted");
            assert_single_tool_call(&response, "union_tool_00", &json!({ "value": null }));
        },
    )
    .await;
}
