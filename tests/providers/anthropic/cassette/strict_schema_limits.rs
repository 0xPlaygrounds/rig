//! Live boundary tests for Anthropic's published strict-schema complexity limits.

use rig::completion::{CompletionError, CompletionModel, ToolDefinition};
use rig::message::{AssistantContent, ToolChoice};
use rig::prelude::*;
use rig::providers::anthropic;
use serde_json::{Map, Value, json};

use super::super::support::with_anthropic_cassette;

fn empty_tool(name: impl Into<String>) -> ToolDefinition {
    ToolDefinition {
        name: name.into(),
        description: "A strict no-argument boundary-test tool.".to_string(),
        parameters: json!({ "type": "object", "properties": {} }),
    }
}

fn optional_parameters_schema(count: usize) -> Value {
    let properties = (0..count)
        .map(|index| (format!("optional_{index:02}"), json!({ "type": "string" })))
        .collect::<Map<_, _>>();
    json!({ "type": "object", "properties": properties })
}

fn union_parameters_schema(count: usize) -> Value {
    let properties = (0..count)
        .map(|index| {
            (
                format!("union_{index:02}"),
                json!({ "type": ["string", "null"] }),
            )
        })
        .collect::<Map<_, _>>();
    let required = properties.keys().cloned().collect::<Vec<_>>();
    json!({
        "type": "object",
        "properties": properties,
        "required": required
    })
}

fn assert_invalid_request(error: &CompletionError) {
    let status = error
        .provider_response_status()
        .expect("provider status should be preserved");
    assert_eq!(status.as_u16(), 400, "unexpected provider status: {status}");
    let body = error
        .provider_response_json()
        .expect("provider error should contain JSON")
        .expect("provider error JSON should be preserved");
    assert_eq!(body["type"], "error");
    assert_eq!(body["error"]["type"], "invalid_request_error");
}

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
    assert_eq!(&calls[0].function.arguments, expected_arguments);
}

#[tokio::test]
async fn twenty_strict_tools_are_accepted() {
    with_anthropic_cassette(
        "strict_schema_limits/twenty_strict_tools_are_accepted",
        |client| async move {
            let model = client
                .completion_model(anthropic::completion::CLAUDE_SONNET_4_6)
                .with_strict_tools();
            let tools = (0..20)
                .map(|index| empty_tool(format!("boundary_tool_{index:02}")))
                .collect::<Vec<_>>();
            let request = model
                .completion_request("Call boundary_tool_19 with an empty object.")
                .preamble("Call only the specifically selected tool.".to_string())
                .max_tokens(1024)
                .tools(tools)
                .tool_choice(ToolChoice::Specific {
                    function_names: vec!["boundary_tool_19".to_string()],
                })
                .build();

            let response = model
                .completion(request)
                .await
                .expect("the documented twenty-strict-tool boundary should succeed");
            assert_single_tool_call(&response, "boundary_tool_19", &json!({}));
        },
    )
    .await;
}

#[tokio::test]
async fn twenty_one_strict_tools_are_rejected() {
    with_anthropic_cassette(
        "strict_schema_limits/twenty_one_strict_tools_are_rejected",
        |client| async move {
            let model = client
                .completion_model(anthropic::completion::CLAUDE_SONNET_4_6)
                .with_strict_tools();
            let request = model
                .completion_request("Call boundary_tool_20 with an empty object.")
                .max_tokens(64)
                .tools(
                    (0..21)
                        .map(|index| empty_tool(format!("boundary_tool_{index:02}")))
                        .collect(),
                )
                .tool_choice(ToolChoice::Specific {
                    function_names: vec!["boundary_tool_20".to_string()],
                })
                .build();

            let error = model
                .completion(request)
                .await
                .expect_err("twenty-one strict tools should exceed the provider limit");
            assert_invalid_request(&error);
        },
    )
    .await;
}

#[tokio::test]
async fn twenty_four_optional_parameters_in_one_schema_hit_internal_limit() {
    with_anthropic_cassette(
        "strict_schema_limits/twenty_four_optional_parameters_in_one_schema_hit_internal_limit",
        |client| async move {
            let model = client
                .completion_model(anthropic::completion::CLAUDE_SONNET_4_6)
                .with_strict_tools();
            let request = model
                .completion_request(
                    "Call optional_boundary with an empty object; omit every optional field.",
                )
                .max_tokens(64)
                .tool_choice(ToolChoice::Required)
                .tool(ToolDefinition {
                    name: "optional_boundary".to_string(),
                    description: "Exercise the strict optional-parameter boundary.".to_string(),
                    parameters: optional_parameters_schema(24),
                })
                .build();

            let error = model
                .completion(request)
                .await
                .expect_err("the provider's internal grammar cap should reject this shape");
            assert_invalid_request(&error);
        },
    )
    .await;
}

#[tokio::test]
async fn twenty_five_optional_parameters_are_rejected() {
    with_anthropic_cassette(
        "strict_schema_limits/twenty_five_optional_parameters_are_rejected",
        |client| async move {
            let model = client
                .completion_model(anthropic::completion::CLAUDE_SONNET_4_6)
                .with_strict_tools();
            let request = model
                .completion_request("Call optional_boundary with an empty object.")
                .max_tokens(64)
                .tool_choice(ToolChoice::Required)
                .tool(ToolDefinition {
                    name: "optional_boundary".to_string(),
                    description: "Exceed the strict optional-parameter boundary.".to_string(),
                    parameters: optional_parameters_schema(25),
                })
                .build();

            let error = model
                .completion(request)
                .await
                .expect_err("twenty-five optional parameters should exceed the provider limit");
            assert_invalid_request(&error);
        },
    )
    .await;
}

#[tokio::test]
async fn sixteen_union_parameters_in_one_schema_hit_internal_limit() {
    with_anthropic_cassette(
        "strict_schema_limits/sixteen_union_parameters_in_one_schema_hit_internal_limit",
        |client| async move {
            let model = client
                .completion_model(anthropic::completion::CLAUDE_SONNET_4_6)
                .with_strict_tools();
            let request = model
                .completion_request(
                    "Call union_boundary and set every union_00 through union_15 field to null.",
                )
                .max_tokens(64)
                .tool_choice(ToolChoice::Required)
                .tool(ToolDefinition {
                    name: "union_boundary".to_string(),
                    description: "Exercise the strict union-parameter boundary.".to_string(),
                    parameters: union_parameters_schema(16),
                })
                .build();

            let error = model
                .completion(request)
                .await
                .expect_err("the provider's internal grammar cap should reject this shape");
            assert_invalid_request(&error);
        },
    )
    .await;
}

#[tokio::test]
async fn twenty_four_optional_parameters_across_tools_are_accepted() {
    with_anthropic_cassette(
        "strict_schema_limits/twenty_four_optional_parameters_across_tools_are_accepted",
        |client| async move {
            let model = client
                .completion_model(anthropic::completion::CLAUDE_SONNET_4_6)
                .with_strict_tools();
            let tools = (0..12)
                .map(|tool_index| ToolDefinition {
                    name: format!("optional_tool_{tool_index:02}"),
                    description: "A tool with two optional parameters.".to_string(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "optional_a": { "type": "string" },
                            "optional_b": { "type": "string" }
                        }
                    }),
                })
                .collect::<Vec<_>>();
            let request = model
                .completion_request("Call optional_tool_00 with an empty object.")
                .max_tokens(1024)
                .tools(tools)
                .tool_choice(ToolChoice::Specific {
                    function_names: vec!["optional_tool_00".to_string()],
                })
                .build();

            let response = model
                .completion(request)
                .await
                .expect("twenty-four simple optional parameters should be accepted");
            assert_single_tool_call(&response, "optional_tool_00", &json!({}));
        },
    )
    .await;
}

#[tokio::test]
async fn sixteen_union_parameters_across_tools_are_accepted() {
    with_anthropic_cassette(
        "strict_schema_limits/sixteen_union_parameters_across_tools_are_accepted",
        |client| async move {
            let model = client
                .completion_model(anthropic::completion::CLAUDE_SONNET_4_6)
                .with_strict_tools();
            let tools = (0..16)
                .map(|tool_index| ToolDefinition {
                    name: format!("union_tool_{tool_index:02}"),
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
            let request = model
                .completion_request("Call union_tool_00 with value = null.")
                .max_tokens(1024)
                .tools(tools)
                .tool_choice(ToolChoice::Specific {
                    function_names: vec!["union_tool_00".to_string()],
                })
                .build();

            let response = model
                .completion(request)
                .await
                .expect("sixteen simple union parameters should be accepted");
            assert_single_tool_call(&response, "union_tool_00", &json!({ "value": null }));
        },
    )
    .await;
}

#[tokio::test]
async fn seventeen_union_parameters_are_rejected() {
    with_anthropic_cassette(
        "strict_schema_limits/seventeen_union_parameters_are_rejected",
        |client| async move {
            let model = client
                .completion_model(anthropic::completion::CLAUDE_SONNET_4_6)
                .with_strict_tools();
            let request = model
                .completion_request("Call union_boundary with every field set to null.")
                .max_tokens(64)
                .tool_choice(ToolChoice::Required)
                .tool(ToolDefinition {
                    name: "union_boundary".to_string(),
                    description: "Exceed the strict union-parameter boundary.".to_string(),
                    parameters: union_parameters_schema(17),
                })
                .build();

            let error = model
                .completion(request)
                .await
                .expect_err("seventeen union parameters should exceed the provider limit");
            assert_invalid_request(&error);
        },
    )
    .await;
}
