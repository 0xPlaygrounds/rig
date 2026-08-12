//! Live-recorded streaming coverage for Anthropic strict tools.

use rig::completion::ToolDefinition;
use rig::message::ToolChoice;
use rig::prelude::*;
use rig::providers::anthropic;
use serde_json::{Value, json};

use super::super::support::with_anthropic_cassette;
use crate::support::collect_raw_stream_observation;

async fn assert_streaming_strict_tool_call(
    client: anthropic::Client,
    tool_name: &str,
    prompt: &str,
    parameters: Value,
    tool_choice: ToolChoice,
    expected_arguments: Value,
) {
    let model = client
        .completion_model(anthropic::completion::CLAUDE_SONNET_4_6)
        .with_strict_tools();
    assert_model_streaming_tool_call(
        model,
        tool_name,
        prompt,
        parameters,
        tool_choice,
        expected_arguments,
        None,
    )
    .await;
}

async fn assert_model_streaming_tool_call(
    model: anthropic::completion::CompletionModel,
    tool_name: &str,
    prompt: &str,
    parameters: Value,
    tool_choice: ToolChoice,
    expected_arguments: Value,
    output_schema: Option<schemars::Schema>,
) {
    let request = model
        .completion_request(prompt)
        .preamble("Call the requested tool exactly once with the requested values.".to_string())
        .max_tokens(1024)
        .tool_choice(tool_choice)
        .tool(ToolDefinition {
            name: tool_name.to_string(),
            description: "Record the requested values in a streaming strict tool call.".to_string(),
            parameters,
        })
        .output_schema_opt(output_schema)
        .build();

    let observation = collect_raw_stream_observation(
        model
            .stream(request)
            .await
            .expect("strict streaming request should start"),
    )
    .await;
    assert!(
        observation.errors.is_empty(),
        "strict stream should not emit errors: {:?}",
        observation.errors
    );
    assert!(
        observation.got_final,
        "strict stream should emit a final event"
    );
    assert_eq!(
        observation.tool_calls.len(),
        1,
        "strict stream should contain exactly one tool call"
    );
    assert_eq!(observation.tool_calls[0].function.name, tool_name);
    assert_eq!(
        observation.tool_calls[0].function.arguments,
        expected_arguments
    );
}

#[tokio::test]
async fn zero_argument_strict_tool_streams() {
    with_anthropic_cassette(
        "strict_schema_streaming/zero_argument_strict_tool_streams",
        |client| async move {
            assert_streaming_strict_tool_call(
                client,
                "stream_ping",
                "Call stream_ping with an empty object.",
                json!({ "type": "object", "properties": {} }),
                ToolChoice::Required,
                json!({}),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn nested_optional_object_streams_with_omission() {
    with_anthropic_cassette(
        "strict_schema_streaming/nested_optional_object_streams_with_omission",
        |client| async move {
            assert_streaming_strict_tool_call(
                client,
                "stream_profile",
                "Record profile.name = Ada and omit profile.nickname and the optional trace_id.",
                json!({
                    "type": "object",
                    "properties": {
                        "profile": {
                            "type": "object",
                            "properties": {
                                "name": { "type": "string" },
                                "nickname": { "type": "string" }
                            },
                            "required": ["name"]
                        },
                        "trace_id": { "type": "string" }
                    },
                    "required": ["profile"]
                }),
                ToolChoice::Required,
                json!({ "profile": { "name": "Ada" } }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn one_of_and_const_stream_with_specific_choice() {
    with_anthropic_cassette(
        "strict_schema_streaming/one_of_and_const_stream_with_specific_choice",
        |client| async move {
            assert_streaming_strict_tool_call(
                client,
                "stream_event",
                "Record a created event with id = 9.",
                json!({
                    "type": "object",
                    "properties": {
                        "event": {
                            "oneOf": [
                                {
                                    "type": "object",
                                    "properties": {
                                        "kind": { "const": "created" },
                                        "id": { "type": "integer" }
                                    },
                                    "required": ["kind", "id"]
                                },
                                {
                                    "type": "object",
                                    "properties": {
                                        "kind": { "const": "deleted" },
                                        "id": { "type": "integer" }
                                    },
                                    "required": ["kind", "id"]
                                }
                            ]
                        }
                    },
                    "required": ["event"]
                }),
                ToolChoice::Specific {
                    function_names: vec!["stream_event".to_string()],
                },
                json!({ "event": { "kind": "created", "id": 9 } }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn automatic_choice_streams_strict_tool_call() {
    with_anthropic_cassette(
        "strict_schema_streaming/automatic_choice_streams_strict_tool_call",
        |client| async move {
            assert_streaming_strict_tool_call(
                client,
                "stream_auto",
                "You must call stream_auto with value = selected.",
                json!({
                    "type": "object",
                    "properties": { "value": { "type": "string" } },
                    "required": ["value"]
                }),
                ToolChoice::Auto,
                json!({ "value": "selected" }),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn structured_output_and_strict_tool_use_stream_together() {
    with_anthropic_cassette(
        "strict_schema_streaming/structured_output_and_strict_tool_use_stream_together",
        |client| async move {
            let model = client
                .completion_model(anthropic::completion::CLAUDE_SONNET_4_6)
                .with_strict_tools();
            let output_schema = serde_json::from_value(json!({
                "type": "object",
                "properties": { "summary": { "type": "string" } },
                "required": ["summary"]
            }))
            .expect("output schema should parse");
            assert_model_streaming_tool_call(
                model,
                "stream_combined",
                "Call stream_combined with value = combined-stream.",
                json!({
                    "type": "object",
                    "properties": { "value": { "type": "string" } },
                    "required": ["value"]
                }),
                ToolChoice::Required,
                json!({ "value": "combined-stream" }),
                Some(output_schema),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn manual_prompt_caching_and_strict_tools_stream_together() {
    with_anthropic_cassette(
        "strict_schema_streaming/manual_prompt_caching_and_strict_tools_stream_together",
        |client| async move {
            let model = client
                .completion_model(anthropic::completion::CLAUDE_SONNET_4_6)
                .with_prompt_caching()
                .with_strict_tools();
            assert_model_streaming_tool_call(
                model,
                "stream_cached",
                "Call stream_cached with value = manual-stream.",
                json!({
                    "type": "object",
                    "properties": { "value": { "type": "string" } },
                    "required": ["value"]
                }),
                ToolChoice::Required,
                json!({ "value": "manual-stream" }),
                None,
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn automatic_prompt_caching_and_strict_tools_stream_together() {
    with_anthropic_cassette(
        "strict_schema_streaming/automatic_prompt_caching_and_strict_tools_stream_together",
        |client| async move {
            let model = client
                .completion_model(anthropic::completion::CLAUDE_SONNET_4_6)
                .with_automatic_caching()
                .with_strict_tools();
            assert_model_streaming_tool_call(
                model,
                "stream_cached",
                "Call stream_cached with value = automatic-stream.",
                json!({
                    "type": "object",
                    "properties": { "value": { "type": "string" } },
                    "required": ["value"]
                }),
                ToolChoice::Required,
                json!({ "value": "automatic-stream" }),
                None,
            )
            .await;
        },
    )
    .await;
}
