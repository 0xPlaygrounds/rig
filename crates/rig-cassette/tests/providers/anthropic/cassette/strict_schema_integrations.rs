//! Live-recorded integration coverage for strict tools and adjacent Anthropic features.

use rig::completion::{CacheRetention, ToolDefinition};
use rig::message::{AssistantContent, ToolChoice};
use rig::providers::anthropic;
use rig_test_support::cassette_models::MapWire;
use serde_json::{Value, json};

use super::super::support::with_anthropic_cassette;
use rig::completion::CompletionRequest;

fn strict_value_tool(name: &str) -> ToolDefinition {
    ToolDefinition {
        name: rig_core::message::ToolName::new(name).expect("tool name"),
        description: "Record one exact string value.".to_string(),
        parameters: json!({
            "type": "object",
            "properties": { "value": { "type": "string" } },
            "required": ["value"]
        }),
    }
}

fn tool_calls(response: &rig::completion::CompletionResponse) -> Vec<(&str, Value)> {
    response
        .choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::ToolCall(tool_call) => Some((
                tool_call.function.name.as_str(),
                tool_call.function.arguments_value(),
            )),
            _ => None,
        })
        .collect()
}

fn assert_one_call(
    response: &rig::completion::CompletionResponse,
    expected_name: &str,
    expected_arguments: &Value,
) {
    let calls = tool_calls(response);
    assert_eq!(calls.len(), 1, "exactly one tool call is expected");
    assert_eq!(calls[0].0, expected_name);
    assert_eq!(&calls[0].1, expected_arguments);
}

#[tokio::test]
async fn static_prefix_ttl_caching_coexists_with_strict_tools() {
    with_anthropic_cassette(
        "strict_schema_integrations/static_prefix_ttl_caching_coexists_with_strict_tools",
        |client| async move {
            let model = client
                .completion(anthropic::completion::CLAUDE_SONNET_4_6)
                .map_wire(|wire| {
                    wire.with_static_prefix_cache_ttl(
                        rig::providers::anthropic::completion::CacheTtl::OneHour,
                    )
                    .with_strict_tools()
                });
            // The preamble must clear the model's minimum cacheable prompt
            // length or the API silently skips caching and the recorded
            // counters prove nothing.
            let padding = "This strict-tool cache fixture paragraph is stable provider test \
                           padding about request routing, tool schemas, system instructions, \
                           and deterministic replay behavior. "
                .repeat(60);
            let request = CompletionRequest::new("Call cached_strict with value = static-prefix.")
                .cache(CacheRetention::Short)
                .preamble(format!("Use the strict tool exactly once.\n{padding}"))
                .max_tokens(1024)
                .tool_choice(ToolChoice::Required)
                .tool(strict_value_tool("cached_strict"));

            let response = model
                .call(request)
                .await
                .expect("a 1h static prefix and strict tools should coexist");
            assert_one_call(
                &response,
                "cached_strict",
                &json!({ "value": "static-prefix" }),
            );
        },
    )
    .await;
}

#[tokio::test]
async fn one_hour_automatic_caching_coexists_with_strict_tools() {
    with_anthropic_cassette(
        "strict_schema_integrations/one_hour_automatic_caching_coexists_with_strict_tools",
        |client| async move {
            let model = client
                .completion(anthropic::completion::CLAUDE_SONNET_4_6)
                .map_wire(|wire| wire.with_strict_tools());
            let request = CompletionRequest::new("Call cached_strict with value = one-hour.")
                .cache(CacheRetention::Long)
                .max_tokens(1024)
                .tool_choice(ToolChoice::Required)
                .tool(strict_value_tool("cached_strict"));

            let response = model
                .call(request)
                .await
                .expect("one-hour automatic caching and strict tools should coexist");
            assert_one_call(&response, "cached_strict", &json!({ "value": "one-hour" }));
        },
    )
    .await;
}

#[tokio::test]
async fn structured_output_and_strict_tool_use_coexist() {
    with_anthropic_cassette(
        "strict_schema_integrations/structured_output_and_strict_tool_use_coexist",
        |client| async move {
            let model = client
                .completion(anthropic::completion::CLAUDE_SONNET_4_6)
                .map_wire(|wire| wire.with_strict_tools());
            let output_schema: schemars::Schema = serde_json::from_value(json!({
                "type": "object",
                "properties": { "summary": { "type": "string" } },
                "required": ["summary"]
            }))
            .expect("output schema should parse");
            let request = CompletionRequest::new("Call combined_strict with value = combined.")
                .max_tokens(1024)
                .tool_choice(ToolChoice::Required)
                .tool(strict_value_tool("combined_strict"))
                .output_schema(output_schema);

            let response = model
                .call(request)
                .await
                .expect("structured output and strict tool use should coexist");
            assert_one_call(
                &response,
                "combined_strict",
                &json!({ "value": "combined" }),
            );
        },
    )
    .await;
}
