//! Live-recorded integration coverage for strict tools and adjacent Anthropic features.

use rig::completion::{CompletionModel, ToolDefinition};
use rig::message::{AssistantContent, ToolChoice};
use rig::prelude::*;
use rig::providers::anthropic;
use serde_json::{Value, json};

use super::super::support::with_anthropic_cassette;

fn strict_value_tool(name: &str) -> ToolDefinition {
    ToolDefinition {
        name: name.to_string(),
        description: "Record one exact string value.".to_string(),
        parameters: json!({
            "type": "object",
            "properties": { "value": { "type": "string" } },
            "required": ["value"]
        }),
    }
}

fn empty_tool(name: impl Into<String>) -> ToolDefinition {
    ToolDefinition {
        name: name.into(),
        description: "A no-argument strict tool.".to_string(),
        parameters: json!({ "type": "object", "properties": {} }),
    }
}

fn tool_calls(response: &rig::completion::CompletionResponse) -> Vec<(&str, &Value)> {
    response
        .choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::ToolCall(tool_call) => Some((
                tool_call.function.name.as_str(),
                &tool_call.function.arguments,
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
    assert_eq!(calls[0].1, expected_arguments);
}

#[tokio::test]
async fn default_model_remains_non_strict() {
    with_anthropic_cassette(
        "strict_schema_integrations/default_model_remains_non_strict",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6);
            let request = model
                .completion_request("Call default_mode with value = unchanged.")
                .max_tokens(1024)
                .tool_choice(ToolChoice::Required)
                .tool(strict_value_tool("default_mode"))
                .build();

            let response = model
                .completion(request)
                .await
                .expect("ordinary non-strict tool use should remain valid");
            assert_one_call(&response, "default_mode", &json!({ "value": "unchanged" }));
        },
    )
    .await;
}

#[tokio::test]
async fn strict_mode_without_tools_is_a_noop() {
    with_anthropic_cassette(
        "strict_schema_integrations/strict_mode_without_tools_is_a_noop",
        |client| async move {
            let model = client
                .completion_model(anthropic::completion::CLAUDE_SONNET_4_6)
                .with_strict_tools();
            let response = model
                .completion(
                    model
                        .completion_request("Reply with exactly: no tools needed")
                        .max_tokens(32)
                        .build(),
                )
                .await
                .expect("strict mode without tools should still complete");
            let text = response
                .choice
                .iter()
                .filter_map(|content| match content {
                    AssistantContent::Text(text) => Some(text.text.as_str()),
                    _ => None,
                })
                .collect::<String>();
            assert!(text.to_ascii_lowercase().contains("no tools needed"));
        },
    )
    .await;
}

#[tokio::test]
async fn automatic_choice_calls_a_strict_tool() {
    with_anthropic_cassette(
        "strict_schema_integrations/automatic_choice_calls_a_strict_tool",
        |client| async move {
            let model = client
                .completion_model(anthropic::completion::CLAUDE_SONNET_4_6)
                .with_strict_tools();
            let request = model
                .completion_request("You must call strict_auto with value = automatic.")
                .preamble("Obey the request by using the provided tool.".to_string())
                .max_tokens(1024)
                .tool_choice(ToolChoice::Auto)
                .tool(strict_value_tool("strict_auto"))
                .build();

            let response = model
                .completion(request)
                .await
                .expect("automatic strict tool choice should succeed");
            assert_one_call(&response, "strict_auto", &json!({ "value": "automatic" }));
        },
    )
    .await;
}

#[tokio::test]
async fn none_choice_suppresses_a_strict_tool() {
    with_anthropic_cassette(
        "strict_schema_integrations/none_choice_suppresses_a_strict_tool",
        |client| async move {
            let model = client
                .completion_model(anthropic::completion::CLAUDE_SONNET_4_6)
                .with_strict_tools();
            let request = model
                .completion_request("Reply with exactly: strict tool suppressed")
                .max_tokens(32)
                .tool_choice(ToolChoice::None)
                .tool(strict_value_tool("unused_strict_tool"))
                .build();

            let response = model
                .completion(request)
                .await
                .expect("tool_choice none with a strict tool should succeed");
            assert!(tool_calls(&response).is_empty());
        },
    )
    .await;
}

#[tokio::test]
async fn specific_choice_selects_one_of_multiple_strict_tools() {
    with_anthropic_cassette(
        "strict_schema_integrations/specific_choice_selects_one_of_multiple_strict_tools",
        |client| async move {
            let model = client
                .completion_model(anthropic::completion::CLAUDE_SONNET_4_6)
                .with_strict_tools();
            let request = model
                .completion_request("Call strict_second with value = chosen.")
                .max_tokens(1024)
                .tools(vec![
                    strict_value_tool("strict_first"),
                    strict_value_tool("strict_second"),
                ])
                .tool_choice(ToolChoice::Specific {
                    function_names: vec!["strict_second".to_string()],
                })
                .build();

            let response = model
                .completion(request)
                .await
                .expect("specific strict tool choice should succeed");
            assert_one_call(&response, "strict_second", &json!({ "value": "chosen" }));
        },
    )
    .await;
}

#[tokio::test]
async fn rig_strict_and_provider_non_strict_tools_coexist() {
    with_anthropic_cassette(
        "strict_schema_integrations/rig_strict_and_provider_non_strict_tools_coexist",
        |client| async move {
            let model = client
                .completion_model(anthropic::completion::CLAUDE_SONNET_4_6)
                .with_strict_tools();
            let request = model
                .completion_request("Call raw_provider_tool with value = raw.")
                .max_tokens(1024)
                .tool(strict_value_tool("rig_strict_tool"))
                .tool_choice(ToolChoice::Specific {
                    function_names: vec!["raw_provider_tool".to_string()],
                })
                .additional_params(json!({
                    "tools": [{
                        "name": "raw_provider_tool",
                        "description": "A provider-specific non-strict tool.",
                        "input_schema": {
                            "type": "object",
                            "properties": { "value": { "type": "string" } },
                            "required": ["value"]
                        }
                    }]
                }))
                .build();

            let response = model
                .completion(request)
                .await
                .expect("Rig strict and provider-specific tools should coexist");
            assert_one_call(&response, "raw_provider_tool", &json!({ "value": "raw" }));
        },
    )
    .await;
}

#[tokio::test]
async fn twenty_rig_strict_plus_one_provider_non_strict_tool_is_accepted() {
    with_anthropic_cassette(
        "strict_schema_integrations/twenty_rig_strict_plus_one_provider_non_strict_tool_is_accepted",
        |client| async move {
            let model = client
                .completion_model(anthropic::completion::CLAUDE_SONNET_4_6)
                .with_strict_tools();
            let request = model
                .completion_request("Call raw_boundary_tool with an empty object.")
                .max_tokens(1024)
                .tools(
                    (0..20)
                        .map(|index| empty_tool(format!("strict_boundary_{index:02}")))
                        .collect(),
                )
                .tool_choice(ToolChoice::Specific {
                    function_names: vec!["raw_boundary_tool".to_string()],
                })
                .additional_params(json!({
                    "tools": [{
                        "name": "raw_boundary_tool",
                        "description": "A non-strict tool that does not count toward the strict limit.",
                        "input_schema": { "type": "object", "properties": {} }
                    }]
                }))
                .build();

            let response = model
                .completion(request)
                .await
                .expect("a non-strict twenty-first tool should not exceed the strict limit");
            assert_one_call(&response, "raw_boundary_tool", &json!({}));
        },
    )
    .await;
}

#[tokio::test]
async fn manual_prompt_caching_coexists_with_strict_tools() {
    with_anthropic_cassette(
        "strict_schema_integrations/manual_prompt_caching_coexists_with_strict_tools",
        |client| async move {
            let model = client
                .completion_model(anthropic::completion::CLAUDE_SONNET_4_6)
                .with_prompt_caching()
                .with_strict_tools();
            let request = model
                .completion_request("Call cached_strict with value = manual.")
                .preamble("Use the strict tool exactly once.".to_string())
                .max_tokens(1024)
                .tool_choice(ToolChoice::Required)
                .tool(strict_value_tool("cached_strict"))
                .build();

            let response = model
                .completion(request)
                .await
                .expect("manual prompt caching and strict tools should coexist");
            assert_one_call(&response, "cached_strict", &json!({ "value": "manual" }));
        },
    )
    .await;
}

#[tokio::test]
async fn automatic_prompt_caching_coexists_with_strict_tools() {
    with_anthropic_cassette(
        "strict_schema_integrations/automatic_prompt_caching_coexists_with_strict_tools",
        |client| async move {
            let model = client
                .completion_model(anthropic::completion::CLAUDE_SONNET_4_6)
                .with_automatic_caching()
                .with_strict_tools();
            let request = model
                .completion_request("Call cached_strict with value = automatic.")
                .max_tokens(1024)
                .tool_choice(ToolChoice::Required)
                .tool(strict_value_tool("cached_strict"))
                .build();

            let response = model
                .completion(request)
                .await
                .expect("automatic prompt caching and strict tools should coexist");
            assert_one_call(&response, "cached_strict", &json!({ "value": "automatic" }));
        },
    )
    .await;
}

#[tokio::test]
async fn static_prefix_ttl_caching_coexists_with_strict_tools() {
    with_anthropic_cassette(
        "strict_schema_integrations/static_prefix_ttl_caching_coexists_with_strict_tools",
        |client| async move {
            let model = client
                .completion_model(anthropic::completion::CLAUDE_SONNET_4_6)
                .with_automatic_caching()
                .with_static_prefix_cache_ttl(
                    rig::providers::anthropic::completion::CacheTtl::OneHour,
                )
                .with_strict_tools();
            // The preamble must clear the model's minimum cacheable prompt
            // length or the API silently skips caching and the recorded
            // counters prove nothing.
            let padding = "This strict-tool cache fixture paragraph is stable provider test \
                           padding about request routing, tool schemas, system instructions, \
                           and deterministic replay behavior. "
                .repeat(60);
            let request = model
                .completion_request("Call cached_strict with value = static-prefix.")
                .preamble(format!("Use the strict tool exactly once.\n{padding}"))
                .max_tokens(1024)
                .tool_choice(ToolChoice::Required)
                .tool(strict_value_tool("cached_strict"))
                .build();

            let response = model
                .completion(request)
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
                .completion_model(anthropic::completion::CLAUDE_SONNET_4_6)
                .with_automatic_caching_1h()
                .with_strict_tools();
            let request = model
                .completion_request("Call cached_strict with value = one-hour.")
                .max_tokens(1024)
                .tool_choice(ToolChoice::Required)
                .tool(strict_value_tool("cached_strict"))
                .build();

            let response = model
                .completion(request)
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
                .completion_model(anthropic::completion::CLAUDE_SONNET_4_6)
                .with_strict_tools();
            let output_schema = serde_json::from_value(json!({
                "type": "object",
                "properties": { "summary": { "type": "string" } },
                "required": ["summary"]
            }))
            .expect("output schema should parse");
            let request = model
                .completion_request("Call combined_strict with value = combined.")
                .max_tokens(1024)
                .tool_choice(ToolChoice::Required)
                .tool(strict_value_tool("combined_strict"))
                .output_schema(output_schema)
                .build();

            let response = model
                .completion(request)
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

#[tokio::test]
async fn parallel_strict_tool_calls_preserve_each_schema() {
    with_anthropic_cassette(
        "strict_schema_integrations/parallel_strict_tool_calls_preserve_each_schema",
        |client| async move {
            let model = client
                .completion_model(anthropic::completion::CLAUDE_SONNET_4_6)
                .with_strict_tools();
            let request = model
                .completion_request(
                    "Call strict_alpha with value = A and strict_beta with value = B in parallel. Call both tools.",
                )
                .preamble("Use every tool requested by the user in the same turn.".to_string())
                .max_tokens(2048)
                .tool_choice(ToolChoice::Auto)
                .tools(vec![
                    strict_value_tool("strict_alpha"),
                    strict_value_tool("strict_beta"),
                ])
                .build();

            let response = model
                .completion(request)
                .await
                .expect("parallel strict tool calls should succeed");
            let mut calls = tool_calls(&response);
            calls.sort_by_key(|(name, _)| *name);
            assert_eq!(
                calls,
                vec![
                    ("strict_alpha", &json!({ "value": "A" })),
                    ("strict_beta", &json!({ "value": "B" }))
                ]
            );
        },
    )
    .await;
}
