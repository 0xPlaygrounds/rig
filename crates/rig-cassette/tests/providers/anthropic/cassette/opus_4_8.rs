//! Dedicated Claude Opus 4.8 cassette coverage.

use rig::completion::{
    AssistantContent, CompletionResponse as RigCompletionResponse, Message, ProviderToolDefinition,
};
use rig::providers::anthropic::completion::CLAUDE_OPUS_4_8;
use rig::providers::anthropic::extension::AnthropicExt;
use serde::Deserialize;
use serde_json::Value;
use serde_json::json;

use crate::support::{assert_contains_any_case_insensitive, assistant_text_response};
use rig::completion::CompletionRequest;

/// The text Anthropic's own reply carried, read back out of
/// [`RigCompletionResponse::raw`] — the value the deleted `raw_completion`
/// returned. These cells fall back to it when rig's normalized choice holds
/// no assistant text, and one request carries both views.
fn provider_text(response: &RigCompletionResponse) -> Option<String> {
    let text: String = response.raw["content"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|block| block["type"] == "text")
        .filter_map(|block| block["text"].as_str())
        .collect();
    (!text.is_empty()).then_some(text)
}

const SERVER_TOOL_USE_SYSTEM_INSTRUCTION: &str =
    "For the rest of this conversation, answer in Spanish only.";

#[tokio::test]
async fn web_search_with_dynamic_filtering_succeeds() {
    super::super::support::with_anthropic_cassette(
        "opus_4_8/web_search_with_dynamic_filtering_succeeds",
        |client| async move {
            let model = client.completion(CLAUDE_OPUS_4_8);
            let request = CompletionRequest::new(
                    "Search for the current prices of AAPL and GOOGL, then calculate which has a better P/E ratio.",
                )
                .provider_tool(
                    ProviderToolDefinition::new("web_search_20260209")
                        .with_config("name", json!("web_search")),
                )
                .max_tokens(1024);
            // One request, two views: the normalized response is what the
            // model returns, and `raw` carries Anthropic's own reply, so the
            // provider-text fallback below still costs a single interaction.
            let response: RigCompletionResponse = model
                .call(request)
                .await
                .expect("Opus 4.8 dynamic web-search request should succeed");
            let raw_text = provider_text(&response);

            assert!(
                response.choice.iter().any(|content| {
                    content_raw_type(content) == Some("code_execution_tool_result")
                }),
                "dynamic web-search response should preserve a code_execution_tool_result block",
            );
            assert!(
                assistant_text_response(&response.choice)
                    .or(raw_text)
                    .is_some_and(|text| !text.trim().is_empty()),
                "dynamic web-search response should contain assistant text",
            );

            let extras = response
                .extras::<AnthropicExt>()
                .expect("an Anthropic reply")
                .expect("the extras read the recorded reply");
            assert_eq!(extras.stop_reason.as_deref(), Some("end_turn"));
            assert_eq!(extras.stop_sequence, None);
            assert_eq!(extras.stop_details, None);
            assert_eq!(extras.service_tier.as_deref(), Some("standard"));
            assert_eq!(extras.inference_geo.as_deref(), Some("global"));
            assert_eq!(extras.speed, None);
            assert_eq!(extras.fallback_model, None);
            assert_eq!(
                extras.server_tool_use.map(|tools| (
                    tools.web_search_requests,
                    tools.web_fetch_requests
                )),
                Some((2, 0))
            );
            let container = extras.container.expect("the reply ran in a container");
            assert_eq!(container.id, "container_01L1N4CD1Sc9WMM81fQweA9p");
            assert_eq!(container.expires_at, "2026-09-22T18:07:11.235261Z");
        },
    )
    .await;
}

#[tokio::test]
async fn messages_preserve_system_role_after_server_tool_result() {
    super::super::support::with_anthropic_cassette(
        "opus_4_8/messages_preserve_system_role_after_server_tool_result",
        |client| async move {
            let model = client.completion(CLAUDE_OPUS_4_8);
            let first_response = model
                .call(CompletionRequest::new(
                    "Use web search to check the color of a clear daytime sky. Keep the final answer under five words.",
                )
                .provider_tool(
                    ProviderToolDefinition::new("web_search_20250305")
                        .with_config("name", json!("web_search")),
                )
                .max_tokens(128))
                .await
                .expect("Opus 4.8 web-search request should produce a server-tool transcript");
            let server_tool_assistant_message =
                server_tool_assistant_message_from_response(&first_response);

            let request = CompletionRequest::new(
                    "What color is a clear daytime sky? Reply with one lowercase Spanish word.",
                )
                .messages([
                    server_tool_assistant_message,
                    Message::system(SERVER_TOOL_USE_SYSTEM_INSTRUCTION),
                    Message::assistant("Entendido."),
                ])
                .max_tokens(64);
            let response: RigCompletionResponse = model.call(request).await.expect(
                "Opus 4.8 request with system role after server tool result should succeed",
            );
            let raw_text = provider_text(&response);

            let text = assistant_text_response(&response.choice)
                .or(raw_text)
                .expect("response should contain assistant text");
            assert_contains_any_case_insensitive(&text, &["azul"]);
        },
    )
    .await;

    assert_cassette_preserves_system_role_after_server_tool_result(
        "opus_4_8/messages_preserve_system_role_after_server_tool_result",
        SERVER_TOOL_USE_SYSTEM_INSTRUCTION,
    );
}

/// The first reply's server-tool blocks as an assistant turn of the model
/// that produced them, so they replay as they came.
fn server_tool_assistant_message_from_response(response: &RigCompletionResponse) -> Message {
    let raw_blocks = response
        .choice
        .iter()
        .filter(|content| content_raw_type(content).is_some())
        .cloned()
        .collect::<Vec<_>>();

    assert!(
        raw_blocks
            .iter()
            .any(|content| content_raw_type(content) == Some("server_tool_use")),
        "first Anthropic response should contain a preserved server_tool_use block"
    );
    assert!(
        raw_blocks
            .last()
            .is_some_and(|content| content_raw_type(content) == Some("web_search_tool_result")),
        "first Anthropic response should end the preserved raw transcript with a server-tool result"
    );

    Message::Assistant(response.head().with_content(raw_blocks))
}

/// The Anthropic block type of an opaque provider item.
fn content_raw_type(content: &AssistantContent) -> Option<&str> {
    match content {
        AssistantContent::Opaque(opaque) => opaque.kind(),
        _ => None,
    }
}

#[derive(Deserialize)]
struct RecordedInteraction {
    when: RecordedRequest,
}

#[derive(Deserialize)]
struct RecordedRequest {
    body: Option<String>,
}

fn assert_cassette_preserves_system_role_after_server_tool_result(
    scenario: &str,
    expected_system_text: &str,
) {
    let request_bodies = recorded_request_bodies(scenario);
    let continuation = request_bodies
        .iter()
        .find(|body| {
            body.get("messages")
                .and_then(Value::as_array)
                .is_some_and(|messages| {
                    messages.iter().any(|message| {
                        message.get("role").and_then(Value::as_str) == Some("system")
                            && message_contains_text(message, expected_system_text)
                    })
                })
        })
        .unwrap_or_else(|| {
            panic!("expected cassette {scenario} to contain the continuation request")
        });

    let top_level_system_contains_instruction = continuation
        .get("system")
        .and_then(Value::as_array)
        .is_some_and(|system| {
            system
                .iter()
                .any(|block| block_contains_text(block, expected_system_text))
        });
    assert!(
        !top_level_system_contains_instruction,
        "expected cassette {scenario} not to hoist the continuation system instruction",
    );

    let messages = continuation
        .get("messages")
        .and_then(Value::as_array)
        .expect("continuation request should contain messages[]");
    let roles = messages
        .iter()
        .map(|message| message.get("role").and_then(Value::as_str))
        .collect::<Vec<_>>();
    assert_eq!(
        roles,
        [
            Some("assistant"),
            Some("system"),
            Some("assistant"),
            Some("user")
        ],
        "expected continuation request to preserve assistant -> system -> assistant -> user order",
    );

    assert!(
        message_content_has_type(&messages[0], "server_tool_use"),
        "expected first continuation message to contain a preserved server_tool_use block",
    );
    assert!(
        message_content_has_type(&messages[0], "web_search_tool_result"),
        "expected first continuation message to contain a preserved web_search_tool_result block",
    );
    assert!(
        message_contains_text(&messages[1], expected_system_text),
        "expected second continuation message to contain the system instruction",
    );
}

fn recorded_request_bodies(scenario: &str) -> Vec<Value> {
    let cassette_path = crate::cassettes::cassette_path("anthropic", scenario);
    let contents = std::fs::read_to_string(&cassette_path).unwrap_or_else(|error| {
        panic!(
            "provider cassette {} should be readable after recording: {error}",
            cassette_path.display()
        )
    });

    serde_yaml::Deserializer::from_str(&contents)
        .filter_map(|document| {
            let interaction = RecordedInteraction::deserialize(document)
                .expect("cassette interaction should deserialize");
            interaction
                .when
                .body
                .and_then(|body| serde_json::from_str::<Value>(&body).ok())
        })
        .collect()
}

fn message_contains_text(message: &Value, expected_text: &str) -> bool {
    message
        .get("content")
        .and_then(Value::as_array)
        .is_some_and(|content| {
            content
                .iter()
                .any(|block| block_contains_text(block, expected_text))
        })
}

fn message_content_has_type(message: &Value, expected_type: &str) -> bool {
    message
        .get("content")
        .and_then(Value::as_array)
        .is_some_and(|content| {
            content
                .iter()
                .any(|block| block.get("type").and_then(Value::as_str) == Some(expected_type))
        })
}

fn block_contains_text(block: &Value, expected_text: &str) -> bool {
    block.get("text").and_then(Value::as_str) == Some(expected_text)
}
