//! Dedicated Claude Opus 4.8 cassette coverage.

use rig::client::CompletionClient;
use rig::completion::{
    AssistantContent, CompletionModel, Document, Message, ProviderToolDefinition,
};
use rig::message::Text;
use rig::providers::anthropic::completion::CLAUDE_OPUS_4_8;
use rig::telemetry::ProviderResponseExt;
use serde::Deserialize;
use serde_json::Value;
use serde_json::json;

use crate::support::{assert_contains_any_case_insensitive, assistant_text_response};

const SYSTEM_ROLE_INSTRUCTION: &str = "For the rest of this conversation, answer in Spanish only.";
const DOCUMENT_GLOBAL_SYSTEM_INSTRUCTION: &str = "Answer in Spanish only. Use one short sentence.";
const SERVER_TOOL_USE_SYSTEM_INSTRUCTION: &str =
    "For the rest of this conversation, answer in Spanish only.";

#[tokio::test]
async fn messages_preserve_mid_conversation_system_role() {
    super::super::support::with_anthropic_cassette(
        "opus_4_8/messages_preserve_mid_conversation_system_role",
        |client| async move {
            let model = client.completion_model(CLAUDE_OPUS_4_8);
            let response = model
                .completion_request(
                    "What color is a clear daytime sky? Reply with one lowercase Spanish word.",
                )
                .messages([
                    Message::user("Start a short language compliance check."),
                    Message::system(SYSTEM_ROLE_INSTRUCTION),
                    Message::assistant("Entendido."),
                ])
                .max_tokens(64)
                .send()
                .await
                .expect("Opus 4.8 system-role request should succeed");

            let text = assistant_text_response(&response.choice)
                .or_else(|| response.raw_response.get_text_response())
                .expect("response should contain assistant text");
            assert_contains_any_case_insensitive(&text, &["azul"]);
        },
    )
    .await;

    assert_cassette_preserves_system_role_message(
        "opus_4_8/messages_preserve_mid_conversation_system_role",
        SYSTEM_ROLE_INSTRUCTION,
    );
}

#[tokio::test]
async fn messages_preserve_system_role_after_server_tool_result() {
    super::super::support::with_anthropic_cassette(
        "opus_4_8/messages_preserve_system_role_after_server_tool_result",
        |client| async move {
            let model = client.completion_model(CLAUDE_OPUS_4_8);
            let first_response = model
                .completion_request(
                    "Use web search to check the color of a clear daytime sky. Keep the final answer under five words.",
                )
                .provider_tool(
                    ProviderToolDefinition::new("web_search_20250305")
                        .with_config("name", json!("web_search")),
                )
                .max_tokens(128)
                .send()
                .await
                .expect("Opus 4.8 web-search request should produce a server-tool transcript");
            let server_tool_assistant_message =
                server_tool_assistant_message_from_response(first_response.choice);

            let response = model
                .completion_request(
                    "What color is a clear daytime sky? Reply with one lowercase Spanish word.",
                )
                .messages([
                    server_tool_assistant_message,
                    Message::system(SERVER_TOOL_USE_SYSTEM_INSTRUCTION),
                    Message::assistant("Entendido."),
                ])
                .max_tokens(64)
                .send()
                .await
                .expect(
                    "Opus 4.8 request with system role after server tool result should succeed",
                );

            let text = assistant_text_response(&response.choice)
                .or_else(|| response.raw_response.get_text_response())
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

#[tokio::test]
async fn documents_keep_leading_system_message_top_level() {
    super::super::support::with_anthropic_cassette(
        "opus_4_8/documents_keep_leading_system_message_top_level",
        |client| async move {
            let model = client.completion_model(CLAUDE_OPUS_4_8);
            let response = model
                .completion_request(
                    "According to the document, what color is the clear daytime sky?",
                )
                .messages([
                    Message::system(DOCUMENT_GLOBAL_SYSTEM_INSTRUCTION),
                    Message::assistant("Entendido."),
                ])
                .document(Document {
                    id: "sky-note".to_string(),
                    text: "A clear daytime sky is blue.".to_string(),
                    additional_props: Default::default(),
                })
                .max_tokens(64)
                .send()
                .await
                .expect(
                    "Opus 4.8 request with documents and a leading system message should succeed",
                );

            let text = assistant_text_response(&response.choice)
                .or_else(|| response.raw_response.get_text_response())
                .expect("response should contain assistant text");
            assert_contains_any_case_insensitive(&text, &["azul"]);
        },
    )
    .await;

    assert_cassette_hoists_system_instruction(
        "opus_4_8/documents_keep_leading_system_message_top_level",
        DOCUMENT_GLOBAL_SYSTEM_INSTRUCTION,
    );
    assert_cassette_document_request_order(
        "opus_4_8/documents_keep_leading_system_message_top_level",
        DOCUMENT_GLOBAL_SYSTEM_INSTRUCTION,
    );
}

fn server_tool_assistant_message_from_response(
    content: rig::OneOrMany<AssistantContent>,
) -> Message {
    let raw_blocks = content
        .into_iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) if anthropic_raw_content_type(&text).is_some() => {
                Some(AssistantContent::Text(text))
            }
            _ => None,
        })
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

    Message::Assistant {
        id: None,
        content: rig::OneOrMany::many(raw_blocks)
            .expect("server tool assistant message should be non-empty"),
    }
}

fn content_raw_type(content: &AssistantContent) -> Option<&str> {
    let AssistantContent::Text(text) = content else {
        return None;
    };

    anthropic_raw_content_type(text)
}

fn anthropic_raw_content_type(text: &Text) -> Option<&str> {
    text.additional_params
        .as_ref()
        .and_then(|params| params.get("anthropic_content"))
        .and_then(|raw_content| raw_content.get("type"))
        .and_then(Value::as_str)
}

#[derive(Deserialize)]
struct RecordedInteraction {
    when: RecordedRequest,
}

#[derive(Deserialize)]
struct RecordedRequest {
    body: Option<String>,
}

fn assert_cassette_preserves_system_role_message(scenario: &str, expected_system_text: &str) {
    let preserves_system_role = recorded_request_bodies(scenario).iter().any(|body| {
        body.get("messages")
            .and_then(Value::as_array)
            .is_some_and(|messages| {
                messages.iter().any(|message| {
                    message.get("role").and_then(Value::as_str) == Some("system")
                        && message_contains_text(message, expected_system_text)
                })
            })
    });

    assert!(
        preserves_system_role,
        "expected cassette {scenario} to contain an Anthropic messages[] entry with role=system",
    );
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

fn assert_cassette_hoists_system_instruction(scenario: &str, expected_system_text: &str) {
    let request_bodies = recorded_request_bodies(scenario);
    let top_level_system_contains_instruction = request_bodies.iter().any(|body| {
        body.get("system")
            .and_then(Value::as_array)
            .is_some_and(|system| {
                system
                    .iter()
                    .any(|block| block_contains_text(block, expected_system_text))
            })
    });
    let messages_contain_system_role_instruction = request_bodies.iter().any(|body| {
        body.get("messages")
            .and_then(Value::as_array)
            .is_some_and(|messages| {
                messages.iter().any(|message| {
                    message.get("role").and_then(Value::as_str) == Some("system")
                        && message_contains_text(message, expected_system_text)
                })
            })
    });

    assert!(
        top_level_system_contains_instruction,
        "expected cassette {scenario} to contain the leading system instruction in top-level system",
    );
    assert!(
        !messages_contain_system_role_instruction,
        "expected cassette {scenario} not to send the leading system instruction as messages[] role=system",
    );
}

fn assert_cassette_document_request_order(scenario: &str, expected_system_text: &str) {
    let request_bodies = recorded_request_bodies(scenario);
    let body = request_bodies
        .iter()
        .find(|body| {
            body.get("system")
                .and_then(Value::as_array)
                .is_some_and(|system| {
                    system
                        .iter()
                        .any(|block| block_contains_text(block, expected_system_text))
                })
        })
        .unwrap_or_else(|| {
            panic!("expected cassette {scenario} to include document ordering request")
        });

    let messages = body
        .get("messages")
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("expected cassette {scenario} request to contain messages[]"));
    let roles = messages
        .iter()
        .map(|message| message.get("role").and_then(Value::as_str))
        .collect::<Vec<_>>();
    assert_eq!(
        roles,
        [Some("user"), Some("assistant"), Some("user")],
        "expected document request to preserve document -> assistant -> prompt order",
    );
    assert!(
        message_content_has_type(&messages[0], "document"),
        "expected first message to contain the normalized document block",
    );
    assert!(
        message_contains_text(&messages[1], "Entendido."),
        "expected second message to preserve prior assistant turn",
    );
    assert!(
        message_contains_text(
            &messages[2],
            "According to the document, what color is the clear daytime sky?",
        ),
        "expected final message to remain the prompt",
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
