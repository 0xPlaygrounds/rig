//! Dedicated Claude Opus 4.8 cassette coverage.

use rig::client::CompletionClient;
use rig::completion::{CompletionModel, Message};
use rig::providers::anthropic::completion::CLAUDE_OPUS_4_8;
use rig::telemetry::ProviderResponseExt;
use serde::Deserialize;
use serde_json::Value;

use crate::support::{assert_contains_any_case_insensitive, assistant_text_response};

const SYSTEM_ROLE_INSTRUCTION: &str = "For the rest of this conversation, answer in Spanish only.";

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

#[derive(Deserialize)]
struct RecordedInteraction {
    when: RecordedRequest,
}

#[derive(Deserialize)]
struct RecordedRequest {
    body: Option<String>,
}

fn assert_cassette_preserves_system_role_message(scenario: &str, expected_system_text: &str) {
    let cassette_path = crate::cassettes::cassette_path("anthropic", scenario);
    let contents = std::fs::read_to_string(&cassette_path).unwrap_or_else(|error| {
        panic!(
            "provider cassette {} should be readable after recording: {error}",
            cassette_path.display()
        )
    });

    let preserves_system_role = serde_yaml::Deserializer::from_str(&contents).any(|document| {
        let interaction = RecordedInteraction::deserialize(document)
            .expect("cassette interaction should deserialize");
        let Some(body) = interaction.when.body else {
            return false;
        };
        let Ok(body) = serde_json::from_str::<Value>(&body) else {
            return false;
        };

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
        "expected cassette {} to contain an Anthropic messages[] entry with role=system",
        cassette_path.display()
    );
}

fn message_contains_text(message: &Value, expected_text: &str) -> bool {
    message
        .get("content")
        .and_then(Value::as_array)
        .is_some_and(|content| {
            content
                .iter()
                .any(|block| block.get("text").and_then(Value::as_str) == Some(expected_text))
        })
}
