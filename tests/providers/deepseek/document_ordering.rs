//! Focused DeepSeek cassette coverage for request document ordering.

use rig::OneOrMany;
use rig::completion::{AssistantContent, CompletionRequest, Document, Message};
use rig::http_runtime::HttpRuntime;
use rig::providers::deepseek;
use serde::Deserialize;
use serde_json::Value;

use super::support::with_deepseek_cassette;
use crate::support::assert_contains_any_case_insensitive;

const SYSTEM_INSTRUCTION: &str = "Answer with the exact token from the document only.";
const DOCUMENT_ANSWER: &str = "violet-needle";
const PROMPT: &str = "According to the document, what is the ordering token?";

#[derive(Deserialize)]
struct RecordedInteraction {
    when: RecordedRequest,
}

#[derive(Deserialize)]
struct RecordedRequest {
    body: Option<String>,
}

fn ordering_document() -> Document {
    Document {
        id: "ordering-note".to_string(),
        text: format!("The ordering token is {DOCUMENT_ANSWER}."),
        additional_props: Default::default(),
    }
}

fn assistant_text(choice: &OneOrMany<AssistantContent>) -> String {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

#[tokio::test]
async fn chat_completions_keeps_documents_after_system_before_history() {
    with_deepseek_cassette(
        "document_ordering/chat_completions_keeps_documents_after_system_before_history",
        |env| async move {
            let model_cfg = env.config(deepseek::DEEPSEEK_V4_FLASH);
            let rt = HttpRuntime::new();
            let request = CompletionRequest {
                documents: vec![ordering_document()],
                temperature: Some(0.0),
                // Needs headroom for deepseek-v4-flash's thinking tokens now
                // that max_tokens is actually forwarded to the API.
                max_tokens: Some(512),
                ..CompletionRequest::with_history(
                    None,
                    vec![
                        Message::system(SYSTEM_INSTRUCTION),
                        Message::assistant("Acknowledged."),
                    ],
                    PROMPT,
                )
            };
            let response = deepseek::functions::complete(&model_cfg, &rt, request)
                .await
                .expect("DeepSeek document ordering request should succeed");

            assert_contains_any_case_insensitive(
                &assistant_text(&response.choice),
                &[DOCUMENT_ANSWER],
            );
        },
    )
    .await;

    assert_chat_request_order(
        "document_ordering/chat_completions_keeps_documents_after_system_before_history",
    );
}

fn recorded_request_body(scenario: &str) -> Value {
    let cassette_path = crate::cassettes::cassette_path("deepseek", scenario);
    let contents = std::fs::read_to_string(&cassette_path).unwrap_or_else(|error| {
        panic!(
            "provider cassette {} should be readable after recording: {error}",
            cassette_path.display()
        )
    });

    serde_yaml::Deserializer::from_str(&contents)
        .find_map(|document| {
            let interaction = RecordedInteraction::deserialize(document)
                .expect("cassette interaction should deserialize");
            interaction
                .when
                .body
                .and_then(|body| serde_json::from_str::<Value>(&body).ok())
        })
        .unwrap_or_else(|| panic!("expected cassette {scenario} to contain a JSON request body"))
}

fn assert_chat_request_order(scenario: &str) {
    let body = recorded_request_body(scenario);
    let messages = body["messages"]
        .as_array()
        .expect("DeepSeek request should contain messages[]");
    assert_eq!(
        messages.len(),
        4,
        "expected system, document, assistant history, and prompt messages: {body:#}"
    );
    assert_eq!(messages[0]["role"], "system");
    assert!(messages[0].to_string().contains(SYSTEM_INSTRUCTION));
    assert_eq!(messages[1]["role"], "user");
    assert!(
        messages[1].to_string().contains("<file id: ordering-note>"),
        "expected second message to contain normalized document: {body:#}"
    );
    assert_eq!(messages[2]["role"], "assistant");
    assert!(messages[2].to_string().contains("Acknowledged."));
    assert_eq!(messages[3]["role"], "user");
    assert!(messages[3].to_string().contains(PROMPT));
}
