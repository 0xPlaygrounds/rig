//! Focused OpenAI cassette coverage for request document ordering.

use rig::OneOrMany;
use rig::client::CompletionClient;
use rig::completion::{AssistantContent, CompletionModel, Document, Message};
use rig::providers::openai;
use serde::Deserialize;
use serde_json::Value;

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
async fn responses_keeps_documents_after_system_before_history() {
    super::super::support::with_openai_cassette(
        "document_ordering/responses_keeps_documents_after_system_before_history",
        |client| async move {
            let response = client
                .completion_model(openai::GPT_4O)
                .completion_request(PROMPT)
                .message(Message::system(SYSTEM_INSTRUCTION))
                .message(Message::assistant("Acknowledged."))
                .document(ordering_document())
                .temperature(0.0)
                .max_tokens(32)
                .send()
                .await
                .expect("OpenAI Responses document ordering request should succeed");

            assert_contains_any_case_insensitive(
                &assistant_text(&response.choice),
                &[DOCUMENT_ANSWER],
            );
        },
    )
    .await;

    assert_responses_request_order(
        "document_ordering/responses_keeps_documents_after_system_before_history",
    );
}

#[tokio::test]
async fn chat_completions_keeps_documents_after_system_before_history() {
    super::super::support::with_openai_completions_cassette(
        "document_ordering/chat_completions_keeps_documents_after_system_before_history",
        |client| async move {
            let response = client
                .completion_model(openai::GPT_4O)
                .completion_request(PROMPT)
                .message(Message::system(SYSTEM_INSTRUCTION))
                .message(Message::assistant("Acknowledged."))
                .document(ordering_document())
                .temperature(0.0)
                .max_tokens(32)
                .send()
                .await
                .expect("OpenAI Chat Completions document ordering request should succeed");

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
    let cassette_path = crate::cassettes::cassette_path("openai", scenario);
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

fn assert_responses_request_order(scenario: &str) {
    let body = recorded_request_body(scenario);
    let input = body["input"]
        .as_array()
        .expect("OpenAI Responses request should contain input[]");
    assert_eq!(
        input.len(),
        4,
        "expected system, document, assistant history, and prompt input items: {body:#}"
    );
    assert_eq!(input[0]["role"], "system");
    assert!(input[0].to_string().contains(SYSTEM_INSTRUCTION));
    assert_eq!(input[1]["role"], "user");
    assert!(
        input[1].to_string().contains("<file id: ordering-note>"),
        "expected second input item to contain normalized document: {body:#}"
    );
    assert_eq!(input[2]["role"], "assistant");
    assert_eq!(input[2]["content"], "Acknowledged.");
    assert!(input[2].get("status").is_none());
    assert_eq!(input[3]["role"], "user");
    assert!(input[3].to_string().contains(PROMPT));
}

fn assert_chat_request_order(scenario: &str) {
    let body = recorded_request_body(scenario);
    let messages = body["messages"]
        .as_array()
        .expect("OpenAI Chat Completions request should contain messages[]");
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
