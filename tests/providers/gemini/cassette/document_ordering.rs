//! Focused Gemini cassette coverage for request document ordering.

use rig::OneOrMany;
use rig::client::CompletionClient;
use rig::completion::{AssistantContent, CompletionModel, Document, Message};
use rig::providers::gemini;
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
async fn generate_content_keeps_documents_after_system_before_history() {
    super::super::support::with_gemini_cassette(
        "document_ordering/generate_content_keeps_documents_after_system_before_history",
        |client| async move {
            let response = client
                .completion_model(gemini::completion::GEMINI_2_5_FLASH)
                .completion_request(PROMPT)
                .message(Message::system(SYSTEM_INSTRUCTION))
                .message(Message::assistant("Acknowledged."))
                .document(ordering_document())
                .temperature(0.0)
                .max_tokens(32)
                .send()
                .await
                .expect("Gemini document ordering request should succeed");

            assert_contains_any_case_insensitive(
                &assistant_text(&response.choice),
                &[DOCUMENT_ANSWER],
            );
        },
    )
    .await;

    assert_generate_content_request_order(
        "document_ordering/generate_content_keeps_documents_after_system_before_history",
    );
}

#[tokio::test]
async fn interactions_keeps_documents_after_system_before_history() {
    super::super::support::with_gemini_interactions_cassette(
        "document_ordering/interactions_keeps_documents_after_system_before_history",
        |client| async move {
            let response = client
                .completion_model("gemini-3-flash-preview")
                .completion_request(PROMPT)
                .message(Message::system(SYSTEM_INSTRUCTION))
                .message(Message::assistant("Acknowledged."))
                .document(ordering_document())
                .temperature(0.0)
                .max_tokens(512)
                .send()
                .await
                .expect("Gemini interactions document ordering request should succeed");

            assert_contains_any_case_insensitive(
                &assistant_text(&response.choice),
                &[DOCUMENT_ANSWER],
            );
        },
    )
    .await;

    assert_interactions_request_order(
        "document_ordering/interactions_keeps_documents_after_system_before_history",
    );
}

fn recorded_request_body(scenario: &str) -> Value {
    let cassette_path = crate::cassettes::cassette_path("gemini", scenario);
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

fn assert_generate_content_request_order(scenario: &str) {
    let body = recorded_request_body(scenario);
    assert!(
        body["systemInstruction"]
            .to_string()
            .contains(SYSTEM_INSTRUCTION),
        "expected Gemini systemInstruction to contain leading system message: {body:#}",
    );

    let contents = body["contents"]
        .as_array()
        .expect("Gemini request should contain contents[]");
    assert_eq!(
        contents.len(),
        3,
        "expected document, assistant history, and prompt turns: {body:#}"
    );
    assert_eq!(contents[0]["role"], "user");
    assert!(
        contents[0].to_string().contains("<file id: ordering-note>"),
        "expected first content turn to contain normalized document: {body:#}"
    );
    assert_eq!(contents[1]["role"], "model");
    assert!(
        contents[1].to_string().contains("Acknowledged."),
        "expected second content turn to preserve assistant history: {body:#}"
    );
    assert_eq!(contents[2]["role"], "user");
    assert!(
        contents[2].to_string().contains(PROMPT),
        "expected final content turn to remain prompt: {body:#}"
    );
}

fn assert_interactions_request_order(scenario: &str) {
    let body = recorded_request_body(scenario);
    assert_eq!(body["system_instruction"], SYSTEM_INSTRUCTION);

    let input = body["input"]
        .as_array()
        .expect("Gemini interactions request should contain input[]");
    assert_eq!(
        input.len(),
        3,
        "expected document, assistant history, and prompt turns: {body:#}"
    );
    assert_eq!(input[0]["type"], "user_input");
    assert!(
        input[0].to_string().contains("<file id: ordering-note>"),
        "expected first input turn to contain normalized document context text: {body:#}"
    );
    assert_eq!(input[1]["type"], "model_output");
    assert!(
        input[1].to_string().contains("Acknowledged."),
        "expected second input turn to preserve assistant history: {body:#}"
    );
    assert_eq!(input[2]["type"], "user_input");
    assert!(
        input[2].to_string().contains(PROMPT),
        "expected final input turn to remain prompt: {body:#}"
    );
}
