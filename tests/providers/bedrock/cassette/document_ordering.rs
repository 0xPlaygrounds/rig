//! Focused AWS Bedrock cassette coverage for request document ordering.

use base64::{Engine, prelude::BASE64_STANDARD};
use rig::OneOrMany;
use rig::bedrock;
use rig::completion::{AssistantContent, CompletionModel, Document, Message};
use rig::prelude::*;
use serde::Deserialize;
use serde_json::Value;

use super::super::support::with_bedrock_cassette;
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
async fn documents_are_prepended_before_history() {
    with_bedrock_cassette(
        "document_ordering/documents_are_prepended_before_history",
        |client| async move {
            let response = client
                .completion_model(bedrock::completion::AMAZON_NOVA_LITE)
                .completion_request(PROMPT)
                .message(Message::system(SYSTEM_INSTRUCTION))
                .message(Message::assistant("Acknowledged."))
                .document(ordering_document())
                .temperature(0.0)
                .max_tokens(32)
                .send()
                .await
                .expect("Bedrock document ordering request should succeed");

            assert_contains_any_case_insensitive(
                &assistant_text(&response.choice),
                &[DOCUMENT_ANSWER],
            );
        },
    )
    .await;

    assert_bedrock_request_order("document_ordering/documents_are_prepended_before_history");
}

fn recorded_request_body(scenario: &str) -> Value {
    let cassette_path = crate::cassettes::cassette_path("bedrock", scenario);
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

fn assert_bedrock_request_order(scenario: &str) {
    let body = recorded_request_body(scenario);
    let system = body["system"]
        .as_array()
        .expect("Bedrock request should contain system[]");
    assert!(
        system
            .iter()
            .any(|block| block.to_string().contains(SYSTEM_INSTRUCTION)),
        "expected system blocks to contain the system instruction: {body:#}"
    );

    let messages = body["messages"]
        .as_array()
        .expect("Bedrock request should contain messages[]");
    assert_eq!(
        messages.len(),
        3,
        "expected document, assistant history, and prompt messages: {body:#}"
    );
    assert_eq!(messages[0]["role"], "user");
    let document_bytes = messages[0]["content"]
        .as_array()
        .and_then(|content| {
            content
                .iter()
                .find_map(|block| block["document"]["source"]["bytes"].as_str())
        })
        .expect("expected first message to contain a Bedrock document source");
    let document_text = String::from_utf8(
        BASE64_STANDARD
            .decode(document_bytes)
            .expect("recorded Bedrock document source should be base64"),
    )
    .expect("recorded Bedrock document source should be UTF-8");
    assert!(
        document_text.contains(DOCUMENT_ANSWER),
        "expected first message document to contain normalized document text: {body:#}"
    );
    assert_eq!(messages[1]["role"], "assistant");
    assert!(
        messages[1].to_string().contains("Acknowledged."),
        "expected second message to contain assistant history: {body:#}"
    );
    assert_eq!(messages[2]["role"], "user");
    assert!(
        messages[2].to_string().contains(PROMPT),
        "expected final message to contain the user prompt: {body:#}"
    );
}
