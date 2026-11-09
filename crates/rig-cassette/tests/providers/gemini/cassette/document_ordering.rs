//! Focused Gemini cassette coverage for request document ordering.
use rig::completion::{AssistantContent, Document, Message};

use crate::support::assert_contains_any_case_insensitive;
use rig::completion::CompletionRequest;

const SYSTEM_INSTRUCTION: &str = "Answer with the exact token from the document only.";
const DOCUMENT_ANSWER: &str = "violet-needle";
const PROMPT: &str = "According to the document, what is the ordering token?";

fn ordering_document() -> Document {
    Document {
        id: "ordering-note".to_string(),
        text: format!("The ordering token is {DOCUMENT_ANSWER}."),
        additional_props: Default::default(),
    }
}

fn assistant_text(choice: &[AssistantContent]) -> String {
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
async fn interactions_keeps_documents_after_system_before_history() {
    super::super::support::with_gemini_cassette(
        "document_ordering/interactions_keeps_documents_after_system_before_history",
        |client| async move {
            let response = client
                .interactions("gemini-3-flash-preview")
                .call(
                    CompletionRequest::new(PROMPT)
                        .message(Message::system(SYSTEM_INSTRUCTION))
                        .message(Message::assistant("Acknowledged."))
                        .document(ordering_document())
                        .temperature(0.0)
                        .max_tokens(512),
                )
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

fn assert_interactions_request_order(scenario: &str) {
    let body = crate::cassettes::recorded_json_request("gemini", scenario);
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
