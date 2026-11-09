//! Focused OpenRouter cassette coverage for request document ordering.
use rig::completion::{AssistantContent, Document, Message};

use super::super::{DEFAULT_MODEL, support::with_openrouter_cassette};
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
async fn chat_completions_keeps_documents_after_system_before_history() {
    with_openrouter_cassette(
        "document_ordering/chat_completions_keeps_documents_after_system_before_history",
        |client| async move {
            let response = client
                .completion(DEFAULT_MODEL)
                .call(
                    CompletionRequest::new(PROMPT)
                        .message(Message::system(SYSTEM_INSTRUCTION))
                        .message(Message::assistant("Acknowledged."))
                        .document(ordering_document())
                        .temperature(0.0)
                        .max_tokens(32),
                )
                .await
                .expect("OpenRouter document ordering request should succeed");

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

fn assert_chat_request_order(scenario: &str) {
    let body = crate::cassettes::recorded_json_request("openrouter", scenario);
    let messages = body["messages"]
        .as_array()
        .expect("OpenRouter request should contain messages[]");
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
