//! Focused OpenAI cassette coverage for request document ordering.
use rig::completion::{AssistantContent, Document, Message};
use rig::providers::openai;

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
async fn responses_keeps_documents_after_system_before_history() {
    super::super::support::with_openai_cassette(
        "document_ordering/responses_keeps_documents_after_system_before_history",
        |client| async move {
            let response = client
                .openai
                .completion(openai::GPT_4O)
                .call(
                    CompletionRequest::new(PROMPT)
                        .message(Message::system(SYSTEM_INSTRUCTION))
                        .message(Message::assistant("Acknowledged."))
                        .document(ordering_document())
                        .temperature(0.0)
                        .max_tokens(32)
                        .provider_options(super::super::support::stateless()),
                )
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

fn assert_responses_request_order(scenario: &str) {
    let body = crate::cassettes::recorded_json_request("openai", scenario);
    let input = body["input"]
        .as_array()
        .expect("OpenAI Responses request should contain input[]");
    assert_eq!(body["instructions"], SYSTEM_INSTRUCTION);
    assert_eq!(
        input.len(),
        3,
        "expected document, assistant history, and prompt input items after lifting system instructions: {body:#}"
    );
    assert_eq!(input[0]["role"], "user");
    assert!(
        input[0].to_string().contains("<file id: ordering-note>"),
        "expected first input item to contain normalized document: {body:#}"
    );
    // A hand-built turn is rebuilt as a completed message item.
    assert_eq!(input[1]["role"], "assistant");
    assert_eq!(input[1]["content"][0]["type"], "output_text");
    assert_eq!(input[1]["content"][0]["text"], "Acknowledged.");
    assert_eq!(input[2]["role"], "user");
    assert!(input[2].to_string().contains(PROMPT));
}
