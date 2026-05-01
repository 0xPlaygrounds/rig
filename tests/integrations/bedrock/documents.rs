//! AWS Bedrock document prompt smoke tests inspired by Anthropic document tests.

use rig::OneOrMany;
use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::message::{Document, DocumentMediaType, DocumentSourceKind, Message, UserContent};

use super::{
    BEDROCK_COMPLETION_MODEL, client,
    support::{assert_contains_any_case_insensitive, assert_nonempty_response},
};

fn rust_document() -> String {
    r#"
The Rust Programming Language

Rust is a systems programming language focused on three goals: safety, speed,
and concurrency. It accomplishes these goals without a garbage collector.

Key Features:
- Zero-cost abstractions
- Move semantics
- Guaranteed memory safety
- Threads without data races
"#
    .trim()
    .to_string()
}

#[tokio::test]
#[ignore = "requires AWS credentials and Bedrock model access"]
async fn plaintext_document_prompt() {
    let agent = client()
        .agent(BEDROCK_COMPLETION_MODEL)
        .preamble("Summarize the provided document.")
        .temperature(0.5)
        .build();

    let document = Document {
        data: DocumentSourceKind::String(rust_document()),
        media_type: Some(DocumentMediaType::TXT),
        additional_params: None,
    };
    let response = agent
        .prompt(document)
        .await
        .expect("document prompt should succeed");

    assert_nonempty_response(&response);
    assert_contains_any_case_insensitive(&response, &["safety", "speed", "concurrency"]);
}

#[tokio::test]
#[ignore = "requires AWS credentials and Bedrock model access"]
async fn plaintext_document_with_instruction() {
    let agent = client()
        .agent(BEDROCK_COMPLETION_MODEL)
        .preamble("Answer from the provided document.")
        .temperature(0.5)
        .build();

    let response = agent
        .prompt(Message::User {
            content: OneOrMany::many(vec![
                UserContent::document(rust_document(), Some(DocumentMediaType::TXT)),
                UserContent::text("List the three main goals of Rust mentioned in this document."),
            ])
            .expect("content should be non-empty"),
        })
        .await
        .expect("instruction prompt should succeed");

    assert_contains_any_case_insensitive(&response, &["safety", "speed", "concurrency"]);
}
