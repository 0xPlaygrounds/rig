//! Migrated from `examples/anthropic_plaintext_document.rs`.

use rig::OneOrMany;
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::message::{Document, DocumentMediaType, DocumentSourceKind, Message, UserContent};
use rig::providers::anthropic;

use crate::support::{assert_contains_any_case_insensitive, assert_nonempty_response};

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
#[ignore = "requires ANTHROPIC_API_KEY"]
async fn plaintext_document_prompt() {
    let client = anthropic::Client::from_env();
    let agent = client
        .agent(anthropic::completion::CLAUDE_SONNET_4_6)
        .preamble("You are a helpful assistant that analyzes documents.")
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
#[ignore = "requires ANTHROPIC_API_KEY"]
async fn plaintext_document_with_instruction() {
    let client = anthropic::Client::from_env();
    let agent = client
        .agent(anthropic::completion::CLAUDE_SONNET_4_6)
        .preamble("You are a helpful assistant that analyzes documents.")
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
