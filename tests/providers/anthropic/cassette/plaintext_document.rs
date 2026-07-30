//! Migrated from `examples/anthropic_plaintext_document.rs`.

use rig::OneOrMany;
use rig::completion::CompletionRequest;
use rig::message::{Document, DocumentMediaType, DocumentSourceKind, Message, UserContent};
use rig::providers::anthropic::completion::Citation;
use rig::providers::anthropic::completion::{self as anthropic_completion, CLAUDE_SONNET_4_6};
use rig::telemetry::ProviderResponseExt;

use serde_json::json;

use crate::support::{
    assert_contains_any_case_insensitive, assert_nonempty_response, collect_stream_final_response,
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

fn cited_rust_document() -> Document {
    Document {
        data: DocumentSourceKind::String(rust_document()),
        media_type: Some(DocumentMediaType::TXT),
        additional_params: Some(json!({
            "title": "Rust Goals",
            "citations": { "enabled": true }
        })),
    }
}

fn citation_prompt() -> Message {
    Message::User {
        content: OneOrMany::many(vec![
            UserContent::Document(cited_rust_document()),
            UserContent::text(
                "Using citations, answer in one sentence: what three goals does Rust focus on?",
            ),
        ])
        .expect("citation prompt content should be non-empty"),
    }
}

fn assistant_text(choice: &OneOrMany<rig::message::AssistantContent>) -> String {
    choice
        .iter()
        .filter_map(|content| match content {
            rig::message::AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect()
}

fn collect_anthropic_citations(
    choice: &OneOrMany<rig::message::AssistantContent>,
) -> Vec<Citation> {
    choice
        .iter()
        .filter_map(|content| match content {
            rig::message::AssistantContent::Text(text) => Some(text),
            _ => None,
        })
        .flat_map(|text| {
            anthropic_completion::anthropic_citations(text)
                .expect("citations should decode from Anthropic text metadata")
        })
        .collect()
}

#[tokio::test]
async fn plaintext_document_prompt() {
    super::super::support::with_anthropic_cassette(
        "plaintext_document/plaintext_document_prompt",
        |client| async move {
            let agent = client
                .agent(CLAUDE_SONNET_4_6)
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
        },
    )
    .await;
}

#[tokio::test]
async fn plaintext_document_with_instruction() {
    super::super::support::with_anthropic_cassette(
        "plaintext_document/plaintext_document_with_instruction",
        |client| async move {
            let agent = client
                .agent(CLAUDE_SONNET_4_6)
                .preamble("You are a helpful assistant that analyzes documents.")
                .temperature(0.5)
                .build();

            let response = agent
                .prompt(Message::User {
                    content: OneOrMany::many(vec![
                        UserContent::document(rust_document(), Some(DocumentMediaType::TXT)),
                        UserContent::text(
                            "List the three main goals of Rust mentioned in this document.",
                        ),
                    ])
                    .expect("content should be non-empty"),
                })
                .await
                .expect("instruction prompt should succeed");

            assert_contains_any_case_insensitive(&response, &["safety", "speed", "concurrency"]);
        },
    )
    .await;
}

#[tokio::test]
async fn streaming_document_citations_accepts_null_citation_start() {
    super::super::support::with_anthropic_cassette(
        "plaintext_document/streaming_document_citations_accepts_null_citation_start",
        |client| async move {
            let agent = client
                .agent(CLAUDE_SONNET_4_6)
                .preamble("Answer using the supplied document and citation metadata.")
                .temperature(0.0)
                .build();

            let mut stream = Box::pin(agent.runner(citation_prompt()).stream_run());
            let response = collect_stream_final_response(&mut stream)
                .await
                .expect("streaming document citations should accept null citations on text start");

            assert_contains_any_case_insensitive(&response, &["safety", "speed", "concurrency"]);
        },
    )
    .await;
}

#[tokio::test]
async fn document_citations_followup_preserves_assistant_citation_history() {
    super::super::support::with_anthropic_cassette(
        "plaintext_document/document_citations_followup_preserves_history",
        |client| async move {
            let model = client.completion_model(CLAUDE_SONNET_4_6);
            let prompt = citation_prompt();

            let first_request = CompletionRequest {
                max_tokens: Some(256),
                temperature: Some(0.0),
                ..CompletionRequest::with_history(
                    Some("Answer using the supplied document and preserve citation metadata."),
                    Vec::new(),
                    prompt.clone(),
                )
            };
            let first_turn = model
                .completion(first_request)
                .await
                .expect("first document citation turn should succeed");

            let first_turn_text = assistant_text(&first_turn.choice);
            assert_nonempty_response(&first_turn_text);
            assert_contains_any_case_insensitive(
                &first_turn_text,
                &["safety", "speed", "concurrency"],
            );
            // The rig response no longer carries the provider wire struct, so
            // re-check the wire-level text against the recorded first-turn body.
            let recorded_bodies = crate::cassettes::recorded_response_bodies(
                "anthropic",
                "plaintext_document/document_citations_followup_preserves_history",
            );
            let first_wire: anthropic_completion::CompletionResponse = serde_json::from_str(
                recorded_bodies
                    .first()
                    .expect("cassette should record the first citation turn"),
            )
            .expect("recorded first-turn body should deserialize as a Messages response");
            assert_eq!(
                first_wire.get_text_response().as_deref(),
                Some(first_turn_text.as_str())
            );

            let citations = collect_anthropic_citations(&first_turn.choice);
            assert!(!citations.is_empty(), "expected citations: {first_turn:?}");
            assert!(citations.iter().any(|citation| match citation {
                Citation::CharLocation {
                    cited_text,
                    document_index,
                    document_title,
                    ..
                } => {
                    *document_index == 0
                        && document_title.as_deref() == Some("Rust Goals")
                        && ["safety", "speed", "concurrency"]
                            .iter()
                            .any(|needle| cited_text.to_lowercase().contains(needle))
                }
                _ => false,
            }));

            let followup_request = CompletionRequest {
                max_tokens: Some(64),
                temperature: Some(0.0),
                ..CompletionRequest::with_history(
                    Some("Answer using the supplied document and preserve citation metadata."),
                    vec![
                        prompt,
                        Message::Assistant {
                            id: first_turn.message_id.clone(),
                            content: first_turn.choice.clone(),
                        },
                    ],
                    "Reply exactly: citations follow-up ok",
                )
            };
            let followup = model
                .completion(followup_request)
                .await
                .expect("follow-up citation history turn should succeed");

            assert_contains_any_case_insensitive(&assistant_text(&followup.choice), &["ok"]);
        },
    )
    .await;
}
