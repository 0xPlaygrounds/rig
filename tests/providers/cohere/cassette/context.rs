//! Cassette-backed Cohere context-document coverage.

use rig::completion::{CompletionModel, Document, Prompt};
use rig::prelude::*;
use std::collections::HashMap;

use super::super::{CASSETTE_MODEL, support::with_cohere_cassette};
use crate::support::{CONTEXT_DOCS, CONTEXT_PROMPT, assert_contains_any_case_insensitive};

#[tokio::test]
async fn context_documents_are_accepted() {
    with_cohere_cassette("context/context_documents_are_accepted", |client| async move {
        let agent = CONTEXT_DOCS
            .iter()
            .copied()
            .fold(client.agent(CASSETTE_MODEL), |builder, doc| {
                builder.context(doc)
            })
            .preamble("Use the provided context documents as the authoritative source. Answer concisely.")
            .build();

        let response = agent
            .prompt(CONTEXT_PROMPT)
            .await
            .expect("context prompt should succeed");

        assert_contains_any_case_insensitive(
            &response,
            &[
                "ancient tool",
                "farming tool",
                "farm the land",
                "used by the ancestors",
            ],
        );
    })
    .await;
}

#[tokio::test]
async fn document_metadata_and_multiple_documents_are_accepted() {
    with_cohere_cassette(
        "context/document_metadata_and_multiple_documents_are_accepted",
        |client| async move {
            let model = client.completion_model(CASSETTE_MODEL);
            let request = model
                .completion_request("Which dock is assigned beacon code amber-73?")
                .document(Document {
                    id: "harbor-record-1".to_string(),
                    text: "Beacon code amber-73 is assigned to Dock Seven.".to_string(),
                    additional_props: HashMap::from([
                        ("source".to_string(), "harbor-registry".to_string()),
                        ("region".to_string(), "north-bay".to_string()),
                    ]),
                })
                .document(Document {
                    id: "harbor-record-2".to_string(),
                    text: "Beacon code violet-19 is assigned to Dock Three.".to_string(),
                    additional_props: HashMap::from([(
                        "source".to_string(),
                        "harbor-registry".to_string(),
                    )]),
                })
                .max_tokens(32)
                .build();

            let response = model
                .completion(request)
                .await
                .expect("documents with metadata should be accepted");
            let text = response
                .choice
                .iter()
                .filter_map(|content| match content {
                    rig::completion::AssistantContent::Text(text) => Some(text.text.as_str()),
                    _ => None,
                })
                .collect::<String>();

            assert_contains_any_case_insensitive(&text, &["dock seven", "dock 7"]);
        },
    )
    .await;
}
