//! Cassette-backed Cohere context-document coverage.

use rig::completion::Document;
use std::collections::HashMap;

use super::super::{CASSETTE_MODEL, support::with_cohere_cassette};
use crate::support::assert_contains_any_case_insensitive;
use rig::completion::CompletionRequest;

#[tokio::test]
async fn document_metadata_and_multiple_documents_are_accepted() {
    with_cohere_cassette(
        "context/document_metadata_and_multiple_documents_are_accepted",
        |client| async move {
            let model = client.completion(CASSETTE_MODEL);
            let request = CompletionRequest::new("Which dock is assigned beacon code amber-73?")
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
                .max_tokens(32);

            let response = model
                .call(request)
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
            // Documents send the request to the native API, which cites them.
            assert_eq!(response.origin.api.as_str(), "cohere.chat");
            let cited = response
                .choice
                .iter()
                .filter_map(rig::completion::AssistantContent::native_item)
                .any(|item| item.to_string().contains("harbor-record-1"));
            assert!(
                cited,
                "the answer cites its document: {:?}",
                response.choice
            );
        },
    )
    .await;
}
