//! Migrated from `examples/gemini_interactions_api.rs`.

use futures::StreamExt;
use rig::OneOrMany;
use rig::completion::{CompletionModel, CompletionRequest};
use rig::message::{
    AssistantContent, Message, ToolCall, ToolChoice, ToolResultContent, UserContent,
};
use rig::prelude::*;
use rig::providers::gemini::interactions_api::{AdditionalParameters, Tool};
use rig::streaming::StreamedAssistantContent;

use crate::support::assert_nonempty_response;

fn extract_text(choice: &OneOrMany<AssistantContent>) -> String {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

fn first_tool_call(choice: &OneOrMany<AssistantContent>) -> Option<ToolCall> {
    choice.iter().find_map(|content| match content {
        AssistantContent::ToolCall(tool_call) => Some(tool_call.clone()),
        _ => None,
    })
}

#[tokio::test]
async fn basic_interaction_returns_id() {
    super::super::support::with_gemini_interactions_cassette(
        "interactions_api/basic_interaction_returns_id",
        |client| async move {
            let model = client.completion_model("gemini-3-flash-preview");
            let params = AdditionalParameters {
                store: Some(true),
                ..Default::default()
            };
            let request = CompletionRequest {
                additional_params: Some(
                    serde_json::to_value(params).expect("params should serialize"),
                ),
                ..CompletionRequest::with_history(
                    Some("Be concise."),
                    Vec::new(),
                    "Give me two fun facts about hummingbirds.",
                )
            };
            let response = model
                .completion(request)
                .await
                .expect("completion should succeed");

            assert_nonempty_response(&extract_text(&response.choice));
            assert!(
                response
                    .message_id
                    .as_deref()
                    .is_some_and(|id| !id.is_empty()),
                "interactions api should return an interaction id"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn followup_with_previous_interaction_id() {
    super::super::support::with_gemini_interactions_cassette(
        "interactions_api/followup_with_previous_interaction_id",
        |client| async move {
            let model = client.completion_model("gemini-3-flash-preview");
            let initial = model
                .completion(CompletionRequest {
                    additional_params: Some(
                        serde_json::to_value(AdditionalParameters {
                            store: Some(true),
                            ..Default::default()
                        })
                        .expect("params should serialize"),
                    ),
                    ..CompletionRequest::from_prompt(
                        "Give me one short fact about hummingbirds.",
                    )
                })
                .await
                .expect("initial completion should succeed");
            let interaction_id = initial
                .message_id
                .clone()
                .expect("expected an interaction id");
            assert!(!interaction_id.is_empty(), "expected an interaction id");

            let followup = model
                .completion(CompletionRequest {
                    additional_params: Some(
                        serde_json::to_value(AdditionalParameters {
                            previous_interaction_id: Some(interaction_id),
                            ..Default::default()
                        })
                        .expect("params should serialize"),
                    ),
                    ..CompletionRequest::from_prompt("Now answer with a short analogy.")
                })
                .await
                .expect("followup completion should succeed");

            assert_nonempty_response(&extract_text(&followup.choice));
        },
    )
    .await;
}

#[tokio::test]
async fn google_search_tool_interaction() {
    super::super::support::with_gemini_interactions_cassette(
        "interactions_api/google_search_tool_interaction",
        |client| async move {
            let model = client.completion_model("gemini-3-flash-preview");
            let response = model
                .completion(CompletionRequest {
                    additional_params: Some(
                        serde_json::to_value(AdditionalParameters {
                            tools: Some(vec![Tool::GoogleSearch]),
                            ..Default::default()
                        })
                        .expect("params should serialize"),
                    ),
                    ..CompletionRequest::from_prompt("Who won the Euro 2024 tournament?")
                })
                .await
                .expect("search completion should succeed");

            assert_nonempty_response(&extract_text(&response.choice));
            // The facade response no longer carries the wire payload; the
            // search exchange is wire-specific, so deserialize the recorded
            // interaction body from the cassette itself.
            let bodies = crate::cassettes::recorded_response_bodies(
                "gemini",
                "interactions_api/google_search_tool_interaction",
            );
            let interaction: rig::providers::gemini::interactions_api::Interaction =
                serde_json::from_str(
                    bodies.first().expect("cassette should record one exchange"),
                )
                .expect("recorded body should deserialize as an Interaction");
            assert!(
                !interaction.google_search_exchanges().is_empty(),
                "expected a search-backed exchange"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn tool_result_roundtrip() {
    super::super::support::with_gemini_interactions_cassette(
        "interactions_api/tool_result_roundtrip",
        |client| async move {
            let model = client.completion_model("gemini-3-flash-preview");
            let tool = rig::completion::ToolDefinition {
                name: "add".to_string(),
                description: "Add two numbers together".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "x": { "type": "number" },
                        "y": { "type": "number" }
                    },
                    "required": ["x", "y"]
                }),
            };

            let initial = model
                .completion(CompletionRequest {
                    tools: vec![tool],
                    tool_choice: Some(ToolChoice::Required),
                    additional_params: Some(
                        serde_json::to_value(AdditionalParameters {
                            store: Some(true),
                            ..Default::default()
                        })
                        .expect("params should serialize"),
                    ),
                    ..CompletionRequest::from_prompt("Use the add tool to sum 7 and 11.")
                })
                .await
                .expect("tool call completion should succeed");

            let tool_call = first_tool_call(&initial.choice).expect("expected a tool call");
            let call_id = tool_call
                .call_id
                .clone()
                .unwrap_or_else(|| tool_call.id.clone());
            let interaction_id = initial
                .message_id
                .clone()
                .expect("expected an interaction id");
            assert!(!interaction_id.is_empty(), "expected an interaction id");

            let followup = model
                .completion(CompletionRequest {
                    additional_params: Some(
                        serde_json::to_value(AdditionalParameters {
                            previous_interaction_id: Some(interaction_id),
                            ..Default::default()
                        })
                        .expect("params should serialize"),
                    ),
                    ..CompletionRequest::from_prompt(Message::from(
                        UserContent::tool_result_with_call_id(
                            tool_call.function.name,
                            call_id,
                            OneOrMany::one(ToolResultContent::json(
                                serde_json::json!({ "sum": 18.0 }),
                            )),
                        ),
                    ))
                })
                .await
                .expect("tool result followup should succeed");

            assert_nonempty_response(&extract_text(&followup.choice));
        },
    )
    .await;
}

#[tokio::test]
async fn streaming_interaction() {
    super::super::support::with_gemini_interactions_cassette(
        "interactions_api/streaming_interaction",
        |client| async move {
            let model = client.completion_model("gemini-3-flash-preview");
            let request = CompletionRequest {
                temperature: Some(0.4),
                ..CompletionRequest::from_prompt("Write a 3-line poem about rust and rivers.")
            };
            let mut stream = model.stream(request).await.expect("stream should start");

            let mut text = String::new();
            let mut saw_usage = false;
            while let Some(chunk) = stream.next().await {
                match chunk.expect("stream chunk should succeed") {
                    StreamedAssistantContent::Text(delta) => text.push_str(&delta.text),
                    StreamedAssistantContent::Final(response) => {
                        saw_usage = response.usage.has_values();
                    }
                    _ => {}
                }
            }

            assert_nonempty_response(&text);
            assert!(
                saw_usage,
                "expected the final response to expose token usage"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn streaming_final_metadata_exposes_model_version() {
    super::super::support::with_gemini_interactions_cassette(
        "interactions_api/streaming_final_metadata_exposes_model_version",
        |client| async move {
            let model = client.completion_model("gemini-3-flash-preview");
            let request = CompletionRequest {
                temperature: Some(0.0),
                ..CompletionRequest::from_prompt("Reply with exactly: interaction metadata ok")
            };
            let mut stream = model.stream(request).await.expect("stream should start");

            let mut text = String::new();
            let mut final_model_version = None;
            let mut final_response_count = 0;
            let mut saw_usage = false;
            while let Some(chunk) = stream.next().await {
                match chunk.expect("stream chunk should succeed") {
                    StreamedAssistantContent::Text(delta) => text.push_str(&delta.text),
                    StreamedAssistantContent::Final(response) => {
                        final_response_count += 1;
                        saw_usage = response.usage.has_values();
                        final_model_version = response.model.clone();
                    }
                    _ => {}
                }
            }

            assert_nonempty_response(&text);
            assert_eq!(
                final_response_count, 1,
                "stream should yield exactly one final response"
            );
            assert_eq!(
                final_model_version.as_deref(),
                Some("gemini-3-flash-preview"),
                "expected Interactions stream final response to expose Interaction.model"
            );
            assert!(
                saw_usage,
                "expected final response to expose Interactions token usage"
            );
        },
    )
    .await;
}
