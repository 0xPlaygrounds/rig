//! Migrated from `examples/gemini_interactions_api.rs`.

use futures::StreamExt;
use rig::completion::CompletionModel;
use rig::message::{
    AssistantContent, Message, ToolCall, ToolChoice, ToolResultContent, UserContent,
};
use rig::prelude::*;
use rig::providers::gemini::interactions_api::{AdditionalParameters, Tool};
use rig::streaming::StreamedAssistantContent;

use crate::support::assert_nonempty_response;

fn extract_text(choice: &[AssistantContent]) -> String {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

fn first_tool_call(choice: &[AssistantContent]) -> Option<ToolCall> {
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
            let request = model
                .completion_request("Give me two fun facts about hummingbirds.")
                .preamble("Be concise.".to_string())
                .additional_params(serde_json::to_value(params).expect("params should serialize"))
                .build();
            // The interaction id is Gemini's continuation handle (fed back as
            // `previous_interaction_id`), not an assistant message id, so it
            // lives on the provider's own response rather than on the
            // normalized one.
            let raw = model
                .raw_completion(request)
                .await
                .expect("completion should succeed");

            let response: rig::completion::CompletionResponse =
                raw.clone().try_into().expect("response should normalize");
            assert_nonempty_response(&extract_text(&response.choice));
            assert!(
                !raw.id.is_empty(),
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
                .raw_completion(
                    model
                        .completion_request("Give me one short fact about hummingbirds.")
                        .additional_params(
                            serde_json::to_value(AdditionalParameters {
                                store: Some(true),
                                ..Default::default()
                            })
                            .expect("params should serialize"),
                        )
                        .build(),
                )
                .await
                .expect("initial completion should succeed");
            // Gemini's continuation handle lives on the provider's own
            // response; it is what `previous_interaction_id` echoes back.
            let interaction_id = initial.id.clone();
            assert!(!interaction_id.is_empty(), "expected an interaction id");

            let followup = model
                .completion(
                    model
                        .completion_request("Now answer with a short analogy.")
                        .additional_params(
                            serde_json::to_value(AdditionalParameters {
                                previous_interaction_id: Some(interaction_id),
                                ..Default::default()
                            })
                            .expect("params should serialize"),
                        )
                        .build(),
                )
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
            // The hosted-tool exchange log is provider-specific, so this asserts
            // against the Interactions payload itself and then normalizes that same
            // payload — one request, exactly as the cassette recorded it.
            let raw = model
                .raw_completion(
                    model
                        .completion_request("Who won the Euro 2024 tournament?")
                        .additional_params(
                            serde_json::to_value(AdditionalParameters {
                                tools: Some(vec![Tool::GoogleSearch]),
                                ..Default::default()
                            })
                            .expect("params should serialize"),
                        )
                        .build(),
                )
                .await
                .expect("search completion should succeed");

            assert!(
                !raw.google_search_exchanges().is_empty(),
                "expected a search-backed exchange"
            );

            let response: rig::completion::CompletionResponse = raw
                .try_into()
                .expect("interaction should normalize into a completion response");
            assert_nonempty_response(&extract_text(&response.choice));
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
                .raw_completion(
                    model
                        .completion_request("Use the add tool to sum 7 and 11.")
                        .tool(tool)
                        .tool_choice(ToolChoice::Required)
                        .additional_params(
                            serde_json::to_value(AdditionalParameters {
                                store: Some(true),
                                ..Default::default()
                            })
                            .expect("params should serialize"),
                        )
                        .build(),
                )
                .await
                .expect("tool call completion should succeed");

            // Gemini's continuation handle lives on the provider's own
            // response; the normalized view of that same value supplies the
            // tool call, so this still costs one interaction.
            let interaction_id = initial.id.clone();
            assert!(!interaction_id.is_empty(), "expected an interaction id");
            let initial: rig::completion::CompletionResponse =
                initial.try_into().expect("response should normalize");

            let tool_call = first_tool_call(&initial.choice).expect("expected a tool call");

            let followup = model
                .completion(
                    model
                        .completion_request(Message::from(UserContent::tool_result_for(
                            tool_call.id.clone(),
                            tool_call.provider.clone(),
                            tool_call.function.name.clone(),
                            vec![ToolResultContent::json(serde_json::json!({ "sum": 18.0 }))],
                        )))
                        .additional_params(
                            serde_json::to_value(AdditionalParameters {
                                previous_interaction_id: Some(interaction_id),
                                ..Default::default()
                            })
                            .expect("params should serialize"),
                        )
                        .build(),
                )
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
            let request = model
                .completion_request("Write a 3-line poem about rust and rivers.")
                .temperature(0.4)
                .build();
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
            let request = model
                .completion_request("Reply with exactly: interaction metadata ok")
                .temperature(0.0)
                .build();
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
