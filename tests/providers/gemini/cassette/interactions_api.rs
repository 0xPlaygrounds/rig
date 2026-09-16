//! Migrated from `examples/gemini_interactions_api.rs`.

use futures::StreamExt;
use rig::completion::CompletionModel;
use rig::message::{
    AssistantContent, Message, ToolCall, ToolChoice, ToolResultContent, UserContent,
};
use rig::providers::gemini::interactions_api::{AdditionalParameters, Interaction, Tool};
use rig::streaming::{Delta, StreamEvent};
use serde::Deserialize;

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
            let model = client.map_wire(|config| config.interactions("gemini-3-flash-preview"));
            let params = AdditionalParameters {
                store: Some(true),
                ..Default::default()
            };
            let request = model
                .completion_request("Give me two fun facts about hummingbirds.")
                .preamble("Be concise.".to_string())
                .additional_params(serde_json::to_value(params).expect("params should serialize"))
                .build();
            let response = model
                .completion(request)
                .await
                .expect("completion should succeed");

            assert_nonempty_response(&extract_text(&response.choice));
            // The interaction id is Gemini's continuation handle (fed back as
            // `previous_interaction_id`), not an assistant message id: it is
            // on the interaction document `raw` carries, and the decoder
            // reports that very value as the response id.
            let document = Interaction::deserialize(&response.raw)
                .expect("raw is the Interactions API's own document");
            assert!(
                !document.id.is_empty(),
                "interactions api should return an interaction id"
            );
            assert_eq!(
                response.response_id.as_deref(),
                Some(document.id.as_str()),
                "the continuation handle is what the normalized response names"
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
            let model = client.map_wire(|config| config.interactions("gemini-3-flash-preview"));
            let initial = model
                .completion(
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
            // Gemini's continuation handle is the interaction id, which the
            // decoder reports as the response id; it is what
            // `previous_interaction_id` echoes back.
            let interaction_id = initial
                .response_id
                .clone()
                .expect("expected an interaction id");
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
            let model = client.map_wire(|config| config.interactions("gemini-3-flash-preview"));
            // The hosted-tool exchange log is provider-specific, so this
            // asserts against the interaction document `raw` carries and then
            // against the normalized view the decoder folded from the same
            // bytes — one request, exactly as the cassette recorded it.
            let response = model
                .completion(
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

            let document = Interaction::deserialize(&response.raw)
                .expect("raw is the Interactions API's own document");
            assert!(
                !document.google_search_exchanges().is_empty(),
                "expected a search-backed exchange"
            );

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
            let model = client.map_wire(|config| config.interactions("gemini-3-flash-preview"));
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
                .completion(
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

            // Gemini's continuation handle is the interaction id the decoder
            // reports as the response id, and the same response supplies the
            // tool call — so this still costs one interaction.
            let interaction_id = initial
                .response_id
                .clone()
                .expect("expected an interaction id");
            assert!(!interaction_id.is_empty(), "expected an interaction id");

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
            let model = client.map_wire(|config| config.interactions("gemini-3-flash-preview"));
            let request = model
                .completion_request("Write a 3-line poem about rust and rivers.")
                .temperature(0.4)
                .build();
            let mut stream = model.stream(request).await.expect("stream should start");

            let mut text = String::new();
            let mut saw_usage = false;
            while let Some(chunk) = stream.next().await {
                match chunk.expect("stream chunk should succeed") {
                    StreamEvent::BlockDelta {
                        delta: Delta::Text { text: delta },
                        ..
                    } => text.push_str(&delta),
                    StreamEvent::Final(response) => {
                        saw_usage = response.usage.is_reported();
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
            let model = client.map_wire(|config| config.interactions("gemini-3-flash-preview"));
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
                    StreamEvent::BlockDelta {
                        delta: Delta::Text { text: delta },
                        ..
                    } => text.push_str(&delta),
                    StreamEvent::Final(response) => {
                        final_response_count += 1;
                        saw_usage = response.usage.is_reported();
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

/// The Interactions surface must surface every token counter it is sent.
///
/// Replays a cassette recorded long before this assertion existed, which is what
/// makes it a genuine regression cell: the recorded wire reports
/// `total_input_tokens: 14`, `total_output_tokens: 34`, `total_tokens: 270` and
/// `total_thought_tokens: 222`. Rig's `InteractionUsage` had three fields, so
/// the 222 thinking tokens — the *majority* of the spend — were dropped on the
/// floor, and the normalized triple could not explain its own total. The wire
/// also carries `total_cached_tokens`, which meant `cached_input_tokens` was
/// structurally zero on this surface no matter what Gemini reported.
///
/// On `origin/main` this fails with `reasoning_tokens == 0`.
#[tokio::test]
async fn interactions_usage_surfaces_thinking_and_cached_tokens() {
    super::super::support::with_gemini_interactions_cassette(
        "interactions_api/basic_interaction_returns_id",
        |client| async move {
            let model = client.map_wire(|config| config.interactions("gemini-3-flash-preview"));
            let params = AdditionalParameters {
                store: Some(true),
                ..Default::default()
            };
            let request = model
                .completion_request("Give me two fun facts about hummingbirds.")
                .preamble("Be concise.".to_string())
                .additional_params(serde_json::to_value(params).expect("params should serialize"))
                .build();

            let response = rig::completion::CompletionModel::completion(&model, request)
                .await
                .expect("completion should succeed");
            let usage = response.usage;

            assert!(
                usage.reasoning_tokens.is_some_and(|n| n > 0),
                "the recorded interaction reports total_thought_tokens; dropping it loses the \
                 largest component of the spend. got {usage:?}"
            );
            assert_eq!(
                Some(
                    usage.input_tokens.unwrap_or(0)
                        + usage.output_tokens.unwrap_or(0)
                        + usage.reasoning_tokens.unwrap_or(0)
                ),
                usage.total_tokens,
                "on this surface thinking is reported beside input/output, so the components \
                 should account for the provider's own total: {usage:?}"
            );
        },
    )
    .await;
}
