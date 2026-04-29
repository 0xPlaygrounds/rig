//! Migrated from `examples/gemini_interactions_api.rs`.

use futures::StreamExt;
use rig_core::OneOrMany;
use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::completion::{CompletionModel, GetTokenUsage};
use rig_core::message::{AssistantContent, Message, ToolCall, ToolChoice};
use rig_core::providers::gemini;
use rig_core::providers::gemini::interactions_api::{AdditionalParameters, Tool};
use rig_core::streaming::StreamedAssistantContent;

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
#[ignore = "requires GEMINI_API_KEY"]
async fn basic_interaction_returns_id() {
    let model = gemini::InteractionsClient::from_env()
        .expect("client should build")
        .completion_model("gemini-3-flash-preview");
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
    assert!(
        !response.raw_response.id.is_empty(),
        "interactions api should return an interaction id"
    );
}

#[tokio::test]
#[ignore = "requires GEMINI_API_KEY"]
async fn followup_with_previous_interaction_id() {
    let model = gemini::InteractionsClient::from_env()
        .expect("client should build")
        .completion_model("gemini-3-flash-preview");
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
    let interaction_id = initial.raw_response.id.clone();
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
}

#[tokio::test]
#[ignore = "requires GEMINI_API_KEY"]
async fn google_search_tool_interaction() {
    let model = gemini::InteractionsClient::from_env()
        .expect("client should build")
        .completion_model("gemini-3-flash-preview");
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

    assert_nonempty_response(&extract_text(&response.choice));
    assert!(
        !response.raw_response.google_search_exchanges().is_empty(),
        "expected a search-backed exchange"
    );
}

#[tokio::test]
#[ignore = "requires GEMINI_API_KEY"]
async fn tool_result_roundtrip() {
    let model = gemini::InteractionsClient::from_env()
        .expect("client should build")
        .completion_model("gemini-3-flash-preview");
    let tool = rig_core::completion::ToolDefinition {
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

    let tool_call = first_tool_call(&initial.choice).expect("expected a tool call");
    let call_id = tool_call
        .call_id
        .clone()
        .unwrap_or_else(|| tool_call.id.clone());
    let interaction_id = initial.raw_response.id.clone();
    assert!(!interaction_id.is_empty(), "expected an interaction id");

    let followup = model
        .completion(
            model
                .completion_request(Message::tool_result_with_call_id(
                    tool_call.function.name,
                    Some(call_id),
                    serde_json::json!({ "sum": 18.0 }).to_string(),
                ))
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
}

#[tokio::test]
#[ignore = "requires GEMINI_API_KEY"]
async fn streaming_interaction() {
    let model = gemini::InteractionsClient::from_env()
        .expect("client should build")
        .completion_model("gemini-3-flash-preview");
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
                saw_usage = response.token_usage().is_some();
            }
            _ => {}
        }
    }

    assert_nonempty_response(&text);
    assert!(
        saw_usage,
        "expected the final response to expose token usage"
    );
}
