//! Migrated from `examples/gemini_interactions_api.rs`.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures::StreamExt;
use rig::OneOrMany;
use rig::agent::MultiTurnStreamItem;
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::{CompletionModel, GetTokenUsage, ToolDefinition};
use rig::message::{AssistantContent, Message, ToolCall, ToolChoice};
use rig::providers::gemini;
use rig::providers::gemini::interactions_api::{
    AdditionalParameters, GenerationConfig, ThinkingLevel, ThinkingSummaries,
    Tool as InteractionsTool,
};
use rig::streaming::{StreamedAssistantContent, StreamedUserContent, StreamingChat};
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use thiserror::Error;

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

#[derive(Debug, Deserialize)]
struct OrderLookupArgs {
    order_id: String,
}

#[derive(Debug, Serialize)]
struct OrderLookupResult {
    order_id: String,
    status: String,
    eta_days: u64,
}

#[derive(Debug, Error)]
#[error("Order lookup unavailable")]
struct OrderLookupError;

#[derive(Clone)]
struct OrderLookupTool {
    call_count: Arc<AtomicUsize>,
}

impl OrderLookupTool {
    fn new(call_count: Arc<AtomicUsize>) -> Self {
        Self { call_count }
    }
}

impl Tool for OrderLookupTool {
    const NAME: &'static str = "lookup_order";

    type Error = OrderLookupError;
    type Args = OrderLookupArgs;
    type Output = OrderLookupResult;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Look up an order by id and return its status JSON.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "order_id": { "type": "string" }
                },
                "required": ["order_id"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        Ok(OrderLookupResult {
            order_id: args.order_id,
            status: "backordered".to_string(),
            eta_days: 3,
        })
    }
}

#[tokio::test]
#[ignore = "requires GEMINI_API_KEY"]
async fn basic_interaction_returns_id() {
    let model = gemini::InteractionsClient::from_env().completion_model("gemini-3-flash-preview");
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
    let model = gemini::InteractionsClient::from_env().completion_model("gemini-3-flash-preview");
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
    let model = gemini::InteractionsClient::from_env().completion_model("gemini-3-flash-preview");
    let response = model
        .completion(
            model
                .completion_request("Who won the Euro 2024 tournament?")
                .additional_params(
                    serde_json::to_value(AdditionalParameters {
                        tools: Some(vec![InteractionsTool::GoogleSearch]),
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
    let model = gemini::InteractionsClient::from_env().completion_model("gemini-3-flash-preview");
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
    let model = gemini::InteractionsClient::from_env().completion_model("gemini-3-flash-preview");
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

#[tokio::test]
#[ignore = "requires GEMINI_API_KEY"]
async fn streaming_tool_result_roundtrip_stateless() {
    let call_count = Arc::new(AtomicUsize::new(0));
    let agent = gemini::InteractionsClient::from_env()
        .agent("gemini-2.5-flash")
        .preamble(
            "You are an order support assistant. You must call the lookup_order tool exactly once \
             for this request. Do not answer before calling the tool. After the tool returns, \
             answer in one concise sentence using only the tool JSON fields order_id, status, \
             and eta_days. Do not call the tool again.",
        )
        .tool(OrderLookupTool::new(call_count.clone()))
        .tool_choice(ToolChoice::Auto)
        .additional_params(
            serde_json::to_value(AdditionalParameters {
                generation_config: Some(GenerationConfig {
                    thinking_level: Some(ThinkingLevel::High),
                    thinking_summaries: Some(ThinkingSummaries::None),
                    ..Default::default()
                }),
                ..Default::default()
            })
            .expect("params should serialize"),
        )
        .build();

    let stream = agent
        .stream_chat(
            "Check order A-17. Use the tool exactly once, then answer from the tool result.",
            Vec::<Message>::new(),
        )
        .multi_turn(3)
        .await;

    futures::pin_mut!(stream);

    let mut tool_calls = 0usize;
    let mut tool_results = 0usize;
    let mut final_text = String::new();
    let mut final_response_text = None;

    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::StreamAssistantItem(content)) => match content {
                StreamedAssistantContent::ToolCall { tool_call, .. } => {
                    tool_calls += 1;
                    assert_eq!(tool_call.function.name, OrderLookupTool::NAME);
                }
                StreamedAssistantContent::Text(text) => final_text.push_str(&text.text),
                _ => {}
            },
            Ok(MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult { .. })) => {
                tool_results += 1;
                final_text.clear();
            }
            Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                final_response_text = Some(response.response().to_owned());
            }
            Ok(_) => {}
            Err(error) => panic!("stream should succeed: {error}"),
        }
    }

    assert!(
        call_count.load(Ordering::SeqCst) >= 1,
        "tool was never invoked"
    );
    assert!(tool_calls >= 1, "expected at least one tool call in stream");
    assert!(
        tool_results >= 1,
        "expected at least one tool result in stream"
    );

    let final_text = if final_text.trim().is_empty() {
        final_response_text.clone().unwrap_or_default()
    } else {
        final_text
    };

    assert_nonempty_response(&final_text);
    let final_text_lower = final_text.to_ascii_lowercase();
    assert!(
        final_text.contains("A-17")
            || final_text_lower.contains("backordered")
            || final_text.contains("3"),
        "expected final text to reference tool output, got {:?}",
        final_text
    );

    let final_response_text = final_response_text.unwrap_or_default();
    assert_nonempty_response(&final_response_text);
}
