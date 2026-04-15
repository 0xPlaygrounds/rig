//! Migrated from `examples/gemini_interactions_api.rs`.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures::StreamExt;
use rig::OneOrMany;
use rig::agent::{Agent, MultiTurnStreamItem, StreamingError};
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::{CompletionModel, GetTokenUsage, ToolDefinition};
use rig::message::{AssistantContent, Message, ToolCall, ToolChoice};
use rig::providers::gemini;
use rig::providers::gemini::interactions_api::{
    AdditionalParameters, ContentDelta, GenerationConfig, InteractionSseEvent,
    InteractionsCompletionModel, StreamingContentStart, ThinkingLevel, ThinkingSummaries,
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

type InteractionsAgent = Agent<InteractionsCompletionModel>;

const ORDER_SUPPORT_PREAMBLE: &str = "You are an order support assistant. If the conversation \
already contains a lookup_order tool result for the requested order, answer only from that \
existing tool result and do not call the tool again. Otherwise, you must call the lookup_order \
tool exactly once for the current order before answering. After the tool returns, answer in one \
concise sentence using only the tool JSON fields order_id, status, and eta_days.";
const ORDER_SUPPORT_PROMPT: &str =
    "Check order A-17. Use the tool exactly once, then answer from the tool result.";
const ORDER_SUPPORT_FOLLOWUP_PROMPT: &str = "Using only the existing conversation history, restate the order status in four words or fewer. Do not call any tool.";
const RAW_STREAM_TOOL_PROMPT: &str = "Check order A-17. Call the lookup_order tool exactly once and stop after emitting the tool call. Do not answer with prose yet.";

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

fn order_lookup_tool_definition() -> ToolDefinition {
    ToolDefinition {
        name: OrderLookupTool::NAME.to_string(),
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

fn roundtrip_additional_params() -> serde_json::Value {
    serde_json::to_value(AdditionalParameters {
        generation_config: Some(GenerationConfig {
            thinking_level: Some(ThinkingLevel::High),
            thinking_summaries: Some(ThinkingSummaries::None),
            ..Default::default()
        }),
        ..Default::default()
    })
    .expect("params should serialize")
}

fn build_stateless_roundtrip_agent(call_count: Arc<AtomicUsize>) -> InteractionsAgent {
    gemini::InteractionsClient::from_env()
        .agent("gemini-2.5-flash")
        .preamble(ORDER_SUPPORT_PREAMBLE)
        .tool(OrderLookupTool::new(call_count))
        .tool_choice(ToolChoice::Auto)
        .additional_params(roundtrip_additional_params())
        .build()
}

#[derive(Default)]
struct StreamObservation {
    tool_call_names: Vec<String>,
    tool_results: usize,
    reasoning_items: usize,
    final_turn_text: String,
    final_response_text: Option<String>,
    final_history: Option<Vec<Message>>,
}

impl StreamObservation {
    fn resolved_final_text(&self) -> String {
        if self.final_turn_text.trim().is_empty() {
            self.final_response_text.clone().unwrap_or_default()
        } else {
            self.final_turn_text.clone()
        }
    }
}

async fn collect_stream_observation<R, S>(stream: S) -> StreamObservation
where
    S: futures::Stream<Item = Result<MultiTurnStreamItem<R>, StreamingError>>,
{
    futures::pin_mut!(stream);

    let mut observation = StreamObservation::default();

    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::StreamAssistantItem(content)) => match content {
                StreamedAssistantContent::ToolCall { tool_call, .. } => {
                    observation.tool_call_names.push(tool_call.function.name);
                }
                StreamedAssistantContent::Reasoning(_) => {
                    observation.reasoning_items += 1;
                }
                StreamedAssistantContent::Text(text) => {
                    observation.final_turn_text.push_str(&text.text);
                }
                _ => {}
            },
            Ok(MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult { .. })) => {
                observation.tool_results += 1;
                observation.final_turn_text.clear();
            }
            Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                observation.final_response_text = Some(response.response().to_owned());
                observation.final_history = response.history().map(|history| history.to_vec());
            }
            Ok(_) => {}
            Err(error) => panic!("stream should succeed: {error}"),
        }
    }

    observation
}

fn assert_final_text_mentions_tool_output(final_text: &str) {
    assert_nonempty_response(final_text);
    let final_text_lower = final_text.to_ascii_lowercase();
    assert!(
        final_text.contains("A-17")
            || final_text_lower.contains("backordered")
            || final_text.contains("3"),
        "expected final text to reference tool output, got {:?}",
        final_text
    );
}

fn assistant_tool_turn_content(history: &[Message]) -> &OneOrMany<AssistantContent> {
    history
        .windows(2)
        .find_map(|window| match (&window[0], &window[1]) {
            (
                Message::Assistant { content, .. },
                Message::User {
                    content: user_content,
                },
            ) if user_content
                .iter()
                .any(|item| matches!(item, rig::message::UserContent::ToolResult(_))) =>
            {
                Some(content)
            }
            _ => None,
        })
        .expect("expected an assistant tool-call turn followed by a user tool result")
}

fn assert_history_replays_reasoning_before_tool_call(history: &[Message]) {
    let assistant_content = assistant_tool_turn_content(history)
        .iter()
        .collect::<Vec<_>>();
    let reasoning_index = assistant_content
        .iter()
        .position(|item| {
            matches!(
                item,
                AssistantContent::Reasoning(reasoning) if reasoning.first_signature().is_some()
            )
        })
        .expect("assistant tool turn should contain signed reasoning");
    let tool_call_index = assistant_content
        .iter()
        .position(|item| {
            matches!(
                item,
                AssistantContent::ToolCall(tool_call) if tool_call.function.name == OrderLookupTool::NAME
            )
        })
        .expect("assistant tool turn should contain the lookup_order tool call");

    assert!(
        reasoning_index < tool_call_index,
        "assistant tool turn should replay reasoning before the tool call: {:?}",
        assistant_content
    );
}

impl Tool for OrderLookupTool {
    const NAME: &'static str = "lookup_order";

    type Error = OrderLookupError;
    type Args = OrderLookupArgs;
    type Output = OrderLookupResult;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        order_lookup_tool_definition()
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
    let agent = build_stateless_roundtrip_agent(call_count.clone());

    let observation = collect_stream_observation(
        agent
            .stream_chat(ORDER_SUPPORT_PROMPT, Vec::<Message>::new())
            .multi_turn(3)
            .await,
    )
    .await;

    assert!(
        call_count.load(Ordering::SeqCst) >= 1,
        "tool was never invoked"
    );
    assert!(
        observation
            .tool_call_names
            .iter()
            .any(|name| name == OrderLookupTool::NAME),
        "expected at least one lookup_order tool call in stream"
    );
    assert!(
        observation.tool_results >= 1,
        "expected at least one tool result in stream"
    );
    assert_final_text_mentions_tool_output(&observation.resolved_final_text());
}

#[tokio::test]
#[ignore = "requires GEMINI_API_KEY"]
async fn streaming_tool_result_roundtrip_stateless_history_replays_reasoning() {
    let call_count = Arc::new(AtomicUsize::new(0));
    let agent = build_stateless_roundtrip_agent(call_count.clone());

    let observation = collect_stream_observation(
        agent
            .stream_chat(ORDER_SUPPORT_PROMPT, Vec::<Message>::new())
            .multi_turn(3)
            .await,
    )
    .await;

    assert!(
        call_count.load(Ordering::SeqCst) >= 1,
        "tool was never invoked"
    );
    assert!(
        observation.reasoning_items >= 1,
        "expected at least one streamed reasoning item before the tool call"
    );

    let final_history = observation
        .final_history
        .clone()
        .expect("expected final response history from stream_chat");
    assert!(
        final_history.len() >= 4,
        "expected at least [user, assistant tool turn, user tool result, assistant answer], got {:?}",
        final_history
    );
    assert_history_replays_reasoning_before_tool_call(&final_history);
    assert_final_text_mentions_tool_output(&observation.resolved_final_text());
}

#[tokio::test]
#[ignore = "requires GEMINI_API_KEY"]
async fn streaming_tool_result_roundtrip_stateless_history_is_reusable_without_tool_recall() {
    let call_count = Arc::new(AtomicUsize::new(0));
    let agent = build_stateless_roundtrip_agent(call_count.clone());

    let first_observation = collect_stream_observation(
        agent
            .stream_chat(ORDER_SUPPORT_PROMPT, Vec::<Message>::new())
            .multi_turn(3)
            .await,
    )
    .await;
    let initial_calls = call_count.load(Ordering::SeqCst);
    assert!(
        initial_calls >= 1,
        "tool was never invoked on the first turn"
    );

    let history = first_observation
        .final_history
        .clone()
        .expect("expected final response history from first stream");
    assert_history_replays_reasoning_before_tool_call(&history);

    let second_observation = collect_stream_observation(
        agent
            .stream_chat(ORDER_SUPPORT_FOLLOWUP_PROMPT, history)
            .multi_turn(3)
            .await,
    )
    .await;

    assert_eq!(
        second_observation.tool_call_names.len(),
        0,
        "expected stateless follow-up to answer from replayed history without another tool call"
    );
    assert_eq!(
        second_observation.tool_results, 0,
        "expected stateless follow-up to avoid another tool result turn"
    );
    assert_eq!(
        call_count.load(Ordering::SeqCst),
        initial_calls,
        "tool should not be invoked again when replayed history already contains the tool result"
    );
    assert_final_text_mentions_tool_output(&second_observation.resolved_final_text());
}

#[tokio::test]
#[ignore = "requires GEMINI_API_KEY"]
async fn raw_stream_tool_turn_emits_thought_signature_and_function_call_lifecycle() {
    let model = gemini::InteractionsClient::from_env().completion_model("gemini-2.5-flash");
    let request = model
        .completion_request(RAW_STREAM_TOOL_PROMPT)
        .preamble(ORDER_SUPPORT_PREAMBLE.to_string())
        .tool(order_lookup_tool_definition())
        .tool_choice(ToolChoice::Auto)
        .additional_params(roundtrip_additional_params())
        .build();

    let mut stream = model
        .stream_interaction_events(request)
        .await
        .expect("raw interaction stream should start");

    let mut function_call_indexes = HashSet::new();
    let mut saw_function_call_payload = false;
    let mut saw_function_call_stop = false;
    let mut saw_thought_signature = false;
    let mut saw_interaction_complete = false;

    while let Some(event) = stream.next().await {
        match event.expect("raw interaction stream should succeed") {
            InteractionSseEvent::ContentStart { index, content, .. } => match content {
                StreamingContentStart::FunctionCall(_) => {
                    function_call_indexes.insert(index);
                    saw_function_call_payload = true;
                }
                StreamingContentStart::Thought(thought) => {
                    if thought.signature.is_some() {
                        saw_thought_signature = true;
                    }
                }
                _ => {}
            },
            InteractionSseEvent::ContentDelta { index, delta, .. } => match delta {
                ContentDelta::FunctionCall(_) => {
                    function_call_indexes.insert(index);
                    saw_function_call_payload = true;
                }
                ContentDelta::ThoughtSignature(_) => {
                    saw_thought_signature = true;
                }
                _ => {}
            },
            InteractionSseEvent::ContentStop { index, .. } => {
                if function_call_indexes.contains(&index) {
                    saw_function_call_stop = true;
                }
            }
            InteractionSseEvent::InteractionComplete { .. } => {
                saw_interaction_complete = true;
            }
            InteractionSseEvent::Error { error, .. } => {
                panic!("raw interaction stream should not error: {error:?}");
            }
            _ => {}
        }
    }

    assert!(
        saw_function_call_payload,
        "expected raw SSE stream to include function call payload events"
    );
    assert!(
        saw_function_call_stop,
        "expected raw SSE stream to include a content.stop event for the function call"
    );
    assert!(
        saw_thought_signature,
        "expected raw SSE stream to include a Gemini thought signature when thinking summaries are disabled"
    );
    assert!(
        saw_interaction_complete,
        "expected raw SSE stream to reach interaction.complete"
    );
}
