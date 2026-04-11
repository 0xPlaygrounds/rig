//! Shared helpers for provider-backed reasoning-enabled integration tests.
//!
//! These tests verify that providers can handle reasoning-enabled requests,
//! preserve multi-turn history, and complete tool roundtrips. Visible reasoning
//! is recorded for diagnostics when a provider emits it, but is not required.
#![allow(dead_code)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures::StreamExt;
use rig::OneOrMany;
use rig::agent::{MultiTurnStreamItem, StreamingError};
use rig::completion::request::ToolDefinition;
use rig::completion::{self, CompletionModel};
use rig::message::{AssistantContent, Message, Reasoning, ReasoningContent, UserContent};
use rig::streaming::{StreamedAssistantContent, StreamedUserContent};
use rig::tool::Tool;
use rig::wasm_compat::WasmCompatSend;
use serde::Deserialize;
use serde_json::json;

pub(crate) const ROUNDTRIP_PREAMBLE: &str = "You are a helpful math tutor. Be concise.";

const ROUNDTRIP_TURN1_TEXT: &str = "\
A train leaves Station A at 60 km/h. Another train leaves Station B \
(300 km away) 30 minutes later at 90 km/h heading toward Station A. \
At what time do they meet, and how far from Station A? Show your work.";

const ROUNDTRIP_TURN2_TEXT: &str = "\
Now suppose both trains slow down by 10 km/h after traveling half \
the original distance. When do they meet now?";

pub(crate) struct ReasoningRoundtripAgent<M: CompletionModel> {
    pub(crate) model: M,
    pub(crate) preamble: String,
    pub(crate) additional_params: Option<serde_json::Value>,
}

impl<M> ReasoningRoundtripAgent<M>
where
    M: CompletionModel,
{
    pub(crate) fn new(model: M, additional_params: Option<serde_json::Value>) -> Self {
        Self {
            model,
            preamble: ROUNDTRIP_PREAMBLE.to_owned(),
            additional_params,
        }
    }
}

pub(crate) async fn run_reasoning_roundtrip_streaming<M>(agent: ReasoningRoundtripAgent<M>)
where
    M: CompletionModel,
    M::StreamingResponse: WasmCompatSend,
{
    let turn1_prompt = Message::User {
        content: OneOrMany::one(UserContent::text(ROUNDTRIP_TURN1_TEXT)),
    };

    let request = completion::CompletionRequest {
        preamble: Some(agent.preamble.clone()),
        chat_history: OneOrMany::one(turn1_prompt.clone()),
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: agent.additional_params.clone(),
        model: None,
        output_schema: None,
    };

    let mut stream = agent.model.stream(request).await.expect("Turn 1 stream");

    let mut assistant_content = Vec::new();
    let mut saw_reasoning_block = false;
    let mut reasoning_delta_text = String::new();
    let mut streamed_text = String::new();

    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(StreamedAssistantContent::Text(text)) => {
                streamed_text.push_str(&text.text);
            }
            Ok(StreamedAssistantContent::Reasoning(reasoning)) => {
                saw_reasoning_block = true;
                assistant_content.push(AssistantContent::Reasoning(reasoning));
            }
            Ok(StreamedAssistantContent::ReasoningDelta { reasoning, .. }) => {
                reasoning_delta_text.push_str(&reasoning);
            }
            Ok(_) => {}
            Err(error) => panic!("Turn 1 stream error: {error}"),
        }
    }

    // Providers like Gemini 2.5 emit thinking as deltas without signatures,
    // so turn the deltas into a single reasoning block before round-tripping.
    if !saw_reasoning_block && !reasoning_delta_text.is_empty() {
        assistant_content.push(AssistantContent::Reasoning(Reasoning::new(
            &reasoning_delta_text,
        )));
    }

    assert!(!streamed_text.is_empty(), "Turn 1 produced no text output.");

    assistant_content.push(AssistantContent::text(&streamed_text));

    let turn1_assistant = Message::Assistant {
        id: stream.message_id.clone(),
        content: OneOrMany::many(assistant_content).expect("non-empty"),
    };

    let turn2_prompt = Message::User {
        content: OneOrMany::one(UserContent::text(ROUNDTRIP_TURN2_TEXT)),
    };

    let request2 = completion::CompletionRequest {
        preamble: Some(agent.preamble.clone()),
        chat_history: OneOrMany::many(vec![turn1_prompt, turn1_assistant, turn2_prompt])
            .expect("non-empty"),
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: agent.additional_params.clone(),
        model: None,
        output_schema: None,
    };

    let mut stream2 = agent.model.stream(request2).await.expect("Turn 2 stream");
    let mut turn2_text = String::new();

    while let Some(chunk) = stream2.next().await {
        match chunk {
            Ok(StreamedAssistantContent::Text(text)) => {
                turn2_text.push_str(&text.text);
            }
            Ok(_) => {}
            Err(error) => panic!("Turn 2 stream error: {error}"),
        }
    }

    assert!(
        !turn2_text.is_empty(),
        "Turn 2 produced no text output. \
         Provider may have rejected the request with reasoning in chat history."
    );

    let trimmed = turn2_text.trim();
    assert!(
        trimmed.len() >= 20,
        "Turn 2 text suspiciously short ({} chars: {:?}). \
         Provider may not have processed the multi-turn context.",
        trimmed.len(),
        &trimmed[..trimmed.len().min(100)]
    );
}

pub(crate) async fn run_reasoning_roundtrip_nonstreaming<M>(agent: ReasoningRoundtripAgent<M>)
where
    M: CompletionModel,
{
    let turn1_prompt = Message::User {
        content: OneOrMany::one(UserContent::text(ROUNDTRIP_TURN1_TEXT)),
    };

    let request = completion::CompletionRequest {
        preamble: Some(agent.preamble.clone()),
        chat_history: OneOrMany::one(turn1_prompt.clone()),
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: agent.additional_params.clone(),
        model: None,
        output_schema: None,
    };

    let response = agent
        .model
        .completion(request)
        .await
        .expect("Turn 1 completion");

    let mut text_parts = String::new();

    for content in response.choice.iter() {
        match content {
            AssistantContent::Reasoning(_) => {}
            AssistantContent::Text(text) => {
                text_parts.push_str(&text.text);
            }
            _ => {}
        }
    }

    assert!(
        !text_parts.is_empty(),
        "Turn 1 non-streaming response has no text output."
    );

    let turn1_assistant = Message::Assistant {
        id: response.message_id,
        content: response.choice,
    };

    let turn2_prompt = Message::User {
        content: OneOrMany::one(UserContent::text(ROUNDTRIP_TURN2_TEXT)),
    };

    let request2 = completion::CompletionRequest {
        preamble: Some(agent.preamble.clone()),
        chat_history: OneOrMany::many(vec![turn1_prompt, turn1_assistant, turn2_prompt])
            .expect("non-empty"),
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: agent.additional_params.clone(),
        model: None,
        output_schema: None,
    };

    let response2 = agent
        .model
        .completion(request2)
        .await
        .expect("Turn 2 completion - provider may have rejected reasoning in chat history");

    let turn2_text: String = response2
        .choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect();

    assert!(
        !turn2_text.is_empty(),
        "Turn 2 non-streaming response has no text. \
         Provider may have rejected the request with reasoning in chat history."
    );

    let trimmed = turn2_text.trim();
    assert!(
        trimmed.len() >= 20,
        "Turn 2 text suspiciously short ({} chars: {:?}). \
         Provider may not have processed the multi-turn context.",
        trimmed.len(),
        &trimmed[..trimmed.len().min(100)]
    );
}

#[derive(Debug, thiserror::Error)]
#[error("Weather service unavailable")]
pub(crate) struct WeatherError;

#[derive(Deserialize)]
pub(crate) struct WeatherArgs {
    pub(crate) city: String,
}

pub(crate) struct WeatherTool {
    call_count: Arc<AtomicUsize>,
}

impl WeatherTool {
    pub(crate) fn new(call_count: Arc<AtomicUsize>) -> Self {
        Self { call_count }
    }
}

impl Tool for WeatherTool {
    const NAME: &'static str = "get_weather";
    type Error = WeatherError;
    type Args = WeatherArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "get_weather".to_string(),
            description:
                "Get the current weather for a city. Must be called for weather questions."
                    .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name to get weather for"
                    }
                },
                "required": ["city"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        Ok(format!(
            "Weather in {}: 72F (22C), sunny with light clouds, humidity 45%, wind 8 mph NW",
            args.city
        ))
    }
}

pub(crate) const TOOL_SYSTEM_PROMPT: &str = "\
You are a weather assistant. You have access to a get_weather tool. \
You must call the get_weather tool for any weather question and never guess weather data. \
After receiving the tool result, provide a concise summary of the weather.";

pub(crate) const TOOL_USER_PROMPT: &str = "\
I'm planning a trip. What is the current weather in Tokyo, Japan? \
Based on the weather conditions, should I pack an umbrella or sunscreen? \
Use the get_weather tool to check before answering.";

pub(crate) struct StreamStats {
    pub(crate) reasoning_block_count: usize,
    pub(crate) reasoning_delta_count: usize,
    pub(crate) reasoning_content_types: Vec<&'static str>,
    pub(crate) reasoning_has_signature: bool,
    pub(crate) reasoning_has_encrypted: bool,
    pub(crate) tool_calls_in_stream: Vec<String>,
    pub(crate) tool_results_in_stream: usize,
    pub(crate) text_chunks: usize,
    pub(crate) final_turn_text: String,
    pub(crate) final_response_text: Option<String>,
    pub(crate) got_final_response: bool,
    pub(crate) errors: Vec<String>,
    pub(crate) events: Vec<&'static str>,
}

impl StreamStats {
    fn new() -> Self {
        Self {
            reasoning_block_count: 0,
            reasoning_delta_count: 0,
            reasoning_content_types: vec![],
            reasoning_has_signature: false,
            reasoning_has_encrypted: false,
            tool_calls_in_stream: vec![],
            tool_results_in_stream: 0,
            text_chunks: 0,
            final_turn_text: String::new(),
            final_response_text: None,
            got_final_response: false,
            errors: vec![],
            events: vec![],
        }
    }

    pub(crate) fn total_reasoning(&self) -> usize {
        self.reasoning_block_count + self.reasoning_delta_count
    }

    pub(crate) fn reasoning_before_first_tool_call(&self) -> bool {
        let first_reasoning = self
            .events
            .iter()
            .position(|event| event.starts_with("reasoning"));
        let first_tool_call = self.events.iter().position(|event| *event == "tool_call");

        match (first_reasoning, first_tool_call) {
            (Some(reasoning), Some(tool_call)) => reasoning < tool_call,
            (Some(_), None) => true,
            _ => false,
        }
    }
}

fn record_reasoning(stats: &mut StreamStats, reasoning: &Reasoning, provider: &str) {
    stats.reasoning_block_count += 1;
    stats.events.push("reasoning_block");

    for content in &reasoning.content {
        let type_name = match content {
            ReasoningContent::Text { signature, .. } => {
                if signature.is_some() {
                    stats.reasoning_has_signature = true;
                }
                "Text"
            }
            ReasoningContent::Encrypted(_) => {
                stats.reasoning_has_encrypted = true;
                "Encrypted"
            }
            ReasoningContent::Summary(_) => "Summary",
            ReasoningContent::Redacted { .. } => "Redacted",
            _ => "Unknown",
        };
        stats.reasoning_content_types.push(type_name);
    }

    eprintln!(
        "[{provider}] Reasoning block: id={:?}, types={:?}",
        reasoning.id, stats.reasoning_content_types
    );
}

pub(crate) async fn collect_stream_stats<R>(
    stream: impl futures::Stream<Item = Result<MultiTurnStreamItem<R>, StreamingError>>,
    provider: &str,
) -> StreamStats
where
    R: std::fmt::Debug,
{
    let mut stats = StreamStats::new();

    futures::pin_mut!(stream);

    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::StreamAssistantItem(content)) => match content {
                StreamedAssistantContent::Reasoning(ref reasoning) => {
                    record_reasoning(&mut stats, reasoning, provider);
                }
                StreamedAssistantContent::ReasoningDelta { .. } => {
                    stats.reasoning_delta_count += 1;
                    if stats.events.last() != Some(&"reasoning_delta") {
                        stats.events.push("reasoning_delta");
                    }
                }
                StreamedAssistantContent::ToolCall { ref tool_call, .. } => {
                    stats
                        .tool_calls_in_stream
                        .push(tool_call.function.name.clone());
                    stats.events.push("tool_call");
                }
                StreamedAssistantContent::Text(ref text) => {
                    stats.text_chunks += 1;
                    stats.final_turn_text.push_str(&text.text);
                    if stats.events.last() != Some(&"text") {
                        stats.events.push("text");
                    }
                }
                StreamedAssistantContent::ToolCallDelta { .. } => {
                    if stats.events.last() != Some(&"tool_call_delta") {
                        stats.events.push("tool_call_delta");
                    }
                }
                StreamedAssistantContent::Final(_) => {
                    stats.events.push("final");
                }
            },
            Ok(MultiTurnStreamItem::StreamUserItem(ref content)) => match content {
                StreamedUserContent::ToolResult { .. } => {
                    stats.tool_results_in_stream += 1;
                    stats.final_turn_text.clear();
                    stats.events.push("tool_result");
                }
            },
            Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                stats.final_response_text = Some(response.response().to_owned());
                stats.got_final_response = true;
            }
            Ok(_) => {}
            Err(error) => {
                stats.errors.push(error.to_string());
            }
        }
    }

    stats
}

pub(crate) fn assert_universal(
    stats: &StreamStats,
    tool_invocations: &AtomicUsize,
    provider: &str,
) {
    assert!(
        stats.errors.is_empty(),
        "[{provider}] Stream had errors: {:?}",
        stats.errors
    );

    let invocations = tool_invocations.load(Ordering::SeqCst);
    assert!(
        invocations >= 1,
        "[{provider}] Tool was never invoked (count=0). Stream tool calls: {:?}",
        stats.tool_calls_in_stream
    );

    assert!(
        !stats.tool_calls_in_stream.is_empty(),
        "[{provider}] No tool-call events in stream."
    );

    assert!(
        stats
            .tool_calls_in_stream
            .iter()
            .any(|name| name == "get_weather"),
        "[{provider}] No get_weather tool call. Saw: {:?}",
        stats.tool_calls_in_stream
    );

    assert!(
        stats.tool_results_in_stream >= 1,
        "[{provider}] No tool-result events in stream. Tool invoked {invocations} times."
    );

    assert!(
        !stats.final_turn_text.trim().is_empty(),
        "[{provider}] Final text is empty."
    );

    let trimmed = stats.final_turn_text.trim();
    assert!(
        trimmed.len() >= 30,
        "[{provider}] Final text suspiciously short ({} chars): {:?}",
        trimmed.len(),
        &trimmed[..trimmed.len().min(100)]
    );

    let text_lower = stats.final_turn_text.to_ascii_lowercase();
    let references_tool_output = text_lower.contains("72")
        || text_lower.contains("22")
        || text_lower.contains("sunny")
        || text_lower.contains("tokyo")
        || text_lower.contains("weather")
        || text_lower.contains("temperature");
    assert!(
        references_tool_output,
        "[{provider}] Final text does not reference tool output: {:?}",
        &trimmed[..trimmed.len().min(200)]
    );

    assert!(
        stats.got_final_response,
        "[{provider}] Stream did not emit FinalResponse."
    );

    assert_eq!(
        stats.final_response_text.as_deref(),
        Some(stats.final_turn_text.as_str()),
        "[{provider}] FinalResponse.response() diverged from streamed text."
    );
}

pub(crate) fn assert_nonstreaming_universal(
    result: &str,
    tool_invocations: &AtomicUsize,
    provider: &str,
) {
    let invocations = tool_invocations.load(Ordering::SeqCst);
    assert!(
        invocations >= 1,
        "[{provider}] Tool was never invoked (count=0)."
    );

    let trimmed = result.trim();
    assert!(
        !trimmed.is_empty(),
        "[{provider}] Agent returned empty response."
    );

    assert!(
        trimmed.len() >= 30,
        "[{provider}] Response suspiciously short ({} chars): {:?}",
        trimmed.len(),
        &trimmed[..trimmed.len().min(100)]
    );

    let text_lower = result.to_ascii_lowercase();
    let references_tool_output = text_lower.contains("72")
        || text_lower.contains("22")
        || text_lower.contains("sunny")
        || text_lower.contains("tokyo")
        || text_lower.contains("weather")
        || text_lower.contains("temperature");
    assert!(
        references_tool_output,
        "[{provider}] Response does not reference tool output: {:?}",
        &trimmed[..trimmed.len().min(200)]
    );
}
