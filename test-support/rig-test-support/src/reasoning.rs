//! Shared helpers for provider-backed reasoning-enabled integration tests.
//!
//! These tests verify that providers can handle reasoning-enabled requests,
//! preserve multi-turn history, and complete tool roundtrips. Visible reasoning
//! is recorded for diagnostics when a provider emits it, but is not required.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use futures::StreamExt;

use rig_agent::agent::AgentBuilder;

use rig_agent::agent::AgentHook;

use rig_agent::agent::HookContext;

use rig_agent::agent::MultiTurnStreamItem;

use rig_agent::agent::ObservationAction;

use rig_agent::agent::ReasoningDelta;

use rig_agent::agent::StepEventKind;

use rig_agent::completion::PromptError;

use rig_agent::completion;

use rig_core::message::AssistantContent;

use rig_core::message::Message;

use rig_core::message::ReasoningContent;

use rig_core::message::ToolResultContent;

use rig_core::message::UserContent;

use rig_core::streaming::{Item, StreamEvent};

use rig_core::streaming::StreamedUserContent;

use rig_core::tool::Tool;

use serde::Deserialize;
use serde_json::json;

/// System instruction shared by the two-turn reasoning roundtrip.
pub const ROUNDTRIP_PREAMBLE: &str = "You are a helpful math tutor. Be concise.";

const REASONING_DELTA_HOOK_PROMPT: &str = "\
How many positive integers n < 400 are divisible by 6 but not by 9? \
Think through the counting carefully, then answer with only the integer.";

#[derive(Clone, Debug, PartialEq, Eq)]
struct ReasoningDeltaSnapshot {
    part: usize,
    delta: String,
    aggregated: Option<String>,
    turn: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ReasoningDeltaTimelineItem {
    Hook(ReasoningDeltaSnapshot),
    Stream(ReasoningDeltaSnapshot),
}

#[derive(Clone, Default)]
struct ReasoningDeltaHookRecorder {
    timeline: Arc<Mutex<Vec<ReasoningDeltaTimelineItem>>>,
}

impl ReasoningDeltaHookRecorder {
    fn record_stream_delta(&self, part: usize, delta: String) {
        self.timeline
            .lock()
            .expect("reasoning delta timeline lock")
            .push(ReasoningDeltaTimelineItem::Stream(ReasoningDeltaSnapshot {
                part,
                delta,
                aggregated: None,
                turn: 1,
            }));
    }

    fn snapshot(&self) -> Vec<ReasoningDeltaTimelineItem> {
        self.timeline
            .lock()
            .expect("reasoning delta timeline lock")
            .clone()
    }
}

impl AgentHook for ReasoningDeltaHookRecorder {
    async fn on_reasoning_delta(
        &self,
        ctx: &HookContext,
        event: ReasoningDelta<'_>,
    ) -> ObservationAction {
        assert!(
            ctx.is_streaming(),
            "ReasoningDelta must only be dispatched on the streaming surface"
        );
        self.timeline
            .lock()
            .expect("reasoning delta timeline lock")
            .push(ReasoningDeltaTimelineItem::Hook(ReasoningDeltaSnapshot {
                part: event.part.index(),
                delta: event.delta.to_owned(),
                aggregated: Some(event.aggregated.to_owned()),
                turn: ctx.turn(),
            }));
        ObservationAction::continue_run()
    }

    fn observes(&self, kind: StepEventKind) -> bool {
        kind == StepEventKind::ReasoningDelta
    }
}

/// Drive one real provider stream through the managed agent surface and pin
/// the `ReasoningDelta` hook contract against the emitted normalized deltas.
pub async fn run_reasoning_delta_hook_streaming(
    model: impl Into<rig_core::DynModel<rig_core::operation::Completion>>,
    additional_params: serde_json::Value,
    provider: &str,
) {
    let hook = ReasoningDeltaHookRecorder::default();
    let probe = hook.clone();
    let agent = AgentBuilder::new(model)
        .preamble("Reason carefully before giving a concise final answer.")
        .max_tokens(4096)
        .additional_params(additional_params)
        .build();
    let mut stream = agent
        .prompt(REASONING_DELTA_HOOK_PROMPT)
        .add_hook(hook)
        .stream();
    let mut final_text = None;

    while let Some(item) = stream.next().await {
        match item.unwrap_or_else(|error| panic!("[{provider}] agent stream failed: {error}")) {
            MultiTurnStreamItem::StreamAssistantItem(Item::Event(StreamEvent::Reasoning {
                part,
                text: reasoning,
            })) => {
                probe.record_stream_delta(part.index(), reasoning);
            }
            MultiTurnStreamItem::FinalResponse(response) => {
                final_text = Some(response.output().to_owned());
            }
            _ => {}
        }
    }

    let final_text = final_text.unwrap_or_else(|| panic!("[{provider}] missing final response"));
    assert!(
        final_text.contains("44"),
        "[{provider}] final response should contain the expected answer 44, got {final_text:?}"
    );

    let timeline = probe.snapshot();
    assert!(
        timeline.len() >= 4,
        "[{provider}] expected multiple hook/emitted reasoning-delta pairs, got {timeline:#?}"
    );
    assert_eq!(
        timeline.len() % 2,
        0,
        "[{provider}] every hooked reasoning delta must be emitted"
    );

    let mut aggregates = HashMap::<usize, String>::new();
    for pair in timeline.chunks_exact(2) {
        let (ReasoningDeltaTimelineItem::Hook(hooked), ReasoningDeltaTimelineItem::Stream(emitted)) =
            (&pair[0], &pair[1])
        else {
            panic!(
                "[{provider}] reasoning hooks must run immediately before outward emission: {pair:#?}"
            );
        };

        assert_eq!(hooked.turn, 1, "[{provider}] unexpected hook turn");
        assert_eq!(hooked.part, emitted.part, "[{provider}] part drift");
        assert_eq!(hooked.delta, emitted.delta, "[{provider}] delta drift");

        let expected_aggregate = aggregates.entry(hooked.part).or_default();
        expected_aggregate.push_str(&hooked.delta);
        assert_eq!(
            hooked.aggregated.as_deref(),
            Some(expected_aggregate.as_str()),
            "[{provider}] aggregate must contain exactly this part's deltas through the current fragment"
        );
    }
}

const ROUNDTRIP_TURN1_TEXT: &str = "\
A train leaves Station A at 60 km/h. Another train leaves Station B \
(300 km away) 30 minutes later at 90 km/h heading toward Station A. \
At what time do they meet, and how far from Station A? Show your work.";

const ROUNDTRIP_TURN2_TEXT: &str = "\
Now suppose both trains slow down by 10 km/h after traveling half \
the original distance. When do they meet now?";

/// Model and request configuration for a two-turn reasoning-history check.
pub struct ReasoningRoundtripAgent {
    /// Completion model used for both turns.
    pub model: rig_core::DynModel<rig_core::operation::Completion>,
    /// System instruction included in the roundtrip requests.
    pub preamble: String,
    /// Provider-specific parameters included in both requests.
    pub additional_params: Option<serde_json::Value>,
    /// Opt-in capability flag. Most providers stream reasoning as unsigned
    /// deltas (or emit none at all), so the shared roundtrip only records
    /// reasoning for diagnostics. A provider whose wire is known to carry a
    /// replay-required signature opts in here, and the streaming roundtrip
    /// then asserts that a complete `Reasoning` block with a signature
    /// reached the caller and was round-tripped into turn 2.
    pub expects_signed_reasoning_block: bool,
}

impl ReasoningRoundtripAgent {
    /// Configure the roundtrip with the shared preamble and unsigned-reasoning default.
    pub fn new(
        model: impl Into<rig_core::DynModel<rig_core::operation::Completion>>,
        additional_params: Option<serde_json::Value>,
    ) -> Self {
        let model: rig_core::DynModel<rig_core::operation::Completion> = model.into();
        Self {
            model,
            preamble: ROUNDTRIP_PREAMBLE.to_owned(),
            additional_params,
            expects_signed_reasoning_block: false,
        }
    }

    /// See [`ReasoningRoundtripAgent::expects_signed_reasoning_block`].
    pub fn expecting_signed_reasoning_block(mut self) -> Self {
        self.expects_signed_reasoning_block = true;
        self
    }
}

/// The issuer the reply's reasoning was sealed to, or its provider when it
/// sent none.
fn stream_issuer(response: &completion::CompletionResponse) -> rig_core::message::Issuer {
    response
        .choice
        .iter()
        .find_map(|content| match content {
            AssistantContent::Reasoning(reasoning) => Some(reasoning.issuer().clone()),
            _ => None,
        })
        .unwrap_or_else(|| rig_core::message::Issuer::from(response.provider.clone()))
}

/// Run and assert the two-turn streaming reasoning-history roundtrip.
pub async fn run_reasoning_roundtrip_streaming(agent: ReasoningRoundtripAgent) {
    run_reasoning_roundtrip_streaming_with_final(agent, |_| {}).await;
}

/// Run the streaming roundtrip and inspect turn 1's response with a custom oracle.
pub async fn run_reasoning_roundtrip_streaming_with_final<F>(
    agent: ReasoningRoundtripAgent,
    mut inspect_final: F,
) where
    F: FnMut(&completion::CompletionResponse),
{
    let turn1_prompt = Message::User {
        content: vec![UserContent::text(ROUNDTRIP_TURN1_TEXT)],
    };

    let request = completion::CompletionRequest {
        chat_history: vec![
            Message::system(agent.preamble.clone()),
            turn1_prompt.clone(),
        ],
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: agent.additional_params.clone(),
        model: None,
        output_schema: None,
        record_telemetry_content: false,
    };

    let mut stream = agent.model.stream(request).expect("Turn 1 stream");

    let mut streamed_text = String::new();

    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(Item::Event(StreamEvent::Text { text, .. })) => streamed_text.push_str(&text),
            Ok(_) => {}
            Err(error) => panic!("Turn 1 stream error: {error}"),
        }
    }
    let response = stream.finish().await.expect("Turn 1 finishes");
    inspect_final(&response);

    if agent.expects_signed_reasoning_block {
        let signed = response.choice.iter().any(|content| match content {
            AssistantContent::Reasoning(reasoning) => reasoning
                .open(reasoning.issuer())
                .is_some_and(|reasoning| reasoning.content.iter().any(|block| matches!(block, ReasoningContent::Text { signature, .. } if signature.is_some()))),
            _ => false,
        });
        assert!(
            signed,
            "Provider opted into signed reasoning but no streamed Reasoning part carried a \
             signature (a signature-only part must not be dropped): {:#?}",
            response.choice
        );
    }

    assert!(!streamed_text.is_empty(), "Turn 1 produced no text output.");

    // The history this suite recorded replays the reasoning the provider
    // announced (an id, a signature, or content other than plain text) and
    // the answer's text. A stream that only sent reasoning fragments
    // replays them as one part.
    let reasoning: Vec<_> = response
        .choice
        .iter()
        .filter(|content| match content {
            AssistantContent::Reasoning(reasoning) => {
                reasoning.open(reasoning.issuer()).is_some_and(|reasoning| {
                    reasoning.id.is_some()
                        || reasoning.content.iter().any(|block| {
                            !matches!(
                                block,
                                ReasoningContent::Text {
                                    signature: None,
                                    ..
                                }
                            )
                        })
                })
            }
            _ => false,
        })
        .cloned()
        .collect();
    let mut assistant_content = if reasoning.is_empty() {
        let fragments = response.reasoning();
        if fragments.is_empty() {
            Vec::new()
        } else {
            vec![AssistantContent::Reasoning(
                rig_core::message::Reasoning::new(&fragments).sealed(stream_issuer(&response)),
            )]
        }
    } else {
        reasoning
    };
    assistant_content.push(AssistantContent::text(&streamed_text));
    let turn1_assistant = Message::Assistant {
        id: response.message_id.clone(),
        content: assistant_content,
    };

    let turn2_prompt = Message::User {
        content: vec![UserContent::text(ROUNDTRIP_TURN2_TEXT)],
    };

    let request2 = completion::CompletionRequest {
        chat_history: vec![
            Message::system(agent.preamble.clone()),
            turn1_prompt,
            turn1_assistant,
            turn2_prompt,
        ],
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: agent.additional_params.clone(),
        model: None,
        output_schema: None,
        record_telemetry_content: false,
    };

    let mut stream2 = agent.model.stream(request2).expect("Turn 2 stream");
    let mut turn2_text = String::new();

    while let Some(chunk) = stream2.next().await {
        match chunk {
            Ok(Item::Event(StreamEvent::Text { text, .. })) => turn2_text.push_str(&text),
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

/// Run and assert the two-turn nonstreaming reasoning-history roundtrip.
pub async fn run_reasoning_roundtrip_nonstreaming(agent: ReasoningRoundtripAgent) {
    let turn1_prompt = Message::User {
        content: vec![UserContent::text(ROUNDTRIP_TURN1_TEXT)],
    };

    let request = completion::CompletionRequest {
        chat_history: vec![
            Message::system(agent.preamble.clone()),
            turn1_prompt.clone(),
        ],
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: agent.additional_params.clone(),
        model: None,
        output_schema: None,
        record_telemetry_content: false,
    };

    let response = agent.model.call(request).await.expect("Turn 1 completion");

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
        content: vec![UserContent::text(ROUNDTRIP_TURN2_TEXT)],
    };

    let request2 = completion::CompletionRequest {
        chat_history: vec![
            Message::system(agent.preamble.clone()),
            turn1_prompt,
            turn1_assistant,
            turn2_prompt,
        ],
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: agent.additional_params.clone(),
        model: None,
        output_schema: None,
        record_telemetry_content: false,
    };

    let response2 = agent
        .model
        .call(request2)
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
/// Error type required by the deterministic weather tool.
pub struct WeatherError;

#[derive(Deserialize)]
/// Arguments accepted by the deterministic weather tool.
pub struct WeatherArgs {
    /// City included in the fixed weather report.
    pub city: String,
}

/// Weather tool returning fixed conditions and counting invocations.
pub struct WeatherTool {
    call_count: Arc<AtomicUsize>,
}

impl WeatherTool {
    /// Create a weather tool that increments the supplied invocation counter.
    pub fn new(call_count: Arc<AtomicUsize>) -> Self {
        Self { call_count }
    }
}

impl Tool for WeatherTool {
    const NAME: &'static str = "get_weather";
    type Error = WeatherError;
    type Args = WeatherArgs;
    type Output = String;

    fn description(&self) -> String {
        "Get the current weather for a city. Must be called for weather questions.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name to get weather for"
                }
            },
            "required": ["city"]
        })
    }

    async fn call(
        &self,
        _context: &mut rig_core::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        Ok(format!(
            "Weather in {}: 72F (22C), sunny with light clouds, humidity 45%, wind 8 mph NW",
            args.city
        ))
    }
}

/// System instruction requiring the weather tool before answering.
pub const TOOL_SYSTEM_PROMPT: &str = "\
You are a weather assistant. You have access to a get_weather tool. \
You must call the get_weather tool for any weather question and never guess weather data. \
After receiving the tool result, provide a concise summary of the weather.";

/// Tokyo weather question used to exercise reasoning and tool continuation.
pub const TOOL_USER_PROMPT: &str = "\
I'm planning a trip. What is the current weather in Tokyo, Japan? \
Based on the weather conditions, should I pack an umbrella or sunscreen? \
Use the get_weather tool to check before answering.";

/// Reasoning, tool, text, and termination observations from an agent stream.
#[derive(Default)]
pub struct StreamStats {
    /// Number of complete reasoning blocks observed.
    pub reasoning_block_count: usize,
    /// Reasoning content variant names in observation order.
    pub reasoning_content_types: Vec<&'static str>,
    /// Whether any reasoning text carried a signature.
    pub reasoning_has_signature: bool,
    /// Whether any reasoning content was encrypted.
    pub reasoning_has_encrypted: bool,
    /// Tool-call names in stream order.
    pub tool_calls_in_stream: Vec<String>,
    /// Number of tool-result events observed.
    pub tool_results_in_stream: usize,
    /// Number of text delta events observed.
    pub text_chunks: usize,
    /// Text accumulated for the final model turn.
    pub final_turn_text: String,
    /// Output carried by the agent's final response, when present.
    pub final_response_text: Option<String>,
    /// Whether the stream emitted an agent final response.
    pub got_final_response: bool,
    /// Formatted stream errors in observation order.
    pub errors: Vec<String>,
}

fn record_reasoning(
    stats: &mut StreamStats,
    reasoning: &rig_core::message::Reasoning,
    provider: &str,
) {
    stats.reasoning_block_count += 1;

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
        };
        stats.reasoning_content_types.push(type_name);
    }

    eprintln!(
        "[{provider}] Reasoning block: id={:?}, types={:?}",
        reasoning.id, stats.reasoning_content_types
    );
}

/// Drain an agent stream into observations, retaining errors for later assertions.
pub async fn collect_stream_stats(
    stream: impl futures::Stream<Item = Result<MultiTurnStreamItem, PromptError>>,
    provider: &str,
) -> StreamStats {
    let mut stats = StreamStats::default();

    futures::pin_mut!(stream);

    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::ToolCall { ref tool_call, .. }) => {
                stats
                    .tool_calls_in_stream
                    .push(tool_call.function.name.clone().into());
            }
            Ok(MultiTurnStreamItem::StreamAssistantItem(content)) => match content {
                Item::Event(StreamEvent::End {
                    content: AssistantContent::Reasoning(ref reasoning),
                    ..
                }) => {
                    if let Some(reasoning) = reasoning.open(reasoning.issuer()) {
                        record_reasoning(&mut stats, reasoning, provider);
                    }
                }
                Item::Event(StreamEvent::Text { ref text, .. }) => {
                    stats.text_chunks += 1;
                    stats.final_turn_text.push_str(text);
                }
                Item::Event(_) | Item::Unknown(_) => {}
            },
            Ok(MultiTurnStreamItem::StreamUserItem(ref content)) => match content {
                StreamedUserContent::ToolResult { .. } => {
                    stats.tool_results_in_stream += 1;
                    stats.final_turn_text.clear();
                }
            },
            Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                stats.final_response_text = Some(response.output().to_owned());
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

/// Assert a successful streamed weather-tool roundtrip and consistent final text.
pub fn assert_universal(stats: &StreamStats, tool_invocations: &AtomicUsize, provider: &str) {
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
        "[{provider}] FinalResponse.output() diverged from streamed text."
    );
}

/// Assert tool execution and a substantive nonstreaming weather answer.
pub fn assert_nonstreaming_universal(result: &str, tool_invocations: &AtomicUsize, provider: &str) {
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

/// Assert that history retains the prompt, reasoning, tool exchange, and final answer in order.
pub fn assert_chat_history_preserves_reasoning_tool_roundtrip(
    chat_history: &[Message],
    result: &str,
    provider: &str,
) {
    assert!(
        chat_history.len() >= 4,
        "[{provider}] Chat history should contain at least user prompt, assistant tool call, tool result, and final assistant response. Got: {chat_history:#?}"
    );

    let result = result.trim();
    let mut prompt_index = None;
    let mut reasoning_index = None;
    let mut tool_call_index = None;
    let mut tool_result_index = None;
    let mut final_response_index = None;
    let mut tool_result_text = String::new();

    for (index, message) in chat_history.iter().enumerate() {
        match message {
            Message::User { content } => {
                for item in content.iter() {
                    match item {
                        UserContent::Text(text)
                            if text.text.contains("Tokyo")
                                || text.text.contains("get_weather")
                                || text.text.contains("weather") =>
                        {
                            prompt_index.get_or_insert(index);
                        }
                        UserContent::ToolResult(tool_result) => {
                            tool_result_index.get_or_insert(index);
                            for content in tool_result.content.iter() {
                                match content {
                                    ToolResultContent::Text(text) => {
                                        tool_result_text.push_str(&text.text);
                                    }
                                    ToolResultContent::Json { value } => {
                                        tool_result_text.push_str(&value.to_string());
                                    }
                                    ToolResultContent::Image(_) => {}
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            Message::Assistant { content, .. } => {
                let mut assistant_text = String::new();

                for item in content.iter() {
                    match item {
                        AssistantContent::Reasoning(_) => {
                            reasoning_index.get_or_insert(index);
                        }
                        AssistantContent::ToolCall(tool_call)
                            if tool_call.function.name == WeatherTool::NAME =>
                        {
                            tool_call_index.get_or_insert(index);
                        }
                        AssistantContent::Text(text) => {
                            assistant_text.push_str(&text.text);
                        }
                        _ => {}
                    }
                }

                let assistant_text = assistant_text.trim();
                if !assistant_text.is_empty()
                    && (assistant_text == result
                        || assistant_text.contains(result)
                        || result.contains(assistant_text))
                {
                    final_response_index.get_or_insert(index);
                }
            }
            Message::System { .. } => {}
        }
    }

    let prompt_index = prompt_index.unwrap_or_else(|| {
        panic!("[{provider}] Chat history is missing the original user prompt: {chat_history:#?}")
    });
    reasoning_index.unwrap_or_else(|| {
        panic!(
            "[{provider}] Chat history is missing assistant reasoning content: {chat_history:#?}"
        )
    });
    let tool_call_index = tool_call_index.unwrap_or_else(|| {
        panic!("[{provider}] Chat history is missing the get_weather tool call: {chat_history:#?}")
    });
    let tool_result_index = tool_result_index.unwrap_or_else(|| {
        panic!(
            "[{provider}] Chat history is missing the get_weather tool result: {chat_history:#?}"
        )
    });
    let final_response_index = final_response_index.unwrap_or_else(|| {
        panic!(
            "[{provider}] Chat history is missing the returned final assistant response {result:?}: {chat_history:#?}"
        )
    });

    assert!(
        prompt_index < tool_call_index,
        "[{provider}] Tool call should appear after the user prompt: {chat_history:#?}"
    );
    assert!(
        tool_call_index < tool_result_index,
        "[{provider}] Tool result should appear after the assistant tool call: {chat_history:#?}"
    );
    assert!(
        tool_result_index < final_response_index,
        "[{provider}] Final assistant response should appear after the tool result: {chat_history:#?}"
    );

    let tool_result_lower = tool_result_text.to_ascii_lowercase();
    assert!(
        tool_result_lower.contains("tokyo")
            && (tool_result_lower.contains("72") || tool_result_lower.contains("sunny")),
        "[{provider}] Tool result content was not preserved in chat history: {tool_result_text:?}"
    );
}
