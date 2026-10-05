//! Anthropic streaming tools smoke test.

use rig::message::AssistantContent;
use rig::streaming::Item;
use rig_cassette::agent::AgentReplayExt;

use futures::StreamExt;
use rig::agent::{MultiTurnStreamItem, StreamingResult};
use rig::message::{CallId, Message, UserContent};
use rig::providers::anthropic;
use rig::streaming::StreamEvent;
use rig::tool::Tool;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering::SeqCst};

use super::super::support::with_anthropic_cassette;
use crate::support::{
    ALPHA_SIGNAL_OUTPUT, Adder, AlphaSignal, BETA_SIGNAL_OUTPUT, BetaSignal, EmptyArgs, MathError,
    STREAMING_TOOLS_PREAMBLE, STREAMING_TOOLS_PROMPT, Subtract, TWO_TOOL_STREAM_PREAMBLE,
    TWO_TOOL_STREAM_PROMPT, assert_mentions_expected_number, collect_stream_final_response,
};

#[derive(Clone, Default)]
pub(super) struct OutOfOrderSignalOrder {
    gate: Arc<tokio::sync::Notify>,
    order: Arc<AtomicU32>,
}

impl OutOfOrderSignalOrder {
    async fn wait_until_this_tool_should_finish(&self) {
        let nth = self.order.fetch_add(1, SeqCst);
        if nth == 0 {
            self.gate.notified().await;
        } else {
            self.gate.notify_one();
        }
    }
}

#[derive(Clone)]
pub(super) struct OutOfOrderAlphaSignal(pub(super) OutOfOrderSignalOrder);

impl Tool for OutOfOrderAlphaSignal {
    const NAME: &'static str = AlphaSignal::NAME;
    type Error = MathError;
    type Args = EmptyArgs;
    type Output = String;

    fn description(&self) -> String {
        AlphaSignal.description()
    }

    fn parameters(&self) -> serde_json::Value {
        AlphaSignal.parameters()
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        self.0.wait_until_this_tool_should_finish().await;
        Ok(ALPHA_SIGNAL_OUTPUT.to_string())
    }
}

#[derive(Clone)]
pub(super) struct OutOfOrderBetaSignal(pub(super) OutOfOrderSignalOrder);

impl Tool for OutOfOrderBetaSignal {
    const NAME: &'static str = BetaSignal::NAME;
    type Error = MathError;
    type Args = EmptyArgs;
    type Output = String;

    fn description(&self) -> String {
        BetaSignal.description()
    }

    fn parameters(&self) -> serde_json::Value {
        BetaSignal.parameters()
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        self.0.wait_until_this_tool_should_finish().await;
        Ok(BETA_SIGNAL_OUTPUT.to_string())
    }
}

#[derive(Default)]
struct ConcurrentToolObservation {
    tool_calls: Vec<String>,
    streamed_tool_results: Vec<String>,
    history_tool_results: Vec<String>,
    last_history_tool_result_message: Vec<String>,
    final_response_text: Option<String>,
    errors: Vec<String>,
    got_final_response: bool,
    events: Vec<&'static str>,
}

async fn collect_concurrent_tool_observation(
    stream: &mut StreamingResult,
) -> ConcurrentToolObservation {
    let mut observation = ConcurrentToolObservation::default();
    let mut tool_names_by_id = HashMap::new();

    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::ToolCall { tool_call, .. }) => {
                tool_names_by_id.insert(tool_call.id.clone(), tool_call.function.name.to_string());
                observation.tool_calls.push(tool_call.function.name.into());
                observation.events.push("tool_call");
            }
            Ok(MultiTurnStreamItem::ToolExecutionCommitted { .. }) => {
                observation.events.push("tool_execution_committed");
            }
            Ok(MultiTurnStreamItem::ToolResult { tool_result, .. }) => {
                observation
                    .streamed_tool_results
                    .push(tool_name_for_result(&tool_names_by_id, &tool_result.call));
                observation.events.push("tool_result");
            }
            Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                observation.final_response_text = Some(response.output().to_owned());
                observation.got_final_response = true;
                let history = response.messages();
                observation.history_tool_results =
                    tool_result_names_in_history(history, &tool_names_by_id);
                observation.last_history_tool_result_message =
                    last_tool_result_message_names(history, &tool_names_by_id);
                observation.events.push("final_response");
            }
            Ok(MultiTurnStreamItem::StreamAssistantItem(Item::Event(StreamEvent::Text {
                text: _,
                ..
            }))) => {
                observation.events.push("text");
            }
            Ok(MultiTurnStreamItem::StreamAssistantItem(Item::Event(StreamEvent::Arguments {
                ..
            }))) => {
                observation.events.push("tool_call_delta");
            }
            Ok(MultiTurnStreamItem::StreamAssistantItem(Item::Event(StreamEvent::End {
                content: AssistantContent::Reasoning(_),
                ..
            }))) => {
                observation.events.push("reasoning");
            }
            Ok(MultiTurnStreamItem::StreamAssistantItem(Item::Event(StreamEvent::Reasoning {
                ..
            }))) => {
                observation.events.push("reasoning_delta");
            }
            Ok(MultiTurnStreamItem::StreamAssistantItem(Item::Unknown(_))) => {
                observation.events.push("unknown");
            }
            Ok(MultiTurnStreamItem::CompletionCall(_)) => {}
            Ok(_) => {}
            Err(error) => {
                observation.errors.push(error.to_string());
                observation.events.push("error");
            }
        }
    }

    observation
}

fn tool_result_names_in_history(
    history: &[Message],
    tool_names_by_id: &HashMap<CallId, String>,
) -> Vec<String> {
    history
        .iter()
        .flat_map(|message| match message {
            Message::User { content } => content
                .iter()
                .filter_map(|item| match item {
                    UserContent::ToolResult(tool_result) => {
                        Some(tool_name_for_result(tool_names_by_id, &tool_result.call))
                    }
                    _ => None,
                })
                .collect::<Vec<_>>(),
            _ => Vec::new(),
        })
        .collect()
}

fn last_tool_result_message_names(
    history: &[Message],
    tool_names_by_id: &HashMap<CallId, String>,
) -> Vec<String> {
    history
        .iter()
        .rev()
        .find_map(|message| match message {
            Message::User { content }
                if content
                    .iter()
                    .any(|item| matches!(item, UserContent::ToolResult(_))) =>
            {
                Some(
                    content
                        .iter()
                        .filter_map(|item| match item {
                            UserContent::ToolResult(tool_result) => {
                                Some(tool_name_for_result(tool_names_by_id, &tool_result.call))
                            }
                            _ => None,
                        })
                        .collect(),
                )
            }
            _ => None,
        })
        .unwrap_or_default()
}

fn tool_name_for_result(tool_names_by_id: &HashMap<CallId, String>, call: &CallId) -> String {
    tool_names_by_id
        .get(call)
        .cloned()
        .unwrap_or_else(|| format!("<unknown tool result id {call}>"))
}

/// Golden `anthropic_streaming_with_events`: a streamed tool turn recorded
/// with its stream events kept, so the corpus pins the event sequence, not
/// only the folded completion.
#[tokio::test]
async fn streaming_tools_effect_log_is_the_golden_fixture() {
    with_anthropic_cassette(
        "streaming_tools/streaming_tools_smoke",
        |client| async move {
            let recorder = rig_cassette::effect_log::EffectLogRecorder::keeping_stream_events();
            let agent =
                rig::AgentBuilder::new(client.completion(anthropic::completion::CLAUDE_SONNET_4_6))
                    .name("golden")
                    .preamble(STREAMING_TOOLS_PREAMBLE)
                    .tool(Adder)
                    .tool(Subtract)
                    .default_max_turns(2)
                    .record_to(recorder.clone())
                    .build();
            let mut stream = agent.prompt(STREAMING_TOOLS_PROMPT).stream();
            let response = collect_stream_final_response(&mut stream)
                .await
                .expect("streaming tool prompt should succeed");
            assert_mentions_expected_number(&response, -3);
            let log = agent.stamp(recorder.take());
            assert!(
                log.records.iter().any(|record| record
                    .events
                    .as_ref()
                    .is_some_and(|events| !events.is_empty())),
                "a streamed completion keeps its events"
            );
            crate::goldens::golden_effects("anthropic_streaming_with_events", &log);
        },
    )
    .await;
}

/// Golden `anthropic_concurrent_tools_serial`: two tool calls in one turn
/// dispatched concurrently by the runner and served one at a time per key
/// (`serial_per_handler: true`) — the cassette-ordered property, recorded.
#[tokio::test]
async fn concurrent_tools_serial_effect_log_is_the_golden_fixture() {
    with_anthropic_cassette(
        "streaming_tools/streaming_tool_concurrency_emits_results_as_completed_but_persists_call_order",
        |client| async move {
            let order = OutOfOrderSignalOrder::default();
            let recorder = rig_cassette::effect_log::EffectLogRecorder::new();
            let agent = rig::AgentBuilder::new(client.completion(anthropic::completion::CLAUDE_SONNET_4_6))
                .name("golden")
                .configure_bus(rig_core::serve::ServingPolicy {
                    serial_per_handler: true,
                    ..rig_core::serve::ServingPolicy::default()
                })
                .preamble(TWO_TOOL_STREAM_PREAMBLE)
                .tool(OutOfOrderAlphaSignal(order.clone()))
                .tool(OutOfOrderBetaSignal(order)).record_to(recorder.clone()).build();
            let mut stream = agent
                .prompt(TWO_TOOL_STREAM_PROMPT)
                .max_turns(8)
                .tool_concurrency(2)
                .stream();
            let observation = tokio::time::timeout(
                std::time::Duration::from_secs(5),
                collect_concurrent_tool_observation(&mut stream),
            )
            .await
            .expect("serial serving is per key; two tools never wait on each other");
            assert!(observation.errors.is_empty(), "{:?}", observation.errors);
            assert!(observation.got_final_response);
            let log = agent.stamp(recorder.take());
            assert_eq!(
                log.header.bus.map(|bus| bus.serial_per_handler),
                Some(true),
                "the header says how it was served"
            );
            crate::goldens::golden_effects("anthropic_concurrent_tools_serial", &log);
        },
    )
    .await;
}
