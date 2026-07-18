//! Anthropic streaming tools smoke test.

use futures::StreamExt;
use rig::agent::{MultiTurnStreamItem, StreamingError, StreamingResult};
use rig::client::CompletionClient;
use rig::message::{Message, UserContent};
use rig::providers::anthropic;
use rig::streaming::{StreamedAssistantContent, StreamedUserContent, StreamingPrompt};
use rig::tool::Tool;
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering::SeqCst};

use super::super::support::with_anthropic_cassette;
use crate::support::{
    ALPHA_SIGNAL_OUTPUT, Adder, AlphaSignal, BETA_SIGNAL_OUTPUT, BetaSignal, EmptyArgs, MathError,
    STREAMING_TOOLS_PREAMBLE, STREAMING_TOOLS_PROMPT, Subtract, TWO_TOOL_STREAM_PREAMBLE,
    TWO_TOOL_STREAM_PROMPT, assert_contains_all_case_insensitive, assert_mentions_expected_number,
    collect_stream_final_response, collect_stream_observation,
};

#[tokio::test]
async fn streaming_tools_smoke() {
    with_anthropic_cassette(
        "streaming_tools/streaming_tools_smoke",
        |client| async move {
            let agent = client
                .agent(anthropic::completion::CLAUDE_SONNET_4_6)
                .preamble(STREAMING_TOOLS_PREAMBLE)
                .tool(Adder)
                .tool(Subtract)
                .default_max_turns(2)
                .build();

            let mut stream = agent.stream_prompt(STREAMING_TOOLS_PROMPT).await;
            let response = collect_stream_final_response(&mut stream)
                .await
                .expect("streaming tool prompt should succeed");

            assert_mentions_expected_number(&response, -3);
        },
    )
    .await;
}

#[cfg(feature = "bevy")]
#[tokio::test]
async fn bevy_streaming_tool_roundtrip_uses_portable_tools() {
    use rig::bevy::{AgentSpec, BevyRuntime, topology::TenantId};

    with_anthropic_cassette(
        "streaming_tools/streaming_tools_smoke",
        |client| async move {
            let runtime = BevyRuntime::default();
            let add_revision = runtime.register_tool(TenantId::default(), Adder);
            let subtract_revision = runtime.register_tool(TenantId::default(), Subtract);
            let agent = runtime.spawn_agent(
                AgentSpec::new(client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6))
                    .preamble(STREAMING_TOOLS_PREAMBLE)
                    .max_calls(2)
                    .grant_tool("add", add_revision)
                    .grant_tool("subtract", subtract_revision),
            );

            let outcome = agent
                .stream_prompt(STREAMING_TOOLS_PROMPT, |_| {})
                .await
                .expect("Bevy tool stream should succeed");

            assert_mentions_expected_number(&format!("{:?}", outcome.choice), -3);
        },
    )
    .await;
}

#[tokio::test]
async fn streaming_tools_batches_multiple_tool_results_in_one_followup_message() {
    with_anthropic_cassette(
        "streaming_tools/streaming_tools_batches_multiple_tool_results_in_one_followup_message",
        |client| async move {
            let agent = client
                .agent(anthropic::completion::CLAUDE_SONNET_4_6)
                .preamble(TWO_TOOL_STREAM_PREAMBLE)
                .tool(AlphaSignal)
                .tool(BetaSignal)
                .build();

            let mut stream = agent
                .stream_prompt(TWO_TOOL_STREAM_PROMPT)
                .max_turns(8)
                .await;
            let observation = collect_stream_observation(&mut stream).await;

            assert!(
                observation.errors.is_empty(),
                "stream should not emit errors: {:?}",
                observation.errors
            );
            assert!(
                observation.got_final_response,
                "stream should emit a final response"
            );
            assert!(
                observation.tool_results >= 2,
                "expected at least 2 tool-result events, got {}",
                observation.tool_results
            );
            for expected_tool in ["lookup_harbor_label", "lookup_orchard_label"] {
                assert!(
                    observation
                        .tool_calls
                        .iter()
                        .any(|tool_call| tool_call == expected_tool),
                    "expected tool call for {expected_tool}, saw {:?}",
                    observation.tool_calls
                );
            }
            assert_contains_all_case_insensitive(
                observation
                    .final_response_text
                    .as_deref()
                    .expect("stream should produce a final response string"),
                &[ALPHA_SIGNAL_OUTPUT, BETA_SIGNAL_OUTPUT],
            );
        },
    )
    .await;

    assert_cassette_groups_multiple_tool_results(
        "streaming_tools/streaming_tools_batches_multiple_tool_results_in_one_followup_message",
        &["lookup_harbor_label", "lookup_orchard_label"],
    );
}

#[tokio::test]
async fn streaming_tool_concurrency_surfaces_results_in_call_order_after_batch_settles() {
    // The cassette file name predates the atomic-batch semantics; the recorded
    // provider interaction is unchanged, only the assertions below.
    with_anthropic_cassette(
        "streaming_tools/streaming_tool_concurrency_emits_results_as_completed_but_persists_call_order",
        |client| async move {
        let order = OutOfOrderSignalOrder::default();
        let agent = client
            .agent(anthropic::completion::CLAUDE_SONNET_4_6)
            .preamble(TWO_TOOL_STREAM_PREAMBLE)
            .tool(OutOfOrderAlphaSignal(order.clone()))
            .tool(OutOfOrderBetaSignal(order))
            .build();

        let mut stream = agent
            .stream_prompt(TWO_TOOL_STREAM_PROMPT)
            .max_turns(8)
            .tool_concurrency(2)
            .await;
        let observation = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            collect_concurrent_tool_observation(&mut stream),
        )
        .await
        .expect("streamed tools must run concurrently, not deadlock on the first tool");

        assert!(
            observation.errors.is_empty(),
            "stream should not emit errors: {:?}",
            observation.errors
        );
        assert!(
            observation.got_final_response,
            "stream should emit a final response"
        );
        assert_eq!(
            observation.tool_calls,
            ["lookup_harbor_label", "lookup_orchard_label"],
            "provider should emit the two tool calls in call order"
        );
        assert_eq!(
            observation.streamed_tool_results,
            ["lookup_harbor_label", "lookup_orchard_label"],
            "tool-result stream items surface in call order, atomically after the \
             whole batch settles (not local completion order)"
        );
        assert_eq!(
            observation.history_tool_results,
            ["lookup_harbor_label", "lookup_orchard_label"],
            "FinalResponse history should persist tool results in provider call order"
        );
        assert_eq!(
            observation.last_history_tool_result_message,
            ["lookup_harbor_label", "lookup_orchard_label"],
            "the history message used for the follow-up request should keep provider call order"
        );
        assert_events_emit_all_tool_calls_before_results(&observation.events);
        assert_contains_all_case_insensitive(
            observation
                .final_response_text
                .as_deref()
                .expect("stream should produce a final response string"),
            &[ALPHA_SIGNAL_OUTPUT, BETA_SIGNAL_OUTPUT],
        );
        },
    )
    .await;

    assert_cassette_tool_results_follow_assistant_tool_use_order(
        "streaming_tools/streaming_tool_concurrency_emits_results_as_completed_but_persists_call_order",
        &["lookup_harbor_label", "lookup_orchard_label"],
    );
}

#[derive(Clone, Default)]
struct OutOfOrderSignalOrder {
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
struct OutOfOrderAlphaSignal(OutOfOrderSignalOrder);

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

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.0.wait_until_this_tool_should_finish().await;
        Ok(ALPHA_SIGNAL_OUTPUT.to_string())
    }
}

#[derive(Clone)]
struct OutOfOrderBetaSignal(OutOfOrderSignalOrder);

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

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
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

async fn collect_concurrent_tool_observation<R>(
    stream: &mut StreamingResult<R>,
) -> ConcurrentToolObservation {
    let mut observation = ConcurrentToolObservation::default();
    let mut tool_names_by_id = HashMap::new();

    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::ToolCall {
                tool_call,
                ..
            })) => {
                tool_names_by_id.insert(tool_call.id.clone(), tool_call.function.name.clone());
                observation.tool_calls.push(tool_call.function.name);
                observation.events.push("tool_call");
            }
            Ok(MultiTurnStreamItem::ToolExecutionCommitted { .. }) => {
                observation.events.push("tool_execution_committed");
            }
            Ok(MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult {
                tool_result,
                ..
            })) => {
                observation
                    .streamed_tool_results
                    .push(tool_name_for_result(&tool_names_by_id, &tool_result.id));
                observation.events.push("tool_result");
            }
            Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                observation.final_response_text = Some(response.output().to_owned());
                observation.got_final_response = true;
                if let Some(history) = response.messages() {
                    observation.history_tool_results =
                        tool_result_names_in_history(history, &tool_names_by_id);
                    observation.last_history_tool_result_message =
                        last_tool_result_message_names(history, &tool_names_by_id);
                }
                observation.events.push("final_response");
            }
            Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(_))) => {
                observation.events.push("text");
            }
            Ok(MultiTurnStreamItem::StreamAssistantItem(
                StreamedAssistantContent::ToolCallDelta { .. },
            )) => {
                observation.events.push("tool_call_delta");
            }
            Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Reasoning(
                _,
            ))) => {
                observation.events.push("reasoning");
            }
            Ok(MultiTurnStreamItem::StreamAssistantItem(
                StreamedAssistantContent::ReasoningDelta { .. },
            )) => {
                observation.events.push("reasoning_delta");
            }
            Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Final(_))) => {
                observation.events.push("stream_final");
            }
            Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Unknown(_))) => {
                observation.events.push("unknown");
            }
            Ok(MultiTurnStreamItem::CompletionCall(_)) => {}
            Ok(_) => {}
            Err(error) => {
                observation.errors.push(streaming_error_to_string(error));
                observation.events.push("error");
            }
        }
    }

    observation
}

fn streaming_error_to_string(error: StreamingError) -> String {
    error.to_string()
}

fn tool_result_names_in_history(
    history: &[Message],
    tool_names_by_id: &HashMap<String, String>,
) -> Vec<String> {
    history
        .iter()
        .flat_map(|message| match message {
            Message::User { content } => content
                .iter()
                .filter_map(|item| match item {
                    UserContent::ToolResult(tool_result) => {
                        Some(tool_name_for_result(tool_names_by_id, &tool_result.id))
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
    tool_names_by_id: &HashMap<String, String>,
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
                                Some(tool_name_for_result(tool_names_by_id, &tool_result.id))
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

fn tool_name_for_result(tool_names_by_id: &HashMap<String, String>, id: &str) -> String {
    tool_names_by_id
        .get(id)
        .cloned()
        .unwrap_or_else(|| format!("<unknown tool result id {id}>"))
}

fn assert_events_emit_all_tool_calls_before_results(events: &[&'static str]) {
    let first_result = events
        .iter()
        .position(|event| *event == "tool_result")
        .expect("stream should emit at least one tool result");
    let tool_calls_before_first_result = events[..first_result]
        .iter()
        .filter(|event| **event == "tool_call")
        .count();
    assert_eq!(
        tool_calls_before_first_result, 2,
        "concurrent streaming should emit every tool call before the first tool result; events: {events:?}"
    );
}

#[derive(Deserialize)]
struct RecordedInteraction {
    when: RecordedRequest,
}

#[derive(Deserialize)]
struct RecordedRequest {
    body: Option<String>,
}

pub(super) fn assert_cassette_groups_multiple_tool_results(
    scenario: &str,
    expected_tools: &[&str],
) {
    let cassette_path = crate::cassettes::cassette_path("anthropic", scenario);
    let contents = std::fs::read_to_string(&cassette_path).unwrap_or_else(|error| {
        panic!(
            "provider cassette {} should be readable after recording: {error}",
            cassette_path.display()
        )
    });

    let request_bodies = serde_yaml::Deserializer::from_str(&contents)
        .map(|document| {
            RecordedInteraction::deserialize(document)
                .expect("cassette interaction should deserialize")
                .when
                .body
        })
        .filter_map(|body| body.and_then(|body| serde_json::from_str::<Value>(&body).ok()))
        .collect::<Vec<_>>();

    let grouped = request_bodies.iter().any(|body| {
        body.get("messages")
            .and_then(Value::as_array)
            .is_some_and(|messages| messages_group_tool_results(messages, expected_tools))
    });

    assert!(
        grouped,
        "expected cassette {} to contain an Anthropic request where the two tool_result blocks \
         from one assistant turn are grouped into one user message",
        cassette_path.display()
    );
}

fn messages_group_tool_results(messages: &[Value], expected_tools: &[&str]) -> bool {
    messages.windows(2).any(|window| {
        let assistant = &window[0];
        let user = &window[1];

        role(assistant) == Some("assistant")
            && expected_tools
                .iter()
                .all(|tool| assistant_tool_use_names(assistant).any(|name| name == *tool))
            && role(user) == Some("user")
            && content_type_count(user, "tool_result") >= expected_tools.len()
    })
}

pub(super) fn assert_cassette_tool_results_follow_assistant_tool_use_order(
    scenario: &str,
    expected_tools: &[&str],
) {
    let cassette_path = crate::cassettes::cassette_path("anthropic", scenario);
    let contents = std::fs::read_to_string(&cassette_path).unwrap_or_else(|error| {
        panic!(
            "provider cassette {} should be readable after recording: {error}",
            cassette_path.display()
        )
    });

    let request_bodies = serde_yaml::Deserializer::from_str(&contents)
        .map(|document| {
            RecordedInteraction::deserialize(document)
                .expect("cassette interaction should deserialize")
                .when
                .body
        })
        .filter_map(|body| body.and_then(|body| serde_json::from_str::<Value>(&body).ok()))
        .collect::<Vec<_>>();

    let ordered = request_bodies.iter().any(|body| {
        body.get("messages")
            .and_then(Value::as_array)
            .is_some_and(|messages| {
                messages_tool_results_follow_assistant_order(messages, expected_tools)
            })
    });

    assert!(
        ordered,
        "expected cassette {} to contain an Anthropic continuation request where tool_result \
         blocks are sent in the same order as the preceding assistant tool_use blocks",
        cassette_path.display()
    );
}

fn messages_tool_results_follow_assistant_order(
    messages: &[Value],
    expected_tools: &[&str],
) -> bool {
    messages.windows(2).any(|window| {
        let assistant = &window[0];
        let user = &window[1];
        if role(assistant) != Some("assistant") || role(user) != Some("user") {
            return false;
        }

        let assistant_ids = assistant_tool_use_id_names(assistant);
        let assistant_order = assistant_ids
            .iter()
            .map(|(_, name)| name.as_str())
            .collect::<Vec<_>>();
        if assistant_order != expected_tools {
            return false;
        }

        let names_by_id = assistant_ids.into_iter().collect::<HashMap<_, _>>();
        let user_order = user_tool_result_ids(user)
            .filter_map(|id| names_by_id.get(id).map(String::as_str))
            .collect::<Vec<_>>();
        user_order == expected_tools
    })
}

fn role(message: &Value) -> Option<&str> {
    message.get("role").and_then(Value::as_str)
}

fn assistant_tool_use_names(message: &Value) -> impl Iterator<Item = &str> {
    content_items(message).filter_map(|content| {
        (content.get("type").and_then(Value::as_str) == Some("tool_use"))
            .then(|| content.get("name").and_then(Value::as_str))
            .flatten()
    })
}

fn assistant_tool_use_id_names(message: &Value) -> Vec<(String, String)> {
    content_items(message)
        .filter(|content| content.get("type").and_then(Value::as_str) == Some("tool_use"))
        .filter_map(|content| {
            let id = content.get("id").and_then(Value::as_str)?;
            let name = content.get("name").and_then(Value::as_str)?;
            Some((id.to_string(), name.to_string()))
        })
        .collect()
}

fn user_tool_result_ids(message: &Value) -> impl Iterator<Item = &str> {
    content_items(message).filter_map(|content| {
        (content.get("type").and_then(Value::as_str) == Some("tool_result"))
            .then(|| content.get("tool_use_id").and_then(Value::as_str))
            .flatten()
    })
}

fn content_type_count(message: &Value, expected_type: &str) -> usize {
    content_items(message)
        .filter(|content| content.get("type").and_then(Value::as_str) == Some(expected_type))
        .count()
}

fn content_items(message: &Value) -> impl Iterator<Item = &Value> {
    message
        .get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
}
