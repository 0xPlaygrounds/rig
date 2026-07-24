//! OpenAI Responses API long-session regression tests.
//!
//! These tests lock down multi-turn, multi-tool agent sessions against the
//! Responses API: sequential tool roundtrips, parallel tool calls in a single
//! model turn, long chat-history replay, reasoning-enabled tool sessions, and
//! usage accounting across turns.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures::StreamExt;
use rig::agent::MultiTurnStreamItem;
use rig::completion::{Chat, CompletionModel, Message};
use rig::message::{AssistantContent, UserContent};
use rig::prelude::*;
use rig::providers::openai;
use rig::streaming::{StreamingChat, StreamingPrompt};
use rig::tool::Tool;

use super::super::support::with_openai_cassette;
use crate::reasoning::{self, WeatherTool};
use crate::support::{
    ALPHA_SIGNAL_OUTPUT, Adder, AlphaSignal, BETA_SIGNAL_OUTPUT, BetaSignal,
    ORDERED_TOOL_STREAM_PREAMBLE, ORDERED_TOOL_STREAM_PROMPT, Subtract, TWO_TOOL_STREAM_PREAMBLE,
    TWO_TOOL_STREAM_PROMPT, assert_mentions_expected_number, assert_two_tool_roundtrip_contract,
    collect_stream_observation,
};

const SEQUENTIAL_TOOLS_PREAMBLE: &str = "\
You are a calculator. Use the provided tools instead of doing arithmetic yourself. \
Call exactly one tool at a time and wait for its result before deciding the next step.";

const SEQUENTIAL_TOOLS_PROMPT: &str = "\
First use the add tool to compute 3 + 4. After you receive that result, use the \
subtract tool to subtract 5 from it. Then state the final number in one short sentence.";

/// A recorded tool event from a caller-owned chat history: the message index
/// it appeared at plus the identifiers needed to pair calls with results.
struct ToolEvent {
    message_index: usize,
    name_or_id: String,
    call_id: Option<String>,
}

fn history_tool_calls(history: &[Message]) -> Vec<ToolEvent> {
    let mut calls = Vec::new();
    for (message_index, message) in history.iter().enumerate() {
        if let Message::Assistant { content, .. } = message {
            for item in content.iter() {
                if let AssistantContent::ToolCall(tool_call) = item {
                    calls.push(ToolEvent {
                        message_index,
                        name_or_id: tool_call.function.name.clone(),
                        call_id: tool_call.call_id.clone().or(Some(tool_call.id.clone())),
                    });
                }
            }
        }
    }
    calls
}

fn history_tool_results(history: &[Message]) -> Vec<ToolEvent> {
    let mut results = Vec::new();
    for (message_index, message) in history.iter().enumerate() {
        if let Message::User { content } = message {
            for item in content.iter() {
                if let UserContent::ToolResult(tool_result) = item {
                    results.push(ToolEvent {
                        message_index,
                        name_or_id: tool_result.id.clone(),
                        call_id: tool_result.call_id.clone().or(Some(tool_result.id.clone())),
                    });
                }
            }
        }
    }
    results
}

fn result_index_for_call(results: &[ToolEvent], call: &ToolEvent) -> usize {
    results
        .iter()
        .find(|result| result.call_id == call.call_id)
        .unwrap_or_else(|| {
            panic!(
                "chat history is missing the tool result answering call {:?} (call_id {:?})",
                call.name_or_id, call.call_id
            )
        })
        .message_index
}

#[tokio::test]
async fn sequential_tool_calls_nonstreaming() {
    with_openai_cassette(
        "responses_sessions/sequential_tool_calls_nonstreaming",
        |client| async move {
            let agent = client
                .agent(openai::GPT_4O)
                .preamble(SEQUENTIAL_TOOLS_PREAMBLE)
                .tool(Adder)
                .tool(Subtract)
                .default_max_turns(6)
                .build();
            let mut history = Vec::<Message>::new();

            let result = agent
                .chat(SEQUENTIAL_TOOLS_PROMPT, &mut history)
                .await
                .expect("sequential tool chat should succeed");

            assert_mentions_expected_number(&result, 2);

            let calls = history_tool_calls(&history);
            let results = history_tool_results(&history);
            let add_call = calls
                .iter()
                .find(|call| call.name_or_id == Adder::NAME)
                .expect("history should contain an add tool call");
            let subtract_call = calls
                .iter()
                .find(|call| call.name_or_id == Subtract::NAME)
                .expect("history should contain a subtract tool call");
            let add_result_index = result_index_for_call(&results, add_call);
            let subtract_result_index = result_index_for_call(&results, subtract_call);

            assert!(
                add_call.message_index < add_result_index,
                "add result should follow the add call"
            );
            assert!(
                add_result_index < subtract_call.message_index,
                "subtract call should only happen after the add result (sequential turns)"
            );
            assert!(
                subtract_call.message_index < subtract_result_index,
                "subtract result should follow the subtract call"
            );

            let final_assistant_text = history
                .iter()
                .skip(subtract_result_index + 1)
                .filter_map(|message| match message {
                    Message::Assistant { content, .. } => Some(
                        content
                            .iter()
                            .filter_map(|item| match item {
                                AssistantContent::Text(text) => Some(text.text.clone()),
                                _ => None,
                            })
                            .collect::<String>(),
                    ),
                    _ => None,
                })
                .collect::<String>();
            assert!(
                !final_assistant_text.trim().is_empty(),
                "history should record a final assistant answer after the last tool result"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn sequential_tool_calls_streaming() {
    with_openai_cassette(
        "responses_sessions/sequential_tool_calls_streaming",
        |client| async move {
            let agent = client
                .agent(openai::GPT_4O)
                .preamble(SEQUENTIAL_TOOLS_PREAMBLE)
                .tool(Adder)
                .tool(Subtract)
                .build();

            let mut stream = agent
                .stream_chat(SEQUENTIAL_TOOLS_PROMPT, Vec::<Message>::new())
                .max_turns(6)
                .await;
            let observation = collect_stream_observation(&mut stream).await;

            assert!(
                observation.errors.is_empty(),
                "stream should not emit errors: {:?}",
                observation.errors
            );
            assert_eq!(
                observation.tool_calls,
                vec![Adder::NAME.to_string(), Subtract::NAME.to_string()],
                "expected exactly one add call followed by one subtract call"
            );
            assert_eq!(
                observation.tool_results, 2,
                "expected one tool result per tool call"
            );
            assert!(
                observation.got_final_response,
                "stream should emit a final response"
            );
            let response = observation
                .final_response_text
                .as_deref()
                .expect("stream should produce final response text");
            assert_mentions_expected_number(response, 2);
        },
    )
    .await;
}

#[tokio::test]
async fn parallel_tool_calls_single_turn_nonstreaming() {
    with_openai_cassette(
        "responses_sessions/parallel_tool_calls_single_turn_nonstreaming",
        |client| async move {
            let agent = client
                .agent(openai::GPT_4O)
                .preamble(TWO_TOOL_STREAM_PREAMBLE)
                .tool(AlphaSignal)
                .tool(BetaSignal)
                .default_max_turns(5)
                .build();
            let mut history = Vec::<Message>::new();

            let result = agent
                .chat(TWO_TOOL_STREAM_PROMPT, &mut history)
                .await
                .expect("parallel tool chat should succeed");

            let lowered = result.to_ascii_lowercase();
            assert!(
                lowered.contains(ALPHA_SIGNAL_OUTPUT) && lowered.contains(BETA_SIGNAL_OUTPUT),
                "final response should include both tool outputs, got {result:?}"
            );

            let calls = history_tool_calls(&history);
            let results = history_tool_results(&history);
            assert_eq!(
                calls.len(),
                2,
                "expected exactly two tool calls in history, got {:?}",
                calls
                    .iter()
                    .map(|call| call.name_or_id.as_str())
                    .collect::<Vec<_>>()
            );
            assert_eq!(results.len(), 2, "expected exactly two tool results");

            for call in &calls {
                let result_index = result_index_for_call(&results, call);
                assert!(
                    call.message_index < result_index,
                    "each tool result should follow its call"
                );
            }

            // Results must come back in the same order as the calls they answer.
            let call_order: Vec<_> = calls.iter().map(|call| call.call_id.clone()).collect();
            let result_order: Vec<_> = results
                .iter()
                .map(|result| result.call_id.clone())
                .collect();
            assert_eq!(
                call_order, result_order,
                "tool results should be recorded in tool-call order"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn parallel_tool_calls_single_turn_streaming() {
    with_openai_cassette(
        "responses_sessions/parallel_tool_calls_single_turn_streaming",
        |client| async move {
            let agent = client
                .agent(openai::GPT_4O)
                .preamble(TWO_TOOL_STREAM_PREAMBLE)
                .tool(AlphaSignal)
                .tool(BetaSignal)
                .build();

            let mut stream = agent
                .stream_prompt(TWO_TOOL_STREAM_PROMPT)
                .max_turns(5)
                .await;
            let observation = collect_stream_observation(&mut stream).await;

            assert_two_tool_roundtrip_contract(
                &observation,
                &[AlphaSignal::NAME, BetaSignal::NAME],
                &[ALPHA_SIGNAL_OUTPUT, BETA_SIGNAL_OUTPUT],
            );
        },
    )
    .await;
}

#[tokio::test]
async fn long_history_replay_nonstreaming() {
    with_openai_cassette(
        "responses_sessions/long_history_replay_nonstreaming",
        |client| async move {
            let model = client.completion_model(openai::GPT_4O);
            let preamble = "You are a concise assistant with perfect recall of this conversation.";

            // First turn: obtain a real tool call so the follow-up can echo
            // its call_id back, the way a caller-owned history would.
            let first_request = model
                .completion_request("Look up the harbor label with the tool.")
                .preamble(preamble.to_string())
                .tool(rig::tool::tool_definition(&AlphaSignal))
                .build();
            let first_response = model
                .completion(first_request)
                .await
                .expect("first turn should succeed");
            let tool_call = first_response
                .choice
                .iter()
                .find_map(|content| match content {
                    AssistantContent::ToolCall(tool_call) => Some(tool_call.clone()),
                    _ => None,
                })
                .expect("first turn should call lookup_harbor_label");
            let call_id = tool_call
                .call_id
                .clone()
                .unwrap_or_else(|| tool_call.id.clone());

            // Follow-up: replay a long client-owned history around that tool
            // roundtrip. The tool call is re-tagged with a local item ID (not
            // the provider's `fc_...` ID) — the request must still be accepted
            // because non-native IDs are omitted and calls pair by call_id.
            let request = model
                .completion_request(
                    "In one short sentence: what is my favorite color, and what was the \
                     harbor label you looked up earlier?",
                )
                .preamble(preamble.to_string())
                .message(Message::user(
                    "My favorite color is teal. Please remember it.",
                ))
                .message(Message::assistant("Noted - your favorite color is teal."))
                .message(Message::user("Now look up the harbor label with the tool."))
                .message(Message::Assistant {
                    id: None,
                    content: rig::OneOrMany::one(AssistantContent::tool_call_with_call_id(
                        "history_tool_1",
                        call_id.clone(),
                        AlphaSignal::NAME,
                        serde_json::json!({}),
                    )),
                })
                .message(Message::tool_result_with_call_id(
                    "history_tool_1",
                    Some(call_id),
                    ALPHA_SIGNAL_OUTPUT,
                ))
                .message(Message::assistant("The harbor label is crimson-harbor."))
                .tool(rig::tool::tool_definition(&AlphaSignal))
                .build();

            let response = model
                .completion(request)
                .await
                .expect("long history replay should be accepted by the Responses API");

            let text: String = response
                .choice
                .iter()
                .filter_map(|content| match content {
                    AssistantContent::Text(text) => Some(text.text.as_str()),
                    _ => None,
                })
                .collect();
            let lowered = text.to_ascii_lowercase();
            assert!(
                lowered.contains("teal"),
                "answer should recall the user fact from early history, got {text:?}"
            );
            assert!(
                lowered.contains(ALPHA_SIGNAL_OUTPUT),
                "answer should recall the replayed tool result, got {text:?}"
            );
            assert!(
                response.usage.input_tokens > 0 && response.usage.output_tokens > 0,
                "usage should be populated, got {:?}",
                response.usage
            );
        },
    )
    .await;
}

#[tokio::test]
async fn reasoning_session_two_tool_calls_streaming() {
    with_openai_cassette(
        "responses_sessions/reasoning_session_two_tool_calls_streaming",
        |client| async move {
            let call_count = Arc::new(AtomicUsize::new(0));
            let agent = client
                .agent(openai::GPT_5_2)
                .preamble(reasoning::TOOL_SYSTEM_PROMPT)
                .max_tokens(6000)
                .tool(WeatherTool::new(call_count.clone()))
                .additional_params(serde_json::json!({
                    "reasoning": { "effort": "low" }
                }))
                .build();

            let stream = agent
                .stream_chat(
                    "I need the current weather in Tokyo and in Paris. Use the get_weather \
                     tool once per city, then compare the two cities in one short paragraph \
                     that mentions both city names.",
                    Vec::<Message>::new(),
                )
                .max_turns(5)
                .await;

            let stats = reasoning::collect_stream_stats(stream, "openai").await;

            assert!(
                stats.errors.is_empty(),
                "stream had errors: {:?}",
                stats.errors
            );
            let invocations = call_count.load(Ordering::SeqCst);
            assert!(
                invocations >= 2,
                "expected get_weather to run once per city, got {invocations}"
            );
            assert!(
                stats
                    .tool_calls_in_stream
                    .iter()
                    .filter(|name| name.as_str() == WeatherTool::NAME)
                    .count()
                    >= 2,
                "expected at least two get_weather calls in the stream, saw {:?}",
                stats.tool_calls_in_stream
            );
            assert!(
                stats.tool_results_in_stream >= 2,
                "expected a tool result per call, got {}",
                stats.tool_results_in_stream
            );
            assert!(
                stats.reasoning_block_count >= 1,
                "expected reasoning output from a reasoning-enabled session"
            );
            assert!(
                stats.got_final_response,
                "stream should emit a final response"
            );
            let final_text = stats.final_turn_text.to_ascii_lowercase();
            assert!(
                final_text.contains("tokyo") && final_text.contains("paris"),
                "final answer should mention both cities, got {:?}",
                stats.final_turn_text
            );
        },
    )
    .await;
}

#[tokio::test]
async fn usage_accumulates_across_streaming_multi_turn() {
    with_openai_cassette(
        "responses_sessions/usage_accumulates_across_streaming_multi_turn",
        |client| async move {
            let agent = client
                .agent(openai::GPT_4O)
                .preamble(ORDERED_TOOL_STREAM_PREAMBLE)
                .tool(AlphaSignal)
                .build();

            let mut stream = agent
                .stream_prompt(ORDERED_TOOL_STREAM_PROMPT)
                .max_turns(5)
                .await;

            let mut saw_tool_result = false;
            let mut final_usage = None;

            while let Some(item) = stream.next().await {
                match item.expect("stream item should be ok") {
                    MultiTurnStreamItem::StreamUserItem(_) => saw_tool_result = true,
                    MultiTurnStreamItem::FinalResponse(response) => {
                        final_usage = Some(response.usage());
                    }
                    _ => {}
                }
            }

            assert!(
                saw_tool_result,
                "session should include a tool roundtrip so usage spans two model turns"
            );
            let usage = final_usage.expect("stream should emit a final response with usage");
            assert!(
                usage.input_tokens > 0,
                "aggregated input tokens should be nonzero: {usage:?}"
            );
            assert!(
                usage.output_tokens > 0,
                "aggregated output tokens should be nonzero: {usage:?}"
            );
            assert!(
                usage.total_tokens >= usage.output_tokens,
                "total tokens should cover output tokens: {usage:?}"
            );
        },
    )
    .await;
}
