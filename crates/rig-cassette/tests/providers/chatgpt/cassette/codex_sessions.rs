//! ChatGPT/Codex Responses backend long-session regression tests.
//!
//! These tests lock down multi-turn, multi-tool agent sessions against the
//! Responses API: sequential tool roundtrips, parallel tool calls in a single
//! model turn, long chat-history replay, reasoning-enabled tool sessions, and
//! usage accounting across turns.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use rig::completion::CompletionRequest;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures::StreamExt;
use rig::agent::MultiTurnStreamItem;
use rig::completion::Message;
use rig::message::{AssistantContent, UserContent};
use rig::providers::chatgpt;
use rig::tool::Tool;

use super::super::support::with_chatgpt_cassette;
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
    name: String,
    call_id: String,
}

fn history_tool_calls(history: &[Message]) -> Vec<ToolEvent> {
    let mut calls = Vec::new();
    for (message_index, message) in history.iter().enumerate() {
        if let Message::Assistant { content, .. } = message {
            for item in content.iter() {
                if let AssistantContent::ToolCall(tool_call) = item {
                    calls.push(ToolEvent {
                        message_index,
                        name: tool_call.function.name.clone().into(),
                        call_id: tool_call.id.to_string(),
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
                        name: tool_result.name.clone().into(),
                        call_id: tool_result.call.to_string(),
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
                call.name, call.call_id
            )
        })
        .message_index
}

#[tokio::test]
async fn sequential_tool_calls_nonstreaming() {
    with_chatgpt_cassette(
        "codex_sessions/sequential_tool_calls_nonstreaming",
        |client| async move {
            let agent = rig::AgentBuilder::new(client.completion(chatgpt::GPT_5_4))
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

            assert_mentions_expected_number(&result.output, 2);

            let calls = history_tool_calls(&history);
            let results = history_tool_results(&history);
            let add_call = calls
                .iter()
                .find(|call| call.name == Adder::NAME)
                .expect("history should contain an add tool call");
            let subtract_call = calls
                .iter()
                .find(|call| call.name == Subtract::NAME)
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
    with_chatgpt_cassette(
        "codex_sessions/sequential_tool_calls_streaming",
        |client| async move {
            let agent = rig::AgentBuilder::new(client.completion(chatgpt::GPT_5_4))
                .preamble(SEQUENTIAL_TOOLS_PREAMBLE)
                .tool(Adder)
                .tool(Subtract)
                .build();

            let mut stream = agent
                .prompt(SEQUENTIAL_TOOLS_PROMPT)
                .history(Vec::<Message>::new())
                .max_turns(6)
                .stream();
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
    with_chatgpt_cassette(
        "codex_sessions/parallel_tool_calls_single_turn_nonstreaming",
        |client| async move {
            let agent = rig::AgentBuilder::new(client.completion(chatgpt::GPT_5_4))
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

            let lowered = result.output.to_ascii_lowercase();
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
                    .map(|call| call.name.as_str())
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
    with_chatgpt_cassette(
        "codex_sessions/parallel_tool_calls_single_turn_streaming",
        |client| async move {
            let agent = rig::AgentBuilder::new(client.completion(chatgpt::GPT_5_4))
                .preamble(TWO_TOOL_STREAM_PREAMBLE)
                .tool(AlphaSignal)
                .tool(BetaSignal)
                .build();

            let mut stream = agent.prompt(TWO_TOOL_STREAM_PROMPT).max_turns(5).stream();
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
    with_chatgpt_cassette(
        "codex_sessions/long_history_replay_nonstreaming",
        |client| async move {
            let model = client.completion(chatgpt::GPT_5_4);
            let preamble = "You are a concise assistant with perfect recall of this conversation.";

            // First turn: obtain a real tool call so the follow-up can echo
            // its call_id back, the way a caller-owned history would.
            let first_request = CompletionRequest::new("Look up the harbor label with the tool.")
                .preamble(preamble.to_string())
                .tool(rig::tool::tool_definition(&AlphaSignal));
            let first_response = model
                .call(first_request)
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
            let call_id = tool_call.id.provider().as_ref().map_or_else(
                || tool_call.id.to_string(),
                |provider| provider.call_id.clone(),
            );

            // Follow-up: replay a long client-owned history around that tool
            // roundtrip. The tool call is re-tagged with a local item ID (not
            // the provider's `fc_...` ID) — the request must still be accepted
            // because non-native IDs are omitted and calls pair by call_id.
            let request = CompletionRequest::new(
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
                content: vec![AssistantContent::tool_call_with_call_id(
                    "history_tool_1",
                    call_id.clone(),
                    rig_core::message::ToolName::new(AlphaSignal::NAME).expect("tool name"),
                    serde_json::json!({}),
                )],
            })
            .message(Message::from(UserContent::tool_result(
                rig_core::message::CallId::from_dual_wire("history_tool_1", call_id),
                rig_core::message::ToolName::new(AlphaSignal::NAME).expect("tool name"),
                vec![rig::message::ToolResultContent::text(ALPHA_SIGNAL_OUTPUT)],
            )))
            .message(Message::assistant("The harbor label is crimson-harbor."))
            .tool(rig::tool::tool_definition(&AlphaSignal));

            let response = model
                .call(request)
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
                response.usage.input_tokens.is_some_and(|n| n > 0)
                    && response.usage.output_tokens.is_some_and(|n| n > 0),
                "usage should be populated, got {:?}",
                response.usage
            );
        },
    )
    .await;
}

#[tokio::test]
async fn reasoning_session_two_tool_calls_streaming() {
    with_chatgpt_cassette(
        "codex_sessions/reasoning_session_two_tool_calls_streaming",
        |client| async move {
            let call_count = Arc::new(AtomicUsize::new(0));
            let agent = rig::AgentBuilder::new(client.completion(chatgpt::GPT_5_4))
                .preamble(reasoning::TOOL_SYSTEM_PROMPT)
                .max_tokens(6000)
                .tool(WeatherTool::new(call_count.clone()))
                .additional_params(serde_json::json!({
                    "reasoning": { "effort": "low" }
                }))
                .build();

            let stream = agent
                .prompt(
                    "I need the current weather in Tokyo and in Paris. Use the get_weather \
                     tool once per city, then compare the two cities in one short paragraph \
                     that mentions both city names.",
                )
                .history(Vec::<Message>::new())
                .max_turns(5)
                .stream();

            let stats = reasoning::collect_stream_stats(stream, "chatgpt").await;

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
    with_chatgpt_cassette(
        "codex_sessions/usage_accumulates_across_streaming_multi_turn",
        |client| async move {
            let agent = rig::AgentBuilder::new(client.completion(chatgpt::GPT_5_4))
                .preamble(ORDERED_TOOL_STREAM_PREAMBLE)
                .tool(AlphaSignal)
                .build();

            let mut stream = agent
                .prompt(ORDERED_TOOL_STREAM_PROMPT)
                .max_turns(5)
                .stream();

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
                usage.input_tokens.is_some_and(|n| n > 0),
                "aggregated input tokens should be nonzero: {usage:?}"
            );
            assert!(
                usage.output_tokens.is_some_and(|n| n > 0),
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

/// A streamed ChatGPT turn's `phase` reaches its text block, and the
/// follow-up re-sends it on the same assistant item, which the backend
/// accepts. Expectations come from the recorded stream: the message's id
/// and `phase` as `output_item.done` stated them.
#[tokio::test]
async fn streamed_phase_round_trips_on_follow_up() {
    const PROMPT: &str = "Remember the codeword ALPHA-17. Reply exactly: ACK-1";
    let mut first = None;
    with_chatgpt_cassette(
        "codex_sessions/streamed_phase_round_trips_on_follow_up",
        |client| {
            let first = &mut first;
            async move {
                // The backend serves this account only its current models.
                let model = client.responses(rig::providers::openai::GPT_6_ASTRA);
                let turn = |request: CompletionRequest| {
                    let model = model.clone();
                    async move {
                        let mut stream = model.stream(request).expect("the stream starts");
                        while let Some(item) = stream.next().await {
                            item.expect("every stream item decodes");
                        }
                        stream.finish().await.expect("the stream ends")
                    }
                };
                let reply = turn(CompletionRequest::new(PROMPT).preamble("Be concise.")).await;
                let mut history = vec![Message::user(PROMPT)];
                history.extend(reply.message());
                turn(
                    CompletionRequest::new("Reply with exactly the remembered codeword.")
                        .preamble("Be concise.")
                        .messages(history),
                )
                .await;
                *first = Some(reply);
            }
        },
    )
    .await;
    let first = first.expect("turn 1 ran");

    let recorded = crate::cassettes::recorded_interaction_bodies(
        "chatgpt",
        "codex_sessions/streamed_phase_round_trips_on_follow_up",
    );
    assert_eq!(recorded.len(), 2, "two turns recorded");
    let delivered: Vec<(String, String)> = crate::cassettes::recorded_sse_json_frames(
        "chatgpt",
        "codex_sessions/streamed_phase_round_trips_on_follow_up",
    )
    .into_iter()
    .filter(|event| {
        event["type"] == "response.output_item.done" && event["item"]["type"] == "message"
    })
    .map(|event| {
        (
            event["item"]["id"]
                .as_str()
                .filter(|id| !id.is_empty())
                .expect("the recorded message has an id")
                .to_owned(),
            event["item"]["phase"]
                .as_str()
                .expect("the recorded message states a phase")
                .to_owned(),
        )
    })
    .collect();
    assert_eq!(delivered.len(), 1, "one message item: {delivered:?}");
    let (id, phase) = &delivered[0];

    let texts: Vec<Option<&str>> = first
        .choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(
                text.additional_params
                    .as_ref()
                    .and_then(|params| params.wire_extras("openai_responses"))
                    .and_then(|extras| extras.get("phase"))
                    .and_then(serde_json::Value::as_str),
            ),
            _ => None,
        })
        .collect();
    assert!(
        !texts.is_empty() && texts.iter().all(|text| *text == Some(phase.as_str())),
        "the streamed text carries the message's phase: {texts:?}"
    );

    let request: serde_json::Value = serde_json::from_str(&recorded[1].0).expect("request is JSON");
    let phased: Vec<(&str, &str)> = request["input"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|item| item.get("phase").is_some())
        .map(|item| {
            (
                item["id"].as_str().unwrap_or_default(),
                item["phase"].as_str().unwrap_or_default(),
            )
        })
        .collect();
    assert_eq!(
        phased,
        [(id.as_str(), phase.as_str())],
        "the follow-up re-sends the phase on the same assistant item, and on no other"
    );
    assert!(
        recorded[1].1.contains("\"type\":\"response.completed\""),
        "the backend accepted the follow-up"
    );
}
