//! Anthropic Messages API long-session regression tests.
//!
//! These tests lock down multi-turn, multi-tool agent sessions against the
//! Messages API: sequential tool roundtrips, parallel tool_use blocks in a
//! single assistant turn with batched tool_result grouping, long chat-history
//! replay (including assistant text before and after tool use), and usage
//! accounting across turns.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use futures::StreamExt;
use rig::agent::MultiTurnStreamItem;
use rig::agent::tool::Tool;
use rig::client::CompletionClient;
use rig::completion::{Chat, CompletionModel, Message};
use rig::message::{AssistantContent, UserContent};
use rig::providers::anthropic;
use rig::streaming::{StreamingChat, StreamingPrompt};

use super::super::support::with_anthropic_cassette;
use super::streaming_tools::assert_cassette_groups_multiple_tool_results;
use crate::support::{
    ALPHA_SIGNAL_OUTPUT, Adder, AlphaSignal, BETA_SIGNAL_OUTPUT, BetaSignal,
    ORDERED_TOOL_STREAM_PREAMBLE, ORDERED_TOOL_STREAM_PROMPT, Subtract, TWO_TOOL_STREAM_PREAMBLE,
    TWO_TOOL_STREAM_PROMPT, assert_mentions_expected_number, collect_stream_observation,
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
    with_anthropic_cassette(
        "messages_sessions/sequential_tool_calls_nonstreaming",
        |client| async move {
            let agent = client
                .agent(anthropic::completion::CLAUDE_SONNET_4_6)
                .preamble(SEQUENTIAL_TOOLS_PREAMBLE)
                .max_tokens(2048)
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
        },
    )
    .await;
}

#[tokio::test]
async fn sequential_tool_calls_streaming() {
    with_anthropic_cassette(
        "messages_sessions/sequential_tool_calls_streaming",
        |client| async move {
            let agent = client
                .agent(anthropic::completion::CLAUDE_SONNET_4_6)
                .preamble(SEQUENTIAL_TOOLS_PREAMBLE)
                .max_tokens(2048)
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
async fn parallel_tool_use_single_turn_nonstreaming() {
    with_anthropic_cassette(
        "messages_sessions/parallel_tool_use_single_turn_nonstreaming",
        |client| async move {
            let agent = client
                .agent(anthropic::completion::CLAUDE_SONNET_4_6)
                .preamble(TWO_TOOL_STREAM_PREAMBLE)
                .max_tokens(2048)
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

    // Wire-format contract: the non-streaming agent loop must also batch both
    // tool_result blocks from one assistant turn into a single user message.
    assert_cassette_groups_multiple_tool_results(
        "messages_sessions/parallel_tool_use_single_turn_nonstreaming",
        &[AlphaSignal::NAME, BetaSignal::NAME],
    );
}

#[tokio::test]
async fn long_history_replay_nonstreaming() {
    with_anthropic_cassette(
        "messages_sessions/long_history_replay_nonstreaming",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6);
            let preamble = "You are a concise assistant with perfect recall of this conversation.";

            // First turn: obtain a real tool_use so the follow-up can echo its
            // id back, the way a caller-owned history would.
            let first_request = model
                .completion_request("Look up the harbor label with the tool.")
                .preamble(preamble.to_string())
                .max_tokens(1024)
                .tool(rig::agent::tool::tool_definition(&AlphaSignal))
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
            assert_eq!(
                first_response.raw_response.stop_reason.as_deref(),
                Some("tool_use"),
                "a tool-using turn should preserve the tool_use stop reason"
            );

            // Follow-up: replay a long client-owned history around that tool
            // roundtrip, including assistant text before the tool_use (in the
            // same assistant message) and assistant text after the result.
            let request = model
                .completion_request(
                    "In one short sentence: what is my favorite color, and what was the \
                     harbor label you looked up earlier?",
                )
                .preamble(preamble.to_string())
                .max_tokens(1024)
                .message(Message::user(
                    "My favorite color is teal. Please remember it.",
                ))
                .message(Message::assistant("Noted - your favorite color is teal."))
                .message(Message::user("Now look up the harbor label with the tool."))
                .message(Message::Assistant {
                    id: None,
                    content: rig::OneOrMany::many(vec![
                        AssistantContent::text("Checking the harbor label now."),
                        AssistantContent::ToolCall(tool_call.clone()),
                    ])
                    .expect("assistant content should be non-empty"),
                })
                .message(Message::tool_result_with_call_id(
                    tool_call.id.clone(),
                    tool_call.call_id.clone(),
                    ALPHA_SIGNAL_OUTPUT,
                ))
                .message(Message::assistant("The harbor label is crimson-harbor."))
                .tool(rig::agent::tool::tool_definition(&AlphaSignal))
                .build();

            let response = model
                .completion(request)
                .await
                .expect("long history replay should be accepted by the Messages API");

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
            assert_eq!(
                response.raw_response.stop_reason.as_deref(),
                Some("end_turn"),
                "a plain answer should preserve the end_turn stop reason"
            );
            assert!(
                !response.raw_response.model.is_empty() && !response.raw_response.id.is_empty(),
                "provider response should preserve model and message id"
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
async fn usage_accumulates_across_streaming_multi_turn() {
    with_anthropic_cassette(
        "messages_sessions/usage_accumulates_across_streaming_multi_turn",
        |client| async move {
            let agent = client
                .agent(anthropic::completion::CLAUDE_SONNET_4_6)
                .preamble(ORDERED_TOOL_STREAM_PREAMBLE)
                .max_tokens(2048)
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
