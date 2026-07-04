//! Gemini generateContent long-session regression tests.
//!
//! These tests lock down multi-turn, multi-tool agent sessions: sequential
//! tool roundtrips with strict ordering, long client-owned history replay
//! (model text before and after functionCall parts), and thinking-enabled
//! usage accounting (thoughts token surfacing).
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use rig::client::CompletionClient;
use rig::completion::{Chat, CompletionModel, Message};
use rig::message::{AssistantContent, UserContent};
use rig::providers::gemini;
use rig::streaming::StreamingChat;
use rig::tool::Tool;

use super::super::support::with_gemini_cassette;
use crate::support::{
    ALPHA_SIGNAL_OUTPUT, Adder, AlphaSignal, Subtract, assert_mentions_expected_number,
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
async fn sequential_tool_calls_ordering_nonstreaming() {
    with_gemini_cassette(
        "generate_sessions/sequential_tool_calls_ordering_nonstreaming",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(SEQUENTIAL_TOOLS_PREAMBLE)
                .temperature(0.0)
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
async fn sequential_tool_calls_ordering_streaming() {
    with_gemini_cassette(
        "generate_sessions/sequential_tool_calls_ordering_streaming",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(SEQUENTIAL_TOOLS_PREAMBLE)
                .temperature(0.0)
                .tool(Adder)
                .tool(Subtract)
                .build();

            let mut stream = agent
                .stream_chat(SEQUENTIAL_TOOLS_PROMPT, Vec::<Message>::new())
                .multi_turn(6)
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
async fn long_history_replay_nonstreaming() {
    with_gemini_cassette(
        "generate_sessions/long_history_replay_nonstreaming",
        |client| async move {
            let model = client.completion_model(gemini::completion::GEMINI_2_5_FLASH);

            // A finished prior session replayed statelessly: Gemini pairs
            // functionResponse parts to functionCall parts by name, so a fully
            // client-constructed history (including model text before the
            // functionCall and after the functionResponse) must be accepted.
            let request = model
                .completion_request(
                    "In one short sentence: what is my favorite color, and what was the \
                     harbor label you looked up earlier?",
                )
                .preamble(
                    "You are a concise assistant with perfect recall of this conversation."
                        .to_string(),
                )
                .temperature(0.0)
                .message(Message::user(
                    "My favorite color is teal. Please remember it.",
                ))
                .message(Message::assistant("Noted - your favorite color is teal."))
                .message(Message::user("Now look up the harbor label with the tool."))
                .message(Message::Assistant {
                    id: None,
                    content: rig::OneOrMany::many(vec![
                        AssistantContent::text("Checking the harbor label now."),
                        AssistantContent::tool_call(
                            AlphaSignal::NAME,
                            AlphaSignal::NAME,
                            serde_json::json!({}),
                        ),
                    ])
                    .expect("assistant content should be non-empty"),
                })
                .message(Message::tool_result(AlphaSignal::NAME, ALPHA_SIGNAL_OUTPUT))
                .message(Message::assistant("The harbor label is crimson-harbor."))
                .tool(AlphaSignal.definition(String::new()).await)
                .build();

            let response = model
                .completion(request)
                .await
                .expect("long history replay should be accepted by generateContent");

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
            assert!(
                response
                    .raw_response
                    .model_version
                    .as_deref()
                    .is_some_and(|version| !version.is_empty()),
                "provider response should preserve the model version"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn thinking_session_reports_thought_tokens_in_usage() {
    with_gemini_cassette(
        "generate_sessions/thinking_session_reports_thought_tokens_in_usage",
        |client| async move {
            let model = client.completion_model(gemini::completion::GEMINI_2_5_FLASH);
            let request = model
                .completion_request(
                    "A farmer has 17 sheep. All but 9 run away. How many sheep are left? \
                     Think it through, then answer in one short sentence.",
                )
                .temperature(0.0)
                .additional_params(serde_json::json!({
                    "generationConfig": {
                        "thinkingConfig": { "thinkingBudget": 1024, "includeThoughts": true }
                    }
                }))
                .build();

            let response = model
                .completion(request)
                .await
                .expect("thinking-enabled completion should succeed");

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
                lowered.contains('9') || lowered.contains("nine"),
                "the visible answer should state the result, got {text:?}"
            );
            assert!(
                response.usage.reasoning_tokens > 0,
                "thoughtsTokenCount should surface as reasoning tokens: {:?}",
                response.usage
            );
            assert!(
                response.usage.total_tokens
                    >= response.usage.input_tokens + response.usage.output_tokens,
                "total tokens should cover prompt and candidate tokens: {:?}",
                response.usage
            );
        },
    )
    .await;
}
