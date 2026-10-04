//! OpenAI Responses API long-session regression tests.
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

use rig::completion::Message;
use rig::message::{AssistantContent, UserContent};
use rig::providers::openai;
use rig::tool::Tool;

use super::super::support::with_openai_cassette;
use crate::reasoning::{self, WeatherTool};
use crate::support::{ALPHA_SIGNAL_OUTPUT, AlphaSignal};

#[tokio::test]
async fn long_history_replay_nonstreaming() {
    with_openai_cassette(
        "responses_sessions/long_history_replay_nonstreaming",
        |client| async move {
            let model = client.openai.completion(openai::GPT_4O);
            let preamble = "You are a concise assistant with perfect recall of this conversation.";

            // First turn: obtain a real tool call so the follow-up can echo
            // its call_id back, the way a caller-owned history would.
            let first_request = CompletionRequest::new("Look up the harbor label with the tool.")
                .preamble(preamble.to_string())
                .tool(rig::tool::tool_definition(&AlphaSignal))
                .additional_params(serde_json::json!({ "store": false }));
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
                |provider| provider.as_str().to_owned(),
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
            .message(Message::Assistant(
                rig_core::message::AssistantMessage::new(vec![AssistantContent::tool_call(
                    call_id.clone(),
                    rig_core::message::ToolName::new(AlphaSignal::NAME).expect("tool name"),
                    serde_json::json!({}),
                )]),
            ))
            .message(Message::from(UserContent::tool_result(
                rig_core::message::CallId::from_wire(call_id),
                rig_core::message::ToolName::new(AlphaSignal::NAME).expect("tool name"),
                vec![rig::message::ToolResultContent::text(ALPHA_SIGNAL_OUTPUT)],
            )))
            .message(Message::assistant("The harbor label is crimson-harbor."))
            .tool(rig::tool::tool_definition(&AlphaSignal))
            .additional_params(serde_json::json!({ "store": false }));

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
    with_openai_cassette(
        "responses_sessions/reasoning_session_two_tool_calls_streaming",
        |client| async move {
            let call_count = Arc::new(AtomicUsize::new(0));
            let agent = rig::AgentBuilder::new(client.openai.completion(openai::GPT_5_2))
                .preamble(reasoning::TOOL_SYSTEM_PROMPT)
                .max_tokens(6000)
                .tool(WeatherTool::new(call_count.clone()))
                .additional_params(serde_json::json!({
                    "reasoning": { "effort": "low" },
                    "store": false
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
