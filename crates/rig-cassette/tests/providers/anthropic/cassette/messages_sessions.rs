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

use rig::completion::{CompletionRequest, FinishReason, Message};
use rig::message::{AssistantContent, UserContent};
use rig::providers::anthropic;

use super::super::support::with_anthropic_cassette;
use crate::support::{ALPHA_SIGNAL_OUTPUT, AlphaSignal};

#[tokio::test]
async fn long_history_replay_nonstreaming() {
    with_anthropic_cassette(
        "messages_sessions/long_history_replay_nonstreaming",
        |client| async move {
            let model = client.completion(anthropic::completion::CLAUDE_SONNET_4_6);
            let preamble = "You are a concise assistant with perfect recall of this conversation.";

            // First turn: obtain a real tool_use so the follow-up can echo its
            // id back, the way a caller-owned history would.
            let first_request = CompletionRequest::new("Look up the harbor label with the tool.")
                .preamble(preamble.to_string())
                .max_tokens(1024)
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
            assert_eq!(
                first_response.finish_reason(),
                Some(FinishReason::ToolCalls),
                "a tool-using turn should preserve the tool_use stop reason"
            );

            // Follow-up: replay a long client-owned history around that tool
            // roundtrip, including assistant text before the tool_use (in the
            // same assistant message) and assistant text after the result.
            let request = CompletionRequest::new(
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
            .message(Message::Assistant(rig::message::AssistantMessage::new(
                vec![
                    AssistantContent::text("Checking the harbor label now."),
                    AssistantContent::ToolCall(tool_call.clone()),
                ],
            )))
            .message(Message::from(UserContent::tool_result(
                tool_call.id.clone(),
                tool_call.function.name.clone(),
                vec![rig::message::ToolResultContent::text(ALPHA_SIGNAL_OUTPUT)],
            )))
            .message(Message::assistant("The harbor label is crimson-harbor."))
            .tool(rig::tool::tool_definition(&AlphaSignal));

            let response = model
                .call(request)
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
                response.finish_reason(),
                Some(FinishReason::Stop),
                "a plain answer should preserve the end_turn stop reason"
            );
            assert!(
                response.model().is_some_and(|model| !model.is_empty())
                    && response.response_id().is_some_and(|id| !id.is_empty()),
                "provider response should preserve model and message id"
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
