//! Ollama reasoning + tool roundtrip tests (thinking model calls a tool, then
//! answers from the tool result).
//!
//! Replays by default; set `RIG_PROVIDER_TEST_MODE=record` to record against a
//! local Ollama server.
//!
//! The non-streaming test additionally asserts the assistant's reasoning is
//! preserved in the caller-owned chat history — this is the direct regression
//! for #1926 (non-streaming responses used to drop `thinking`, so it never
//! entered history and was never sent back to Ollama).

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use rig::client::CompletionClient;
use rig::completion::{Chat, Message};
use rig::message::AssistantContent;
use rig::streaming::StreamingChat;

use super::super::support::with_ollama_cassette;
use crate::reasoning::{self, WeatherTool};

const MODEL: &str = "qwen3:4b";

fn think_params() -> serde_json::Value {
    serde_json::json!({ "think": true })
}

#[tokio::test]
async fn nonstreaming() {
    let call_count = Arc::new(AtomicUsize::new(0));
    with_ollama_cassette(
        "reasoning_tool_roundtrip/nonstreaming",
        |client| async move {
            let agent = client
                .agent(MODEL)
                .preamble(reasoning::TOOL_SYSTEM_PROMPT)
                .tool(WeatherTool::new(call_count.clone()))
                .additional_params(think_params())
                .build();

            let mut chat_history = Vec::<Message>::new();
            let result = agent
                .chat(reasoning::TOOL_USER_PROMPT, &mut chat_history)
                .await
                .expect("[ollama] non-streaming chat failed");

            reasoning::assert_nonstreaming_universal(&result, &call_count, "ollama");
            // #1926: the tool-call turn's `thinking` must survive into history as
            // an AssistantContent::Reasoning. Pre-fix, the non-streaming choice
            // contained only the ToolCall and this assertion failed.
            reasoning::assert_chat_history_preserves_reasoning_tool_roundtrip(
                &chat_history,
                &result,
                "ollama",
            );
            assert!(
                chat_history.iter().any(|msg| matches!(
                    msg,
                    Message::Assistant { content, .. }
                        if content.iter().any(|c| matches!(c, AssistantContent::Reasoning(_)))
                )),
                "[ollama] expected at least one assistant turn carrying Reasoning in history",
            );
        },
    )
    .await;
}

#[tokio::test]
async fn streaming() {
    let call_count = Arc::new(AtomicUsize::new(0));
    with_ollama_cassette("reasoning_tool_roundtrip/streaming", |client| async move {
        let agent = client
            .agent(MODEL)
            .preamble(reasoning::TOOL_SYSTEM_PROMPT)
            .tool(WeatherTool::new(call_count.clone()))
            .additional_params(think_params())
            .build();

        let stream = agent
            .stream_chat(reasoning::TOOL_USER_PROMPT, Vec::<Message>::new())
            .multi_turn(3)
            .await;

        let stats = reasoning::collect_stream_stats(stream, "ollama").await;
        reasoning::assert_universal(&stats, &call_count, "ollama");
    })
    .await;
}
