//! Cassette-backed OpenRouter reasoning tool roundtrip tests.

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use rig::client::CompletionClient;
use rig::completion::{Chat, Message};
use rig::streaming::StreamingChat;

use crate::reasoning::{self, WeatherTool};

use super::super::support::with_openrouter_cassette;

#[tokio::test]
async fn streaming() {
    with_openrouter_cassette("reasoning_tool_roundtrip/streaming", |client| async move {
        let call_count = Arc::new(AtomicUsize::new(0));
        let agent = client
            .agent("openai/gpt-5.2")
            .preamble(reasoning::TOOL_SYSTEM_PROMPT)
            .max_tokens(4096)
            .tool(WeatherTool::new(call_count.clone()))
            .additional_params(serde_json::json!({
                "reasoning": { "effort": "high" },
                "include_reasoning": true
            }))
            .build();

        let stream = agent
            .stream_chat(reasoning::TOOL_USER_PROMPT, Vec::<Message>::new())
            .multi_turn(3)
            .await;

        let stats = reasoning::collect_stream_stats(stream, "openrouter").await;
        reasoning::assert_universal(&stats, &call_count, "openrouter");
    })
    .await;
}

#[tokio::test]
async fn nonstreaming() {
    with_openrouter_cassette(
        "reasoning_tool_roundtrip/nonstreaming",
        |client| async move {
            let call_count = Arc::new(AtomicUsize::new(0));
            let agent = client
                .agent("openai/gpt-5.2")
                .preamble(reasoning::TOOL_SYSTEM_PROMPT)
                .max_tokens(4096)
                .tool(WeatherTool::new(call_count.clone()))
                .additional_params(serde_json::json!({
                    "reasoning": { "effort": "high" },
                    "include_reasoning": true
                }))
                .build();

            let result = agent
                .chat(reasoning::TOOL_USER_PROMPT, &mut Vec::<Message>::new())
                .await
                .expect(
                    "[openrouter] Non-streaming chat failed - likely 400 from dropped reasoning",
                );

            reasoning::assert_nonstreaming_universal(&result, &call_count, "openrouter");
        },
    )
    .await;
}
