//! DeepSeek reasoning-enabled tool roundtrip tests.

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use rig::client::CompletionClient;
use rig::completion::{Chat, Message};
use rig::providers::deepseek;
use rig::streaming::StreamingChat;

use super::support::with_deepseek_cassette;
use crate::reasoning::{self, WeatherTool};

fn thinking_params() -> serde_json::Value {
    serde_json::json!({
        "thinking": { "type": "enabled" }
    })
}

#[tokio::test]
async fn streaming() {
    with_deepseek_cassette("reasoning_tool_roundtrip/streaming", |client| async move {
        let call_count = Arc::new(AtomicUsize::new(0));
        let agent = client
            .agent(deepseek::DEEPSEEK_V4_FLASH)
            .preamble(reasoning::TOOL_SYSTEM_PROMPT)
            .max_tokens(4096)
            .tool(WeatherTool::new(call_count.clone()))
            .additional_params(thinking_params())
            .build();

        let stream = agent
            .stream_chat(reasoning::TOOL_USER_PROMPT, Vec::<Message>::new())
            .multi_turn(3)
            .await;

        let stats = reasoning::collect_stream_stats(stream, "deepseek").await;
        reasoning::assert_universal(&stats, &call_count, "deepseek");

        if stats.reasoning_block_count > 0 {
            assert!(
                stats.reasoning_content_types.contains(&"Text"),
                "[deepseek] Expected text reasoning content. Got: {:?}",
                stats.reasoning_content_types
            );
        }
    })
    .await;
}

#[tokio::test]
async fn nonstreaming() {
    with_deepseek_cassette(
        "reasoning_tool_roundtrip/nonstreaming",
        |client| async move {
            let call_count = Arc::new(AtomicUsize::new(0));
            let agent = client
                .agent(deepseek::DEEPSEEK_V4_FLASH)
                .preamble(reasoning::TOOL_SYSTEM_PROMPT)
                .max_tokens(4096)
                .tool(WeatherTool::new(call_count.clone()))
                .additional_params(thinking_params())
                .build();

            let result = agent
                .chat(reasoning::TOOL_USER_PROMPT, &mut Vec::<Message>::new())
                .await
                .expect("[deepseek] Non-streaming chat failed - likely 400 from dropped reasoning");

            reasoning::assert_nonstreaming_universal(&result, &call_count, "deepseek");
        },
    )
    .await;
}
