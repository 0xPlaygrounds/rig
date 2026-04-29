//! DeepSeek reasoning-enabled tool roundtrip tests.

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::completion::{Chat, Message};
use rig_core::providers::deepseek;
use rig_core::streaming::StreamingChat;

use crate::reasoning::{self, WeatherTool};

fn thinking_params() -> serde_json::Value {
    serde_json::json!({
        "thinking": { "type": "enabled" }
    })
}

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY"]
async fn streaming() {
    let call_count = Arc::new(AtomicUsize::new(0));
    let client = deepseek::Client::from_env().expect("client should build");
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
}

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY"]
async fn nonstreaming() {
    let call_count = Arc::new(AtomicUsize::new(0));
    let client = deepseek::Client::from_env().expect("client should build");
    let agent = client
        .agent(deepseek::DEEPSEEK_V4_FLASH)
        .preamble(reasoning::TOOL_SYSTEM_PROMPT)
        .max_tokens(4096)
        .tool(WeatherTool::new(call_count.clone()))
        .additional_params(thinking_params())
        .build();

    let result = agent
        .chat(reasoning::TOOL_USER_PROMPT, Vec::<Message>::new())
        .await
        .expect("[deepseek] Non-streaming chat failed - likely 400 from dropped reasoning");

    reasoning::assert_nonstreaming_universal(&result, &call_count, "deepseek");
}
