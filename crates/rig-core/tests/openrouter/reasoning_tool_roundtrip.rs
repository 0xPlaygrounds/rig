//! OpenRouter reasoning tool roundtrip tests.
//!
//! Run only these cases with:
//! `cargo test -p rig-core --test openrouter openrouter::reasoning_tool_roundtrip::streaming -- --ignored --nocapture`

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::completion::{Chat, Message};
use rig_core::providers::openrouter;
use rig_core::streaming::StreamingChat;

use crate::reasoning::{self, WeatherTool};

#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY"]
async fn streaming() {
    let call_count = Arc::new(AtomicUsize::new(0));
    let client = openrouter::Client::from_env().expect("client should build");
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
}

#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY"]
async fn nonstreaming() {
    let call_count = Arc::new(AtomicUsize::new(0));
    let client = openrouter::Client::from_env().expect("client should build");
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
        .chat(reasoning::TOOL_USER_PROMPT, Vec::<Message>::new())
        .await
        .expect("[openrouter] Non-streaming chat failed - likely 400 from dropped reasoning");

    reasoning::assert_nonstreaming_universal(&result, &call_count, "openrouter");
}
