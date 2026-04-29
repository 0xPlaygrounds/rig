//! xAI reasoning tool roundtrip tests.
//!
//! Run only these cases with:
//! `cargo test -p rig-core --test xai xai::reasoning_tool_roundtrip::streaming -- --ignored --nocapture`

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::completion::{Chat, Message};
use rig_core::providers::xai;
use rig_core::streaming::StreamingChat;

use crate::reasoning::{self, WeatherTool};

#[tokio::test]
#[ignore = "requires XAI_API_KEY - validate with grok-4-0725 once key is available"]
async fn streaming() {
    let call_count = Arc::new(AtomicUsize::new(0));
    let client = xai::Client::from_env().expect("client should build");
    let agent = client
        .agent(xai::GROK_3_MINI)
        .preamble(reasoning::TOOL_SYSTEM_PROMPT)
        .max_tokens(4096)
        .tool(WeatherTool::new(call_count.clone()))
        .build();

    let stream = agent
        .stream_chat(reasoning::TOOL_USER_PROMPT, Vec::<Message>::new())
        .multi_turn(3)
        .await;

    let stats = reasoning::collect_stream_stats(stream, "xai").await;
    reasoning::assert_universal(&stats, &call_count, "xai");
}

#[tokio::test]
#[ignore = "requires XAI_API_KEY - validate with grok-4-0725 once key is available"]
async fn nonstreaming() {
    let call_count = Arc::new(AtomicUsize::new(0));
    let client = xai::Client::from_env().expect("client should build");
    let agent = client
        .agent(xai::GROK_3_MINI)
        .preamble(reasoning::TOOL_SYSTEM_PROMPT)
        .max_tokens(4096)
        .tool(WeatherTool::new(call_count.clone()))
        .build();

    let result = agent
        .chat(reasoning::TOOL_USER_PROMPT, Vec::<Message>::new())
        .await
        .expect("[xai] Non-streaming chat failed - likely 400 from dropped reasoning");

    reasoning::assert_nonstreaming_universal(&result, &call_count, "xai");
}
