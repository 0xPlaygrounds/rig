//! Anthropic reasoning-enabled tool roundtrip tests.
//!
//! Run only these cases with:
//! `cargo test -p rig-core --test anthropic anthropic::reasoning_tool_roundtrip::streaming -- --ignored --nocapture`

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::completion::{Chat, Message};
use rig_core::providers::anthropic::{self, completion::CLAUDE_SONNET_4_6};
use rig_core::streaming::StreamingChat;

use crate::reasoning::{self, WeatherTool};

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY"]
async fn streaming() {
    let call_count = Arc::new(AtomicUsize::new(0));
    let client = anthropic::Client::from_env().expect("client should build");
    let agent = client
        .agent(CLAUDE_SONNET_4_6)
        .preamble(reasoning::TOOL_SYSTEM_PROMPT)
        .max_tokens(16384)
        .tool(WeatherTool::new(call_count.clone()))
        .additional_params(serde_json::json!({
            "thinking": { "type": "adaptive" }
        }))
        .build();

    let stream = agent
        .stream_chat(reasoning::TOOL_USER_PROMPT, Vec::<Message>::new())
        .multi_turn(3)
        .await;

    let stats = reasoning::collect_stream_stats(stream, "anthropic").await;
    reasoning::assert_universal(&stats, &call_count, "anthropic");

    if stats.reasoning_block_count > 0 {
        assert!(
            stats.reasoning_has_signature,
            "[anthropic] Thinking blocks should have signatures. Content types: {:?}",
            stats.reasoning_content_types
        );
        assert!(
            stats.reasoning_content_types.contains(&"Text"),
            "[anthropic] Expected text reasoning content. Got: {:?}",
            stats.reasoning_content_types
        );
    }
}

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY"]
async fn nonstreaming() {
    let call_count = Arc::new(AtomicUsize::new(0));
    let client = anthropic::Client::from_env().expect("client should build");
    let agent = client
        .agent(CLAUDE_SONNET_4_6)
        .preamble(reasoning::TOOL_SYSTEM_PROMPT)
        .max_tokens(16384)
        .tool(WeatherTool::new(call_count.clone()))
        .additional_params(serde_json::json!({
            "thinking": { "type": "adaptive" }
        }))
        .build();

    let result = agent
        .chat(reasoning::TOOL_USER_PROMPT, Vec::<Message>::new())
        .await
        .expect("[anthropic] Non-streaming chat failed - likely 400 from dropped reasoning");

    reasoning::assert_nonstreaming_universal(&result, &call_count, "anthropic");
}
