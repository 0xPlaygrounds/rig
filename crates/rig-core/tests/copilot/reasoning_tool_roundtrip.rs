//! Copilot reasoning-enabled tool roundtrip tests.

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use rig_core::client::CompletionClient;
use rig_core::completion::{Chat, Message};
use rig_core::streaming::StreamingChat;

use crate::copilot::{live_client, live_responses_model};
use crate::reasoning::{self, WeatherTool};

#[tokio::test]
#[ignore = "requires Copilot credentials or existing OAuth cache"]
async fn streaming() {
    let call_count = Arc::new(AtomicUsize::new(0));
    let agent = live_client()
        .agent(live_responses_model())
        .preamble(reasoning::TOOL_SYSTEM_PROMPT)
        .max_tokens(4096)
        .tool(WeatherTool::new(call_count.clone()))
        .additional_params(serde_json::json!({
            "reasoning": { "effort": "high" }
        }))
        .build();

    let stream = agent
        .stream_chat(reasoning::TOOL_USER_PROMPT, Vec::<Message>::new())
        .multi_turn(3)
        .await;

    let stats = reasoning::collect_stream_stats(stream, "copilot").await;
    reasoning::assert_universal(&stats, &call_count, "copilot");

    if stats.reasoning_block_count > 0 {
        assert!(
            stats.reasoning_has_encrypted || stats.reasoning_content_types.contains(&"Summary"),
            "[copilot] Expected encrypted or summary reasoning content. Got: {:?}",
            stats.reasoning_content_types
        );
    }
}

#[tokio::test]
#[ignore = "requires Copilot credentials or existing OAuth cache"]
async fn nonstreaming() {
    let call_count = Arc::new(AtomicUsize::new(0));
    let agent = live_client()
        .agent(live_responses_model())
        .preamble(reasoning::TOOL_SYSTEM_PROMPT)
        .max_tokens(4096)
        .tool(WeatherTool::new(call_count.clone()))
        .additional_params(serde_json::json!({
            "reasoning": { "effort": "high" }
        }))
        .build();

    let result = agent
        .chat(reasoning::TOOL_USER_PROMPT, Vec::<Message>::new())
        .await
        .expect("[copilot] Non-streaming chat failed");

    reasoning::assert_nonstreaming_universal(&result, &call_count, "copilot");
}
