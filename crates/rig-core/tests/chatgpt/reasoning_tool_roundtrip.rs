//! ChatGPT reasoning-enabled tool roundtrip tests.

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use rig_core::client::CompletionClient;
use rig_core::completion::Message;
use rig_core::streaming::StreamingChat;

use crate::chatgpt::{LIVE_MODEL, live_client};
use crate::reasoning::{self, WeatherTool};

#[tokio::test]
#[ignore = "requires ChatGPT credentials or existing OAuth cache"]
async fn streaming() {
    let call_count = Arc::new(AtomicUsize::new(0));
    let agent = live_client()
        .agent(LIVE_MODEL)
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

    let stats = reasoning::collect_stream_stats(stream, "chatgpt").await;
    reasoning::assert_universal(&stats, &call_count, "chatgpt");

    if stats.reasoning_block_count > 0 {
        assert!(
            stats.reasoning_has_encrypted || stats.reasoning_content_types.contains(&"Summary"),
            "[chatgpt] Expected encrypted or summary reasoning content. Got: {:?}",
            stats.reasoning_content_types
        );
    }
}
