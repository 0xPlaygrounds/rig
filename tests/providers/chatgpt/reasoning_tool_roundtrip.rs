//! ChatGPT reasoning-enabled tool roundtrip tests.

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use rig::completion::Message;

use crate::chatgpt::{LIVE_MODEL, live_agent};
use crate::reasoning::{self, WeatherTool};

#[tokio::test]
#[ignore = "requires ChatGPT credentials or existing OAuth cache"]
async fn streaming() {
    let call_count = Arc::new(AtomicUsize::new(0));
    let agent = live_agent(LIVE_MODEL)
        .await
        .preamble(reasoning::TOOL_SYSTEM_PROMPT)
        .max_tokens(4096)
        .tool(WeatherTool::new(call_count.clone()))
        .additional_params(serde_json::json!({
            "reasoning": { "effort": "high" }
        }))
        .build();

    let stream = agent
        .runner(reasoning::TOOL_USER_PROMPT)
        .history(Vec::<Message>::new())
        .max_turns(3)
        .stream_run();

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
