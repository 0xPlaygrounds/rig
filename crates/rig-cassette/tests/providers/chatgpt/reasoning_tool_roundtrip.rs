//! ChatGPT reasoning-enabled tool roundtrip tests.

use rig::wire::Wire as _;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use rig::completion::Message;

use crate::chatgpt::{LIVE_MODEL, live_client};
use crate::reasoning::{self, WeatherTool};

#[tokio::test]
#[ignore = "requires ChatGPT credentials or existing OAuth cache"]
async fn streaming() {
    let call_count = Arc::new(AtomicUsize::new(0));
    let agent = rig::AgentBuilder::new(
        live_client()
            .await
            .completion(LIVE_MODEL)
            .on(rig::transport()),
    )
    .preamble(reasoning::TOOL_SYSTEM_PROMPT)
    .max_tokens(4096)
    .tool(WeatherTool::new(call_count.clone()))
    .additional_params(serde_json::json!({
        "reasoning": { "effort": "high" }
    }))
    .build();

    let stream = agent
        .prompt(reasoning::TOOL_USER_PROMPT)
        .history(Vec::<Message>::new())
        .max_turns(3)
        .stream();

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
