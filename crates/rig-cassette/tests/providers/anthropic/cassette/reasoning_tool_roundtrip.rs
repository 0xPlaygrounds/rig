//! Anthropic reasoning-enabled tool roundtrip tests.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use rig::completion::Message;
use rig::providers::anthropic::completion::CLAUDE_SONNET_4_6;

use super::super::support::with_anthropic_cassette;
use crate::reasoning::{self, WeatherTool};

#[tokio::test]
async fn nonstreaming() {
    with_anthropic_cassette(
        "reasoning_tool_roundtrip/nonstreaming",
        |client| async move {
            let call_count = Arc::new(AtomicUsize::new(0));
            let agent = rig::AgentBuilder::new(client.completion(CLAUDE_SONNET_4_6))
                .preamble(reasoning::TOOL_SYSTEM_PROMPT)
                .max_tokens(16384)
                .tool(WeatherTool::new(call_count.clone()))
                .additional_params(serde_json::json!({
                    "thinking": { "type": "adaptive" }
                }))
                .default_max_turns(2)
                .build();

            let result = agent
                .chat(reasoning::TOOL_USER_PROMPT, &mut Vec::<Message>::new())
                .await
                .expect("[anthropic] Non-streaming chat failed - likely 400 from dropped reasoning")
                .output();

            reasoning::assert_nonstreaming_universal(&result, &call_count, "anthropic");
        },
    )
    .await;
}
