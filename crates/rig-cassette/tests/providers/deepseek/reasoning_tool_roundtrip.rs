//! DeepSeek reasoning-enabled tool roundtrip tests.

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use rig::completion::Message;
use rig::providers::deepseek;

use super::support::with_deepseek_cassette;
use crate::reasoning::{self, WeatherTool};

fn thinking_params() -> serde_json::Value {
    serde_json::json!({
        "thinking": { "type": "enabled" }
    })
}

#[tokio::test]
async fn nonstreaming() {
    with_deepseek_cassette(
        "reasoning_tool_roundtrip/nonstreaming",
        |client| async move {
            let call_count = Arc::new(AtomicUsize::new(0));
            let agent = rig::AgentBuilder::new(client.completion(deepseek::DEEPSEEK_V4_FLASH))
                .preamble(reasoning::TOOL_SYSTEM_PROMPT)
                .max_tokens(4096)
                .tool(WeatherTool::new(call_count.clone()))
                .additional_params(thinking_params())
                .default_max_turns(2)
                .build();

            let result = agent
                .chat(reasoning::TOOL_USER_PROMPT, &mut Vec::<Message>::new())
                .await
                .expect("[deepseek] Non-streaming chat failed - likely 400 from dropped reasoning");

            reasoning::assert_nonstreaming_universal(&result.output(), &call_count, "deepseek");
        },
    )
    .await;
}
