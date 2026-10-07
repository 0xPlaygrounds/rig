//! Cassette-backed OpenRouter reasoning tool roundtrip tests.

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use rig::completion::Message;

use crate::reasoning::{self, WeatherTool};

use super::super::support::with_openrouter_cassette;

#[tokio::test]
async fn nonstreaming() {
    with_openrouter_cassette(
        "reasoning_tool_roundtrip/nonstreaming",
        |client| async move {
            let call_count = Arc::new(AtomicUsize::new(0));
            let agent = rig::AgentBuilder::new(client.completion("openai/gpt-5.2"))
                .preamble(reasoning::TOOL_SYSTEM_PROMPT)
                .max_tokens(4096)
                .tool(WeatherTool::new(call_count.clone()))
                .reasoning(rig::completion::Effort::High)
                .additional_params(serde_json::json!({ "include_reasoning": true }))
                .default_max_turns(2)
                .build();

            let result = agent
                .chat(reasoning::TOOL_USER_PROMPT, &mut Vec::<Message>::new())
                .await
                .expect(
                    "[openrouter] Non-streaming chat failed - likely 400 from dropped reasoning",
                );

            reasoning::assert_nonstreaming_universal(&result.output(), &call_count, "openrouter");
        },
    )
    .await;
}
