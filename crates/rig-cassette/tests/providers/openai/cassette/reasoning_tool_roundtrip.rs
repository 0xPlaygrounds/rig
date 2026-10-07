//! OpenAI reasoning-enabled tool roundtrip tests.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use rig::completion::Message;

use super::super::support::{stateless, with_openai_cassette};
use crate::reasoning::{self, WeatherTool};

#[tokio::test]
async fn nonstreaming() {
    with_openai_cassette(
        "reasoning_tool_roundtrip/nonstreaming",
        |client| async move {
            let call_count = Arc::new(AtomicUsize::new(0));
            let agent = rig::AgentBuilder::new(client.openai.completion("gpt-5.2"))
                .preamble(reasoning::TOOL_SYSTEM_PROMPT)
                .max_tokens(4096)
                .tool(WeatherTool::new(call_count.clone()))
                .reasoning(rig::completion::Effort::High)
                .provider_options(stateless())
                .default_max_turns(2)
                .build();

            let result = agent
                .chat(reasoning::TOOL_USER_PROMPT, &mut Vec::<Message>::new())
                .await
                .expect("[openai] Non-streaming chat failed - likely 400 from dropped reasoning")
                .output();

            reasoning::assert_nonstreaming_universal(&result, &call_count, "openai");
        },
    )
    .await;
}
