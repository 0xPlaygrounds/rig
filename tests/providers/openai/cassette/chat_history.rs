//! OpenAI high-level Chat history regression tests.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use rig::client::CompletionClient;
use rig::completion::{Chat, Message};
use rig::providers::openai;

use super::super::support::with_openai_cassette;
use crate::reasoning::{self, WeatherTool};

#[tokio::test]
async fn chat_appends_reasoning_tool_turns_to_caller_history() {
    with_openai_cassette(
        "chat_history/chat_appends_reasoning_tool_turns_to_caller_history",
        |client| async move {
            let call_count = Arc::new(AtomicUsize::new(0));
            let agent = client
                .agent(openai::GPT_5_2)
                .preamble(reasoning::TOOL_SYSTEM_PROMPT)
                .max_tokens(4096)
                .tool(WeatherTool::new(call_count.clone()))
                .additional_params(serde_json::json!({
                    "reasoning": { "effort": "high" }
                }))
                .build();
            let mut chat_history = Vec::<Message>::new();

            let result = agent
                .chat(reasoning::TOOL_USER_PROMPT, &mut chat_history)
                .await
                .expect("[openai] Chat failed before it could update caller-owned history");

            reasoning::assert_nonstreaming_universal(&result, &call_count, "openai");
            reasoning::assert_chat_history_preserves_reasoning_tool_roundtrip(
                &chat_history,
                &result,
                "openai",
            );
        },
    )
    .await;
}
