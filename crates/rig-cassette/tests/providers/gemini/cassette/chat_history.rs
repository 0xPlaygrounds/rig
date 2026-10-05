//! Gemini high-level Chat history regression tests.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use rig::completion::Message;
use rig::providers::gemini;

use crate::reasoning::{self, WeatherTool};

#[tokio::test]
async fn chat_appends_reasoning_tool_turns_to_caller_history() {
    let call_count = Arc::new(AtomicUsize::new(0));
    super::super::support::with_gemini_cassette(
        "chat_history/chat_appends_reasoning_tool_turns_to_caller_history",
        |client| async move {
            let agent =
                rig::AgentBuilder::new(client.completion(gemini::completion::GEMINI_2_5_FLASH))
                    .preamble(reasoning::TOOL_SYSTEM_PROMPT)
                    .max_tokens(4096)
                    .tool(WeatherTool::new(call_count.clone()))
                    .additional_params(serde_json::json!({
                        "generationConfig": {
                            "thinkingConfig": { "thinkingBudget": 4096, "includeThoughts": true }
                        }
                    }))
                    .default_max_turns(2)
                    .build();
            let mut chat_history = Vec::<Message>::new();

            let result = agent
                .chat(reasoning::TOOL_USER_PROMPT, &mut chat_history)
                .await
                .expect("[gemini] Chat failed before it could update caller-owned history");

            reasoning::assert_nonstreaming_universal(&result.output(), &call_count, "gemini");
            reasoning::assert_chat_history_preserves_reasoning_tool_roundtrip(
                &chat_history,
                &result.output(),
                "gemini",
            );
        },
    )
    .await;
}
