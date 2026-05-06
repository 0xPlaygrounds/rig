//! Gemini high-level Chat history regression tests.
//!
//! Run only this case with:
//! `cargo test -p rig --test gemini gemini::chat_history::chat_appends_reasoning_tool_turns_to_caller_history -- --ignored --nocapture`

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use rig::client::{CompletionClient, ProviderClient};
use rig::completion::{Chat, Message};
use rig::providers::gemini;

use crate::reasoning::{self, WeatherTool};

#[tokio::test]
#[ignore = "requires GEMINI_API_KEY"]
async fn chat_appends_reasoning_tool_turns_to_caller_history() {
    let call_count = Arc::new(AtomicUsize::new(0));
    let client = gemini::Client::from_env().expect("client should build");
    let agent = client
        .agent(gemini::completion::GEMINI_2_5_FLASH)
        .preamble(reasoning::TOOL_SYSTEM_PROMPT)
        .max_tokens(4096)
        .tool(WeatherTool::new(call_count.clone()))
        .additional_params(serde_json::json!({
            "generationConfig": {
                "thinkingConfig": { "thinkingBudget": 4096, "includeThoughts": true }
            }
        }))
        .build();
    let mut chat_history = Vec::<Message>::new();

    let result = agent
        .chat(reasoning::TOOL_USER_PROMPT, &mut chat_history)
        .await
        .expect("[gemini] Chat failed before it could update caller-owned history");

    reasoning::assert_nonstreaming_universal(&result, &call_count, "gemini");
    reasoning::assert_chat_history_preserves_reasoning_tool_roundtrip(
        &chat_history,
        &result,
        "gemini",
    );
}
