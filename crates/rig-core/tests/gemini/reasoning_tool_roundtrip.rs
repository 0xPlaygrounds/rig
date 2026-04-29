//! Gemini reasoning tool roundtrip tests.
//!
//! Run only these cases with:
//! `cargo test -p rig-core --test gemini gemini::reasoning_tool_roundtrip::streaming -- --ignored --nocapture`

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::completion::{Chat, Message};
use rig_core::providers::gemini;
use rig_core::streaming::StreamingChat;

use crate::reasoning::{self, WeatherTool};

#[tokio::test]
#[ignore = "requires GEMINI_API_KEY"]
async fn streaming() {
    let call_count = Arc::new(AtomicUsize::new(0));
    let client = gemini::Client::from_env().expect("client should build");
    let agent = client
        .agent("gemini-2.5-flash")
        .preamble(reasoning::TOOL_SYSTEM_PROMPT)
        .max_tokens(4096)
        .tool(WeatherTool::new(call_count.clone()))
        .additional_params(serde_json::json!({
            "generationConfig": {
                "thinkingConfig": { "thinkingBudget": 4096, "includeThoughts": true }
            }
        }))
        .build();

    let stream = agent
        .stream_chat(reasoning::TOOL_USER_PROMPT, Vec::<Message>::new())
        .multi_turn(3)
        .await;

    let stats = reasoning::collect_stream_stats(stream, "gemini").await;
    reasoning::assert_universal(&stats, &call_count, "gemini");
}

#[tokio::test]
#[ignore = "requires GEMINI_API_KEY"]
async fn nonstreaming() {
    let call_count = Arc::new(AtomicUsize::new(0));
    let client = gemini::Client::from_env().expect("client should build");
    let agent = client
        .agent("gemini-2.5-flash")
        .preamble(reasoning::TOOL_SYSTEM_PROMPT)
        .max_tokens(4096)
        .tool(WeatherTool::new(call_count.clone()))
        .additional_params(serde_json::json!({
            "generationConfig": {
                "thinkingConfig": { "thinkingBudget": 4096, "includeThoughts": true }
            }
        }))
        .build();

    let result = agent
        .chat(reasoning::TOOL_USER_PROMPT, Vec::<Message>::new())
        .await
        .expect("[gemini] Non-streaming chat failed - likely 400 from dropped reasoning");

    reasoning::assert_nonstreaming_universal(&result, &call_count, "gemini");
}
