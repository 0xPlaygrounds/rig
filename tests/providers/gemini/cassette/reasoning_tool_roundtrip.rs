//! Gemini reasoning tool roundtrip tests.
//!
//! Run only these cases with:
//! `cargo test -p rig --test gemini gemini::reasoning_tool_roundtrip::streaming -- --ignored --nocapture`

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use rig::client::CompletionClient;
use rig::completion::{Chat, Message};
use rig::streaming::StreamingChat;

use crate::reasoning::{self, WeatherTool};

#[tokio::test]
async fn streaming() {
    let call_count = Arc::new(AtomicUsize::new(0));
    let (cassette, client) =
        super::super::support::gemini_cassette("reasoning_tool_roundtrip/streaming").await;
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

    cassette.finish().await;
}

#[tokio::test]
async fn nonstreaming() {
    let call_count = Arc::new(AtomicUsize::new(0));
    let (cassette, client) =
        super::super::support::gemini_cassette("reasoning_tool_roundtrip/nonstreaming").await;
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
        .chat(reasoning::TOOL_USER_PROMPT, &mut Vec::<Message>::new())
        .await
        .expect("[gemini] Non-streaming chat failed - likely 400 from dropped reasoning");

    reasoning::assert_nonstreaming_universal(&result, &call_count, "gemini");

    cassette.finish().await;
}
