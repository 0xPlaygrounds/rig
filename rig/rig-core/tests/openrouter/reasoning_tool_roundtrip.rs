//! OpenRouter reasoning tool roundtrip tests.
//!
//! Run only these cases with:
//! `cargo test -p rig-core --test openrouter reasoning_tool_roundtrip::streaming -- --ignored --nocapture`

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use rig::client::{CompletionClient, ProviderClient};
use rig::completion::{Chat, Message};
use rig::providers::openrouter;
use rig::streaming::StreamingChat;

use crate::reasoning::{self, WeatherTool};

#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY"]
async fn streaming() {
    let call_count = Arc::new(AtomicUsize::new(0));
    let client = openrouter::Client::from_env();
    let agent = client
        .agent("openai/gpt-5.2")
        .preamble(reasoning::TOOL_SYSTEM_PROMPT)
        .max_tokens(4096)
        .tool(WeatherTool::new(call_count.clone()))
        .additional_params(serde_json::json!({
            "reasoning": { "effort": "high" },
            "include_reasoning": true
        }))
        .build();

    let stream = agent
        .stream_chat(reasoning::TOOL_USER_PROMPT, Vec::<Message>::new())
        .multi_turn(3)
        .await;

    let stats = reasoning::collect_stream_stats(stream, "openrouter").await;

    assert!(
        stats.errors.is_empty(),
        "[openrouter] Stream had errors: {:?}",
        stats.errors
    );

    let invocations = call_count.load(Ordering::SeqCst);
    assert!(
        invocations >= 1,
        "[openrouter] Tool was never invoked (count=0)."
    );
    assert!(
        !stats.tool_calls_in_stream.is_empty(),
        "[openrouter] No tool-call events in stream."
    );
    assert!(
        stats.tool_results_in_stream >= 1,
        "[openrouter] No tool-result events in stream."
    );
    assert!(
        !stats.final_text.trim().is_empty(),
        "[openrouter] Final text is empty."
    );
    assert!(
        stats.got_final_response,
        "[openrouter] Stream did not emit FinalResponse."
    );
}

#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY"]
async fn nonstreaming() {
    let call_count = Arc::new(AtomicUsize::new(0));
    let client = openrouter::Client::from_env();
    let agent = client
        .agent("openai/gpt-5.2")
        .preamble(reasoning::TOOL_SYSTEM_PROMPT)
        .max_tokens(4096)
        .tool(WeatherTool::new(call_count.clone()))
        .additional_params(serde_json::json!({
            "reasoning": { "effort": "high" },
            "include_reasoning": true
        }))
        .build();

    let result = agent
        .chat(reasoning::TOOL_USER_PROMPT, Vec::<Message>::new())
        .await
        .expect("[openrouter] Non-streaming chat failed - likely 400 from dropped reasoning");

    reasoning::assert_nonstreaming_universal(&result, &call_count, "openrouter");
}
