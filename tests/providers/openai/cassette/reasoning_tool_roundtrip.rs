//! OpenAI reasoning-enabled tool roundtrip tests.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use rig::client::CompletionClient;
use rig::completion::{Chat, CompletionModel, Message};
use rig::prelude::AgentClientExt;
use rig::providers::openai;
use rig::streaming::StreamingChat;
use rig_bevy::{LocalRuntime, PortableTool, TenantId};

use super::super::support::with_openai_cassette;
use crate::reasoning::{self, WeatherTool};

#[tokio::test]
async fn streaming() {
    with_openai_cassette("reasoning_tool_roundtrip/streaming", |client| async move {
        let call_count = Arc::new(AtomicUsize::new(0));
        let agent = client
            .agent("gpt-5.2")
            .preamble(reasoning::TOOL_SYSTEM_PROMPT)
            .max_tokens(4096)
            .tool(WeatherTool::new(call_count.clone()))
            .additional_params(serde_json::json!({
                "reasoning": { "effort": "high" }
            }))
            .build();

        let stream = agent
            .stream_chat(reasoning::TOOL_USER_PROMPT, Vec::<Message>::new())
            .max_turns(3)
            .await;

        let stats = reasoning::collect_stream_stats(stream, "openai").await;
        reasoning::assert_universal(&stats, &call_count, "openai");

        if stats.reasoning_block_count > 0 {
            assert!(
                stats.reasoning_has_encrypted || stats.reasoning_content_types.contains(&"Summary"),
                "[openai] Expected encrypted or summary reasoning content. Got: {:?}",
                stats.reasoning_content_types
            );
        }
    })
    .await;
}

#[tokio::test]
async fn nonstreaming() {
    with_openai_cassette(
        "reasoning_tool_roundtrip/nonstreaming",
        |client| async move {
            let call_count = Arc::new(AtomicUsize::new(0));
            let agent = client
                .agent("gpt-5.2")
                .preamble(reasoning::TOOL_SYSTEM_PROMPT)
                .max_tokens(4096)
                .tool(WeatherTool::new(call_count.clone()))
                .additional_params(serde_json::json!({
                    "reasoning": { "effort": "high" }
                }))
                .default_max_turns(2)
                .build();

            let result = agent
                .chat(reasoning::TOOL_USER_PROMPT, &mut Vec::<Message>::new())
                .await
                .expect("[openai] Non-streaming chat failed - likely 400 from dropped reasoning");

            reasoning::assert_nonstreaming_universal(&result, &call_count, "openai");
        },
    )
    .await;
}

#[tokio::test]
async fn bevy_local_executes_portable_tool_roundtrip() {
    with_openai_cassette(
        "reasoning_tool_roundtrip/nonstreaming",
        |client| async move {
            let call_count = Arc::new(AtomicUsize::new(0));
            let model = client.completion_model("gpt-5.2");
            let request = model
                .completion_request(reasoning::TOOL_USER_PROMPT)
                .preamble(reasoning::TOOL_SYSTEM_PROMPT.to_string())
                .max_tokens(4096)
                .additional_params(serde_json::json!({
                    "reasoning": { "effort": "high" }
                }))
                .build();
            let mut runtime = LocalRuntime::new(model, TenantId::new());
            let result = runtime
                .run_with_tools(
                    request,
                    2,
                    vec![Arc::new(PortableTool::new(WeatherTool::new(
                        call_count.clone(),
                    )))],
                )
                .await
                .expect("Bevy tool roundtrip");
            assert_eq!(call_count.load(std::sync::atomic::Ordering::SeqCst), 1);
            assert!(!result.snapshot.output.is_empty());
            let _: openai::responses_api::CompletionResponse = result.raw_response;
        },
    )
    .await;
}
