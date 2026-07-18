//! Anthropic reasoning-enabled tool roundtrip tests.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use rig::client::CompletionClient;
use rig::completion::{Chat, CompletionModel, Message};
use rig::prelude::AgentClientExt;
use rig::providers::anthropic::{self, completion::CLAUDE_SONNET_4_6};
use rig::streaming::StreamingChat;
use rig_bevy::{LocalRuntime, PortableTool, TenantId};

use super::super::support::with_anthropic_cassette;
use crate::reasoning::{self, WeatherTool};

#[tokio::test]
async fn streaming() {
    with_anthropic_cassette("reasoning_tool_roundtrip/streaming", |client| async move {
        let call_count = Arc::new(AtomicUsize::new(0));
        let agent = client
            .agent(CLAUDE_SONNET_4_6)
            .preamble(reasoning::TOOL_SYSTEM_PROMPT)
            .max_tokens(16384)
            .tool(WeatherTool::new(call_count.clone()))
            .additional_params(serde_json::json!({
                "thinking": { "type": "adaptive" }
            }))
            .build();

        let stream = agent
            .stream_chat(reasoning::TOOL_USER_PROMPT, Vec::<Message>::new())
            .max_turns(3)
            .await;

        let stats = reasoning::collect_stream_stats(stream, "anthropic").await;
        reasoning::assert_universal(&stats, &call_count, "anthropic");

        if stats.reasoning_block_count > 0 {
            assert!(
                stats.reasoning_has_signature,
                "[anthropic] Thinking blocks should have signatures. Content types: {:?}",
                stats.reasoning_content_types
            );
            assert!(
                stats.reasoning_content_types.contains(&"Text"),
                "[anthropic] Expected text reasoning content. Got: {:?}",
                stats.reasoning_content_types
            );
        }
    })
    .await;
}

#[tokio::test]
async fn nonstreaming() {
    with_anthropic_cassette(
        "reasoning_tool_roundtrip/nonstreaming",
        |client| async move {
            let call_count = Arc::new(AtomicUsize::new(0));
            let agent = client
                .agent(CLAUDE_SONNET_4_6)
                .preamble(reasoning::TOOL_SYSTEM_PROMPT)
                .max_tokens(16384)
                .tool(WeatherTool::new(call_count.clone()))
                .additional_params(serde_json::json!({
                    "thinking": { "type": "adaptive" }
                }))
                .default_max_turns(2)
                .build();

            let result = agent
                .chat(reasoning::TOOL_USER_PROMPT, &mut Vec::<Message>::new())
                .await
                .expect(
                    "[anthropic] Non-streaming chat failed - likely 400 from dropped reasoning",
                );

            reasoning::assert_nonstreaming_universal(&result, &call_count, "anthropic");
        },
    )
    .await;
}

#[tokio::test]
async fn bevy_local_executes_portable_tool_roundtrip() {
    with_anthropic_cassette(
        "reasoning_tool_roundtrip/nonstreaming",
        |client| async move {
            let call_count = Arc::new(AtomicUsize::new(0));
            let model = client.completion_model(CLAUDE_SONNET_4_6);
            let request = model
                .completion_request(reasoning::TOOL_USER_PROMPT)
                .preamble(reasoning::TOOL_SYSTEM_PROMPT.to_string())
                .max_tokens(16384)
                .additional_params(serde_json::json!({
                    "thinking": { "type": "adaptive" }
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
            let _: anthropic::completion::CompletionResponse = result.raw_response;
        },
    )
    .await;
}
