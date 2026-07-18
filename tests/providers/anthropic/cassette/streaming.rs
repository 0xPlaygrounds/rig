//! Anthropic streaming smoke test.

use rig::client::CompletionClient;
use rig::completion::CompletionModel;
use rig::prelude::AgentClientExt;
use rig::providers::anthropic;
use rig::streaming::StreamingPrompt;
use rig_bevy::{LocalRuntime, TenantId};

use super::super::support::with_anthropic_cassette;
use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, assert_nonempty_response, collect_stream_final_response,
};

#[tokio::test]
async fn streaming_smoke() {
    with_anthropic_cassette("streaming/streaming_smoke", |client| async move {
        let agent = client
            .agent(anthropic::completion::CLAUDE_SONNET_4_6)
            .preamble(STREAMING_PREAMBLE)
            .build();

        let mut stream = agent.stream_prompt(STREAMING_PROMPT).await;
        let response = collect_stream_final_response(&mut stream)
            .await
            .expect("streaming prompt should succeed");

        assert_nonempty_response(&response);
    })
    .await;
}

#[tokio::test]
async fn bevy_local_streaming_preserves_raw_final() {
    with_anthropic_cassette("streaming/streaming_smoke", |client| async move {
        let model = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6);
        let request = model
            .completion_request(STREAMING_PROMPT)
            .preamble(STREAMING_PREAMBLE.to_string())
            .build();
        let mut runtime = LocalRuntime::new(model, TenantId::new());
        let result = runtime.stream(request, 1).await.expect("Bevy stream");
        let _: anthropic::streaming::StreamingCompletionResponse = result.raw_response;
        assert!(!result.provisional.is_empty());
    })
    .await;
}
