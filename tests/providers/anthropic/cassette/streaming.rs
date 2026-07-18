//! Anthropic streaming smoke test.

use rig::client::CompletionClient;
use rig::providers::anthropic;
use rig::streaming::StreamingPrompt;

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

#[cfg(feature = "bevy")]
#[tokio::test]
async fn bevy_streaming_emits_and_returns_typed_provider_final() {
    use rig::bevy::{AgentSpec, BevyRuntime, effects::SubscriptionEvent};

    with_anthropic_cassette("streaming/streaming_smoke", |client| async move {
        let runtime = BevyRuntime::default();
        let agent = runtime.spawn_agent(
            AgentSpec::new(client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6))
                .preamble(STREAMING_PREAMBLE),
        );
        let mut provider_finals = 0;

        let outcome = agent
            .stream_prompt(STREAMING_PROMPT, |event| {
                if matches!(event, SubscriptionEvent::ProviderFinal(_)) {
                    provider_finals += 1;
                }
            })
            .await
            .expect("Bevy stream should succeed");

        assert_eq!(provider_finals, 1);
        assert_nonempty_response(&format!("{:?}", outcome.choice));
        assert!(
            !format!("{:?}", outcome.provider_final).is_empty(),
            "Bevy local streaming should return the concrete Anthropic final"
        );
    })
    .await;
}
