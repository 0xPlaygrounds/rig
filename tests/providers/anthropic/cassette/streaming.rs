//! Anthropic streaming smoke test.

use rig::completion::GetTokenUsage;
use rig::prelude::*;
use rig::providers::anthropic;
use rig::streaming::StreamingPrompt;

use super::super::support::with_anthropic_cassette;
use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, assert_nonempty_response,
    collect_stream_final_response_and_provider_final,
};

#[tokio::test]
async fn streaming_smoke() {
    with_anthropic_cassette("streaming/streaming_smoke", |client| async move {
        let agent = client
            .agent(anthropic::completion::CLAUDE_SONNET_4_6)
            .preamble(STREAMING_PREAMBLE)
            .build();

        let mut stream = agent.stream_prompt(STREAMING_PROMPT).await;
        let (response, provider_final): (_, anthropic::streaming::StreamingCompletionResponse) =
            collect_stream_final_response_and_provider_final(&mut stream)
                .await
                .expect("streaming prompt should succeed");

        assert_nonempty_response(&response);
        assert!(provider_final.token_usage().total_tokens > 0);
    })
    .await;
}

/// Regression: the streaming final response must carry the metadata Anthropic
/// puts on `message_delta`.
///
/// Two things used to be dropped. `stop_reason` never reached the consumer at
/// all, so a `max_tokens` truncation was indistinguishable from a normal stop.
/// And `input_tokens` was read only from `message_start`, which
/// Anthropic-compatible gateways report as `0`, hiding the real prompt size
/// that the provider sends on `message_delta`.
///
/// The cassette reproduces exactly that shape: `message_start` carries
/// `input_tokens: 0` while `message_delta` carries `stop_reason: "max_tokens"`
/// and `input_tokens: 9`.
#[tokio::test]
async fn streaming_metadata_from_message_delta() {
    with_anthropic_cassette(
        "streaming/streaming_metadata_from_message_delta",
        |client| async move {
            let agent = client
                .agent(anthropic::completion::CLAUDE_SONNET_4_6)
                .preamble(STREAMING_PREAMBLE)
                .build();

            let mut stream = agent.stream_prompt(STREAMING_PROMPT).await;
            let (response, provider_final): (_, anthropic::streaming::StreamingCompletionResponse) =
                collect_stream_final_response_and_provider_final(&mut stream)
                    .await
                    .expect("streaming prompt should succeed");

            assert_nonempty_response(&response);
            assert_eq!(
                provider_final.stop_reason.as_deref(),
                Some("max_tokens"),
                "stop_reason from message_delta must reach the consumer"
            );
            assert_eq!(
                provider_final.usage.input_tokens,
                Some(9),
                "input_tokens must come from message_delta when the provider sends it there"
            );
        },
    )
    .await;
}
