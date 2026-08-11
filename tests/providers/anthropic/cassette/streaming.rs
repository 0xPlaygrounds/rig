//! Anthropic streaming smoke test.

use rig::prelude::*;
use rig::providers::anthropic;
use rig::streaming::StreamingPrompt;

use super::super::support::{with_anthropic_cassette, with_anthropic_gateway_cassette};
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
        let (response, provider_final): (_, rig::streaming::StreamFinal) =
            collect_stream_final_response_and_provider_final(&mut stream)
                .await
                .expect("streaming prompt should succeed");

        assert_nonempty_response(&response);
        assert_eq!(provider_final.provider, "anthropic");
        assert!(provider_final.usage.total_tokens > 0);
    })
    .await;
}

/// Regression: the streamed terminal must carry the metadata the provider puts
/// on `message_delta`.
///
/// Two signals used to be dropped. `stop_reason` never reached the consumer, so
/// a `max_tokens` truncation was indistinguishable from a natural stop. And
/// `input_tokens` was read only from `message_start`, which Anthropic-compatible
/// gateways may report as `0` while sending the real prompt size on
/// `message_delta`, silently yielding `Usage { input_tokens: 0 }`.
///
/// Recorded against OpenRouter's Anthropic Messages endpoint rather than
/// `api.anthropic.com`, because the *disagreement* is what Anthropic proper does
/// not produce: it reports the count on both frames and they always match (see
/// every streaming cassette under `tests/cassettes/anthropic/`), so a recording
/// from Anthropic passes whether or not the bug is present and cannot witness
/// it. `max_tokens` is capped low so one recording carries both signals.
#[tokio::test]
async fn gateway_reports_input_tokens_on_message_delta() {
    with_anthropic_gateway_cassette(
        "streaming/gateway_message_delta_metadata",
        |client| async move {
            let agent = client
                .agent("anthropic/claude-haiku-4.5")
                .preamble(STREAMING_PREAMBLE)
                .max_tokens(16)
                .build();

            let mut stream = agent.stream_prompt(STREAMING_PROMPT).await;
            let (_response, provider_final): (_, rig::streaming::StreamFinal) =
                collect_stream_final_response_and_provider_final(&mut stream)
                    .await
                    .expect("streaming prompt should succeed");

            assert_eq!(provider_final.provider, "anthropic");
            assert_eq!(
                provider_final.finish_reason,
                Some(rig::completion::FinishReason::Length),
                "a max_tokens truncation must be distinguishable from a natural stop"
            );
            assert_eq!(
                provider_final.usage.input_tokens, 32,
                "the prompt size the gateway reported on message_delta must reach the consumer"
            );
        },
    )
    .await;
}
