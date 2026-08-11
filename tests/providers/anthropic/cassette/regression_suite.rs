//! Live regression cassettes for shipped Anthropic fixes whose *premise* was
//! previously unrecorded.
//!
//! See `many_rigs/rig-regression-cassette-suite-proposal.md` for the catalogue
//! and the rule these follow: pin the premise a fix's comment rests on, not only
//! the behavior the fix produces. A test that asserts only the behavior keeps
//! passing for the wrong reason the moment the premise changes.

use rig::completion::FinishReason;
use rig::prelude::*;
use rig::providers::anthropic;
use rig::streaming::StreamingPrompt;

use super::super::support::with_anthropic_cassette;
use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, collect_stream_final_response_and_provider_final,
};

/// A3 — Regression: a `max_tokens` truncation reaches the consumer as
/// [`FinishReason::Length`].
///
/// Issue #2235 had two halves; the `input_tokens` half is covered in
/// `streaming.rs`. This is the other: `stop_reason` never reached the consumer,
/// so a truncated turn was indistinguishable from a natural stop — a consumer
/// deciding whether to continue generation had nothing to branch on.
///
/// `max_tokens` is capped hard so the model cannot finish the answer.
#[tokio::test]
async fn max_tokens_truncation_surfaces_as_length() {
    with_anthropic_cassette("regression/stop_reason_max_tokens", |client| async move {
        let agent = client
            .agent(anthropic::completion::CLAUDE_SONNET_4_6)
            .preamble(STREAMING_PREAMBLE)
            .max_tokens(8)
            .build();

        let mut stream = agent
            .stream_prompt("Write a detailed five paragraph essay about the ocean.")
            .await;
        let (_response, provider_final): (_, rig::streaming::StreamFinal) =
            collect_stream_final_response_and_provider_final(&mut stream)
                .await
                .expect("streaming prompt should succeed");

        assert_eq!(
            provider_final.finish_reason,
            Some(FinishReason::Length),
            "a max_tokens truncation must be distinguishable from a natural stop"
        );
    })
    .await;
}

/// A4 — Regression: a natural stop normalizes to [`FinishReason::Stop`].
///
/// The counterpart to A3. Without it, A3 alone cannot tell you that `Length` is
/// *specific* to truncation — a wire change mapping every terminal to `Length`
/// would keep A3 green.
#[tokio::test]
async fn natural_stop_surfaces_as_stop() {
    with_anthropic_cassette("regression/stop_reason_end_turn", |client| async move {
        let agent = client
            .agent(anthropic::completion::CLAUDE_SONNET_4_6)
            .preamble(STREAMING_PREAMBLE)
            .max_tokens(512)
            .build();

        let mut stream = agent.stream_prompt(STREAMING_PROMPT).await;
        let (_response, provider_final): (_, rig::streaming::StreamFinal) =
            collect_stream_final_response_and_provider_final(&mut stream)
                .await
                .expect("streaming prompt should succeed");

        assert_eq!(
            provider_final.finish_reason,
            Some(FinishReason::Stop),
            "a turn that ended on its own must not report truncation"
        );
    })
    .await;
}
