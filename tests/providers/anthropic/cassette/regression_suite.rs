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

/// A5 — Regression: on a cache-hit turn, `input_tokens` is the **uncached
/// remainder**, not the prompt size.
///
/// `anthropic/streaming.rs` filters `input_tokens` on the terminal
/// `message_delta` with `> 0`, and its comment states plainly that this is a
/// *heuristic, not an invariant*: with prompt caching, a turn whose prefix is
/// fully cached legitimately bills few (or zero) uncached input tokens while the
/// real prompt size sits in `cache_read_input_tokens`. The comment then warns
/// against extending the `> 0` filter to `message_start` or the cache fields,
/// where a genuine zero would be discarded.
///
/// Nothing pinned that semantic. This records a cached streamed turn and asserts
/// the relationship the comment depends on — cached tokens dominate, and
/// `input_tokens` is much smaller than the true prompt — so a future change that
/// treated `input_tokens` as "the prompt size" has something to fail against.
///
/// Deliberately asserts the *relationship*, not a literal: the exact split is
/// Anthropic's to choose and a re-record must not turn this into a tautology.
#[tokio::test]
async fn cache_hit_turn_reports_uncached_remainder_not_prompt_size() {
    with_anthropic_cassette(
        "regression/cache_hit_zero_uncached_input",
        |client| async move {
            let model = client
                .completion_model(anthropic::completion::CLAUDE_SONNET_4_6)
                .with_prompt_caching();

            // A prefix long enough to clear Anthropic's minimum cacheable size.
            let padding = std::iter::repeat_n(
                "This cache fixture paragraph is stable provider test padding about request \
                 routing, tool schemas, system instructions, and deterministic replay behavior.",
                180,
            )
            .collect::<Vec<_>>()
            .join(" ");

            let send = |model: anthropic::completion::CompletionModel<_>, padding: String| async move {
                let agent = rig::agent::AgentBuilder::new(model)
                    .preamble(&padding)
                    .max_tokens(32)
                    .build();
                let mut stream = agent
                    .stream_prompt("Reply with exactly: cache probe ready")
                    .await;
                let (_text, provider_final): (_, rig::streaming::StreamFinal) =
                    collect_stream_final_response_and_provider_final(&mut stream)
                        .await
                        .expect("cached streaming prompt should succeed");
                provider_final
            };

            // Turn 1 writes the cache; turn 2 reads it.
            let _first = send(model.clone(), padding.clone()).await;
            let second = send(model, padding).await;

            assert!(
                second.usage.cached_input_tokens > 0,
                "the second turn must read the cache for this fixture to say anything \
                 about cache-hit accounting; usage: {:?}",
                second.usage
            );
            assert!(
                second.usage.cached_input_tokens > second.usage.input_tokens,
                "on a cache-hit turn the cached prefix must dominate the uncached \
                 remainder — `input_tokens` is NOT the prompt size, which is exactly \
                 why the `> 0` filter in anthropic/streaming.rs is documented as a \
                 heuristic rather than an invariant; usage: {:?}",
                second.usage
            );
        },
    )
    .await;
}
