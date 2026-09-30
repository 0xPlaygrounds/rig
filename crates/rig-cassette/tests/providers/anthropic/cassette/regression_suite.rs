//! Live regression cassettes for shipped Anthropic fixes whose *premise* was
//! previously unrecorded.
//!
//! See `many_rigs/rig-regression-cassette-suite-proposal.md` for the catalogue
//! and the rule these follow: pin the premise a fix's comment rests on, not only
//! the behavior the fix produces. A test that asserts only the behavior keeps
//! passing for the wrong reason the moment the premise changes.

use rig::completion::FinishReason;
use rig::driver::Model;
use rig::providers::anthropic;
use rig::providers::anthropic::wire::Messages;

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
        let agent =
            rig::AgentBuilder::new(client.completion(anthropic::completion::CLAUDE_SONNET_4_6))
                .preamble(STREAMING_PREAMBLE)
                .max_tokens(8)
                .build();

        let mut stream = agent
            .prompt("Write a detailed five paragraph essay about the ocean.")
            .stream();
        let (_response, provider_final) =
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
        let agent =
            rig::AgentBuilder::new(client.completion(anthropic::completion::CLAUDE_SONNET_4_6))
                .preamble(STREAMING_PREAMBLE)
                .max_tokens(512)
                .build();

        let mut stream = agent.prompt(STREAMING_PROMPT).stream();
        let (_response, provider_final) =
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

/// A5 — Regression: on a cache-hit turn, rig's `input_tokens` is the **prompt
/// size**: Anthropic's uncached remainder plus the cache reads and writes it
/// reports beside it.
///
/// `anthropic/streaming.rs` filters the wire's `input_tokens` on the terminal
/// `message_delta` with `> 0`, and its comment states plainly that this is a
/// *heuristic, not an invariant*: with prompt caching, a turn whose prefix is
/// fully cached legitimately bills few (or zero) uncached input tokens while the
/// real prompt size sits in `cache_read_input_tokens`.
///
/// This records a cached streamed turn and asserts both halves: rig's input
/// covers the cache counters, and the cached prefix dominates the uncached
/// remainder, so a change that reported the remainder as the prompt size, or
/// dropped the cache from it, has something to fail against.
///
/// Deliberately asserts the *relationship*, not a literal: the exact split is
/// Anthropic's to choose and a re-record must not turn this into a tautology.
#[tokio::test]
async fn cache_hit_turn_reports_the_prompt_size_with_its_cached_prefix() {
    with_anthropic_cassette(
        "regression/cache_hit_zero_uncached_input",
        |client| async move {
            let model = client
                .completion(anthropic::completion::CLAUDE_SONNET_4_6)
                .map_wire(|wire| wire.with_prompt_caching());

            // A prefix long enough to clear Anthropic's minimum cacheable size.
            let padding = std::iter::repeat_n(
                "This cache fixture paragraph is stable provider test padding about request \
                 routing, tool schemas, system instructions, and deterministic replay behavior.",
                180,
            )
            .collect::<Vec<_>>()
            .join(" ");

            let send = |model: Model<Messages>, padding: String| async move {
                let agent = rig::agent::AgentBuilder::new(model)
                    .preamble(&padding)
                    .max_tokens(32)
                    .build();
                let mut stream = agent
                    .prompt("Reply with exactly: cache probe ready")
                    .stream();
                let (_text, provider_final) =
                    collect_stream_final_response_and_provider_final(&mut stream)
                        .await
                        .expect("cached streaming prompt should succeed");
                provider_final
            };

            // Turn 1 writes the cache; turn 2 reads it.
            let _first = send(model.clone(), padding.clone()).await;
            let second = send(model, padding).await;

            assert!(
                second.usage.cached_input_tokens.is_some_and(|n| n > 0),
                "the second turn must read the cache for this fixture to say anything \
                 about cache-hit accounting; usage: {:?}",
                second.usage
            );
            let input = second.usage.input_tokens.unwrap_or(0);
            let cached = second.usage.cached_input_tokens.unwrap_or(0);
            let written = second.usage.cache_creation_input_tokens.unwrap_or(0);
            assert!(
                cached + written <= input,
                "`input_tokens` is the prompt size, cache reads and writes included; \
                 usage: {:?}",
                second.usage
            );
            assert!(
                cached > input - cached - written,
                "on a cache-hit turn the cached prefix must dominate the uncached \
                 remainder, which is why the `> 0` filter in anthropic/streaming.rs is \
                 documented as a heuristic rather than an invariant; usage: {:?}",
                second.usage
            );
        },
    )
    .await;
}
