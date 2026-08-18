//! Mistral prompt-caching cassette suite.
//!
//! Mistral's prompt cache is real but **intermittent**, and saying so precisely
//! is the point of this suite.
//!
//! Across repeated live passes over an identical 1,783-token prefix, with rig
//! sending byte-identical requests every time, Mistral produced: a hit on turn 2
//! (1,760 tokens — a 98.7% ratio), on another pass a hit on turn 1 only, and on
//! another no hit at all. The plausible cause is request routing without cache
//! affinity; either way it is not something rig controls, and
//! `assert_prefix_stable` proves on every run that rig's own bytes did not move.
//!
//! So these cells deliberately do **not** assert the strict three-turn
//! conformance — that would be a coin flip, and re-recording until a good run
//! landed would be cherry-picking. They assert the narrower thing that is true
//! and deterministic: when Mistral reports a cache read, rig surfaces it at full
//! magnitude, on the blocking and streaming paths alike. That is a claim about
//! rig's usage mapping rather than about Mistral's hit rate.
//!
//! Rig reads the count from `prompt_tokens_details.cached_tokens`, falling back
//! to the top-level `num_cached_tokens`
//! (`crates/rig-core/src/providers/mistral/client.rs`, `Usage::cached_tokens`);
//! it is a subset of `prompt_tokens`, so the denominator is `input_tokens`.
//!
//! # No agent-loop cell, and why
//!
//! Most providers in the matrix have an `agent_loop` cell driving a real
//! multi-turn agent run with a tool round-trip — the only cell that can catch
//! rig's *driver* disturbing the prefix between iterations. This provider has
//! none, deliberately: its model returns an empty response ("Response contained no message or tool call") when driven through the agent surface with the probe's small output budget. A run that
//! makes one model call records one request, which leaves `assert_prefix_stable`
//! no pair to compare and the cell proving nothing. A cell that cannot fail is
//! worse than an acknowledged gap.
//!
//! The loop-level guarantee is still covered for this provider by the
//! corpus-wide scan in `tests/cassette_cache_prefix.rs`, which compares every
//! recorded multi-turn conversation in its existing suites.
//!
//! # Recording
//!
//! ```text
//! RIG_PROVIDER_TEST_MODE=record cargo test -p rig --test mistral --all-features \
//!     prompt_caching:: -- --exact --test-threads=1
//! ```

use rig::client::CompletionClient as _;
use rig::providers::mistral;

use crate::cache_conformance::{
    CacheAccounting, CacheProbe, CacheSupport, assert_cache_read_is_surfaced, assert_prefix_stable,
    run_cache_probe, run_cache_probe_streaming,
};

use super::support::with_mistral_prompt_caching_cassette;

const CACHE_MODEL: &str = mistral::MISTRAL_LARGE;

const MISTRAL_CACHE_SUPPORT: CacheSupport = CacheSupport {
    provider: "mistral",
    accounting: CacheAccounting::Subset,
    reports_writes: false,
    min_cacheable_tokens: 1024,
    cache_key_field: None,
    hit_ratio_floor: 0.80,
};

/// Mistral's rate limit rejects three back-to-back turns of the default
/// ~4,600-token probe with `429 rate_limited`. Sixty repetitions puts each turn
/// near 1,600 tokens — still comfortably past the 1,024-token floor any prompt
/// cache would need, so "no caching observed" remains a meaningful result.
const MISTRAL_PADDING_REPETITIONS: usize = 60;

fn probe() -> CacheProbe {
    CacheProbe::new("mistral prompt caching")
        .with_padding(MISTRAL_PADDING_REPETITIONS, "mistral prompt caching")
}

#[tokio::test]
async fn blocking_probe_surfaces_the_cache_read_mistral_reports() {
    const SCENARIO: &str = "prompt_caching/blocking_probe";

    with_mistral_prompt_caching_cassette("prompt_caching/blocking_probe", |client| async move {
        let model = client.completion_model(CACHE_MODEL);
        let observation = run_cache_probe(&model, &probe()).await;
        assert_cache_read_is_surfaced(&observation, &MISTRAL_CACHE_SUPPORT, "blocking probe");
    })
    .await;

    assert_prefix_stable("mistral", SCENARIO);
}

#[tokio::test]
async fn streaming_probe_surfaces_the_cache_read_mistral_reports() {
    const SCENARIO: &str = "prompt_caching/streaming_probe";

    with_mistral_prompt_caching_cassette("prompt_caching/streaming_probe", |client| async move {
        let model = client.completion_model(CACHE_MODEL);
        let observation = run_cache_probe_streaming(&model, &probe()).await;
        assert_cache_read_is_surfaced(&observation, &MISTRAL_CACHE_SUPPORT, "streaming probe");
    })
    .await;

    assert_prefix_stable("mistral", SCENARIO);
}
