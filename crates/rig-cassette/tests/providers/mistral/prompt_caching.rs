//! Mistral prompt-caching cassette suite.
//!
//! Mistral's prompt cache is real but intermittent: identical requests over the same prefix hit
//! on some passes and miss on others, plausibly from routing without cache affinity. Rig does
//! not control that, and `assert_prefix_stable` proves on every run that rig's own bytes did not
//! move. So these cells do not assert the strict three-turn conformance, which would be a coin
//! flip. They assert that when Mistral reports a cache read, rig surfaces it at full magnitude
//! on the blocking and streaming paths alike.
//!
//! Rig reads the count from `prompt_tokens_details.cached_tokens`, falling back to the
//! top-level `num_cached_tokens` (`Usage::to_normalized` in the OpenAI completion module); it is
//! a subset of `prompt_tokens`, so the denominator is `input_tokens`.
//!
//! There is no `agent_loop` cell: the model returns an empty response through the agent
//! surface with the probe's small output budget, and a one-call run leaves
//! `assert_prefix_stable` no pair to compare. The corpus-wide scan in
//! `tests/cassette_cache_prefix.rs` covers the loop-level guarantee instead.

use rig::providers::mistral;

use crate::cache_conformance::{
    CacheProbe, CacheSupport, assert_cache_read_is_surfaced, assert_prefix_stable, run_cache_probe,
    run_cache_probe_streaming,
};

use super::support::with_mistral_cassette;

const CACHE_MODEL: &str = mistral::MISTRAL_LARGE;

const MISTRAL_CACHE_SUPPORT: CacheSupport = CacheSupport {
    provider: "mistral",
    explicit_breakpoints: false,
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

    with_mistral_cassette("prompt_caching/blocking_probe", |client| async move {
        let model = client.completion(CACHE_MODEL);
        let observation = run_cache_probe(model, &probe()).await;
        assert_cache_read_is_surfaced(&observation, &MISTRAL_CACHE_SUPPORT, "blocking probe");
    })
    .await;

    assert_prefix_stable("mistral", SCENARIO);
}

#[tokio::test]
async fn streaming_probe_surfaces_the_cache_read_mistral_reports() {
    const SCENARIO: &str = "prompt_caching/streaming_probe";

    with_mistral_cassette("prompt_caching/streaming_probe", |client| async move {
        let model = client.completion(CACHE_MODEL);
        let observation = run_cache_probe_streaming(model, &probe()).await;
        assert_cache_read_is_surfaced(&observation, &MISTRAL_CACHE_SUPPORT, "streaming probe");
    })
    .await;

    assert_prefix_stable("mistral", SCENARIO);
}
