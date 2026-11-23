//! Perplexity prompt-caching cassette suite.
//!
//! Perplexity does no meaningful prefix caching: a long prompt re-sent byte-identically reads
//! zero cached tokens on every turn, and its search-grounded API documents no prompt cache. The
//! cells assert that absence through `assert_no_meaningful_prefix_cache`, which fails the day
//! the provider ships real prefix caching, so the suite can move to `assert_cache_conformance`.
//! Perplexity streams the shared chat-completions usage, so rig has a slot for cached tokens
//! regardless.
//!
//! There is no `agent_loop` cell: the search-grounded model answers the probe's prompt directly
//! and never calls the tool, and a one-call run leaves `assert_prefix_stable` no pair to
//! compare. The corpus-wide scan in `tests/cassette_cache_prefix.rs` covers the loop-level
//! guarantee instead.

use rig::providers::perplexity;

use crate::cache_conformance::{
    CacheProbe, CacheSupport, assert_no_meaningful_prefix_cache, assert_prefix_stable,
    run_cache_probe, run_cache_probe_streaming,
};

use super::super::support::with_perplexity_cassette;

const CACHE_MODEL: &str = perplexity::SONAR;

const PERPLEXITY_CACHE_SUPPORT: CacheSupport = CacheSupport {
    provider: "perplexity",
    explicit_breakpoints: false,
    reports_writes: false,
    min_cacheable_tokens: 1024,
    cache_key_field: None,
    hit_ratio_floor: 0.80,
};

fn probe() -> CacheProbe {
    CacheProbe::new("perplexity prompt caching")
}

#[tokio::test]
async fn blocking_probe_observes_no_meaningful_prefix_cache() {
    const SCENARIO: &str = "prompt_caching/blocking_probe";

    with_perplexity_cassette("prompt_caching/blocking_probe", |client| async move {
        let model = client.completion(CACHE_MODEL);
        let observation = run_cache_probe(model, &probe()).await;
        assert_no_meaningful_prefix_cache(
            &observation,
            &PERPLEXITY_CACHE_SUPPORT,
            "blocking probe",
        );
    })
    .await;

    assert_prefix_stable("perplexity", SCENARIO);
}

#[tokio::test]
async fn streaming_probe_observes_no_meaningful_prefix_cache() {
    const SCENARIO: &str = "prompt_caching/streaming_probe";

    with_perplexity_cassette("prompt_caching/streaming_probe", |client| async move {
        let model = client.completion(CACHE_MODEL);
        let observation = run_cache_probe_streaming(model, &probe()).await;
        assert_no_meaningful_prefix_cache(
            &observation,
            &PERPLEXITY_CACHE_SUPPORT,
            "streaming probe",
        );
    })
    .await;

    assert_prefix_stable("perplexity", SCENARIO);
}
