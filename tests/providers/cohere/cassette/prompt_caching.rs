//! Cohere prompt-caching cassette suite.
//!
//! **This provider does not do meaningful prefix caching**, and this suite is
//! what records that as a measured fact rather than an assumption.
//!
//! Measured, and the single most instructive result in the whole matrix: Cohere
//! reports a **constant 112 cached tokens** against a 6,058-token prompt, on
//! every turn, whether or not the prefix repeats. That is a 1.8% hit ratio.
//!
//! It is non-zero — so the `cached_input_tokens > 0` assertion that this harness
//! exists to replace would pass, and would have reported Cohere as "caching
//! works" while 98% of every prompt was re-billed on every turn. The 112 tokens
//! do not grow with the prefix and do not change between an identical repeat and
//! a grown one, so they are not prefix caching.
//!
//! The cells assert the *absence* through `assert_no_meaningful_prefix_cache`,
//! which is self-invalidating: the day this provider ships real prefix caching,
//! the assertion fails and tells whoever is looking to replace it with the full
//! `assert_cache_conformance` suite and drop the coverage opt-out.
//!
//! Cohere documents `cached_tokens` as a subset of `tokens.input_tokens` and
//! excludes it from `billed_units` (`crates/rig-core/src/providers/cohere/completion.rs`).
//!
//! This suite is also the end-to-end regression cell for the `Document::data`
//! `HashMap` bug: document metadata used to serialize in a random key order on
//! every request, so any Cohere request carrying a document had a different
//! prefix every time.
//!
//! # Recording
//!
//! ```text
//! RIG_PROVIDER_TEST_MODE=record cargo test -p rig --test cohere --all-features \
//!     prompt_caching:: -- --exact --test-threads=1
//! ```

use rig::client::CompletionClient as _;
use rig::providers::cohere;

use crate::cache_conformance::{
    CacheAccounting, CacheProbe, CacheSupport, assert_no_meaningful_prefix_cache,
    assert_prefix_stable, run_cache_probe, run_cache_probe_streaming,
};

use super::super::support::with_cohere_prompt_caching_cassette;

const CACHE_MODEL: &str = cohere::COMMAND_A_03_2025;

const COHERE_CACHE_SUPPORT: CacheSupport = CacheSupport {
    provider: "cohere",
    accounting: CacheAccounting::Subset,
    reports_writes: false,
    explicit_breakpoints: false,
    min_cacheable_tokens: 1024,
    cache_key_field: None,
    hit_ratio_floor: 0.80,
};

fn probe() -> CacheProbe {
    CacheProbe::new("cohere prompt caching")
}

#[tokio::test]
async fn blocking_probe_observes_no_meaningful_prefix_cache() {
    const SCENARIO: &str = "prompt_caching/blocking_probe";

    with_cohere_prompt_caching_cassette("prompt_caching/blocking_probe", |client| async move {
        let model = client.completion_model(CACHE_MODEL);
        let observation = run_cache_probe(&model, &probe()).await;
        assert_no_meaningful_prefix_cache(&observation, &COHERE_CACHE_SUPPORT, "blocking probe");
    })
    .await;

    assert_prefix_stable("cohere", SCENARIO);
}

#[tokio::test]
async fn streaming_probe_observes_no_meaningful_prefix_cache() {
    const SCENARIO: &str = "prompt_caching/streaming_probe";

    with_cohere_prompt_caching_cassette("prompt_caching/streaming_probe", |client| async move {
        let model = client.completion_model(CACHE_MODEL);
        let observation = run_cache_probe_streaming(&model, &probe()).await;
        assert_no_meaningful_prefix_cache(&observation, &COHERE_CACHE_SUPPORT, "streaming probe");
    })
    .await;

    assert_prefix_stable("cohere", SCENARIO);
}
