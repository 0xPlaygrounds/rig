//! Perplexity prompt-caching cassette suite.
//!
//! **This provider does not do meaningful prefix caching**, and this suite is
//! what records that as a measured fact rather than an assumption.
//!
//! Measured: a 4,365-token prompt re-sent byte-identically reads **zero** cached
//! tokens on every turn — as expected, since Perplexity's search-grounded API
//! documents no prompt cache. Recorded rather than assumed so that the day it
//! gains one, this suite says so.
//!
//! The cells assert the *absence* through `assert_no_meaningful_prefix_cache`,
//! which is self-invalidating: the day this provider ships real prefix caching,
//! the assertion fails and tells whoever is looking to replace it with the full
//! `assert_cache_conformance` suite and drop the coverage opt-out.
//!
//! Perplexity reuses `openai::Usage` for streaming, so rig has a slot for cached
//! tokens even though Perplexity's search-grounded API documents no prompt cache.
//!
//! # Recording
//!
//! ```text
//! RIG_PROVIDER_TEST_MODE=record cargo test -p rig --test perplexity --all-features \
//!     prompt_caching:: -- --exact --test-threads=1
//! ```

use rig::client::CompletionClient as _;
use rig::providers::perplexity;

use crate::cache_conformance::{
    CacheAccounting, CacheProbe, CacheSupport, assert_no_meaningful_prefix_cache,
    assert_prefix_stable, run_cache_probe, run_cache_probe_streaming,
};

use super::super::support::with_perplexity_prompt_caching_cassette;

const CACHE_MODEL: &str = perplexity::SONAR;

const PERPLEXITY_CACHE_SUPPORT: CacheSupport = CacheSupport {
    provider: "perplexity",
    accounting: CacheAccounting::Subset,
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

    with_perplexity_prompt_caching_cassette("prompt_caching/blocking_probe", |client| async move {
        let model = client.completion_model(CACHE_MODEL);
        let observation = run_cache_probe(&model, &probe()).await;
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

    with_perplexity_prompt_caching_cassette(
        "prompt_caching/streaming_probe",
        |client| async move {
            let model = client.completion_model(CACHE_MODEL);
            let observation = run_cache_probe_streaming(&model, &probe()).await;
            assert_no_meaningful_prefix_cache(
                &observation,
                &PERPLEXITY_CACHE_SUPPORT,
                "streaming probe",
            );
        },
    )
    .await;

    assert_prefix_stable("perplexity", SCENARIO);
}
