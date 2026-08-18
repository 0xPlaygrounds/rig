//! Doubleword prompt-caching cassette suite.
//!
//! **This provider does not do meaningful prefix caching**, and this suite is
//! what records that as a measured fact rather than an assumption.
//!
//! Measured: a 4,865-token prompt re-sent byte-identically reads **zero** cached
//! tokens on every turn. Doubleword serves an OpenAI-compatible wire and rig
//! would surface `prompt_tokens_details.cached_tokens` if it were populated.
//!
//! The cells assert the *absence* through `assert_no_meaningful_prefix_cache`,
//! which is self-invalidating: the day this provider ships real prefix caching,
//! the assertion fails and tells whoever is looking to replace it with the full
//! `assert_cache_conformance` suite and drop the coverage opt-out.
//!
//! Doubleword serves an OpenAI-compatible wire, so a cache hit would arrive in
//! `prompt_tokens_details.cached_tokens`.
//!
//! # Recording
//!
//! ```text
//! RIG_PROVIDER_TEST_MODE=record cargo test -p rig --test doubleword --all-features \
//!     prompt_caching:: -- --exact --test-threads=1
//! ```

use rig::client::CompletionClient as _;
use rig::prelude::*;
use rig::providers::doubleword;

use crate::cache_conformance::{
    AGENT_CACHE_PROMPT, CacheAccounting, CacheProbe, CacheProbeLookupTool, CacheSupport,
    assert_no_meaningful_prefix_cache, assert_prefix_stable, run_cache_probe,
    run_cache_probe_streaming,
};

use super::super::support::with_doubleword_prompt_caching_cassette;

const CACHE_MODEL: &str = doubleword::QWEN3_5_9B;

const DOUBLEWORD_CACHE_SUPPORT: CacheSupport = CacheSupport {
    provider: "doubleword",
    accounting: CacheAccounting::Subset,
    reports_writes: false,
    min_cacheable_tokens: 1024,
    cache_key_field: None,
    hit_ratio_floor: 0.80,
};

fn probe() -> CacheProbe {
    CacheProbe::new("doubleword prompt caching")
}

#[tokio::test]
async fn blocking_probe_observes_no_meaningful_prefix_cache() {
    const SCENARIO: &str = "prompt_caching/blocking_probe";

    with_doubleword_prompt_caching_cassette("prompt_caching/blocking_probe", |client| async move {
        let model = client.completion_model(CACHE_MODEL);
        let observation = run_cache_probe(&model, &probe()).await;
        assert_no_meaningful_prefix_cache(
            &observation,
            &DOUBLEWORD_CACHE_SUPPORT,
            "blocking probe",
        );
    })
    .await;

    assert_prefix_stable("doubleword", SCENARIO);
}

#[tokio::test]
async fn streaming_probe_observes_no_meaningful_prefix_cache() {
    const SCENARIO: &str = "prompt_caching/streaming_probe";

    with_doubleword_prompt_caching_cassette(
        "prompt_caching/streaming_probe",
        |client| async move {
            let model = client.completion_model(CACHE_MODEL);
            let observation = run_cache_probe_streaming(&model, &probe()).await;
            assert_no_meaningful_prefix_cache(
                &observation,
                &DOUBLEWORD_CACHE_SUPPORT,
                "streaming probe",
            );
        },
    )
    .await;

    assert_prefix_stable("doubleword", SCENARIO);
}

/// A real agent loop with a tool round-trip, asserted on **prefix stability
/// alone**.
///
/// This provider's cache is intermittent or absent (see the module docs), so a
/// cache-growth assertion here would be a coin flip or trivially unsatisfiable.
/// What the cell still buys — and what nothing else in the suite covers — is the
/// loop-level guarantee: across a real multi-turn agent run with a tool
/// round-trip, every outbound request must *extend* its predecessor rather than
/// rewrite it. A driver that re-advertises tools in a different order, rebuilds
/// the system prompt on turn N, or re-normalizes an earlier assistant turn would
/// bust caching on every provider that does cache, and `assert_prefix_stable`
/// catches that here without depending on this provider's hit rate at all.
#[tokio::test]
async fn agent_loop_does_not_move_its_own_prefix() {
    const SCENARIO: &str = "prompt_caching/agent_loop";

    with_doubleword_prompt_caching_cassette("prompt_caching/agent_loop", |client| async move {
        let response = client
            .agent(CACHE_MODEL)
            .preamble(&probe().preamble)
            .tool(CacheProbeLookupTool)
            .temperature(0.0)
            .build()
            .prompt(AGENT_CACHE_PROMPT)
            .max_turns(6)
            .extended_details()
            .await
            .expect("doubleword agent cache probe should complete");

        assert!(
            response.completion_calls().len() >= 2,
            "[doubleword] agent loop: the run made {} completion calls; a tool round-trip is at \
             least two, so the model never called the tool and the prefix never grew",
            response.completion_calls().len()
        );
    })
    .await;

    assert_prefix_stable("doubleword", SCENARIO);
}
