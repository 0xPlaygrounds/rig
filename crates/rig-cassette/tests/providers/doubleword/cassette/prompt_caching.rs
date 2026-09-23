//! Doubleword prompt-caching cassette suite.
//!
//! Doubleword serves an OpenAI-compatible wire and reports cache hits in
//! `prompt_tokens_details.cached_tokens`. It did no measurable prefix caching
//! when this suite was first recorded; re-recorded, every turn of a
//! 4,865-token probe read 4,752 cached tokens (97.7%), turn one included,
//! because the probe's prefix was already warm from an earlier run. The
//! probes assert the full conformance suite.
//!
//! # Recording
//!
//! ```text
//! RIG_PROVIDER_TEST_MODE=record cargo test -p rig --test doubleword --all-features \
//!     prompt_caching:: -- --exact --test-threads=1
//! ```

use rig::prelude::*;
use rig::providers::doubleword;

use crate::cache_conformance::{
    AGENT_CACHE_PROMPT, CacheAccounting, CacheProbe, CacheProbeLookupTool, CacheSupport,
    assert_cache_conformance, assert_prefix_stable, run_cache_probe, run_cache_probe_streaming,
};

use super::super::support::with_doubleword_prompt_caching_cassette;

const CACHE_MODEL: &str = doubleword::QWEN3_5_9B;

const DOUBLEWORD_CACHE_SUPPORT: CacheSupport = CacheSupport {
    provider: "doubleword",
    accounting: CacheAccounting::Subset,
    explicit_breakpoints: false,
    reports_writes: false,
    min_cacheable_tokens: 1024,
    cache_key_field: None,
    hit_ratio_floor: 0.80,
};

fn probe() -> CacheProbe {
    CacheProbe::new("doubleword prompt caching")
}

#[tokio::test]
async fn blocking_probe_hits_and_keeps_hitting_as_the_prefix_grows() {
    const SCENARIO: &str = "prompt_caching/blocking_probe";

    with_doubleword_prompt_caching_cassette("prompt_caching/blocking_probe", |client| async move {
        let model = client.completion(CACHE_MODEL);
        let observation = run_cache_probe(&model, &probe()).await;
        assert_cache_conformance(&observation, &DOUBLEWORD_CACHE_SUPPORT, "blocking probe");
    })
    .await;

    assert_prefix_stable("doubleword", SCENARIO);
}

#[tokio::test]
async fn streaming_probe_survives_the_streaming_accumulator() {
    const SCENARIO: &str = "prompt_caching/streaming_probe";

    with_doubleword_prompt_caching_cassette(
        "prompt_caching/streaming_probe",
        |client| async move {
            let model = client.completion(CACHE_MODEL);
            let observation = run_cache_probe_streaming(&model, &probe()).await;
            assert_cache_conformance(&observation, &DOUBLEWORD_CACHE_SUPPORT, "streaming probe");
        },
    )
    .await;

    assert_prefix_stable("doubleword", SCENARIO);
}

/// A real agent loop with a tool round-trip, asserted on **prefix stability
/// alone**.
///
/// The probes above assert the cache itself. What this cell adds, and what
/// nothing else in the suite covers, is the
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
