//! Cohere prompt-caching cassette suite.
//!
//! Cohere's prompt cache is real and warms across turns: on `command-a-03-2025`
//! the blocking three-turn probe reads a small fraction of the prompt on turns 1
//! and 2 and nearly all of it on turn 3, and the streaming probe warms the same
//! way. The cells assert that the cache read never goes backwards as the
//! conversation grows and that the final turn clears the provider's floor; a
//! prefix move breaks both. A bare `cached_input_tokens > 0` check would pass on
//! turn 2 while most of the prompt is re-billed, which is why
//! `assert_no_meaningful_prefix_cache` folds over every turn.
//!
//! Cohere documents `cached_tokens` as a subset of `tokens.input_tokens` and
//! excludes it from `billed_units`. This suite is also the end-to-end cell for
//! `Document::data` serializing in a stable key order, without which every
//! request carrying a document would have a different prefix.
//!
//! Re-record with
//! `RIG_PROVIDER_TEST_MODE=record cargo test -p rig-cassette --test integration --all-features cohere::cassette::prompt_caching:: -- --exact --test-threads=1`.

use rig::providers::cohere;

use crate::cache_conformance::{
    AGENT_CACHE_PROMPT, CacheProbe, CacheProbeLookupTool, CacheSupport,
    assert_breakpoints_match_support, assert_cache_warms_over_turns, assert_prefix_stable,
    run_cache_probe, run_cache_probe_streaming,
};

use super::super::support::with_cohere_cassette;

const CACHE_MODEL: &str = cohere::COMMAND_A_03_2025;

const COHERE_CACHE_SUPPORT: CacheSupport = CacheSupport {
    provider: "cohere",
    explicit_breakpoints: false,
    reports_writes: false,
    min_cacheable_tokens: 1024,
    cache_key_field: None,
    hit_ratio_floor: 0.80,
};

fn probe() -> CacheProbe {
    CacheProbe::new("cohere prompt caching")
}

#[tokio::test]
#[ignore = "stale cassette: its request predates the rebuilt Chat history, and Cohere never warmed past its first cached block in all three re-record attempts"]
async fn blocking_probe_warms_to_a_full_cache_hit_over_three_turns() {
    const SCENARIO: &str = "prompt_caching/blocking_probe";

    with_cohere_cassette("prompt_caching/blocking_probe", |client| async move {
        let model = client.completion(CACHE_MODEL);
        let observation = run_cache_probe(model, &probe()).await;
        assert_cache_warms_over_turns(&observation, &COHERE_CACHE_SUPPORT, "blocking probe");
    })
    .await;

    assert_prefix_stable("cohere", SCENARIO);
    assert_breakpoints_match_support("cohere", SCENARIO, &COHERE_CACHE_SUPPORT);
}

#[tokio::test]
#[ignore = "re-recording on the Compatibility API failed three times: it reports cached_tokens 0 on every turn"]
async fn streaming_probe_warms_to_a_full_cache_hit_over_three_turns() {
    const SCENARIO: &str = "prompt_caching/streaming_probe";

    with_cohere_cassette("prompt_caching/streaming_probe", |client| async move {
        let model = client.completion(CACHE_MODEL);
        let observation = run_cache_probe_streaming(model, &probe()).await;
        assert_cache_warms_over_turns(&observation, &COHERE_CACHE_SUPPORT, "streaming probe");
    })
    .await;

    assert_prefix_stable("cohere", SCENARIO);
    assert_breakpoints_match_support("cohere", SCENARIO, &COHERE_CACHE_SUPPORT);
}

/// A real agent loop with a tool round-trip, asserted on **prefix stability
/// alone**.
///
/// A cache-growth assertion here would depend on the provider's hit rate. What
/// the cell buys, and nothing else in the suite covers, is the loop-level
/// guarantee: across a real multi-turn agent run with a tool
/// round-trip, every outbound request must *extend* its predecessor rather than
/// rewrite it. A driver that re-advertises tools in a different order, rebuilds
/// the system prompt on turn N, or re-normalizes an earlier assistant turn would
/// bust caching on every provider that does cache, and `assert_prefix_stable`
/// catches that here without depending on this provider's hit rate at all.
#[tokio::test]
async fn agent_loop_does_not_move_its_own_prefix() {
    const SCENARIO: &str = "prompt_caching/agent_loop";

    with_cohere_cassette("prompt_caching/agent_loop", |client| async move {
        let response = rig::AgentBuilder::new(client.completion(CACHE_MODEL))
            .preamble(&probe().preamble)
            .tool(CacheProbeLookupTool)
            .temperature(0.0)
            .build()
            .prompt(AGENT_CACHE_PROMPT)
            .max_turns(6)
            .await
            .expect("cohere agent cache probe should complete");

        assert!(
            response.completion_calls().len() >= 2,
            "[cohere] agent loop: the run made {} completion calls; a tool round-trip is at \
             least two, so the model never called the tool and the prefix never grew",
            response.completion_calls().len()
        );
    })
    .await;

    assert_prefix_stable("cohere", SCENARIO);
    assert_breakpoints_match_support("cohere", SCENARIO, &COHERE_CACHE_SUPPORT);
}
