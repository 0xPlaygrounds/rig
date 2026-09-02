//! Openrouter prompt-caching cassette suite.
//!
//! Rig normalizes `Usage::cached_input_tokens` for this provider, and before
//! this suite no test had ever seen the field be non-zero. The three-turn probe
//! below is what makes the provider's actual behavior observable — including the
//! possibility that it does not cache at all, which is recorded rather than
//! assumed.
//!
//! OpenRouter proxies to an upstream provider and is the one gateway in the
//! matrix that reports cache *writes* as well as reads
//! (`prompt_tokens_details.cache_write_tokens`, mapped in
//! `crates/rig-core/src/providers/openrouter/client.rs`). Routed here to an
//! OpenAI model, so the underlying cache is OpenAI's 1,024-token automatic one.
//!
//! # Recording
//!
//! ```text
//! RIG_PROVIDER_TEST_MODE=record cargo test -p rig --test openrouter --all-features \
//!     prompt_caching:: -- --exact --test-threads=1
//! ```

use rig::client::CompletionClient as _;
use rig::prelude::*;

use crate::cache_conformance::{
    AGENT_CACHE_PROMPT, CacheAccounting, CacheProbe, CacheProbeLookupTool, CacheSupport,
    assert_agent_growth_still_hits, assert_cache_conformance, assert_prefix_stable,
    observation_from_completion_calls, report_and_assert_live, run_cache_probe,
    run_cache_probe_streaming,
};

use super::super::support::with_openrouter_prompt_caching_cassette;

const CACHE_MODEL: &str = "openai/gpt-4o-mini";

const OPENROUTER_CACHE_SUPPORT: CacheSupport = CacheSupport {
    provider: "openrouter",
    accounting: CacheAccounting::Subset,
    // OpenRouter *can* report cache writes (`cache_write_tokens`), and does for
    // upstreams that bill them separately such as Anthropic. This scenario
    // routes to an OpenAI model, whose cache writes are free and unreported, so
    // turn 1 legitimately shows zero for both counters — measured, not assumed.
    explicit_breakpoints: false,
    reports_writes: false,
    min_cacheable_tokens: 1024,
    cache_key_field: None,
    hit_ratio_floor: 0.80,
};

fn probe() -> CacheProbe {
    CacheProbe::new("openrouter prompt caching")
}

#[tokio::test]
async fn blocking_probe_hits_and_keeps_hitting_as_the_prefix_grows() {
    const SCENARIO: &str = "prompt_caching/blocking_probe";

    with_openrouter_prompt_caching_cassette("prompt_caching/blocking_probe", |client| async move {
        let model = client.completion_model(CACHE_MODEL);
        let observation = run_cache_probe(&model, &probe()).await;
        assert_cache_conformance(&observation, &OPENROUTER_CACHE_SUPPORT, "blocking probe");
    })
    .await;

    assert_prefix_stable("openrouter", SCENARIO);
}

#[tokio::test]
async fn streaming_probe_survives_the_streaming_accumulator() {
    const SCENARIO: &str = "prompt_caching/streaming_probe";

    with_openrouter_prompt_caching_cassette(
        "prompt_caching/streaming_probe",
        |client| async move {
            let model = client.completion_model(CACHE_MODEL);
            let observation = run_cache_probe_streaming(&model, &probe()).await;
            assert_cache_conformance(&observation, &OPENROUTER_CACHE_SUPPORT, "streaming probe");
        },
    )
    .await;

    assert_prefix_stable("openrouter", SCENARIO);
}

/// Live economics: run the same probe against the real API.
///
/// A cassette pins what openrouter did at record time. Only a live run catches
/// openrouter changing its cache semantics under us — a shorter TTL, a higher
/// minimum, a different block granularity — which is exactly the kind of change
/// that costs money silently. `#[ignore]`d so it never runs in the key-free
/// gate; run it with `--ignored` and a key present.
#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY and spends real tokens"]
async fn live_cache_economics() {
    let client = rig::providers::openrouter::Client::from_env().expect("OPENROUTER_API_KEY");
    let model = client.completion_model(CACHE_MODEL);
    let observation = run_cache_probe(&model, &probe()).await;
    report_and_assert_live(
        &observation,
        &OPENROUTER_CACHE_SUPPORT,
        "live_cache_economics",
    );
}

/// A real agent loop with a tool round-trip.
///
/// The cell the three-turn probe cannot replace. The probe builds its own
/// history, so it proves the *provider* caches a growing prefix; only a real run
/// proves rig's agent driver does not disturb that prefix between iterations —
/// tool definitions re-advertised in a different order, a system prompt rebuilt
/// differently on turn N, an assistant turn re-normalized on the way back in.
/// Each busts the cache from that point on while every counter stays non-zero.
#[tokio::test]
async fn agent_loop_keeps_hitting_across_tool_turns() {
    const SCENARIO: &str = "prompt_caching/agent_loop";

    with_openrouter_prompt_caching_cassette("prompt_caching/agent_loop", |client| async move {
        let response = client
            .agent(CACHE_MODEL)
            .preamble(&probe().preamble)
            .tool(CacheProbeLookupTool)
            .temperature(0.0)
            .build()
            .prompt(AGENT_CACHE_PROMPT)
            .max_turns(6)
            .await
            .expect("openrouter agent cache probe should complete");

        let observation = observation_from_completion_calls(response.completion_calls());
        assert_agent_growth_still_hits(&observation, &OPENROUTER_CACHE_SUPPORT, "agent loop");
    })
    .await;

    assert_prefix_stable("openrouter", SCENARIO);
}
