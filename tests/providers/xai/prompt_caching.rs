//! Xai prompt-caching cassette suite.
//!
//! Rig normalizes `Usage::cached_input_tokens` for this provider, and before
//! this suite no test had ever seen the field be non-zero. The three-turn probe
//! below is what makes the provider's actual behavior observable — including the
//! possibility that it does not cache at all, which is recorded rather than
//! assumed.
//!
//! xAI reuses the OpenAI-compatible usage shape, so a cache hit would arrive in
//! `prompt_tokens_details.cached_tokens`.
//!
//! # No agent-loop cell, and why
//!
//! Every other caching provider in the matrix has an `agent_loop` cell that
//! drives a real multi-turn agent run with a tool round-trip — the only cell
//! that can catch rig's *driver* disturbing the prefix between iterations.
//! This provider has none, deliberately: its model answers the probe's prompt
//! from its own knowledge and never calls the tool, even with
//! `ToolChoice::Required`. A run that makes one model call records one request,
//! which leaves `assert_prefix_stable` no pair to compare and the cell proving
//! nothing. A cell that cannot fail is worse than an acknowledged gap.
//!
//! The loop-level guarantee is still covered for this provider by the
//! corpus-wide scan in `tests/cassette_cache_prefix.rs`, which compares every
//! recorded multi-turn conversation in its existing suites.
//!
//! # Recording
//!
//! ```text
//! RIG_PROVIDER_TEST_MODE=record cargo test -p rig --test xai --all-features \
//!     prompt_caching:: -- --exact --test-threads=1
//! ```

use rig::client::CompletionClient as _;
use rig::client::DefaultTransportClient as _;
use rig::providers::xai;

use crate::cache_conformance::{
    CacheAccounting, CacheProbe, CacheSupport, assert_cache_conformance, assert_prefix_stable,
    report_and_assert_live, run_cache_probe, run_cache_probe_streaming,
};

use super::support::with_xai_prompt_caching_cassette;

const CACHE_MODEL: &str = xai::GROK_3_MINI;

const XAI_CACHE_SUPPORT: CacheSupport = CacheSupport {
    provider: "xai",
    accounting: CacheAccounting::Subset,
    explicit_breakpoints: false,
    reports_writes: false,
    min_cacheable_tokens: 1024,
    cache_key_field: None,
    hit_ratio_floor: 0.80,
};

fn probe() -> CacheProbe {
    CacheProbe::new("xai prompt caching")
}

#[tokio::test]
async fn blocking_probe_hits_and_keeps_hitting_as_the_prefix_grows() {
    const SCENARIO: &str = "prompt_caching/blocking_probe";

    with_xai_prompt_caching_cassette("prompt_caching/blocking_probe", |client| async move {
        let model = client.completion_model(CACHE_MODEL);
        let observation = run_cache_probe(&model, &probe()).await;
        assert_cache_conformance(&observation, &XAI_CACHE_SUPPORT, "blocking probe");
    })
    .await;

    assert_prefix_stable("xai", SCENARIO);
}

#[tokio::test]
async fn streaming_probe_survives_the_streaming_accumulator() {
    const SCENARIO: &str = "prompt_caching/streaming_probe";

    with_xai_prompt_caching_cassette("prompt_caching/streaming_probe", |client| async move {
        let model = client.completion_model(CACHE_MODEL);
        let observation = run_cache_probe_streaming(&model, &probe()).await;
        assert_cache_conformance(&observation, &XAI_CACHE_SUPPORT, "streaming probe");
    })
    .await;

    assert_prefix_stable("xai", SCENARIO);
}

/// Live economics: run the same probe against the real API.
///
/// A cassette pins what xai did at record time. Only a live run catches
/// xai changing its cache semantics under us — a shorter TTL, a higher
/// minimum, a different block granularity — which is exactly the kind of change
/// that costs money silently. `#[ignore]`d so it never runs in the key-free
/// gate; run it with `--ignored` and a key present.
#[tokio::test]
#[ignore = "requires XAI_API_KEY and spends real tokens"]
async fn live_cache_economics() {
    let client = xai::Client::from_env().expect("XAI_API_KEY");
    let model = client.completion_model(CACHE_MODEL);
    let observation = run_cache_probe(&model, &probe()).await;
    report_and_assert_live(&observation, &XAI_CACHE_SUPPORT, "live_cache_economics");
}
