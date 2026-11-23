//! Groq prompt-caching cassette suite.
//!
//! Groq's prompt cache is real but intermittent: repeated recordings put the hit
//! on different turns while `assert_prefix_stable` proves rig's own bytes never
//! moved, so routing without cache affinity is the plausible cause. These cells
//! assert the narrower deterministic claim: when Groq reports a cache read, rig
//! surfaces it at full magnitude on the blocking and streaming paths. Groq
//! speaks the OpenAI chat shape, so the value arrives in
//! `prompt_tokens_details.cached_tokens`. A fixture must contain a hit for
//! `assert_cache_read_is_surfaced` to prove anything, so re-record until a turn
//! hits, about a minute apart for the tier's rate limit.
//!
//! There is no `agent_loop` cell: the model answers the probe from its own
//! knowledge and never calls the tool, so a run records one request and leaves
//! `assert_prefix_stable` nothing to compare. The corpus-wide scan in
//! `tests/cassette_cache_prefix.rs` covers the loop-level guarantee.
//!
//! Re-record with
//! `RIG_PROVIDER_TEST_MODE=record cargo test -p rig-cassette --test integration --all-features groq::prompt_caching:: -- --exact --test-threads=1`.

use crate::cache_conformance::{
    CacheProbe, CacheSupport, assert_cache_read_is_surfaced, assert_prefix_stable, run_cache_probe,
    run_cache_probe_streaming,
};

use super::support::with_groq_prompt_caching_cassette;

/// A model Groq actually serves today; the cassette was recorded against it.
const CACHE_MODEL: &str = rig::providers::groq::GPT_OSS_20B;

const GROQ_CACHE_SUPPORT: CacheSupport = CacheSupport {
    provider: "groq",
    explicit_breakpoints: false,
    reports_writes: false,
    min_cacheable_tokens: 1024,
    cache_key_field: None,
    hit_ratio_floor: 0.80,
};

/// Groq's free tier allows 8,000 tokens per minute, and three turns of the
/// default ~4,600-token probe blows through that before turn 2 is even sent
/// (`rate_limit_exceeded`, "Limit 8000, Used 4633, Requested 4691"). Sixty
/// repetitions puts each turn near 1,600 tokens, so all three fit in one
/// minute's budget while still clearing the 1,024-token floor a prompt cache
/// would need.
const GROQ_PADDING_REPETITIONS: usize = 60;

/// Room for a reasoning model to finish thinking and answer in text, so
/// each turn's reply goes back and the next request extends it.
const ANSWERING_MAX_TOKENS: u64 = 1024;

fn probe() -> CacheProbe {
    CacheProbe::new("groq prompt caching")
        .with_padding(GROQ_PADDING_REPETITIONS, "groq prompt caching")
}

#[tokio::test]
async fn blocking_probe_surfaces_the_cache_read_groq_reports() {
    const SCENARIO: &str = "prompt_caching/blocking_probe";

    with_groq_prompt_caching_cassette("prompt_caching/blocking_probe", |client| async move {
        let model = client.completion(CACHE_MODEL);
        let probe = CacheProbe {
            max_tokens: ANSWERING_MAX_TOKENS,
            ..probe()
        };
        let observation = run_cache_probe(model, &probe).await;
        assert_cache_read_is_surfaced(&observation, &GROQ_CACHE_SUPPORT, "blocking probe");
    })
    .await;

    assert_prefix_stable("groq", SCENARIO);
}

#[tokio::test]
#[ignore = "stale cassette: its request predates item-shaped history, and Groq's cache never warmed in any re-record attempt"]
async fn streaming_probe_surfaces_the_cache_read_groq_reports() {
    const SCENARIO: &str = "prompt_caching/streaming_probe";

    with_groq_prompt_caching_cassette("prompt_caching/streaming_probe", |client| async move {
        let model = client.completion(CACHE_MODEL);
        let observation = run_cache_probe_streaming(model, &probe()).await;
        assert_cache_read_is_surfaced(&observation, &GROQ_CACHE_SUPPORT, "streaming probe");
    })
    .await;

    assert_prefix_stable("groq", SCENARIO);
}
