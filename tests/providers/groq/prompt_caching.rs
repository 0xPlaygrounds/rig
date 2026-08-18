//! Groq prompt-caching cassette suite.
//!
//! **No meaningful prefix caching was observed**, and this suite records that as
//! a measured fact rather than an assumption.
//!
//! Measured on `openai/gpt-oss-20b`: a 1,795-token prompt re-sent
//! byte-identically reads zero cached tokens on every turn, blocking and
//! streaming alike. Groq reuses `openai::Usage`, so rig would surface
//! `prompt_tokens_details.cached_tokens` if Groq populated it.
//!
//! **Caveat, stated because it bounds the claim:** this account's Groq tier
//! allows 8,000 tokens per minute, and three turns of the default ~4,600-token
//! probe exceed that before turn 2 can be sent. The probe is therefore capped at
//! ~1,800 tokens per turn. That clears the 1,024-token floor a prompt cache
//! would plausibly need, so the result is meaningful — but it does not rule out
//! a cache whose real minimum sits above 1,800 tokens. Re-run on a higher tier
//! to settle that.
//!
//! The cells assert the absence through `assert_no_meaningful_prefix_cache`,
//! which is self-invalidating: the day Groq caches this prefix, the assertion
//! fails and says to replace it with the full `assert_cache_conformance` suite.
//!
//! Groq reuses `openai::Usage`, so rig would surface
//! `prompt_tokens_details.cached_tokens` if Groq populated it.
//!
//! # Recording
//!
//! ```text
//! RIG_PROVIDER_TEST_MODE=record cargo test -p rig --test groq --all-features \
//!     prompt_caching:: -- --exact --test-threads=1
//! ```

use rig::client::CompletionClient as _;

use crate::cache_conformance::{
    CacheAccounting, CacheProbe, CacheSupport, assert_no_meaningful_prefix_cache,
    assert_prefix_stable, run_cache_probe, run_cache_probe_streaming,
};

use super::support::with_groq_prompt_caching_cassette;

/// A model Groq actually serves today.
///
/// Deliberately a literal rather than one of `rig::providers::groq`'s model
/// constants: every one of them (`llama-3.1-8b-instant`, `mixtral-8x7b-32768`,
/// the llama-3.2 previews) now 404s with `model_not_found`. That is its own
/// staleness problem, out of scope here; this suite just needs a live model.
const CACHE_MODEL: &str = "openai/gpt-oss-20b";

const GROQ_CACHE_SUPPORT: CacheSupport = CacheSupport {
    provider: "groq",
    accounting: CacheAccounting::Subset,
    reports_writes: false,
    explicit_breakpoints: false,
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

fn probe() -> CacheProbe {
    CacheProbe::new("groq prompt caching")
        .with_padding(GROQ_PADDING_REPETITIONS, "groq prompt caching")
}

#[tokio::test]
async fn blocking_probe_observes_no_meaningful_prefix_cache() {
    const SCENARIO: &str = "prompt_caching/blocking_probe";

    with_groq_prompt_caching_cassette("prompt_caching/blocking_probe", |client| async move {
        let model = client.completion_model(CACHE_MODEL);
        let observation = run_cache_probe(&model, &probe()).await;
        assert_no_meaningful_prefix_cache(&observation, &GROQ_CACHE_SUPPORT, "blocking probe");
    })
    .await;

    assert_prefix_stable("groq", SCENARIO);
}

#[tokio::test]
async fn streaming_probe_observes_no_meaningful_prefix_cache() {
    const SCENARIO: &str = "prompt_caching/streaming_probe";

    with_groq_prompt_caching_cassette("prompt_caching/streaming_probe", |client| async move {
        let model = client.completion_model(CACHE_MODEL);
        let observation = run_cache_probe_streaming(&model, &probe()).await;
        assert_no_meaningful_prefix_cache(&observation, &GROQ_CACHE_SUPPORT, "streaming probe");
    })
    .await;

    assert_prefix_stable("groq", SCENARIO);
}
