//! Groq prompt-caching cassette suite.
//!
//! **Groq's prompt cache is real but intermittent** — the same signature as
//! Mistral, and it took the strengthened `assert_no_meaningful_prefix_cache` to
//! notice.
//!
//! An earlier revision of this suite concluded Groq did not cache at all. It
//! was reading only the first two turns of one probe; the streaming probe's
//! *first* turn reads 1,792 of 1,795 prompt tokens — 99.8% — against a prefix an
//! earlier scenario had warmed, and then turns 2 and 3 read zero. Repeated
//! recordings put the hit on different turns, with `assert_prefix_stable`
//! proving on every run that rig's own bytes never moved. Routing without cache
//! affinity is the plausible cause; either way it is not something rig controls.
//!
//! For the record: the blocking fixture was re-recorded as part of this
//! reclassification. Its previous recording read zero on all three turns and
//! would have kept passing the "no cache" assertion, which is exactly how the
//! wrong classification survived. Re-recording until a turn hits is what
//! `assert_cache_read_is_surfaced` explicitly asks for — the fixture has to
//! contain a hit for it to prove rig maps the field at all.
//!
//! So these cells assert the narrower thing that is true and deterministic:
//! when Groq reports a cache read, rig surfaces it at full magnitude, on the
//! blocking and streaming paths alike. That is a claim about rig's usage mapping
//! rather than about Groq's hit rate. Groq reuses `openai::Usage`, so the value
//! arrives in `prompt_tokens_details.cached_tokens`.
//!
//! **Rate-limit caveat:** this account's Groq tier allows 8,000 tokens per
//! minute, and three turns of the default ~4,600-token probe exceed that before
//! turn 2 can be sent. The probe is capped at ~1,800 tokens per turn, which
//! clears the 1,024-token floor a prompt cache would plausibly need. Recording
//! these fixtures needs roughly a minute between attempts.
//!
//! # Recording
//!
//! ```text
//! RIG_PROVIDER_TEST_MODE=record cargo test -p rig --test groq --all-features \
//!     prompt_caching:: -- --exact --test-threads=1
//! ```

use rig::client::CompletionClient as _;

use crate::cache_conformance::{
    CacheAccounting, CacheProbe, CacheSupport, assert_cache_read_is_surfaced, assert_prefix_stable,
    run_cache_probe, run_cache_probe_streaming,
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
async fn blocking_probe_surfaces_the_cache_read_groq_reports() {
    const SCENARIO: &str = "prompt_caching/blocking_probe";

    with_groq_prompt_caching_cassette("prompt_caching/blocking_probe", |client| async move {
        let model = client.completion_model(CACHE_MODEL);
        let observation = run_cache_probe(&model, &probe()).await;
        assert_cache_read_is_surfaced(&observation, &GROQ_CACHE_SUPPORT, "blocking probe");
    })
    .await;

    assert_prefix_stable("groq", SCENARIO);
}

#[tokio::test]
async fn streaming_probe_surfaces_the_cache_read_groq_reports() {
    const SCENARIO: &str = "prompt_caching/streaming_probe";

    with_groq_prompt_caching_cassette("prompt_caching/streaming_probe", |client| async move {
        let model = client.completion_model(CACHE_MODEL);
        let observation = run_cache_probe_streaming(&model, &probe()).await;
        assert_cache_read_is_surfaced(&observation, &GROQ_CACHE_SUPPORT, "streaming probe");
    })
    .await;

    assert_prefix_stable("groq", SCENARIO);
}
