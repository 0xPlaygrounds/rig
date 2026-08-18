//! Cohere prompt-caching cassette suite.
//!
//! **Cohere's prompt cache is real, and it warms across turns.** Measured on
//! `command-a-03-2025`, one three-turn probe over a 6,058-token prompt:
//!
//! | turn | prompt | cached | ratio |
//! |---|---:|---:|---:|
//! | 1 (cold) | 6,058 | 112 | 1.8% |
//! | 2 (byte-identical repeat) | 6,058 | 992 | 16.4% |
//! | 3 (prefix grown) | 6,085 | 6,016 | **98.9%** |
//!
//! An earlier revision of this suite got this wrong, and the way it got it wrong
//! is worth recording. It asserted only over turns 1 and 2, saw 112 and
//! concluded Cohere reported "a constant 112 cached tokens … not prefix
//! caching". Turn 3 was never examined, so the mistake would have passed
//! forever. `assert_no_meaningful_prefix_cache` now folds over *every* turn
//! precisely so a late-warming cache cannot hide from it.
//!
//! The cells therefore assert what is actually true: the cache read never goes
//! backwards as the conversation grows, and the final turn clears the
//! provider's floor. A prefix move breaks both.
//!
//! Note that turn 2's 16.4% is exactly the reading a bare
//! `cached_input_tokens > 0` assertion reports as "caching works" — 83% of the
//! prompt re-billed, and no test that only checks for non-zero could tell.
//!
//! Cohere documents `cached_tokens` as a subset of `tokens.input_tokens` and
//! excludes it from `billed_units`
//! (`crates/rig-core/src/providers/cohere/completion.rs`).
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
    CacheAccounting, CacheProbe, CacheSupport, assert_cache_warms_over_turns, assert_prefix_stable,
    run_cache_probe, run_cache_probe_streaming,
};

use super::super::support::with_cohere_prompt_caching_cassette;

const CACHE_MODEL: &str = cohere::COMMAND_A_03_2025;

const COHERE_CACHE_SUPPORT: CacheSupport = CacheSupport {
    provider: "cohere",
    accounting: CacheAccounting::Subset,
    reports_writes: false,
    min_cacheable_tokens: 1024,
    cache_key_field: None,
    hit_ratio_floor: 0.80,
};

fn probe() -> CacheProbe {
    CacheProbe::new("cohere prompt caching")
}

#[tokio::test]
async fn blocking_probe_warms_to_a_full_cache_hit_over_three_turns() {
    const SCENARIO: &str = "prompt_caching/blocking_probe";

    with_cohere_prompt_caching_cassette("prompt_caching/blocking_probe", |client| async move {
        let model = client.completion_model(CACHE_MODEL);
        let observation = run_cache_probe(&model, &probe()).await;
        assert_cache_warms_over_turns(&observation, &COHERE_CACHE_SUPPORT, "blocking probe");
    })
    .await;

    assert_prefix_stable("cohere", SCENARIO);
}

#[tokio::test]
async fn streaming_probe_warms_to_a_full_cache_hit_over_three_turns() {
    const SCENARIO: &str = "prompt_caching/streaming_probe";

    with_cohere_prompt_caching_cassette("prompt_caching/streaming_probe", |client| async move {
        let model = client.completion_model(CACHE_MODEL);
        let observation = run_cache_probe_streaming(&model, &probe()).await;
        assert_cache_warms_over_turns(&observation, &COHERE_CACHE_SUPPORT, "streaming probe");
    })
    .await;

    assert_prefix_stable("cohere", SCENARIO);
}
