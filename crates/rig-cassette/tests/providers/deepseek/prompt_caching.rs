//! DeepSeek prompt-caching cassette suite.
//!
//! DeepSeek caches context automatically and reports hits in its own
//! `prompt_cache_hit_tokens` field, outside the OpenAI-style details object,
//! which rig maps into `Usage::cached_input_tokens`. Hits and misses
//! partition `prompt_tokens`, so the denominator is `input_tokens`.
//!
//! There is no `agent_loop` cell: the model answers the probe from its own
//! knowledge and never calls the tool, even under `ToolChoice::Required`, so
//! a run records one request and `assert_prefix_stable` would have no pair
//! to compare. The corpus-wide scan in `tests/cassette_cache_prefix.rs`
//! covers this provider's recorded multi-turn conversations instead.
//!
//! ```text
//! RIG_PROVIDER_TEST_MODE=record cargo test -p rig-cassette --test integration --all-features \
//!     deepseek::prompt_caching:: -- --exact --test-threads=1
//! ```

use rig::providers::deepseek;
use rig::providers::openai::wire::DEEPSEEK;
use rig_test_support::cassette_models::OpenAiModels;

use crate::cache_conformance::{
    CacheProbe, CacheSupport, assert_cache_conformance, assert_prefix_stable,
    report_and_assert_live, run_cache_probe, run_cache_probe_streaming,
};

use super::support::with_deepseek_cassette;

const CACHE_MODEL: &str = deepseek::DEEPSEEK_V4_FLASH;

const DEEPSEEK_CACHE_SUPPORT: CacheSupport = CacheSupport {
    provider: "deepseek",
    explicit_breakpoints: false,
    reports_writes: false,
    min_cacheable_tokens: 64,
    cache_key_field: None,
    hit_ratio_floor: 0.80,
};

fn probe() -> CacheProbe {
    CacheProbe::new("deepseek prompt caching")
}

#[tokio::test]
async fn streaming_probe_survives_the_streaming_accumulator() {
    const SCENARIO: &str = "prompt_caching/streaming_probe";

    with_deepseek_cassette("prompt_caching/streaming_probe", |client| async move {
        let model = client.completion(CACHE_MODEL);
        let observation = run_cache_probe_streaming(model, &probe()).await;
        assert_cache_conformance(&observation, &DEEPSEEK_CACHE_SUPPORT, "streaming probe");
    })
    .await;

    assert_prefix_stable("deepseek", SCENARIO);
}

/// Live economics: run the same probe against the real API.
///
/// A cassette pins what deepseek did at record time. Only a live run catches
/// deepseek changing its cache semantics under us — a shorter TTL, a higher
/// minimum, a different block granularity — which is exactly the kind of change
/// that costs money silently. `#[ignore]`d so it never runs in the key-free
/// gate; run it with `--ignored` and a key present.
#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY and spends real tokens"]
async fn live_cache_economics() {
    let bound = OpenAiModels::from_env_for(&DEEPSEEK).expect("DEEPSEEK_API_KEY");
    let model = bound.completion(CACHE_MODEL);
    let observation = run_cache_probe(model, &probe()).await;
    report_and_assert_live(
        &observation,
        &DEEPSEEK_CACHE_SUPPORT,
        "live_cache_economics",
    );
}
