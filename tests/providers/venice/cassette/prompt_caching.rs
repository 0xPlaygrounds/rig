//! Venice prompt-caching cassette suite.
//!
//! Rig normalizes `Usage::cached_input_tokens` for this provider, and before
//! this suite no test had ever seen the field be non-zero. The three-turn probe
//! below is what makes the provider's actual behavior observable — including the
//! possibility that it does not cache at all, which is recorded rather than
//! assumed.
//!
//! Venice exposes an explicit `prompt_cache_key` on its request
//! (`crates/rig-core/src/providers/venice/completion.rs`), which nothing pinned
//! on the wire before this suite.
//!
//! # Recording
//!
//! ```text
//! RIG_PROVIDER_TEST_MODE=record cargo test -p rig --test venice --all-features \
//!     prompt_caching:: -- --exact --test-threads=1
//! ```

use rig::client::CompletionClient as _;
use rig::providers::venice;

use crate::cache_conformance::{
    CacheAccounting, CacheProbe, CacheSupport, assert_cache_conformance, assert_cache_key_stable,
    assert_prefix_stable, report_and_assert_live, run_cache_probe, run_cache_probe_streaming,
};

use super::super::support::with_venice_prompt_caching_cassette;

const CACHE_MODEL: &str = venice::QWEN3_5_9B;

/// Venice with the explicit `prompt_cache_key` rig can send.
const VENICE_KEYED_SUPPORT: CacheSupport = CacheSupport {
    cache_key_field: Some("prompt_cache_key"),
    ..VENICE_CACHE_SUPPORT
};

const VENICE_CACHE_SUPPORT: CacheSupport = CacheSupport {
    provider: "venice",
    accounting: CacheAccounting::Subset,
    reports_writes: false,
    explicit_breakpoints: false,
    min_cacheable_tokens: 1024,
    cache_key_field: None,
    hit_ratio_floor: 0.80,
};

fn probe() -> CacheProbe {
    CacheProbe::new("venice prompt caching")
}

#[tokio::test]
async fn blocking_probe_hits_and_keeps_hitting_as_the_prefix_grows() {
    const SCENARIO: &str = "prompt_caching/blocking_probe";

    with_venice_prompt_caching_cassette("prompt_caching/blocking_probe", |client| async move {
        let model = client.completion_model(CACHE_MODEL);
        let observation = run_cache_probe(&model, &probe()).await;
        assert_cache_conformance(&observation, &VENICE_CACHE_SUPPORT, "blocking probe");
    })
    .await;

    assert_prefix_stable("venice", SCENARIO);
}

#[tokio::test]
async fn streaming_probe_survives_the_streaming_accumulator() {
    const SCENARIO: &str = "prompt_caching/streaming_probe";

    with_venice_prompt_caching_cassette("prompt_caching/streaming_probe", |client| async move {
        let model = client.completion_model(CACHE_MODEL);
        let observation = run_cache_probe_streaming(&model, &probe()).await;
        assert_cache_conformance(&observation, &VENICE_CACHE_SUPPORT, "streaming probe");
    })
    .await;

    assert_prefix_stable("venice", SCENARIO);
}

/// Venice's `prompt_cache_key` must reach the wire and stay identical across turns.
///
/// The field exists on Venice's request type
/// (`crates/rig-core/src/providers/venice/completion.rs`) and nothing pinned it
/// on the wire before this cell. A key that changes between turns partitions the
/// cache and guarantees a miss, which is indistinguishable from "caching is off"
/// in the usage numbers alone — so it is asserted against the recorded request
/// bodies rather than the response.
#[tokio::test]
async fn prompt_cache_key_reaches_the_wire_and_is_stable() {
    const SCENARIO: &str = "prompt_caching/cache_key_stable";

    with_venice_prompt_caching_cassette("prompt_caching/cache_key_stable", |client| async move {
        let model = client.completion_model(CACHE_MODEL);
        let probe = probe().with_additional_params(serde_json::json!({
            "prompt_cache_key": "rig-cache-conformance-venice",
        }));
        let observation = run_cache_probe(&model, &probe).await;
        assert_cache_conformance(&observation, &VENICE_KEYED_SUPPORT, "keyed probe");
    })
    .await;

    assert_prefix_stable("venice", SCENARIO);
    assert_cache_key_stable("venice", SCENARIO, &VENICE_KEYED_SUPPORT);
}

/// Live economics: run the same probe against the real API.
///
/// A cassette pins what venice did at record time. Only a live run catches
/// venice changing its cache semantics under us — a shorter TTL, a higher
/// minimum, a different block granularity — which is exactly the kind of change
/// that costs money silently. `#[ignore]`d so it never runs in the key-free
/// gate; run it with `--ignored` and a key present.
#[tokio::test]
#[ignore = "requires VENICE_API_KEY and spends real tokens"]
async fn live_cache_economics() {
    use rig::client::ProviderClient as _;

    let client = venice::Client::from_env().expect("VENICE_API_KEY");
    let model = client.completion_model(CACHE_MODEL);
    let observation = run_cache_probe(&model, &probe()).await;
    report_and_assert_live(&observation, &VENICE_CACHE_SUPPORT, "live_cache_economics");
}
