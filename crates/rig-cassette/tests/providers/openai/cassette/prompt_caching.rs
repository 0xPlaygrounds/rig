//! OpenAI prompt-caching cassette suite.
//!
//! OpenAI is the first provider after Anthropic to have its prompt cache
//! *observed* rather than merely normalized: rig has mapped
//! `prompt_tokens_details.cached_tokens` and
//! `input_tokens_details.cached_tokens` into `Usage::cached_input_tokens` for a
//! long time, and until this suite no test had ever seen either field be
//! non-zero.
//!
//! Both surfaces are covered, separately and deliberately. Chat completions and
//! the Responses API are two different endpoints, with two different request
//! shapes, two different usage payloads, and two different rig mapping
//! functions; a cache result on one says nothing about the other.
//!
//! # Reading the numbers
//!
//! OpenAI reports `cached_tokens` as a **subset** of the prompt-token counter,
//! as rig's `Usage` counts cache reads on every provider, so turn 1's billed
//! prompt is `input_tokens` on its own and the hit ratio is turn 2's cache read
//! over that. It never reports cache *writes* — rig hardcodes
//! `cache_creation_input_tokens` to 0 on both paths through
//! `providers::internal::completion_usage` — so turn 1 legitimately shows zero
//! for both counters, and [`assert_warms`] does not require otherwise.
//!
//! OpenAI caches a prefix only above 1,024 tokens and in 128-token increments,
//! so the tail of the prompt is legitimately uncached and the hit ratio is a
//! floor rather than an equality.
//!
//! # Recording
//!
//! ```text
//! RIG_PROVIDER_TEST_MODE=record cargo test -p rig --test openai --all-features \
//!     prompt_caching:: -- --exact --test-threads=1
//! ```
//!
//! The cache TTL is minutes, so all three turns of a probe must record
//! back-to-back inside one test body — which is exactly how the shared probe
//! runs them. The assertions run identically in record mode, so a session that
//! records a miss fails immediately instead of committing a fixture that pins
//! one.

use rig::providers::openai;
use rig_test_support::cassette_models::OpenAiModels;

use crate::cache_conformance::{
    CacheProbe, CacheSupport, assert_breakpoints_match_support, assert_cache_conformance,
    assert_prefix_stable, assert_warms, report_and_assert_live, run_cache_probe,
    run_cache_probe_streaming,
};

use super::super::support::{shared_options, stateless, with_openai_prompt_caching_cassette};

/// A cheap model that still participates in prompt caching.
pub(super) const CACHE_MODEL: &str = openai::GPT_4O_MINI;

/// Shared descriptor for both OpenAI surfaces.
///
/// `min_cacheable_tokens` is OpenAI's documented 1,024-token floor: below it the
/// API silently declines to cache, and a fixture recorded under it would pin a
/// miss no matter what rig did.
pub(super) const OPENAI_CACHE_SUPPORT: CacheSupport = CacheSupport {
    provider: "openai",
    explicit_breakpoints: false,
    reports_writes: false,
    min_cacheable_tokens: 1024,
    cache_key_field: None,
    hit_ratio_floor: 0.80,
};

/// The Responses surface, whose request carries an explicit `prompt_cache_key`.
const OPENAI_RESPONSES_KEYED_SUPPORT: CacheSupport = CacheSupport {
    cache_key_field: Some("prompt_cache_key"),
    ..OPENAI_CACHE_SUPPORT
};

pub(super) fn probe() -> CacheProbe {
    CacheProbe::new("openai prompt caching")
}

/// The probe on the Responses surface, which stores no response.
fn responses_probe() -> CacheProbe {
    probe().with_provider_options(stateless())
}

/// The Responses probe plus the `prompt_cache_key` that surface needs to route
/// same-prefix traffic to the same cache.
fn keyed_probe() -> CacheProbe {
    probe().with_provider_options(shared_options(
        Some(false),
        Some("rig-cache-conformance-openai"),
    ))
}

/// Measured Responses behavior without `prompt_cache_key`: turn 2 misses.
///
/// This is a **provider** property, not a rig defect, and it is recorded rather
/// than hidden because it changes the advice rig should give its users.
///
/// Two back-to-back recording sessions produced byte-identical counters: a
/// 4,595-token prefix, zero cached on turns 1 *and* 2, then 4,480 cached on
/// turn 3. The request bodies for turns 1 and 2 are byte-identical — the
/// corpus prefix check and `assert_prefix_stable` both confirm it — so rig is
/// not moving the prefix. OpenAI documents `prompt_cache_key` as the way to
/// raise cache hit rates by routing same-prefix traffic consistently, and
/// supplying one makes turn 2 hit
/// ([`responses_blocking_probe_hits_and_keeps_hitting_as_the_prefix_grows`]).
/// The chat-completions surface hits on turn 2 with no key at all, so this is
/// specific to Responses.
///
/// Asserted here as what was actually observed — the cache is cold through turn
/// 2 and warm by turn 3 — so that the day OpenAI makes un-keyed Responses
/// caching hit sooner, this cell fails and tells us.
#[tokio::test]
async fn responses_without_a_cache_key_does_not_hit_until_the_third_turn() {
    const SCENARIO: &str = "prompt_caching/responses_unkeyed_probe";

    with_openai_prompt_caching_cassette("prompt_caching/responses_unkeyed_probe", |client| async move {
        let model = client.openai.completion(CACHE_MODEL);
        let observation = run_cache_probe(model, &responses_probe()).await;

        // Not `assert_cache_conformance`: turn 2 legitimately misses here, and
        // pretending otherwise would either fail forever or force the floor
        // down for every other OpenAI cell.
        assert_warms(
            &observation,
            &OPENAI_CACHE_SUPPORT,
            "responses unkeyed probe",
        );
        let turns = &observation.turns;
        assert_eq!(
            turns[1].cached_input_tokens,
            Some(0),
            "un-keyed Responses caching was observed to still be cold on turn 2; if OpenAI has \
             changed that, this cell is the notification — drop it and fold the scenario into the \
             keyed probe.\n{}",
            observation.report(&OPENAI_CACHE_SUPPORT)
        );
        assert!(
            turns[2].cached_input_tokens.is_some_and(|n| n > 0),
            "un-keyed Responses caching was observed to be warm by turn 3, so a turn-3 miss means \
             caching stopped working on this surface entirely.\n{}",
            observation.report(&OPENAI_CACHE_SUPPORT)
        );
    })
    .await;

    // The point of the cell: rig's own bytes are stable across all three turns,
    // so the turn-2 miss above cannot be blamed on a moved prefix.
    assert_prefix_stable("openai", SCENARIO);
    assert_breakpoints_match_support("openai", SCENARIO, &OPENAI_CACHE_SUPPORT);
}

#[tokio::test]
async fn responses_streaming_probe_survives_the_streaming_accumulator() {
    const SCENARIO: &str = "prompt_caching/responses_streaming_probe";

    with_openai_prompt_caching_cassette(
        "prompt_caching/responses_streaming_probe",
        |client| async move {
            let model = client.openai.completion(CACHE_MODEL);
            let observation = run_cache_probe_streaming(model, &keyed_probe()).await;
            assert_cache_conformance(
                &observation,
                &OPENAI_RESPONSES_KEYED_SUPPORT,
                "responses streaming probe",
            );
        },
    )
    .await;

    assert_prefix_stable("openai", SCENARIO);
    assert_breakpoints_match_support("openai", SCENARIO, &OPENAI_CACHE_SUPPORT);
}

/// Live economics: run the same probe against the real API.
///
/// A cassette pins what openai did at record time. Only a live run catches
/// openai changing its cache semantics under us — a shorter TTL, a higher
/// minimum, a different block granularity — which is exactly the kind of change
/// that costs money silently. `#[ignore]`d so it never runs in the key-free
/// gate; run it with `--ignored` and a key present.
#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and spends real tokens"]
async fn live_cache_economics() {
    let client = OpenAiModels::from_env().expect("OPENAI_API_KEY");
    let model = client.completion(CACHE_MODEL);
    let observation = run_cache_probe(model, &keyed_probe()).await;
    report_and_assert_live(
        &observation,
        &OPENAI_RESPONSES_KEYED_SUPPORT,
        "live_cache_economics",
    );
}
