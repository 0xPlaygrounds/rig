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
//! not alongside it (see [`crate::cache_conformance::CacheAccounting::Subset`]),
//! so turn 1's billed prompt is `input_tokens` on its own and the hit ratio is
//! turn 2's cache read over that. It never reports cache *writes* — rig hardcodes
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

use rig::client::CompletionClient as _;
use rig::prelude::*;
use rig::providers::openai;
use serde_json::json;

use crate::cache_conformance::{
    AGENT_CACHE_PROMPT, CacheAccounting, CacheObservation, CacheProbe, CacheProbeLookupTool,
    CacheSupport, assert_agent_growth_still_hits, assert_breakpoints_match_support,
    assert_cache_conformance, assert_cache_key_stable, assert_prefix_stable, assert_warms,
    observation_from_completion_calls, report_and_assert_live, run_cache_probe,
    run_cache_probe_streaming,
};

use super::super::support::{
    with_openai_completions_prompt_caching_cassette, with_openai_prompt_caching_cassette,
};

/// A cheap model that still participates in prompt caching.
const CACHE_MODEL: &str = openai::GPT_4O_MINI;

/// Shared descriptor for both OpenAI surfaces.
///
/// `min_cacheable_tokens` is OpenAI's documented 1,024-token floor: below it the
/// API silently declines to cache, and a fixture recorded under it would pin a
/// miss no matter what rig did.
const OPENAI_CACHE_SUPPORT: CacheSupport = CacheSupport {
    provider: "openai",
    accounting: CacheAccounting::Subset,
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

fn probe() -> CacheProbe {
    CacheProbe::new("openai prompt caching")
}

/// The probe plus the `prompt_cache_key` the Responses surface needs to route
/// same-prefix traffic to the same cache.
fn keyed_probe() -> CacheProbe {
    probe().with_additional_params(json!({
        "prompt_cache_key": "rig-cache-conformance-openai",
    }))
}

/// The Responses surface caches reliably **when rig sends a cache key**.
///
/// The un-keyed behavior is recorded separately in
/// [`responses_without_a_cache_key_does_not_hit_until_the_third_turn`], which is
/// where the reasoning for keying this cell lives.
#[tokio::test]
async fn responses_blocking_probe_hits_and_keeps_hitting_as_the_prefix_grows() {
    const SCENARIO: &str = "prompt_caching/responses_blocking_probe";

    with_openai_prompt_caching_cassette(
        "prompt_caching/responses_blocking_probe",
        |client| async move {
            let model = client.completion_model(CACHE_MODEL);
            let observation: CacheObservation = run_cache_probe(&model, &keyed_probe()).await;
            assert_cache_conformance(
                &observation,
                &OPENAI_RESPONSES_KEYED_SUPPORT,
                "responses blocking probe",
            );
        },
    )
    .await;

    assert_prefix_stable("openai", SCENARIO);
    assert_breakpoints_match_support("openai", SCENARIO, &OPENAI_CACHE_SUPPORT);
    assert_cache_key_stable("openai", SCENARIO, &OPENAI_RESPONSES_KEYED_SUPPORT);
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
        let model = client.completion_model(CACHE_MODEL);
        let observation = run_cache_probe(&model, &probe()).await;

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
            0,
            "un-keyed Responses caching was observed to still be cold on turn 2; if OpenAI has \
             changed that, this cell is the notification — drop it and fold the scenario into the \
             keyed probe.\n{}",
            observation.report(&OPENAI_CACHE_SUPPORT)
        );
        assert!(
            turns[2].cached_input_tokens > 0,
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
async fn chat_completions_blocking_probe_hits_and_keeps_hitting_as_the_prefix_grows() {
    const SCENARIO: &str = "prompt_caching/chat_completions_blocking_probe";

    with_openai_completions_prompt_caching_cassette(
        "prompt_caching/chat_completions_blocking_probe",
        |client| async move {
            let model = client.completion_model(CACHE_MODEL);
            let observation = run_cache_probe(&model, &probe()).await;
            assert_cache_conformance(
                &observation,
                &OPENAI_CACHE_SUPPORT,
                "chat completions blocking probe",
            );
        },
    )
    .await;

    assert_prefix_stable("openai", SCENARIO);
    assert_breakpoints_match_support("openai", SCENARIO, &OPENAI_CACHE_SUPPORT);
}

#[tokio::test]
async fn responses_streaming_probe_survives_the_streaming_accumulator() {
    const SCENARIO: &str = "prompt_caching/responses_streaming_probe";

    with_openai_prompt_caching_cassette(
        "prompt_caching/responses_streaming_probe",
        |client| async move {
            let model = client.completion_model(CACHE_MODEL);
            let observation = run_cache_probe_streaming(&model, &keyed_probe()).await;
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

#[tokio::test]
async fn chat_completions_streaming_probe_survives_the_streaming_accumulator() {
    const SCENARIO: &str = "prompt_caching/chat_completions_streaming_probe";

    with_openai_completions_prompt_caching_cassette(
        "prompt_caching/chat_completions_streaming_probe",
        |client| async move {
            let model = client.completion_model(CACHE_MODEL);
            let observation = run_cache_probe_streaming(&model, &probe()).await;
            assert_cache_conformance(
                &observation,
                &OPENAI_CACHE_SUPPORT,
                "chat completions streaming probe",
            );
        },
    )
    .await;

    assert_prefix_stable("openai", SCENARIO);
    assert_breakpoints_match_support("openai", SCENARIO, &OPENAI_CACHE_SUPPORT);
}

/// A real agent loop with a tool round-trip, on the chat-completions wire.
///
/// The cell the rest of the suite cannot replace: the three-turn probe builds
/// its own history, so it proves the *provider* caches a growing prefix, but
/// only a real run proves rig's agent driver does not disturb that prefix
/// between iterations. Tool definitions re-advertised in a different order, a
/// system prompt rebuilt differently on turn N, an assistant turn re-normalized
/// on the way back in — each busts the cache from that point on while every
/// counter stays non-zero.
#[tokio::test]
async fn chat_completions_agent_loop_keeps_hitting_across_tool_turns() {
    const SCENARIO: &str = "prompt_caching/chat_completions_agent_loop";

    with_openai_completions_prompt_caching_cassette(
        "prompt_caching/chat_completions_agent_loop",
        |client| async move {
            let response = client
                .agent(CACHE_MODEL)
                .preamble(&probe().preamble)
                .tool(CacheProbeLookupTool)
                .temperature(0.0)
                .build()
                .prompt(AGENT_CACHE_PROMPT)
                .max_turns(6)
                .await
                .expect("openai agent cache probe should complete");

            let observation = observation_from_completion_calls(response.completion_calls());
            assert_agent_growth_still_hits(
                &observation,
                &OPENAI_CACHE_SUPPORT,
                "chat completions agent loop",
            );
        },
    )
    .await;

    // The loop-level prefix guard: every one of the agent's own outbound
    // requests must extend its predecessor rather than rewrite it.
    assert_prefix_stable("openai", SCENARIO);
    assert_breakpoints_match_support("openai", SCENARIO, &OPENAI_CACHE_SUPPORT);
}

/// The Responses twin of [`chat_completions_agent_loop_keeps_hitting_across_tool_turns`].
#[tokio::test]
async fn responses_agent_loop_keeps_hitting_across_tool_turns() {
    const SCENARIO: &str = "prompt_caching/responses_agent_loop";

    with_openai_prompt_caching_cassette(
        "prompt_caching/responses_agent_loop",
        |client| async move {
            let response = client
                .agent(CACHE_MODEL)
                .preamble(&probe().preamble)
                .tool(CacheProbeLookupTool)
                .temperature(0.0)
                .additional_params(json!({
                    "prompt_cache_key": "rig-cache-conformance-openai-agent",
                }))
                .build()
                .prompt(AGENT_CACHE_PROMPT)
                .max_turns(6)
                .await
                .expect("openai responses agent cache probe should complete");

            let observation = observation_from_completion_calls(response.completion_calls());
            assert_agent_growth_still_hits(
                &observation,
                &OPENAI_CACHE_SUPPORT,
                "responses agent loop",
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
    let client = openai::Client::from_env().expect("OPENAI_API_KEY");
    let model = client.completion_model(CACHE_MODEL);
    let observation = run_cache_probe(&model, &keyed_probe()).await;
    report_and_assert_live(
        &observation,
        &OPENAI_RESPONSES_KEYED_SUPPORT,
        "live_cache_economics",
    );
}
