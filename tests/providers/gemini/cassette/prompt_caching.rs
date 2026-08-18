//! Gemini prompt-caching cassette suite.
//!
//! Gemini caches long prefixes *implicitly* — there is no `cache_control` marker
//! to place and no opt-in call to make; a request whose prefix exceeds the
//! model's minimum is eligible, and the provider reports what it reused in
//! `usageMetadata.cachedContentTokenCount`. Rig has mapped that field into
//! `Usage::cached_input_tokens` for a long time
//! (`crates/rig-core/src/providers/gemini/completion.rs`) and, before this
//! suite, no test had ever seen it be non-zero.
//!
//! # This suite is also the regression cell for a real cache bug
//!
//! Gemini's tool schema `properties` used to be a `HashMap`, so its keys
//! serialized in a *different order on every request*. Gemini renders `tools`
//! at the very front of the cacheable prefix, which meant any rig request
//! carrying a tool with two or more properties had a different prefix every time
//! and could never hit the cache — silently, with the full prompt re-billed on
//! every turn. The probe below carries exactly such a tool, so a regression
//! would show up here as a turn-2 miss. (The direct, key-free guard is
//! `gemini_request_serialization_is_deterministic` in
//! `tests/cassette_cache_prefix.rs`.)
//!
//! # Recording note: Gemini's implicit cache needs a warm-up pass
//!
//! Gemini's implicit cache serves a prefix only once an entry for *that exact
//! prefix* has been established, and establishing one is not instantaneous. On a
//! cold run the byte-identical repeat (turn 2) hits while the grown turn-3
//! prefix — which no earlier request ever sent — reads zero. Run the same
//! scenario again and turn 3 hits too, because the first pass created its entry.
//!
//! This was measured, not assumed: three consecutive cold/warm recording passes
//! reproduced it on the blocking, streaming and agent paths.
//!
//! The practical consequence is for whoever re-records these fixtures: **run the
//! scenario twice and keep the second recording.** A cold first pass fails
//! `assert_growth_still_hits` rather than quietly committing a fixture that pins
//! a miss, which is the intended outcome — a recorded miss is worse than a
//! failed recording session.
//!
//! # Reading the numbers
//!
//! `cachedContentTokenCount` is a **subset** of `promptTokenCount`, so turn 1's
//! billed prompt is `input_tokens` on its own. Gemini never reports cache
//! *writes*, so turn 1 legitimately shows zero for both counters.
//!
//! # Recording
//!
//! ```text
//! RIG_PROVIDER_TEST_MODE=record cargo test -p rig --test gemini --all-features \
//!     prompt_caching:: -- --exact --test-threads=1
//! ```

use rig::client::CompletionClient as _;
use rig::prelude::*;
use rig::providers::gemini;

use crate::cache_conformance::{
    AGENT_CACHE_PROMPT, CacheAccounting, CacheProbe, CacheProbeLookupTool, CacheSupport,
    assert_agent_growth_still_hits, assert_cache_conformance, assert_prefix_stable,
    observation_from_completion_calls, run_cache_probe, run_cache_probe_streaming,
};

use super::super::support::with_gemini_prompt_caching_cassette;

/// Gemini 2.5 Flash: implicit caching, and the cheapest model that has it.
const CACHE_MODEL: &str = gemini::completion::GEMINI_2_5_FLASH;

/// Gemini's documented implicit-cache minimum for 2.5 Flash is 1,024 tokens
/// (2.5 Pro's is 2,048). The probe pads well past both.
const GEMINI_CACHE_SUPPORT: CacheSupport = CacheSupport {
    provider: "gemini",
    accounting: CacheAccounting::Subset,
    reports_writes: false,
    explicit_breakpoints: false,
    min_cacheable_tokens: 1024,
    // `cachedContent` names an *explicitly* created cache resource, which is a
    // different feature from the implicit prefix caching under test here. Rig
    // does not send it on this path, so there is no per-turn key to pin.
    cache_key_field: None,
    // Gemini's implicit cache works in coarse blocks and consistently leaves
    // roughly 800 tokens of the tail uncached, which puts the measured ratio
    // right around 0.80 — 3,760/4,633 on the grown turn. The floor is lowered to
    // 0.75 so normal block re-alignment cannot fail the suite, while a genuine
    // prefix move (which collapses the ratio to zero) still does.
    hit_ratio_floor: 0.75,
};

/// The probe, with Gemini 2.5's thinking disabled.
///
/// Gemini 2.5 Flash spends its output budget on thinking before it writes
/// anything, so a 16-token cap produces a response with no message at all
/// ("Response contained no message or tool call"). Zeroing the thinking budget
/// keeps the cheap, short, deterministic answer the probe wants; the thinking
/// tokens are output-side anyway and have no bearing on what gets cached.
fn probe() -> CacheProbe {
    CacheProbe::new("gemini prompt caching").with_additional_params(serde_json::json!({
        "generationConfig": {
            "thinkingConfig": { "thinkingBudget": 0 }
        }
    }))
}

#[tokio::test]
async fn blocking_probe_hits_and_keeps_hitting_as_the_prefix_grows() {
    const SCENARIO: &str = "prompt_caching/blocking_probe";

    with_gemini_prompt_caching_cassette("prompt_caching/blocking_probe", |client| async move {
        let model = client.completion_model(CACHE_MODEL);
        let observation = run_cache_probe(&model, &probe()).await;
        assert_cache_conformance(&observation, &GEMINI_CACHE_SUPPORT, "blocking probe");
    })
    .await;

    assert_prefix_stable("gemini", SCENARIO);
}

#[tokio::test]
async fn streaming_probe_survives_the_streaming_accumulator() {
    const SCENARIO: &str = "prompt_caching/streaming_probe";

    with_gemini_prompt_caching_cassette("prompt_caching/streaming_probe", |client| async move {
        let model = client.completion_model(CACHE_MODEL);
        let observation = run_cache_probe_streaming(&model, &probe()).await;
        assert_cache_conformance(&observation, &GEMINI_CACHE_SUPPORT, "streaming probe");
    })
    .await;

    assert_prefix_stable("gemini", SCENARIO);
}

#[tokio::test]
async fn agent_loop_keeps_hitting_across_tool_turns() {
    const SCENARIO: &str = "prompt_caching/agent_loop";

    with_gemini_prompt_caching_cassette("prompt_caching/agent_loop", |client| async move {
        let response = client
            .agent(CACHE_MODEL)
            .preamble(&probe().preamble)
            .tool(CacheProbeLookupTool)
            .temperature(0.0)
            .build()
            .prompt(AGENT_CACHE_PROMPT)
            .max_turns(6)
            .extended_details()
            .await
            .expect("gemini agent cache probe should complete");

        let observation = observation_from_completion_calls(response.completion_calls());
        assert_agent_growth_still_hits(&observation, &GEMINI_CACHE_SUPPORT, "agent loop");
    })
    .await;

    assert_prefix_stable("gemini", SCENARIO);
}
