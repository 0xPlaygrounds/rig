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
    assert_agent_growth_still_hits, assert_breakpoints_match_support, assert_cache_conformance,
    assert_cache_key_stable, assert_prefix_stable, observation_from_completion_calls,
    report_and_assert_live, run_cache_probe, run_cache_probe_streaming,
};

use super::super::support::with_gemini_prompt_caching_cassette;

/// Gemini 2.5 Flash: implicit caching, and the cheapest model that has it.
const CACHE_MODEL: &str = gemini::completion::GEMINI_2_5_FLASH;

/// Gemini's documented implicit-cache minimum for 2.5 Flash is 1,024 tokens
/// (2.5 Pro's is 2,048). The probe pads well past both.
const GEMINI_CACHE_SUPPORT: CacheSupport = CacheSupport {
    provider: "gemini",
    accounting: CacheAccounting::Subset,
    explicit_breakpoints: false,
    reports_writes: false,
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
    assert_breakpoints_match_support("gemini", SCENARIO, &GEMINI_CACHE_SUPPORT);
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
    assert_breakpoints_match_support("gemini", SCENARIO, &GEMINI_CACHE_SUPPORT);
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
    assert_breakpoints_match_support("gemini", SCENARIO, &GEMINI_CACHE_SUPPORT);
}

/// Live economics: run the same probe against the real API.
///
/// A cassette pins what gemini did at record time. Only a live run catches
/// gemini changing its cache semantics under us — a shorter TTL, a higher
/// minimum, a different block granularity — which is exactly the kind of change
/// that costs money silently. `#[ignore]`d so it never runs in the key-free
/// gate; run it with `--ignored` and a key present.
#[tokio::test]
#[ignore = "requires GEMINI_API_KEY and spends real tokens"]
async fn live_cache_economics() {
    use rig::client::ProviderClient as _;

    let client = gemini::Client::from_env().expect("GEMINI_API_KEY");
    let model = client.completion_model(CACHE_MODEL);

    // Two passes, asserting on the second — the same procedure the module docs
    // prescribe for re-recording. Gemini's implicit cache only serves a prefix
    // once an entry for *that exact prefix* exists, so on a cold run the grown
    // turn-3 prefix (which no earlier request ever sent) reads zero. The first
    // pass establishes it; the second measures steady-state economics, which is
    // what this cell is for.
    let _warm_up = run_cache_probe(&model, &probe()).await;
    let observation = run_cache_probe(&model, &probe()).await;
    report_and_assert_live(&observation, &GEMINI_CACHE_SUPPORT, "live_cache_economics");
}

// ---------------------------------------------------------------------------
// Explicit context caching (`cachedContents`)
// ---------------------------------------------------------------------------
//
// A different feature from the implicit caching every cell above measures, and
// the reason rig grew a `cachedContents` client. Measured live on
// gemini-2.5-flash over one 18.5k-token corpus, the same day these fixtures were
// recorded:
//
//   implicit: 0% cached on five consecutive turns; 99.6% only on a sixth request
//   explicit: 100.0% on turn one, and 100.0% again from an unrelated conversation
//
// Implicit caching keys on a prefix the provider has seen before, so a fresh
// conversation starts cold. Explicit caching keys on a handle, so it does not.

use rig::providers::gemini::cached_content::{CacheExpiry, NewCachedContent};
use std::time::Duration;

/// Explicit caching should serve essentially the whole prefix from the first
/// request — a far higher bar than implicit caching's 0.75 floor, and the reason
/// the feature is worth its storage cost.
const GEMINI_EXPLICIT_SUPPORT: CacheSupport = CacheSupport {
    cache_key_field: Some("cachedContent"),
    hit_ratio_floor: 0.95,
    ..GEMINI_CACHE_SUPPORT
};

/// Corpus held by the cache. Deterministic and committed, like all probe
/// padding — a nonce would churn the fixture and break body matching.
fn cached_corpus() -> String {
    crate::cache_conformance::cache_padding(240)
}

async fn create_probe_cache(
    client: &gemini::Client,
    display_name: &str,
) -> rig::providers::gemini::cached_content::CachedContent {
    client
        .cached_contents()
        .create(
            NewCachedContent::new(CACHE_MODEL)
                .system_instruction(format!(
                    "You are a deterministic cassette test assistant.\n{}",
                    cached_corpus()
                ))
                .display_name(display_name)
                .expiry(CacheExpiry::ttl(Duration::from_secs(600))),
        )
        .await
        .expect("creating a gemini cached content should succeed")
}

/// The whole resource lifecycle, in one recording.
#[tokio::test]
async fn explicit_cache_lifecycle() {
    with_gemini_prompt_caching_cassette(
        "prompt_caching/explicit_cache_lifecycle",
        |client| async move {
            let caches = client.cached_contents();
            let created = create_probe_cache(&client, "rig-lifecycle").await;

            assert!(
                created.name.starts_with("cachedContents/"),
                "a handle should be `cachedContents/<id>`, got {:?}",
                created.name
            );
            assert_eq!(
                created.model,
                format!("models/{CACHE_MODEL}"),
                "the cache is bound to the model it was created for"
            );
            let stored = created
                .usage_metadata
                .as_ref()
                .map(|usage| usage.total_token_count)
                .unwrap_or_default();
            assert!(
                stored >= GEMINI_CACHE_SUPPORT.min_cacheable_tokens as u64,
                "a cache holding {stored} tokens is below the {}-token minimum and would cache \
                 nothing",
                GEMINI_CACHE_SUPPORT.min_cacheable_tokens
            );

            let fetched = caches.get(&created.name).await.expect("get should succeed");
            assert_eq!(fetched.name, created.name);

            let listed = caches.list().await.expect("list should succeed");
            assert!(
                listed.iter().any(|entry| entry.name == created.name),
                "the cache just created should appear in the listing"
            );

            // Expiry is the only mutable part of the resource.
            let extended = caches
                .update_expiry(&created.name, CacheExpiry::ttl(Duration::from_secs(900)))
                .await
                .expect("update should succeed");
            assert_eq!(extended.name, created.name);
            assert!(extended.expire_time.is_some());

            caches
                .delete(&created.name)
                .await
                .expect("delete should succeed — storage bills until it lands");
        },
    )
    .await;
}

/// The headline: explicit caching serves the prefix from turn one.
#[tokio::test]
async fn explicit_cache_serves_the_whole_prefix_from_the_first_turn() {
    with_gemini_prompt_caching_cassette(
        "prompt_caching/explicit_cache_hit_ratio",
        |client| async move {
            let cache = create_probe_cache(&client, "rig-hit-ratio").await;
            let model = client
                .completion_model(CACHE_MODEL)
                .with_cached_content(cache.name.clone());

            // `bare()`: the cache owns the system instruction and tools, and a
            // request that also sends its own is rejected — by rig, before it
            // reaches Gemini.
            let observation = run_cache_probe(&model, &probe().bare()).await;
            assert_cache_conformance(
                &observation,
                &GEMINI_EXPLICIT_SUPPORT,
                "explicit cache probe",
            );

            // Unlike implicit caching, turn 1 already hits — there is no warm-up.
            assert!(
                observation.turns[0].cached_input_tokens > 0,
                "explicit caching should hit on the very first request; that is the property \
                 implicit caching does not have.\n{}",
                observation.report(&GEMINI_EXPLICIT_SUPPORT)
            );

            client
                .cached_contents()
                .delete(&cache.name)
                .await
                .expect("delete should succeed");
        },
    )
    .await;

    assert_prefix_stable("gemini", "prompt_caching/explicit_cache_hit_ratio");
    assert_cache_key_stable(
        "gemini",
        "prompt_caching/explicit_cache_hit_ratio",
        &GEMINI_EXPLICIT_SUPPORT,
    );
}

/// One cache, two conversations that share nothing else.
///
/// This is the property implicit caching structurally cannot have: it keys on a
/// prefix the provider has already seen, so a conversation that opens with
/// different words starts cold. A handle does not care.
#[tokio::test]
async fn explicit_cache_hits_across_unrelated_conversations() {
    with_gemini_prompt_caching_cassette(
        "prompt_caching/explicit_cache_across_conversations",
        |client| async move {
            let cache = create_probe_cache(&client, "rig-cross-conversation").await;
            let model = client
                .completion_model(CACHE_MODEL)
                .with_cached_content(cache.name.clone());

            let mut reads = Vec::new();
            for prompt in [
                "Reply with exactly: alpha",
                "Say only the word beta, nothing else",
            ] {
                let request = rig::completion::CompletionRequest {
                    preamble: None,
                    chat_history: vec![rig::message::Message::User {
                        content: vec![rig::message::UserContent::text(prompt)],
                    }],
                    documents: vec![],
                    tools: vec![],
                    temperature: Some(0.0),
                    max_tokens: Some(16),
                    tool_choice: None,
                    additional_params: Some(serde_json::json!({
                        "generationConfig": { "thinkingConfig": { "thinkingBudget": 0 } }
                    })),
                    model: None,
                    output_schema: None,
                    record_telemetry_content: false,
                };
                let response = rig::completion::CompletionModel::completion(&model, request)
                    .await
                    .expect("a cached-content request should succeed");
                reads.push((
                    response.usage.input_tokens,
                    response.usage.cached_input_tokens,
                ));
            }

            for (index, (prompt_tokens, cached)) in reads.iter().enumerate() {
                let ratio = *cached as f64 / *prompt_tokens as f64;
                assert!(
                    ratio >= GEMINI_EXPLICIT_SUPPORT.hit_ratio_floor,
                    "conversation {} read {cached} of {prompt_tokens} prompt tokens ({:.1}%); a \
                     cache handle should not care that the conversations are unrelated",
                    index + 1,
                    ratio * 100.0
                );
            }

            client
                .cached_contents()
                .delete(&cache.name)
                .await
                .expect("delete should succeed");
        },
    )
    .await;
}

/// `cacheTokensDetails` is parsed and then not surfaced in the normalized
/// `Usage` — and that is correct, which is worth pinning rather than assuming.
///
/// Gemini reports a per-modality breakdown of *what* it cached
/// (`[{"modality":"TEXT","tokenCount":3660}]`). Rig's normalized `Usage` has no
/// modality concept, so there is nowhere for it to go and no way to add one
/// without inventing a cross-provider abstraction that only Gemini populates.
///
/// It is not lost, though: the field lives on `GenerateContentResponse`, which
/// `raw_completion` hands back — rig's documented escape hatch for
/// provider-specific fields. This cell asserts that path stays open, and that
/// the breakdown agrees with the aggregate rig *does* normalize, so the two can
/// never silently disagree.
#[test]
fn cache_tokens_details_are_populated_and_agree_with_the_aggregate() {
    let interactions =
        crate::cassettes::recorded_interaction_bodies("gemini", "prompt_caching/blocking_probe");

    let mut checked = 0usize;
    for (_, response) in &interactions {
        let Ok(body) = serde_json::from_str::<serde_json::Value>(response) else {
            continue;
        };
        let Some(usage) = body.get("usageMetadata") else {
            continue;
        };
        let Some(aggregate) = usage
            .get("cachedContentTokenCount")
            .and_then(serde_json::Value::as_u64)
            .filter(|count| *count > 0)
        else {
            continue;
        };

        let details = usage
            .get("cacheTokensDetails")
            .and_then(serde_json::Value::as_array)
            .expect("a turn reporting cached tokens should report their modality breakdown");
        let summed: u64 = details
            .iter()
            .filter_map(|entry| entry.get("tokenCount").and_then(serde_json::Value::as_u64))
            .sum();

        assert_eq!(
            summed, aggregate,
            "the per-modality breakdown should account for the aggregate rig normalizes: \
             {details:?} vs {aggregate}"
        );
        checked += 1;
    }

    assert!(
        checked > 0,
        "no recorded turn reported cached tokens, so this check proved nothing"
    );
}

// ---------------------------------------------------------------------------
// Threshold edges and prefix mutations
// ---------------------------------------------------------------------------

/// Build a bare Gemini request: no tools, no preamble, thinking off.
fn mutation_request(
    system: Option<&str>,
    tools: Vec<rig::completion::ToolDefinition>,
    history: Vec<rig::message::Message>,
    temperature: f64,
) -> rig::completion::CompletionRequest {
    rig::completion::CompletionRequest {
        preamble: system.map(str::to_owned),
        chat_history: history,
        documents: vec![],
        tools,
        temperature: Some(temperature),
        max_tokens: Some(16),
        tool_choice: None,
        additional_params: Some(serde_json::json!({
            "generationConfig": { "thinkingConfig": { "thinkingBudget": 0 } }
        })),
        model: None,
        output_schema: None,
        record_telemetry_content: false,
    }
}

fn user(text: &str) -> rig::message::Message {
    rig::message::Message::User {
        content: vec![rig::message::UserContent::text(text)],
    }
}

/// A prompt under the model's documented minimum must not cache.
///
/// The cell that gives every other cell's padding its meaning. If Gemini cached
/// a 300-token prefix, the 1,024-token floor in `GEMINI_CACHE_SUPPORT` would be
/// superstition and the padding in every other probe would be proving nothing.
#[tokio::test]
async fn a_prefix_below_the_minimum_does_not_cache() {
    with_gemini_prompt_caching_cassette(
        "prompt_caching/below_minimum_does_not_cache",
        |client| async move {
            let model = client.completion_model(CACHE_MODEL);
            let small = crate::cache_conformance::cache_padding(8);
            let request = || {
                mutation_request(
                    Some(&format!("You are a terse assistant. {small}")),
                    vec![],
                    vec![user("Reply with exactly: small")],
                    0.0,
                )
            };

            let first = rig::completion::CompletionModel::completion(&model, request())
                .await
                .expect("first small request should succeed");
            let second = rig::completion::CompletionModel::completion(&model, request())
                .await
                .expect("second small request should succeed");

            assert!(
                first.usage.input_tokens < GEMINI_CACHE_SUPPORT.min_cacheable_tokens as u64,
                "this probe is supposed to sit *below* the {}-token minimum, but billed {} — \
                 re-tune the padding or the cell proves nothing",
                GEMINI_CACHE_SUPPORT.min_cacheable_tokens,
                first.usage.input_tokens
            );
            assert_eq!(
                second.usage.cached_input_tokens, 0,
                "a prefix below the documented minimum must not cache; if Gemini started caching \
                 it, the minimum in GEMINI_CACHE_SUPPORT is wrong and every other cell's padding \
                 needs revisiting. usage: {:?}",
                second.usage
            );
        },
    )
    .await;
}

/// `generationConfig` is not part of the cached prefix.
///
/// Asserted as a **hit**: temperature changes between turns constantly in real
/// applications, and if that busted the cache it would be a finding worth a
/// warning in rig's docs. It does not.
#[tokio::test]
async fn changing_temperature_still_hits() {
    with_gemini_prompt_caching_cassette(
        "prompt_caching/temperature_change_still_hits",
        |client| async move {
            let model = client.completion_model(CACHE_MODEL);
            let preamble = probe().preamble;
            let history = vec![user("Reply with exactly: temp")];

            let warm = rig::completion::CompletionModel::completion(
                &model,
                mutation_request(Some(&preamble), vec![], history.clone(), 0.0),
            )
            .await
            .expect("warming request should succeed");

            let hotter = rig::completion::CompletionModel::completion(
                &model,
                mutation_request(Some(&preamble), vec![], history, 0.7),
            )
            .await
            .expect("second request should succeed");

            let ratio = hotter.usage.cached_input_tokens as f64 / hotter.usage.input_tokens as f64;
            assert!(
                ratio >= GEMINI_CACHE_SUPPORT.hit_ratio_floor,
                "changing only `temperature` should not move the cached prefix, but the second \
                 turn read {} of {} prompt tokens ({:.1}%). If this ever fails, rig should warn \
                 users that per-request sampling changes cost them their cache.\nwarm: {:?}\nhot: \
                 {:?}",
                hotter.usage.cached_input_tokens,
                hotter.usage.input_tokens,
                ratio * 100.0,
                warm.usage,
                hotter.usage
            );
        },
    )
    .await;
}

/// Changing one word of the system instruction must cost the cache.
///
/// The complement of every hit assertion in this file: those prove caching
/// happens, this proves it is keyed on what it claims to be keyed on. A cache
/// that "hit" here would be serving a prefix the caller did not send.
#[tokio::test]
async fn changing_the_system_instruction_misses() {
    with_gemini_prompt_caching_cassette(
        "prompt_caching/changed_system_instruction_miss",
        |client| async move {
            let model = client.completion_model(CACHE_MODEL);
            let base = probe().preamble;
            let history = vec![user("Reply with exactly: sysinstr")];

            let _warm = rig::completion::CompletionModel::completion(
                &model,
                mutation_request(Some(&base), vec![], history.clone(), 0.0),
            )
            .await
            .expect("warming request should succeed");

            // One word, at the very front of the prefix.
            let mutated = base.replacen("deterministic", "nondeterministic", 1);
            assert_ne!(
                mutated, base,
                "the mutation should actually change the text"
            );

            let after = rig::completion::CompletionModel::completion(
                &model,
                mutation_request(Some(&mutated), vec![], history, 0.0),
            )
            .await
            .expect("mutated request should succeed");

            assert_eq!(
                after.usage.cached_input_tokens, 0,
                "a changed system instruction is a different prefix and must not hit the previous \
                 entry. usage: {:?}",
                after.usage
            );
        },
    )
    .await;
}
