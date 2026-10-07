//! llama.cpp prompt-caching cassette suite.
//!
//! Before this suite, `tests/cassette_cache_prefix.rs` carried a
//! `NO_CACHE_SUITE` exemption for both llama.cpp suites reading "records
//! against a local llama.cpp server that is not running in this environment".
//! That reason was true and is no longer: a server *is* running for this PR,
//! so the exemption is deleted and this is what replaces it.
//!
//! # What llama.cpp's cache is
//!
//! Not a hosted prefix cache with a billing minimum — the server's own **KV
//! slot cache**. It caches from the first token, needs no opt-in beyond the
//! `cache_prompt` default, and reports its work in two independently populated
//! places: `usage.prompt_tokens_details.cached_tokens`, which rig normalizes
//! into [`Usage::cached_input_tokens`](rig::completion::Usage), and
//! `timings.cache_n`, which reaches a caller in `raw`, the reply document.
//!
//! Measured by the fixtures below, against b10964-b29c606e2 with
//! `unsloth/Qwen3-1.7B-GGUF` Q4_K_M: `prompt_caching/blocking_probe` bills
//! 2,825 prompt tokens on turn 1 and reads **2,824** of them back on turn 2 —
//! far above the 0.80 floor every other provider in the matrix is held to.
//! Turn 3 grows the prefix to 2,854 and still reads 2,825.
//!
//! | Cell | Dimension | Pinned |
//! | --- | --- | --- |
//! | [`streaming_probe_survives_the_streaming_accumulator`] | streaming | the terminal frame's usage carries the read |
//! | [`cache_prompt_false_turns_the_cache_off_for_that_turn_only`] | `cache_prompt` | a llama.cpp-only request field; off means 0 cached, and it does not poison the next turn |
//! | [`agent_loop_does_not_move_its_own_prefix`] | agent loop | every outbound request extends its predecessor |
//!
//! # Recording
//!
//! ```text
//! # restart the default server first: turn 1 is only cold on an empty KV cache
//! RIG_PROVIDER_TEST_MODE=record cargo test -p rig --test llamacpp --all-features \
//!     prompt_caching:: -- --test-threads=1
//! ```
//!

use serde_json::Value;

use crate::cache_conformance::{
    AGENT_CACHE_PROMPT, CacheProbe, CacheProbeLookupTool, CacheSupport,
    assert_breakpoints_match_support, assert_cache_conformance, assert_prefix_stable,
    run_cache_probe_streaming,
};
use crate::cassettes::recorded_statuses_and_bodies;

use super::super::cassette_support::*;
use rig::completion::CompletionRequest;

/// llama.cpp's cache, as a descriptor.
///
/// `min_cacheable_tokens` is the harness's padding target rather than a
/// documented provider minimum: llama.cpp has none, and caches from the first
/// token. The 1,024 keeps this probe the same size as every other provider's,
/// which is what makes the hit ratios comparable across the matrix.
const LLAMACPP_CACHE_SUPPORT: CacheSupport = CacheSupport {
    provider: "llamacpp",
    // `openai::Usage`: `prompt_tokens_details.cached_tokens` is a breakdown of
    // `prompt_tokens`, not a sibling of it.
    // The KV cache is automatic; there is no `cache_control` on this wire.
    explicit_breakpoints: false,
    // llama.cpp reports reads only. There is no cache-creation counter, and
    // rig hardcodes `cache_creation_input_tokens` to 0 on this path.
    reports_writes: false,
    min_cacheable_tokens: 1024,
    // No `prompt_cache_key` equivalent: the slot cache keys on the token
    // prefix itself.
    cache_key_field: None,
    hit_ratio_floor: 0.80,
};

/// Padding repetitions for this provider's probe.
///
/// The harness default is 180 (~4,700 prompt tokens), which every hosted
/// provider's context swallows without noticing. The recording server runs
/// `-c 4096` — the value every other cell in this suite was recorded under —
/// so the default probe does not fit and answers `400
/// exceed_context_size_error` before it can cache anything. 100 repetitions is
/// ~2,600 tokens: comfortably above the 1,024 the descriptor asks for, and
/// with room for turn 3 to grow the prefix.
///
/// Groq's suite reduces the padding for the same *kind* of reason (a
/// tokens-per-minute rate limit); the precedent is that the probe adapts to the
/// provider rather than the provider's configuration being bent to the probe.
const LLAMACPP_CACHE_PADDING: usize = 100;

/// One probe per cell, each with its own label.
///
/// llama.cpp's cache lives in the **server**, not in a per-account service, so
/// two cells sharing a preamble share a KV slot and the second one's "turn 1"
/// is already warm — which would make [`assert_warms`]'s cold-start reading
/// and this suite's own cold-miss premise untestable. The label is
/// interpolated into the preamble, so a distinct label per cell is a distinct
/// prefix and a genuinely cold first turn.
fn probe_for(label: &'static str) -> CacheProbe {
    CacheProbe::new(label).with_padding(LLAMACPP_CACHE_PADDING, label)
}

/// Every turn's `(prompt_tokens, cached_tokens, timings.cache_n)`.
fn recorded_cache_counters(scenario: &str) -> Vec<(u64, u64, u64)> {
    recorded_statuses_and_bodies("llamacpp", scenario)
        .into_iter()
        .filter_map(|(status, body)| (status == 200).then_some(body))
        .filter_map(|body| {
            // Streaming scenarios put usage on the terminal SSE frame.
            serde_json::from_str::<Value>(&body).ok().or_else(|| {
                body.lines()
                    .filter_map(|line| line.trim().strip_prefix("data:"))
                    .map(str::trim)
                    .filter(|payload| *payload != "[DONE]")
                    .filter_map(|payload| serde_json::from_str::<Value>(payload).ok())
                    .find(|frame| frame.get("usage").is_some())
            })
        })
        .filter_map(|response| {
            let usage = response.get("usage")?;
            Some((
                usage["prompt_tokens"].as_u64().unwrap_or_default(),
                usage["prompt_tokens_details"]["cached_tokens"]
                    .as_u64()
                    .unwrap_or_default(),
                response["timings"]["cache_n"].as_u64().unwrap_or_default(),
            ))
        })
        .collect()
}

#[tokio::test]
async fn streaming_probe_survives_the_streaming_accumulator() {
    with_llamacpp_prompt_caching_cassette("prompt_caching/streaming_probe", |client| async move {
        let model = client.completion(CASSETTE_MODEL);
        let observation =
            run_cache_probe_streaming(model, &probe_for("llamacpp streaming probe")).await;
        assert_cache_conformance(&observation, &LLAMACPP_CACHE_SUPPORT, "streaming probe");
    })
    .await;

    assert_prefix_stable("llamacpp", "prompt_caching/streaming_probe");
    assert_breakpoints_match_support(
        "llamacpp",
        "prompt_caching/streaming_probe",
        &LLAMACPP_CACHE_SUPPORT,
    );
}

/// `cache_prompt: false` is llama.cpp's own switch, and it is per-turn.
///
/// No OpenAI-compatible provider has this field; it rides `additional_params`.
/// Two claims, and the second is the one worth recording: turning the cache
/// off for a turn reads zero cached tokens, and it does **not** evict what was
/// already there — the turn after it hits again. A switch that poisoned the
/// slot would make "disable caching for this one sensitive prompt" cost the
/// whole conversation.
#[tokio::test]
async fn cache_prompt_false_turns_the_cache_off_for_that_turn_only() {
    with_llamacpp_prompt_caching_cassette(
        "prompt_caching/cache_prompt_disabled",
        |client| async move {
            let model = client.completion(CASSETTE_MODEL);
            let probe = probe_for("llamacpp cache_prompt switch");

            // Warm the slot.
            let warm = model
                .call(
                    CompletionRequest::new(probe.prompt)
                        .preamble(probe.preamble.clone())
                        .temperature(0.0)
                        .max_tokens(16),
                )
                .await
                .expect("the warming turn should succeed");
            let _ = warm;

            let second = model
                .call(
                    CompletionRequest::new(probe.prompt)
                        .preamble(probe.preamble.clone())
                        .temperature(0.0)
                        .max_tokens(16),
                )
                .await
                .expect("the warm turn should succeed");
            assert!(
                second.usage.cached_input_tokens.is_some_and(|n| n > 0),
                "the slot must be warm before the switch is tested: {:?}",
                second.usage
            );

            let disabled = model
                .call(
                    CompletionRequest::new(probe.prompt)
                        .preamble(probe.preamble.clone())
                        .temperature(0.0)
                        .max_tokens(16)
                        .options(
                            rig::completion::GenerationOptions::default()
                                .cache(rig::completion::CacheRetention::None),
                        ),
                )
                .await
                .expect("cache_prompt: false should succeed");
            assert_eq!(
                disabled.usage.cached_input_tokens,
                Some(0),
                "the switch must actually disable the read: {:?}",
                disabled.usage
            );

            let after = model
                .call(
                    CompletionRequest::new(probe.prompt)
                        .preamble(probe.preamble.clone())
                        .temperature(0.0)
                        .max_tokens(16),
                )
                .await
                .expect("the turn after the switch should succeed");
            assert!(
                after.usage.cached_input_tokens.is_some_and(|n| n > 0),
                "one disabled turn must not evict the slot: {:?}",
                after.usage
            );
        },
    )
    .await;

    let counters = recorded_cache_counters("prompt_caching/cache_prompt_disabled");
    assert_eq!(counters.len(), 4, "{counters:?}");
    assert_eq!(counters[0].1, 0, "turn 1 cold: {counters:?}");
    assert!(counters[1].1 > 0, "turn 2 warm: {counters:?}");
    assert_eq!(counters[2].1, 0, "turn 3 disabled: {counters:?}");
    assert!(
        counters[3].1 > 0,
        "turn 4 warm again — the disabled turn did not evict: {counters:?}"
    );
}

/// A real agent loop with a tool round-trip, asserted on prefix stability.
#[tokio::test]
async fn agent_loop_does_not_move_its_own_prefix() {
    with_llamacpp_prompt_caching_cassette("prompt_caching/agent_loop", |client| async move {
        let response = rig::AgentBuilder::new(client.completion(CASSETTE_MODEL))
            .preamble(&probe_for("llamacpp agent loop").preamble)
            .tool(CacheProbeLookupTool)
            .temperature(0.0)
            .build()
            .prompt(AGENT_CACHE_PROMPT)
            .max_turns(6)
            .await
            .expect("llamacpp agent cache probe should complete");

        assert!(
            response.completion_calls().len() >= 2,
            "[llamacpp] agent loop: the run made {} completion calls; a tool \
             round-trip is at least two, so the model never called the tool and \
             the prefix never grew",
            response.completion_calls().len()
        );
    })
    .await;

    assert_prefix_stable("llamacpp", "prompt_caching/agent_loop");
}
