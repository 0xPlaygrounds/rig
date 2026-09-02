//! Edge matrix for OpenRouter's **reasoning-token accounting**.
//!
//! **Bug.** OpenRouter documents usage accounting as always included ("Full
//! usage details are now always included automatically in every response", and
//! "in the last SSE message for streaming responses"), and every reasoning
//! route reports the breakdown:
//!
//! ```json
//! "usage": {"completion_tokens": 540,
//!           "completion_tokens_details": {"reasoning_tokens": 531}, …}
//! ```
//!
//! (verbatim from this matrix's own
//! `blocking_anthropic_routed_reports_reasoning_tokens` fixture)
//!
//! `openrouter::Usage` modeled `prompt_tokens`, `completion_tokens`,
//! `total_tokens`, `cost` and `prompt_tokens_details` — but no
//! `completion_tokens_details` field at all, so the whole object was dropped
//! at deserialization and `From<&Usage> for completion::Usage` ended with a
//! literal `reasoning_tokens: 0`. rig's normalized `Usage` has a first-class
//! `reasoning_tokens` slot that every other reasoning-capable provider fills
//! (openai, deepseek, gemini, anthropic), and it is recorded onto the
//! `gen_ai.usage.reasoning_tokens` telemetry span — so on OpenRouter that
//! span, and every caller reading the field, saw a hardcoded zero no matter
//! how much reasoning the route billed for.
//!
//! The same type is `OpenRouter::StreamingUsage`, so the streaming
//! terminal record lost it too — the zero was consistent across transports,
//! which is exactly why nothing caught it.
//!
//! **How these cells fail on `origin/main`.** Every non-control cell replays
//! its recorded fixture and asserts `usage.reasoning_tokens > 0`; on
//! `origin/main` the field is `0` for all of them.
//!
//! **Recorded upstreams.** Reasoning cells need a reasoning-capable upstream
//! and the *shape* of the breakdown is upstream-specific, so every cell pins
//! `provider.order` + `allow_fallbacks: false` and asserts the recorded
//! `provider`. Three upstream families are covered so the mapping is not
//! proven against one dialect only: `OpenAI` (`openai/o4-mini`,
//! `openai/gpt-5.2`), `Anthropic` (`anthropic/claude-haiku-4.5`, extended
//! thinking) and an open-weight route (`deepseek/deepseek-r1-0528` via a pinned
//! upstream). Non-reasoning controls use `openai/gpt-4o-mini`.
//!
//! Each cell re-reads its own fixture: a reasoning cell fails if the recorded
//! body's `usage.completion_tokens_details.reasoning_tokens` is absent or
//! zero, and a control fails if it is present and non-zero. A route that
//! stopped reporting the breakdown leaves a red test rather than a green one
//! covering nothing.
//!
//! | # | cell | transport | level | upstream | status |
//! |---|------|-----------|-------|----------|--------|
//! | 1 | `blocking_reasoning_tokens_reach_normalized_usage` | blocking | raw model | OpenAI / o4-mini | recorded |
//! | 2 | `streaming_reasoning_tokens_reach_the_terminal_record` | streaming | raw model | OpenAI / o4-mini | recorded |
//! | 3 | `blocking_agent_reports_reasoning_tokens` | blocking | agent | OpenAI / o4-mini | recorded |
//! | 4 | `streaming_agent_reports_reasoning_tokens` | streaming | agent | OpenAI / o4-mini | recorded |
//! | 5 | `blocking_high_effort_reports_reasoning_tokens` | blocking | raw model | OpenAI / o4-mini | recorded |
//! | 6 | `blocking_gpt_5_reports_reasoning_tokens` | blocking | raw model | OpenAI / gpt-5.2 | recorded |
//! | 7 | `streaming_gpt_5_reports_reasoning_tokens` | streaming | raw model | OpenAI / gpt-5.2 | recorded |
//! | 8 | `blocking_anthropic_routed_reports_reasoning_tokens` | blocking | raw model | Anthropic / haiku-4.5 | recorded |
//! | 9 | `streaming_anthropic_routed_reports_reasoning_tokens` | streaming | raw model | Anthropic / haiku-4.5 | recorded |
//! | 10 | `blocking_open_weight_route_reports_reasoning_tokens` | blocking | raw model | DeepInfra / deepseek-r1-0528 | recorded |
//! | 11 | `blocking_excluded_reasoning_still_counts_tokens` | blocking | raw model | OpenAI / o4-mini | recorded |
//! | 12 | `blocking_reasoning_tokens_stay_within_completion_tokens` | blocking | raw model | OpenAI / o4-mini | recorded |
//! | 13 | `blocking_reasoning_tokens_with_tools_in_request` | blocking | raw model | OpenAI / o4-mini | recorded |
//! | 14 | `transports_agree_on_reasoning_tokens` | both | raw model | OpenAI / o4-mini | recorded |
//! | 15 | `blocking_raw_usage_and_normalized_usage_agree` | blocking | raw + normalized | OpenAI / o4-mini | recorded |
//! | 16 | `blocking_cost_and_cache_details_still_map` | blocking | raw model | OpenAI / o4-mini | recorded |
//! | 17 | `control_non_reasoning_model_reports_zero_blocking` | blocking | raw model | OpenAI / gpt-4o-mini | recorded |
//! | 18 | `control_non_reasoning_model_reports_zero_streaming` | streaming | raw model | OpenAI / gpt-4o-mini | recorded |
//!
//! Five unit cells — usage payloads the live gateway will not produce on demand
//! (a real recorded usage object, the breakdown absent / `null` / `{}` /
//! zero, unmodeled siblings tolerated, the `total - prompt` output-token
//! fallback undisturbed, and the field omitted on serialization when absent)
//! — live next to the fix in
//! `crates/rig-core/src/providers/openrouter/client.rs`
//! (`completion_tokens_details_*`).

use rig::client::completion::CompletionClient;
use rig::completion::{CompletionModel, NormalizeCompletionResponse};
use rig::prelude::*;
use rig::telemetry::ProviderResponseExt;
use serde::Deserialize;
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};

use super::super::support::with_openrouter_usage_cassette;
use crate::cassettes;
use crate::support::{
    collect_stream_final_response_and_provider_final, collect_text_and_terminal,
    zero_arg_tool_definition,
};

/// Small enough to be cheap, hard enough that a reasoning route actually
/// spends tokens thinking about it.
const REASONING_PROMPT: &str = "A farmer has 17 sheep; all but 9 run away. He then buys 3 times \
                                as many as remain, sells 5, and splits the rest evenly among 4 \
                                pens. How many sheep per pen? Answer with the number only.";
const PLAIN_PROMPT: &str = "Name one common tree species. One word.";

const O4_MINI: &str = "openai/o4-mini";
const GPT_5: &str = "openai/gpt-5.2";
const CLAUDE_HAIKU: &str = "anthropic/claude-haiku-4.5";
const DEEPSEEK_R1: &str = "deepseek/deepseek-r1-0528";
const PLAIN_MODEL: &str = "openai/gpt-4o-mini";
const CAP: u64 = 2000;

fn openai_reasoning(effort: &str) -> Value {
    json!({
        "reasoning": { "effort": effort },
        "provider": { "order": ["OpenAI"], "allow_fallbacks": false }
    })
}

fn pinned(upstream: &str) -> Value {
    json!({ "provider": { "order": [upstream], "allow_fallbacks": false } })
}

// ---------------------------------------------------------------------------
// The bug: a documented, always-present field with a first-class slot, zeroed.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn blocking_reasoning_tokens_reach_normalized_usage() {
    const SCENARIO: &str =
        "reasoning_usage_matrix/blocking_reasoning_tokens_reach_normalized_usage";

    let delivered = Arc::new(Mutex::new(0u64));
    let recorder = delivered.clone();

    with_openrouter_usage_cassette(
        "reasoning_usage_matrix/blocking_reasoning_tokens_reach_normalized_usage",
        |client| async move {
            let model = client.completion_model(O4_MINI);
            let request = model
                .completion_request(REASONING_PROMPT)
                .max_tokens(CAP)
                .additional_params(openai_reasoning("medium"))
                .build();

            let response = model.completion(request).await.expect("reasoning turn");

            assert!(response.usage.reasoning_tokens > 0);
            *recorder.lock().expect("recorder") = response.usage.reasoning_tokens;
        },
    )
    .await;

    assert_recorded_reasoning_tokens(SCENARIO);
    assert_recorded_provider(SCENARIO, "OpenAI");
    assert_eq!(
        *delivered.lock().expect("recorder"),
        recorded_reasoning_tokens(SCENARIO),
        "the reported reasoning tokens must be exactly what the wire billed"
    );
    // The contrast for cell 11: with reasoning shown, this route returns both
    // detail kinds. If that ever stops being true, cell 11's `exclude` claim
    // stops meaning anything, so both are pinned to their own bytes.
    assert_recorded_reasoning_detail_types(SCENARIO, &["reasoning.encrypted", "reasoning.summary"]);
}

#[tokio::test]
async fn streaming_reasoning_tokens_reach_the_terminal_record() {
    const SCENARIO: &str =
        "reasoning_usage_matrix/streaming_reasoning_tokens_reach_the_terminal_record";

    let delivered = Arc::new(Mutex::new(0u64));
    let recorder = delivered.clone();

    with_openrouter_usage_cassette(
        "reasoning_usage_matrix/streaming_reasoning_tokens_reach_the_terminal_record",
        |client| async move {
            let model = client.completion_model(O4_MINI);
            let request = model
                .completion_request(REASONING_PROMPT)
                .max_tokens(CAP)
                .additional_params(openai_reasoning("medium"))
                .build();

            let stream = model.stream(request).await.expect("stream should connect");
            let (_, terminal) = collect_text_and_terminal(stream).await;
            let terminal = terminal.expect("terminal record");

            assert!(terminal.usage.reasoning_tokens > 0);
            *recorder.lock().expect("recorder") = terminal.usage.reasoning_tokens;
        },
    )
    .await;

    assert_recorded_reasoning_tokens(SCENARIO);
    assert_recorded_provider(SCENARIO, "OpenAI");
    assert_eq!(
        *delivered.lock().expect("recorder"),
        recorded_reasoning_tokens(SCENARIO),
        "the reported reasoning tokens must be exactly what the wire billed"
    );
}

/// The agent surface must report the breakdown too: `prompt()` returns a
/// `PromptResponse` whose `usage` carries the reasoning tokens, so this cell
/// fails if the mapping drops them.
#[tokio::test]
async fn blocking_agent_reports_reasoning_tokens() {
    const SCENARIO: &str = "reasoning_usage_matrix/blocking_agent_reports_reasoning_tokens";

    let delivered = Arc::new(Mutex::new(0u64));
    let recorder = delivered.clone();

    with_openrouter_usage_cassette(
        "reasoning_usage_matrix/blocking_agent_reports_reasoning_tokens",
        |client| async move {
            let agent = client
                .agent(O4_MINI)
                .max_tokens(CAP)
                .additional_params(openai_reasoning("medium"))
                .build();

            let response = agent
                .prompt(REASONING_PROMPT)
                .await
                .expect("agent reasoning turn");

            assert!(response.usage.reasoning_tokens > 0, "{:?}", response.usage);
            *recorder.lock().expect("recorder") = response.usage.reasoning_tokens;
        },
    )
    .await;

    assert_recorded_reasoning_tokens(SCENARIO);
    assert_recorded_provider(SCENARIO, "OpenAI");
    assert_eq!(
        *delivered.lock().expect("recorder"),
        recorded_reasoning_tokens(SCENARIO),
        "the agent's aggregated usage must report exactly what the wire billed"
    );
}

#[tokio::test]
async fn streaming_agent_reports_reasoning_tokens() {
    const SCENARIO: &str = "reasoning_usage_matrix/streaming_agent_reports_reasoning_tokens";

    let delivered = Arc::new(Mutex::new(0u64));
    let recorder = delivered.clone();

    with_openrouter_usage_cassette(
        "reasoning_usage_matrix/streaming_agent_reports_reasoning_tokens",
        |client| async move {
            let agent = client
                .agent(O4_MINI)
                .max_tokens(CAP)
                .additional_params(openai_reasoning("medium"))
                .build();

            let mut stream = agent.stream_prompt(REASONING_PROMPT).stream().await;
            let (_, provider_final) = collect_stream_final_response_and_provider_final(&mut stream)
                .await
                .expect("agent stream should succeed");

            assert!(provider_final.usage.reasoning_tokens > 0);
            *recorder.lock().expect("recorder") = provider_final.usage.reasoning_tokens;
        },
    )
    .await;

    assert_recorded_reasoning_tokens(SCENARIO);
    assert_recorded_provider(SCENARIO, "OpenAI");
    assert_eq!(
        *delivered.lock().expect("recorder"),
        recorded_reasoning_tokens(SCENARIO),
        "the reported reasoning tokens must be exactly what the wire billed"
    );
}

#[tokio::test]
async fn blocking_high_effort_reports_reasoning_tokens() {
    const SCENARIO: &str = "reasoning_usage_matrix/blocking_high_effort_reports_reasoning_tokens";

    let delivered = Arc::new(Mutex::new(0u64));
    let recorder = delivered.clone();

    with_openrouter_usage_cassette(
        "reasoning_usage_matrix/blocking_high_effort_reports_reasoning_tokens",
        |client| async move {
            let model = client.completion_model(O4_MINI);
            let request = model
                .completion_request(REASONING_PROMPT)
                .max_tokens(CAP)
                .additional_params(openai_reasoning("high"))
                .build();

            let response = model.completion(request).await.expect("reasoning turn");
            assert!(response.usage.reasoning_tokens > 0);
            *recorder.lock().expect("recorder") = response.usage.reasoning_tokens;
        },
    )
    .await;

    assert_recorded_reasoning_tokens(SCENARIO);
    assert_recorded_provider(SCENARIO, "OpenAI");
    assert_eq!(
        *delivered.lock().expect("recorder"),
        recorded_reasoning_tokens(SCENARIO),
        "the reported reasoning tokens must be exactly what the wire billed"
    );
}

#[tokio::test]
async fn blocking_gpt_5_reports_reasoning_tokens() {
    const SCENARIO: &str = "reasoning_usage_matrix/blocking_gpt_5_reports_reasoning_tokens";

    let delivered = Arc::new(Mutex::new(0u64));
    let recorder = delivered.clone();

    with_openrouter_usage_cassette(
        "reasoning_usage_matrix/blocking_gpt_5_reports_reasoning_tokens",
        |client| async move {
            let model = client.completion_model(GPT_5);
            let request = model
                .completion_request(REASONING_PROMPT)
                .max_tokens(CAP)
                .additional_params(openai_reasoning("medium"))
                .build();

            let response = model.completion(request).await.expect("reasoning turn");
            assert!(response.usage.reasoning_tokens > 0);
            *recorder.lock().expect("recorder") = response.usage.reasoning_tokens;
        },
    )
    .await;

    assert_recorded_reasoning_tokens(SCENARIO);
    assert_recorded_provider(SCENARIO, "OpenAI");
    assert_eq!(
        *delivered.lock().expect("recorder"),
        recorded_reasoning_tokens(SCENARIO),
        "the reported reasoning tokens must be exactly what the wire billed"
    );
}

#[tokio::test]
async fn streaming_gpt_5_reports_reasoning_tokens() {
    const SCENARIO: &str = "reasoning_usage_matrix/streaming_gpt_5_reports_reasoning_tokens";

    let delivered = Arc::new(Mutex::new(0u64));
    let recorder = delivered.clone();

    with_openrouter_usage_cassette(
        "reasoning_usage_matrix/streaming_gpt_5_reports_reasoning_tokens",
        |client| async move {
            let model = client.completion_model(GPT_5);
            let request = model
                .completion_request(REASONING_PROMPT)
                .max_tokens(CAP)
                .additional_params(openai_reasoning("medium"))
                .build();

            let stream = model.stream(request).await.expect("stream should connect");
            let (_, terminal) = collect_text_and_terminal(stream).await;
            let terminal = terminal.expect("terminal record");

            assert!(terminal.usage.reasoning_tokens > 0);
            *recorder.lock().expect("recorder") = terminal.usage.reasoning_tokens;
        },
    )
    .await;

    assert_recorded_reasoning_tokens(SCENARIO);
    assert_recorded_provider(SCENARIO, "OpenAI");
    assert_eq!(
        *delivered.lock().expect("recorder"),
        recorded_reasoning_tokens(SCENARIO),
        "the reported reasoning tokens must be exactly what the wire billed"
    );
}

// ---------------------------------------------------------------------------
// A second and third upstream family: the breakdown is not an OpenAI-ism.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn blocking_anthropic_routed_reports_reasoning_tokens() {
    const SCENARIO: &str =
        "reasoning_usage_matrix/blocking_anthropic_routed_reports_reasoning_tokens";

    let delivered = Arc::new(Mutex::new(0u64));
    let recorder = delivered.clone();

    with_openrouter_usage_cassette(
        "reasoning_usage_matrix/blocking_anthropic_routed_reports_reasoning_tokens",
        |client| async move {
            let model = client.completion_model(CLAUDE_HAIKU);
            let request = model
                .completion_request(REASONING_PROMPT)
                .max_tokens(CAP)
                .additional_params(json!({
                    "reasoning": { "max_tokens": 1024 },
                    "provider": { "order": ["Anthropic"], "allow_fallbacks": false }
                }))
                .build();

            let response = model.completion(request).await.expect("reasoning turn");
            assert!(response.usage.reasoning_tokens > 0);
            *recorder.lock().expect("recorder") = response.usage.reasoning_tokens;
        },
    )
    .await;

    assert_recorded_reasoning_tokens(SCENARIO);
    assert_recorded_provider(SCENARIO, "Anthropic");
    assert_eq!(
        *delivered.lock().expect("recorder"),
        recorded_reasoning_tokens(SCENARIO),
        "the reported reasoning tokens must be exactly what the wire billed"
    );
}

#[tokio::test]
async fn streaming_anthropic_routed_reports_reasoning_tokens() {
    const SCENARIO: &str =
        "reasoning_usage_matrix/streaming_anthropic_routed_reports_reasoning_tokens";

    let delivered = Arc::new(Mutex::new(0u64));
    let recorder = delivered.clone();

    with_openrouter_usage_cassette(
        "reasoning_usage_matrix/streaming_anthropic_routed_reports_reasoning_tokens",
        |client| async move {
            let model = client.completion_model(CLAUDE_HAIKU);
            let request = model
                .completion_request(REASONING_PROMPT)
                .max_tokens(CAP)
                .additional_params(json!({
                    "reasoning": { "max_tokens": 1024 },
                    "provider": { "order": ["Anthropic"], "allow_fallbacks": false }
                }))
                .build();

            let stream = model.stream(request).await.expect("stream should connect");
            let (_, terminal) = collect_text_and_terminal(stream).await;
            let terminal = terminal.expect("terminal record");

            assert!(terminal.usage.reasoning_tokens > 0);
            *recorder.lock().expect("recorder") = terminal.usage.reasoning_tokens;
        },
    )
    .await;

    assert_recorded_reasoning_tokens(SCENARIO);
    assert_recorded_provider(SCENARIO, "Anthropic");
    assert_eq!(
        *delivered.lock().expect("recorder"),
        recorded_reasoning_tokens(SCENARIO),
        "the reported reasoning tokens must be exactly what the wire billed"
    );
}

#[tokio::test]
async fn blocking_open_weight_route_reports_reasoning_tokens() {
    const SCENARIO: &str =
        "reasoning_usage_matrix/blocking_open_weight_route_reports_reasoning_tokens";

    let delivered = Arc::new(Mutex::new(0u64));
    let recorder = delivered.clone();

    with_openrouter_usage_cassette(
        "reasoning_usage_matrix/blocking_open_weight_route_reports_reasoning_tokens",
        |client| async move {
            let model = client.completion_model(DEEPSEEK_R1);
            let request = model
                .completion_request(REASONING_PROMPT)
                .max_tokens(CAP)
                .additional_params(pinned("DeepInfra"))
                .build();

            let response = model.completion(request).await.expect("reasoning turn");
            assert!(response.usage.reasoning_tokens > 0);
            *recorder.lock().expect("recorder") = response.usage.reasoning_tokens;
        },
    )
    .await;

    assert_recorded_reasoning_tokens(SCENARIO);
    assert_recorded_provider(SCENARIO, "DeepInfra");
    assert_eq!(
        *delivered.lock().expect("recorder"),
        recorded_reasoning_tokens(SCENARIO),
        "the reported reasoning tokens must be exactly what the wire billed"
    );
}

// ---------------------------------------------------------------------------
// Adjacent shapes on the same code path.
// ---------------------------------------------------------------------------

/// `reasoning.exclude: true` asks OpenRouter not to *show* the reasoning, and
/// it changes neither the billing nor this mapping. Recorded census, from this
/// cell's own fixture: on the OpenAI route `exclude` still returns the
/// `reasoning.encrypted` detail (only the human-readable summary is withheld),
/// so the cell asserts the usage rule alone and leaves the block question to
/// the wire.
#[tokio::test]
async fn blocking_excluded_reasoning_still_counts_tokens() {
    const SCENARIO: &str = "reasoning_usage_matrix/blocking_excluded_reasoning_still_counts_tokens";

    let delivered = Arc::new(Mutex::new(0u64));
    let recorder = delivered.clone();

    with_openrouter_usage_cassette(
        "reasoning_usage_matrix/blocking_excluded_reasoning_still_counts_tokens",
        |client| async move {
            let model = client.completion_model(O4_MINI);
            let request = model
                .completion_request(REASONING_PROMPT)
                .max_tokens(CAP)
                .additional_params(json!({
                    "reasoning": { "effort": "medium", "exclude": true },
                    "provider": { "order": ["OpenAI"], "allow_fallbacks": false }
                }))
                .build();

            let response = model.completion(request).await.expect("reasoning turn");

            assert!(response.usage.reasoning_tokens > 0);
            *recorder.lock().expect("recorder") = response.usage.reasoning_tokens;
        },
    )
    .await;

    assert_recorded_reasoning_tokens(SCENARIO);
    assert_recorded_provider(SCENARIO, "OpenAI");
    assert_eq!(
        *delivered.lock().expect("recorder"),
        recorded_reasoning_tokens(SCENARIO),
        "the reported reasoning tokens must be exactly what the wire billed"
    );
    // Pin the census claim in this cell's doc comment to the bytes, so it
    // cannot go stale silently: `exclude` withholds the summary and keeps the
    // encrypted detail.
    assert_recorded_reasoning_detail_types(SCENARIO, &["reasoning.encrypted"]);
}

/// The breakdown is a *share* of `completion_tokens`, not an addition to it —
/// the invariant that says the field was mapped rather than invented.
#[tokio::test]
async fn blocking_reasoning_tokens_stay_within_completion_tokens() {
    const SCENARIO: &str =
        "reasoning_usage_matrix/blocking_reasoning_tokens_stay_within_completion_tokens";

    with_openrouter_usage_cassette(
        "reasoning_usage_matrix/blocking_reasoning_tokens_stay_within_completion_tokens",
        |client| async move {
            let model = client.completion_model(O4_MINI);
            let request = model
                .completion_request(REASONING_PROMPT)
                .max_tokens(CAP)
                .additional_params(openai_reasoning("medium"))
                .build();

            let response = model.completion(request).await.expect("reasoning turn");
            let usage = &response.usage;

            assert!(usage.reasoning_tokens > 0, "{usage:?}");
            assert!(usage.reasoning_tokens <= usage.output_tokens, "{usage:?}");
            assert_eq!(usage.total_tokens, usage.input_tokens + usage.output_tokens);
        },
    )
    .await;

    assert_recorded_reasoning_tokens(SCENARIO);
    assert_recorded_provider(SCENARIO, "OpenAI");
}

#[tokio::test]
async fn blocking_reasoning_tokens_with_tools_in_request() {
    const SCENARIO: &str = "reasoning_usage_matrix/blocking_reasoning_tokens_with_tools_in_request";

    let delivered = Arc::new(Mutex::new(0u64));
    let recorder = delivered.clone();

    with_openrouter_usage_cassette(
        "reasoning_usage_matrix/blocking_reasoning_tokens_with_tools_in_request",
        |client| async move {
            let model = client.completion_model(O4_MINI);
            let request = model
                .completion_request(REASONING_PROMPT)
                .max_tokens(CAP)
                .tools(vec![zero_arg_tool_definition("ping")])
                .additional_params(openai_reasoning("medium"))
                .build();

            let response = model.completion(request).await.expect("reasoning turn");
            assert!(response.usage.reasoning_tokens > 0);
            *recorder.lock().expect("recorder") = response.usage.reasoning_tokens;
        },
    )
    .await;

    assert_recorded_reasoning_tokens(SCENARIO);
    assert_recorded_provider(SCENARIO, "OpenAI");
    assert_eq!(
        *delivered.lock().expect("recorder"),
        recorded_reasoning_tokens(SCENARIO),
        "the reported reasoning tokens must be exactly what the wire billed"
    );
}

/// One scenario, both transports: `openrouter::Usage` is also
/// `OpenRouter::StreamingUsage`, so a fix that only reached the blocking
/// mapping would show up here.
#[tokio::test]
async fn transports_agree_on_reasoning_tokens() {
    const SCENARIO: &str = "reasoning_usage_matrix/transports_agree_on_reasoning_tokens";

    let delivered = Arc::new(Mutex::new((0u64, 0u64)));
    let recorder = delivered.clone();

    with_openrouter_usage_cassette(
        "reasoning_usage_matrix/transports_agree_on_reasoning_tokens",
        |client| async move {
            let model = client.completion_model(O4_MINI);

            let blocking = model
                .completion(
                    model
                        .completion_request(REASONING_PROMPT)
                        .max_tokens(CAP)
                        .additional_params(openai_reasoning("medium"))
                        .build(),
                )
                .await
                .expect("blocking reasoning turn");

            let stream = model
                .stream(
                    model
                        .completion_request(REASONING_PROMPT)
                        .max_tokens(CAP)
                        .additional_params(openai_reasoning("medium"))
                        .build(),
                )
                .await
                .expect("stream should connect");
            let (_, terminal) = collect_text_and_terminal(stream).await;
            let terminal = terminal.expect("terminal record");

            assert!(blocking.usage.reasoning_tokens > 0);
            assert!(terminal.usage.reasoning_tokens > 0);
            *recorder.lock().expect("recorder") = (
                blocking.usage.reasoning_tokens,
                terminal.usage.reasoning_tokens,
            );
        },
    )
    .await;

    assert_recorded_reasoning_tokens(SCENARIO);
    assert_recorded_provider(SCENARIO, "OpenAI");

    let recorded = recorded_reasoning_token_counts(SCENARIO);
    let (blocking, streamed) = *delivered.lock().expect("recorder");
    assert_eq!(
        recorded.len(),
        2,
        "one blocking body and one final SSE usage: {recorded:?}"
    );
    assert!(
        recorded.contains(&blocking),
        "{blocking} not in {recorded:?}"
    );
    assert!(
        recorded.contains(&streamed),
        "{streamed} not in {recorded:?}"
    );
}

/// The provider-native escape hatch and the normalized view must agree: the
/// raw `openrouter::Usage` now carries the breakdown, and `From` maps it.
#[tokio::test]
async fn blocking_raw_usage_and_normalized_usage_agree() {
    const SCENARIO: &str = "reasoning_usage_matrix/blocking_raw_usage_and_normalized_usage_agree";

    with_openrouter_usage_cassette(
        "reasoning_usage_matrix/blocking_raw_usage_and_normalized_usage_agree",
        |client| async move {
            let model = client.completion_model(O4_MINI);
            let request = model
                .completion_request(REASONING_PROMPT)
                .max_tokens(CAP)
                .additional_params(openai_reasoning("medium"))
                .build();

            let raw = model.raw_completion(request).await.expect("raw turn");
            let raw_reasoning = raw
                .usage()
                .and_then(|usage| usage.completion_tokens_details)
                .map(|details| details.reasoning_tokens as u64)
                .expect("the raw usage must model the breakdown");

            let normalized = raw.normalize("openrouter").expect("normalization");

            assert!(raw_reasoning > 0);
            assert_eq!(normalized.usage.reasoning_tokens, raw_reasoning);
        },
    )
    .await;

    assert_recorded_reasoning_tokens(SCENARIO);
    assert_recorded_provider(SCENARIO, "OpenAI");
}

/// The fields the mapping already handled must keep working: adding a field to
/// `Usage` must not disturb cost or the prompt-token breakdown.
///
/// Scope note: this route reports `cached_tokens: 0`, so the cached-token
/// assertion below compares zero against zero and cannot fail on its own. It
/// is kept as a shape check (the `prompt_tokens_details` object still decodes
/// and still reaches the normalized field) — the load-bearing assertions here
/// are `cost > 0` and the presence of *both* detail objects. Proving a
/// non-zero cache hit needs a prompt-caching turn, which this matrix does not
/// record.
#[tokio::test]
async fn blocking_cost_and_cache_details_still_map() {
    const SCENARIO: &str = "reasoning_usage_matrix/blocking_cost_and_cache_details_still_map";

    with_openrouter_usage_cassette(
        "reasoning_usage_matrix/blocking_cost_and_cache_details_still_map",
        |client| async move {
            let model = client.completion_model(O4_MINI);
            let request = model
                .completion_request(REASONING_PROMPT)
                .max_tokens(CAP)
                .additional_params(openai_reasoning("medium"))
                .build();

            let raw = model.raw_completion(request).await.expect("raw turn");
            let usage = raw.usage().expect("usage");

            assert!(usage.cost > 0.0, "{usage:?}");
            assert!(usage.prompt_tokens_details.is_some(), "{usage:?}");
            assert!(usage.completion_tokens_details.is_some(), "{usage:?}");

            let normalized = raw.normalize("openrouter").expect("normalization");
            // Shape check only — see the scope note above.
            assert_eq!(
                normalized.usage.cached_input_tokens,
                usage
                    .prompt_tokens_details
                    .as_ref()
                    .map_or(0, |d| d.cached_tokens as u64)
            );
            // These two do carry weight: the new field must not have displaced
            // the cost mapping, and the reasoning share must still arrive.
            assert!(
                normalized.usage.reasoning_tokens > 0,
                "{:?}",
                normalized.usage
            );
            assert_eq!(
                normalized.usage.output_tokens,
                usage.completion_tokens as u64
            );
        },
    )
    .await;

    assert_recorded_reasoning_tokens(SCENARIO);
    assert_recorded_provider(SCENARIO, "OpenAI");
}

// ---------------------------------------------------------------------------
// Controls — a non-reasoning route must still report zero.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn control_non_reasoning_model_reports_zero_blocking() {
    const SCENARIO: &str =
        "reasoning_usage_matrix/control_non_reasoning_model_reports_zero_blocking";

    with_openrouter_usage_cassette(
        "reasoning_usage_matrix/control_non_reasoning_model_reports_zero_blocking",
        |client| async move {
            let model = client.completion_model(PLAIN_MODEL);
            let request = model
                .completion_request(PLAIN_PROMPT)
                .max_tokens(32)
                .additional_params(pinned("OpenAI"))
                .build();

            let response = model.completion(request).await.expect("plain turn");
            assert_eq!(response.usage.reasoning_tokens, 0);
            assert!(response.usage.output_tokens > 0);
        },
    )
    .await;

    assert_recorded_no_reasoning_tokens(SCENARIO);
    assert_recorded_provider(SCENARIO, "OpenAI");
}

#[tokio::test]
async fn control_non_reasoning_model_reports_zero_streaming() {
    const SCENARIO: &str =
        "reasoning_usage_matrix/control_non_reasoning_model_reports_zero_streaming";

    with_openrouter_usage_cassette(
        "reasoning_usage_matrix/control_non_reasoning_model_reports_zero_streaming",
        |client| async move {
            let model = client.completion_model(PLAIN_MODEL);
            let request = model
                .completion_request(PLAIN_PROMPT)
                .max_tokens(32)
                .additional_params(pinned("OpenAI"))
                .build();

            let stream = model.stream(request).await.expect("stream should connect");
            let (_, terminal) = collect_text_and_terminal(stream).await;
            let terminal = terminal.expect("terminal record");

            assert_eq!(terminal.usage.reasoning_tokens, 0);
            assert!(terminal.usage.output_tokens > 0);
        },
    )
    .await;

    assert_recorded_no_reasoning_tokens(SCENARIO);
    assert_recorded_provider(SCENARIO, "OpenAI");
}

// ---------------------------------------------------------------------------
// Premise assertions — derived from each cell's own recorded bytes.
// ---------------------------------------------------------------------------

fn recorded_response_bodies(scenario: &str) -> Vec<String> {
    let path = cassettes::cassette_path("openrouter", scenario);
    let contents = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "provider cassette {} should be readable after recording: {error}",
            path.display()
        )
    });

    serde_yaml::Deserializer::from_str(&contents)
        .map(|document| serde_yaml::Value::deserialize(document).expect("cassette interaction"))
        .filter_map(|interaction| {
            interaction
                .get("then")
                .and_then(|then| then.get("body"))
                .and_then(serde_yaml::Value::as_str)
                .map(ToOwned::to_owned)
        })
        .collect()
}

/// Every JSON object a recorded body holds — the blocking response itself, or
/// each SSE `data:` frame of a streamed one.
fn recorded_payloads(scenario: &str) -> Vec<Value> {
    recorded_response_bodies(scenario)
        .iter()
        .flat_map(|body| {
            if let Ok(value) = serde_json::from_str::<Value>(body) {
                return vec![value];
            }
            body.lines()
                .filter_map(|line| line.strip_prefix("data:"))
                .map(str::trim)
                .filter(|data| !data.is_empty() && *data != "[DONE]")
                .filter_map(|data| serde_json::from_str::<Value>(data).ok())
                .collect()
        })
        .collect()
}

/// Every non-zero `usage.completion_tokens_details.reasoning_tokens` the
/// cassette recorded.
fn recorded_reasoning_token_counts(scenario: &str) -> Vec<u64> {
    recorded_payloads(scenario)
        .iter()
        .filter_map(|payload| {
            payload
                .get("usage")?
                .get("completion_tokens_details")?
                .get("reasoning_tokens")?
                .as_u64()
                .filter(|count| *count > 0)
        })
        .collect()
}

fn recorded_reasoning_tokens(scenario: &str) -> u64 {
    recorded_reasoning_token_counts(scenario)
        .into_iter()
        .next()
        .unwrap_or_else(|| {
            panic!(
                "cassette {scenario} records no non-zero \
                 `usage.completion_tokens_details.reasoning_tokens`"
            )
        })
}

fn assert_recorded_reasoning_tokens(scenario: &str) {
    assert!(
        !recorded_reasoning_token_counts(scenario).is_empty(),
        "cassette {scenario} no longer records a non-zero \
         `usage.completion_tokens_details.reasoning_tokens`; this cell would \
         pass while covering nothing"
    );
}

/// The exact set of `reasoning_details[].type` values the cassette recorded.
///
/// Some cells make a claim about *which kinds* of reasoning detail a route
/// returns (notably `reasoning.exclude`). Prose in a doc comment cannot fail,
/// so the claim is pinned to the recorded bytes here instead.
fn assert_recorded_reasoning_detail_types(scenario: &str, expected: &[&str]) {
    let mut found = recorded_payloads(scenario)
        .iter()
        .filter_map(|payload| payload.get("choices")?.as_array().cloned())
        .flatten()
        .filter_map(|choice| {
            let message = choice.get("message").or_else(|| choice.get("delta"))?;
            Some(
                message
                    .get("reasoning_details")?
                    .as_array()?
                    .iter()
                    .filter_map(|detail| detail.get("type")?.as_str().map(ToOwned::to_owned))
                    .collect::<Vec<_>>(),
            )
        })
        .flatten()
        .collect::<Vec<_>>();
    found.sort();
    found.dedup();

    let mut expected = expected
        .iter()
        .map(|kind| (*kind).to_owned())
        .collect::<Vec<_>>();
    expected.sort();

    assert_eq!(
        found, expected,
        "cassette {scenario} records reasoning detail types {found:?}, not {expected:?}; \
         the cell's claim about which kinds this route returns is now stale"
    );
}

fn assert_recorded_no_reasoning_tokens(scenario: &str) {
    assert!(
        recorded_reasoning_token_counts(scenario).is_empty(),
        "control cassette {scenario} unexpectedly records non-zero reasoning tokens"
    );
}

fn assert_recorded_provider(scenario: &str, expected: &str) {
    let providers = recorded_payloads(scenario)
        .iter()
        .filter_map(|value| {
            value
                .get("provider")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
        })
        .collect::<Vec<_>>();

    assert!(
        !providers.is_empty(),
        "cassette {scenario} records no `provider` field, so its routing premise is unproven"
    );
    assert!(
        providers
            .iter()
            .all(|provider| provider.as_str() == expected),
        "cassette {scenario} was recorded against {providers:?}, not the pinned {expected}"
    );
}
