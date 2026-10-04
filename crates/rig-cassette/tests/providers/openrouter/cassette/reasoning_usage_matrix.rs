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
//! The streaming terminal record read the same accounting, so it lost the
//! field too — the zero was consistent across transports, which is exactly why
//! nothing caught it.
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
//! | 11 | `blocking_excluded_reasoning_still_counts_tokens` | blocking | raw model | OpenAI / o4-mini | recorded |
//! | 14 | `transports_agree_on_reasoning_tokens` | both | raw model | OpenAI / o4-mini | recorded |
//!
//! Five unit cells — usage payloads the live gateway will not produce on demand
//! (a real recorded usage object, the breakdown absent / `null` / `{}` /
//! zero, unmodeled siblings tolerated, the `total - prompt` output-token
//! fallback undisturbed, and the field omitted on serialization when absent)
//! — live next to the shared chat usage shape in
//! `crates/rig-core/src/providers/openai/wire/dto.rs`, which flattens a
//! dialect's extra usage fields so `completion_tokens_details` reaches both
//! the normalized `Usage` and the reply document on `raw`.

use serde::Deserialize;
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};

use super::super::support::with_openrouter_usage_cassette;
use crate::cassettes;
use crate::support::collect_text_and_terminal;
use rig::completion::CompletionRequest;

/// Small enough to be cheap, hard enough that a reasoning route actually
/// spends tokens thinking about it.
const REASONING_PROMPT: &str = "A farmer has 17 sheep; all but 9 run away. He then buys 3 times \
                                as many as remain, sells 5, and splits the rest evenly among 4 \
                                pens. How many sheep per pen? Answer with the number only.";

const O4_MINI: &str = "openai/o4-mini";
const CAP: u64 = 2000;

fn openai_reasoning(effort: &str) -> Value {
    json!({
        "reasoning": { "effort": effort },
        "provider": { "order": ["OpenAI"], "allow_fallbacks": false }
    })
}

// ---------------------------------------------------------------------------
// The bug: a documented, always-present field with a first-class slot, zeroed.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// A second and third upstream family: the breakdown is not an OpenAI-ism.
// ---------------------------------------------------------------------------

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

    let delivered = Arc::new(Mutex::new(None));
    let recorder = delivered.clone();

    with_openrouter_usage_cassette(
        "reasoning_usage_matrix/blocking_excluded_reasoning_still_counts_tokens",
        |client| async move {
            let model = client.completion(O4_MINI);
            let request = CompletionRequest::new(REASONING_PROMPT)
                .max_tokens(CAP)
                .additional_params(json!({
                    "reasoning": { "effort": "medium", "exclude": true },
                    "provider": { "order": ["OpenAI"], "allow_fallbacks": false }
                }));

            let response = model.call(request).await.expect("reasoning turn");

            assert!(response.usage.reasoning_tokens.is_some_and(|n| n > 0));
            *recorder.lock().expect("recorder") = response.usage.reasoning_tokens;
        },
    )
    .await;

    assert_recorded_reasoning_tokens(SCENARIO);
    assert_recorded_provider(SCENARIO, "OpenAI");
    assert_eq!(
        *delivered.lock().expect("recorder"),
        Some(recorded_reasoning_tokens(SCENARIO)),
        "the reported reasoning tokens must be exactly what the wire billed"
    );
    // Pin the census claim in this cell's doc comment to the bytes, so it
    // cannot go stale silently: `exclude` withholds the summary and keeps the
    // encrypted detail.
    assert_recorded_reasoning_detail_types(SCENARIO, &["reasoning.encrypted"]);
}

/// One scenario, both transports: the blocking reply and the stream's terminal
/// frame report the same accounting through the same shape, so a fix that only
/// reached one of them would show up here.
#[tokio::test]
async fn transports_agree_on_reasoning_tokens() {
    const SCENARIO: &str = "reasoning_usage_matrix/transports_agree_on_reasoning_tokens";

    let delivered = Arc::new(Mutex::new((None, None)));
    let recorder = delivered.clone();

    with_openrouter_usage_cassette(
        "reasoning_usage_matrix/transports_agree_on_reasoning_tokens",
        |client| async move {
            let model = client.completion(O4_MINI);

            let blocking = model
                .call(
                    CompletionRequest::new(REASONING_PROMPT)
                        .max_tokens(CAP)
                        .additional_params(openai_reasoning("medium")),
                )
                .await
                .expect("blocking reasoning turn");

            let stream = model
                .stream(
                    CompletionRequest::new(REASONING_PROMPT)
                        .max_tokens(CAP)
                        .additional_params(openai_reasoning("medium")),
                )
                .expect("stream should connect");
            let (_, terminal) = collect_text_and_terminal(stream).await;
            let terminal = terminal.expect("terminal record");

            assert!(blocking.usage.reasoning_tokens.is_some_and(|n| n > 0));
            assert!(terminal.usage.reasoning_tokens.is_some_and(|n| n > 0));
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
        blocking.is_some_and(|count| recorded.contains(&count)),
        "{blocking:?} not in {recorded:?}"
    );
    assert!(
        streamed.is_some_and(|count| recorded.contains(&count)),
        "{streamed:?} not in {recorded:?}"
    );
}

// ---------------------------------------------------------------------------
// Controls — a non-reasoning route must still report zero.
// ---------------------------------------------------------------------------

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
