//! Raw provider response capture on DeepSeek's blocking chat-completions path.
//!
//! **The feature.** Every blocking completion attaches the provider's own
//! reply to the normalized [`rig::completion::CompletionResponse::raw`].
//! Capture is always on: there is no flag to request it, nothing about it
//! reaches the wire, and a `Value::Null` only ever means a response built by
//! hand with no provider payload behind it.
//!
//! **What `raw` is.** The driver fills it from the reply's bytes, so it is
//! DeepSeek's response *document* rather than a re-serialization of whatever
//! the decoder parsed — which is why [`deepseek::CompletionResponse`]
//! deserializes straight out of it and is the typed escape hatch for the
//! fields the normalized view has no slot for. DeepSeek is worth its own
//! matrix because its usage block carries a `prompt_cache_hit_tokens` /
//! `prompt_cache_miss_tokens` split rig only half-normalizes: the hit count
//! reaches `Usage::cached_input_tokens`, the miss count has no slot at all
//! and is reachable only through `raw`.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `raw_round_trips_deepseek_type` | typed read-back | `raw` deserializes into `deepseek::CompletionResponse`, whose provider-native fields are the normalized response's | recorded |
//! | 2 | `raw_exposes_prompt_cache_miss_tokens` | provider-only field | `raw.usage.prompt_cache_miss_tokens` and `raw.system_fingerprint` equal the fixture body, and neither has a normalized slot | recorded |
//! | 3 | `normalized_fields_match_raw_renormalized` | normalized view | the response reproduces its fixture bytes, and the typed view of its own `raw` agrees with it field for field | recorded |
//! | 4 | `reasoning_raw_round_trips_and_exposes_reasoning_content` | reasoning turn | a thinking-mode turn's `raw` reads back into `deepseek::CompletionResponse`; its `choices[0].message.reasoning_content` is the fixture's reasoning string, which the normalized view carries only as a `Reasoning` block under a different spelling | recorded |
//!
//! The scenario literals — and so the fixture directories — keep the names
//! they were recorded under.
//!
//! Every cell is recorded. Each re-derives its premise from its own fixture
//! after the wrapper returns: cell 2 reads the miss count out of the recorded
//! body rather than trusting the number the typed view reports, cell 3
//! checks the normalized fields against the recorded body — the shared
//! chat-completions field set plus DeepSeek's own cache-hit rule — before
//! comparing them with the typed view of `raw`, and cell 4 reads the
//! reasoning string out of the recorded body (and the `thinking` toggle out
//! of the recorded request), so a recording that stopped carrying a usage
//! block, a finish reason, or a reasoning block fails loudly instead of
//! covering nothing.

use rig::completion::{CompletionModel, CompletionRequest, CompletionResponse};
use rig::message::{AssistantContent, ReasoningContent};
use rig::providers::deepseek;
use serde::Deserialize;
use serde_json::json;

use super::support::with_deepseek_cassette_result;
use crate::cassettes::recorded_json_turn;
use crate::raw_capture::{assert_no_request_id, capture_completion, chat};
use crate::support::{Observed, assert_matches_recorded_token, assistant_text, json_contains_key};

const PROVIDER: &str = "deepseek";
const MODEL: &str = deepseek::DEEPSEEK_V4_FLASH;
const PROMPT: &str = "Reply with the single word: pong";
/// A question small enough that a thinking-mode turn reasons briefly and
/// still answers within the budget.
const REASONING_PROMPT: &str = "What is 17 multiplied by 23? Reply with only the number.";
/// A thinking-mode turn spends most of its budget on reasoning tokens before
/// it answers, so the reasoning cell needs real headroom.
const REASONING_BUDGET: u64 = 640;

fn request(model: &(impl CompletionModel + Clone)) -> CompletionRequest {
    model
        .completion_request(PROMPT)
        .additional_params(json!({ "thinking": { "type": "disabled" } }))
        .max_tokens(16)
        .build()
}

/// The thinking-mode request shape the `reasoning_*` modules use.
fn reasoning_request(model: &(impl CompletionModel + Clone)) -> CompletionRequest {
    model
        .completion_request(REASONING_PROMPT)
        .additional_params(json!({ "thinking": { "type": "enabled" } }))
        .max_tokens(REASONING_BUDGET)
        .build()
}

fn reasoning_text_of(choice: &[AssistantContent]) -> String {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Reasoning(reasoning) => Some(reasoning),
            _ => None,
        })
        .flat_map(|reasoning| reasoning.content.iter())
        .filter_map(|content| match content {
            ReasoningContent::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

/// DeepSeek's own view of a reply, checked against the normalized response it
/// rode on.
///
/// One decoder produces the normalized response, and this is the typed read
/// of the very document that decoder read, so agreement here pins that
/// mapping rather than comparing it with a second mapping of the same bytes.
///
/// Not [`chat::assert_native_matches_normalized`]: that one reads the shared
/// [`rig::providers::openai::CompletionResponse`], and the point of this one
/// is DeepSeek's own type — including the half of its cache split that gets
/// normalized, which no shared chat type models.
fn assert_typed_view_matches(typed: &deepseek::CompletionResponse, response: &CompletionResponse) {
    assert_eq!(
        typed.id.as_deref(),
        response.response_id.as_deref(),
        "response id"
    );
    assert_eq!(typed.model, response.model, "model");
    let choice = typed.choices.first().expect("a reply carries a choice");
    assert_eq!(
        Some(chat::native_finish_reason(&choice.finish_reason)),
        response.finish_reason(),
        "finish reason"
    );
    assert_eq!(
        Some(u64::from(typed.usage.prompt_tokens)),
        response.usage.input_tokens,
        "input tokens"
    );
    assert_eq!(
        Some(u64::from(typed.usage.completion_tokens)),
        response.usage.output_tokens,
        "output tokens"
    );
    assert_eq!(
        Some(u64::from(typed.usage.prompt_cache_hit_tokens)),
        response.usage.cached_input_tokens,
        "the hit count is the half of the split that gets normalized"
    );
    let deepseek::Message::Assistant { content, .. } = &choice.message;
    assert_eq!(&assistant_text(&response.choice), content, "choice text");
}

// ================================================================
// 1. raw reads back as DeepSeek's own type
// ================================================================

#[tokio::test]
async fn raw_round_trips_deepseek_type() {
    const SCENARIO: &str = "raw_capture_matrix/raw_round_trips_deepseek_type";
    let sink = Observed::default();
    with_deepseek_cassette_result(
        "raw_capture_matrix/raw_round_trips_deepseek_type",
        |client| capture_completion(client.completion(MODEL), request, sink.clone()),
    )
    .await
    .expect("raw_round_trips_deepseek_type should replay from its cassette");
    let response = sink.take();

    let typed = deepseek::CompletionResponse::deserialize(&response.raw)
        .expect("raw reads back as DeepSeek's own CompletionResponse");
    assert_typed_view_matches(&typed, &response);
    // And `raw` is the reply document rather than a re-serialization
    // of that parse, so it keeps what neither view models.
    assert!(
        response.raw["usage"]["prompt_cache_miss_tokens"].is_u64(),
        "the document keeps DeepSeek's miss count: {}",
        response.raw
    );

    let (_, response_body) = recorded_json_turn(PROVIDER, SCENARIO);
    assert!(
        response_body["choices"][0]["message"]["content"].is_string(),
        "the recorded turn should be a plain text answer"
    );
}

// ================================================================
// 2. A field the normalized response provably lacks
// ================================================================

#[tokio::test]
async fn raw_exposes_prompt_cache_miss_tokens() {
    const SCENARIO: &str = "raw_capture_matrix/raw_exposes_prompt_cache_miss_tokens";
    let sink = Observed::default();
    with_deepseek_cassette_result(
        "raw_capture_matrix/raw_exposes_prompt_cache_miss_tokens",
        |client| capture_completion(client.completion(MODEL), request, sink.clone()),
    )
    .await
    .expect("raw_exposes_prompt_cache_miss_tokens should replay from its cassette");
    let response = sink.take();

    let (_, body) = recorded_json_turn(PROVIDER, SCENARIO);
    let recorded_miss = body["usage"]["prompt_cache_miss_tokens"]
        .as_u64()
        .expect("DeepSeek reports prompt_cache_miss_tokens on every usage block");
    let recorded_fingerprint = body["system_fingerprint"]
        .as_str()
        .expect("DeepSeek reports a system_fingerprint");

    let raw = &response.raw;
    assert_eq!(
        raw["usage"]["prompt_cache_miss_tokens"],
        json!(recorded_miss)
    );
    assert_matches_recorded_token(
        raw["system_fingerprint"].as_str(),
        Some(recorded_fingerprint),
        "system fingerprint",
    );
    // And the normalized view has no slot for either: the miss count is not
    // any of the normalized usage counters, and there is no fingerprint field.
    let normalized_usage = serde_json::to_value(response.usage).expect("usage serializes");
    assert!(
        normalized_usage.get("prompt_cache_miss_tokens").is_none(),
        "the normalized usage has no miss-count slot: {normalized_usage}"
    );
    let normalized = serde_json::to_value(&response).expect("response serializes");
    assert!(normalized.get("system_fingerprint").is_none());
}

// ================================================================
// 3. The normalized view and raw tell one story
// ================================================================

#[tokio::test]
async fn normalized_fields_match_raw_renormalized() {
    const SCENARIO: &str = "raw_capture_matrix/normalized_fields_match_raw_renormalized";
    let sink = Observed::default();
    with_deepseek_cassette_result(
        "raw_capture_matrix/normalized_fields_match_raw_renormalized",
        |client| capture_completion(client.completion(MODEL), request, sink.clone()),
    )
    .await
    .expect("normalized_fields_match_raw_renormalized should replay from its cassette");
    let response = sink.take();

    let (_, body) = recorded_json_turn(PROVIDER, SCENARIO);
    chat::assert_reproduces_body(&response, PROVIDER, &body, "the recorded body");
    // The shared chat field set stops at the three OpenAI-compatible
    // counters; DeepSeek's cache split is its own rule, and only the hit
    // half reaches a normalized slot.
    assert_eq!(
        response.usage.cached_input_tokens,
        body["usage"]["prompt_cache_hit_tokens"].as_u64(),
        "cached input tokens come from prompt_cache_hit_tokens"
    );
    // The transport id is never on DeepSeek's wire: it contracts no
    // request-id header, so `None` is the documented outcome.
    assert_no_request_id(response.provider_request_id.as_deref(), PROVIDER);

    // One seam, two views: the typed read of the response's own `raw` is the
    // same reply the normalized fields describe. Capture adds a view; there
    // is only one mapping left, and this is what pins it.
    let typed = deepseek::CompletionResponse::deserialize(&response.raw)
        .expect("raw reads back as DeepSeek's own type");
    assert_typed_view_matches(&typed, &response);
}

// ================================================================
// 4. A thinking-mode turn: raw reads back and carries reasoning_content
// ================================================================

#[tokio::test]
async fn reasoning_raw_round_trips_and_exposes_reasoning_content() {
    const SCENARIO: &str =
        "raw_capture_matrix/reasoning_raw_round_trips_and_exposes_reasoning_content";
    let sink = Observed::default();
    with_deepseek_cassette_result(
        "raw_capture_matrix/reasoning_raw_round_trips_and_exposes_reasoning_content",
        |client| capture_completion(client.completion(MODEL), reasoning_request, sink.clone()),
    )
    .await
    .expect(
        "reasoning_raw_round_trips_and_exposes_reasoning_content should replay from its cassette",
    );
    let response = sink.take();

    let (request_body, body) = recorded_json_turn(PROVIDER, SCENARIO);
    // Premise, from the bytes: thinking was asked for and the recorded turn
    // carries a non-empty reasoning string next to its answer.
    assert_eq!(request_body["thinking"], json!({ "type": "enabled" }));
    let recorded_reasoning = body["choices"][0]["message"]["reasoning_content"]
        .as_str()
        .expect("a thinking-mode turn carries message.reasoning_content");
    assert!(
        !recorded_reasoning.trim().is_empty(),
        "the recorded reasoning must not be empty"
    );
    assert!(
        body["choices"][0]["message"]["content"].is_string(),
        "the recorded turn should still carry an answer"
    );

    // The typed view of `raw` carries the wire's own spelling of the
    // reasoning block, and agrees with the normalized response beside it.
    let typed = deepseek::CompletionResponse::deserialize(&response.raw)
        .expect("raw reads back as DeepSeek's own CompletionResponse");
    assert_typed_view_matches(&typed, &response);
    let deepseek::Message::Assistant {
        reasoning_content, ..
    } = &typed
        .choices
        .first()
        .expect("a reply carries a choice")
        .message;
    assert_eq!(
        reasoning_content.as_deref(),
        Some(recorded_reasoning),
        "the typed view carries the fixture's reasoning_content verbatim"
    );
    // The normalized view carries the same text, but only as a `Reasoning`
    // content block: there is no `reasoning_content` key anywhere on it.
    assert_eq!(
        reasoning_text_of(&response.choice),
        recorded_reasoning,
        "the normalized Reasoning block is the same text raw spells reasoning_content"
    );
    let normalized_choice = serde_json::to_value(&response.choice).expect("choice serializes");
    assert!(
        !json_contains_key(&normalized_choice, "reasoning_content"),
        "the normalized choice never spells reasoning_content: {normalized_choice}"
    );
}
