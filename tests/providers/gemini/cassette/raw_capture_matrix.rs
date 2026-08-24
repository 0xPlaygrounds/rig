//! Feature matrix for raw provider response capture on the Gemini REST
//! (`generateContent`) unary seam.
//!
//! # The feature
//!
//! Raw capture is always on: `CompletionModel::completion` serializes the value
//! its inherent `raw_completion` returned — here Gemini's own
//! [`GenerateContentResponse`] — onto [`rig::completion::CompletionResponse::raw`]
//! before `try_into` normalizes it. There is no opt-in and nothing about it
//! reaches the wire; `raw` is `Value::Null` only on a response constructed
//! without a provider payload behind it (hand-built, or persisted before the
//! field existed), never because capture "was not requested".
//!
//! # Matrix
//!
//! `expected` is what the caller observes on the normalized response. Every
//! recorded cell re-derives its premise from its own fixture bytes after the
//! wrapper returns (record mode writes the fixture on the way out): a cell
//! whose recording lost the wire shape it claims to cover must fail loudly.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `raw_roundtrips_generate_content_response` | typed access | `GenerateContentResponse::deserialize(&*raw)` re-serializes equal, and its `try_into` reproduces the normalized response | recorded |
//! | 2 | `raw_exposes_prompt_tokens_details` | un-normalized field | `usageMetadata.promptTokensDetails` == fixture, absent from the normalized response | recorded |
//! | 3 | `raw_exposes_forced_function_call` | forced tool call (`ToolChoice::Specific`) | `raw` round-trips; `candidates[0].content.parts[].functionCall` and `finishMessage` == fixture; raw `finishReason` spelled `"STOP"` while `finish_reason() == ToolCalls` | recorded |
//! | 4 | `raw_exposes_structured_output_turn` | structured output (`responseMimeType: application/json` + `responseJsonSchema`) | `raw` round-trips; raw `finishReason` spelled `"STOP"` and `usageMetadata.promptTokensDetails` == fixture while the normalized response carries neither | recorded |
//!
//! Every cell is recorded: `GEMINI_API_KEY` was available and the seam under
//! test is the plain `generateContent` route, so nothing needs a unit stand-in.
//!
//! Cell 1 also carries the "one story" contract: re-normalizing `raw` by hand
//! lands on the same choice / finish reason / model / usage / identity the
//! typed route reported, so `raw` and the normalized response can never
//! disagree about the turn they describe.
//!
//! The un-normalized field of choice is `usageMetadata.promptTokensDetails`
//! (a per-modality token breakdown): `modelVersion` and `responseId` are
//! normalized into `model` / `response_id`, and `responseId` is scrubbed on
//! the way into the fixture, so neither would prove the raw value survives
//! against the recorded bytes.
//!
//! Cells 3–4 cover the two wire shapes a text turn never produces. A forced
//! `functionCall` turn is where Gemini's own `finishReason` (`"STOP"`, even
//! on a call-only turn) and rig's normalized `ToolCalls` visibly disagree,
//! so `raw` must keep the wire spelling while the normalized response reports
//! the upgraded reason. A structured-output turn (rig's `output_schema` maps
//! onto `generationConfig.responseMimeType` + `responseJsonSchema`) proves the
//! same provider-only fields survive when the response text is schema JSON;
//! the request side of that premise is read back from the recorded request's
//! `generationConfig`.

use rig::completion::{
    AssistantContent, CompletionModel as _, CompletionResponse as RigCompletionResponse,
    FinishReason,
};
use rig::message::ToolChoice;
use rig::prelude::*;
use rig::providers::gemini;
use rig::providers::gemini::completion::gemini_api_types::GenerateContentResponse;
use rig::tool::Tool;
use serde::Deserialize;
use serde_json::Value;
use std::sync::{Arc, Mutex};

use super::super::support::{recorded_request_generation_configs, with_gemini_cassette};
use crate::support::{Adder, STRUCTURED_OUTPUT_PROMPT, SmokeStructuredOutput};

const PROVIDER: &str = "gemini";

/// Cheap, non-thinking, so the recorded body stays small.
const MODEL: &str = "gemini-2.5-flash-lite";

const PROMPT: &str = "Reply with exactly this one word and nothing else: captured";

/// A prompt the forced-tool cell can only satisfy by calling `add`.
const TOOL_PROMPT: &str = "Use the add tool to add 2 and 3.";

fn request(model: &gemini::CompletionModel) -> rig::completion::CompletionRequest {
    model.completion_request(PROMPT).temperature(0.0).build()
}

/// The forced-tool request: `add` is offered and `ToolChoice::Specific` pins
/// the turn to it (Gemini `functionCallingConfig.mode: ANY` with
/// `allowedFunctionNames`), so the recorded turn is a `functionCall` part.
fn forced_tool_request(model: &gemini::CompletionModel) -> rig::completion::CompletionRequest {
    model
        .completion_request(TOOL_PROMPT)
        .temperature(0.0)
        .tool(rig::tool::tool_definition(&Adder))
        .tool_choice(ToolChoice::Specific {
            function_names: vec![Adder::NAME.to_string()],
        })
        .build()
}

/// The structured-output request: rig maps `output_schema` onto
/// `generationConfig.responseMimeType: application/json` +
/// `responseJsonSchema`, Gemini's native structured-output controls.
fn structured_output_request(
    model: &gemini::CompletionModel,
) -> rig::completion::CompletionRequest {
    model
        .completion_request(STRUCTURED_OUTPUT_PROMPT)
        .temperature(0.0)
        .output_schema(schemars::schema_for!(SmokeStructuredOutput))
        .build()
}

/// The premise every cell rests on: the recorded body is a `generateContent`
/// answer whose first candidate stopped naturally and whose usage carries the
/// per-modality prompt breakdown cell 2 reads.
fn assert_recorded_generate_content_body(scenario: &str) -> Value {
    let body = crate::cassettes::recorded_json_response(PROVIDER, scenario);
    assert_eq!(
        body.pointer("/candidates/0/finishReason"),
        Some(&Value::String("STOP".to_string())),
        "{scenario}: the recorded turn should have stopped naturally"
    );
    assert!(
        body.pointer("/usageMetadata/promptTokensDetails")
            .and_then(Value::as_array)
            .is_some_and(|details| !details.is_empty()),
        "{scenario}: the recorded usageMetadata should carry promptTokensDetails, the \
         un-normalized field this matrix reads through `raw`"
    );
    body
}

/// The premise of the forced-tool cell: the recorded body's first candidate
/// carries a `functionCall` part naming `add`, and Gemini still spelled the
/// finish reason `"STOP"` — the wire shape rig upgrades to `ToolCalls`.
fn assert_recorded_function_call_body(scenario: &str) -> Value {
    let body = crate::cassettes::recorded_json_response(PROVIDER, scenario);
    assert_eq!(
        body.pointer("/candidates/0/finishReason"),
        Some(&Value::String("STOP".to_string())),
        "{scenario}: Gemini spells a call-only turn's finishReason STOP; this cell exists to \
         show raw keeps that spelling while the normalized reason is ToolCalls"
    );
    let parts = body
        .pointer("/candidates/0/content/parts")
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{scenario}: the recorded candidate should carry parts"));
    assert!(
        parts
            .iter()
            .any(|part| part.pointer("/functionCall/name") == Some(&Value::String("add".into()))),
        "{scenario}: the recorded turn should carry a functionCall part naming `add`, the wire \
         shape this cell reads through `raw`; got {parts:?}"
    );
    body
}

/// The premise of the structured-output cell, on both sides of the wire: the
/// recorded request asked for `application/json` against a JSON schema, and
/// the recorded body answered with a natural stop and the usage breakdown.
fn assert_recorded_structured_output_turn(scenario: &str) -> Value {
    let configs = recorded_request_generation_configs(scenario);
    assert_eq!(configs.len(), 1, "{scenario}: one recorded turn");
    assert_eq!(
        configs[0].get("responseMimeType"),
        Some(&Value::String("application/json".to_string())),
        "{scenario}: the recorded request should ask Gemini for application/json"
    );
    assert!(
        configs[0]
            .get("responseJsonSchema")
            .and_then(Value::as_object)
            .is_some_and(|schema| schema.contains_key("properties")),
        "{scenario}: the recorded request should carry the responseJsonSchema rig maps \
         output_schema onto; got {:?}",
        configs[0]
    );
    let body = assert_recorded_generate_content_body(scenario);
    let text = body
        .pointer("/candidates/0/content/parts/0/text")
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("{scenario}: the recorded candidate should carry a text part"));
    serde_json::from_str::<SmokeStructuredOutput>(text).unwrap_or_else(|error| {
        panic!("{scenario}: the recorded text should be schema JSON: {error}: {text}")
    });
    body
}

/// Serialize the normalized response with `raw` removed, so a cell can prove a
/// field is *not* reachable through the normalized surface (as opposed to
/// merely re-reading it out of `raw`).
fn normalized_without_raw(response: &RigCompletionResponse) -> Value {
    let mut value = serde_json::to_value(response).expect("normalized response serializes");
    value
        .as_object_mut()
        .expect("normalized response is an object")
        .remove("raw");
    value
}

/// The `(name, arguments)` of every tool call in a choice — the part of a
/// Gemini `functionCall` the wire actually carries (Gemini assigns no call
/// id; rig mints one per normalization).
fn tool_functions(choice: &[AssistantContent]) -> Vec<(String, Value)> {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::ToolCall(call) => {
                Some((call.function.name.clone(), call.function.arguments.clone()))
            }
            _ => None,
        })
        .collect()
}

fn contains_key(value: &Value, needle: &str) -> bool {
    match value {
        Value::Object(map) => map
            .iter()
            .any(|(key, value)| key == needle || contains_key(value, needle)),
        Value::Array(items) => items.iter().any(|item| contains_key(item, needle)),
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// 1: typed access is recoverable, and re-normalizes to the same story
// ---------------------------------------------------------------------------

#[tokio::test]
async fn raw_roundtrips_generate_content_response() {
    const SCENARIO: &str = "raw_capture_matrix/raw_roundtrips_generate_content_response";
    with_gemini_cassette(
        "raw_capture_matrix/raw_roundtrips_generate_content_response",
        |client| async move {
            let model = client.completion_model(MODEL);
            let response = model
                .completion(request(&model))
                .await
                .expect("completion should succeed");

            let raw = &response.raw;

            // `raw` is the value `raw_completion` returned, serialized: Gemini's
            // own type reads it back, and re-serializing that typed value
            // reproduces `raw` exactly — nothing is lost through the escape hatch.
            let typed = GenerateContentResponse::deserialize(raw)
                .expect("raw must deserialize into Gemini's GenerateContentResponse");
            assert_eq!(
                serde_json::to_value(&typed).expect("typed raw re-serializes"),
                *raw,
                "GenerateContentResponse must round-trip through its own Serialize/Deserialize"
            );

            // The typed value agrees with the normalized fields next to it.
            assert_eq!(typed.model_version.as_deref(), response.model.as_deref());
            assert_eq!(
                Some(typed.response_id.as_str()),
                response.response_id.as_deref()
            );
            assert_eq!(
                typed
                    .usage_metadata
                    .as_ref()
                    .map(|usage| usage.prompt_token_count as u64),
                Some(response.usage.input_tokens)
            );

            // And re-normalizing it by hand tells the same story the typed
            // route told: `raw` is additive, never a divergent second view.
            let renormalized: RigCompletionResponse =
                typed.try_into().expect("typed raw should normalize");
            assert_eq!(renormalized.choice, response.choice);
            assert_eq!(renormalized.finish_reason(), response.finish_reason());
            assert_eq!(renormalized.model, response.model);
            assert_eq!(renormalized.usage, response.usage);
            assert_eq!(renormalized.identity(), response.identity());
            assert_eq!(renormalized.provider, response.provider);
        },
    )
    .await;

    assert_recorded_generate_content_body(SCENARIO);
}

// ---------------------------------------------------------------------------
// 2: an un-normalized field is readable and matches the wire
// ---------------------------------------------------------------------------

#[tokio::test]
async fn raw_exposes_prompt_tokens_details() {
    const SCENARIO: &str = "raw_capture_matrix/raw_exposes_prompt_tokens_details";
    // The observed `raw` is compared against the fixture bytes only after the
    // wrapper returns, so it is carried out of the test body.
    let observed: Arc<Mutex<Option<Value>>> = Arc::new(Mutex::new(None));
    let sink = Arc::clone(&observed);
    with_gemini_cassette(
        "raw_capture_matrix/raw_exposes_prompt_tokens_details",
        |client| async move {
            let model = client.completion_model(MODEL);
            let response = model
                .completion(request(&model))
                .await
                .expect("completion should succeed");

            let raw = &response.raw;
            *sink.lock().expect("observation lock") = Some(raw.clone());

            // The normalized response provably lacks the field: it is only
            // reachable through `raw`.
            assert!(
                !contains_key(&normalized_without_raw(&response), "promptTokensDetails"),
                "promptTokensDetails is not part of rig's normalized response — it is exactly \
             the kind of provider detail `raw` exists to expose"
            );
        },
    )
    .await;

    let raw = observed
        .lock()
        .expect("observation lock")
        .take()
        .expect("the test body observed a raw payload");
    let body = assert_recorded_generate_content_body(SCENARIO);
    assert_eq!(
        raw.pointer("/usageMetadata/promptTokensDetails"),
        body.pointer("/usageMetadata/promptTokensDetails"),
        "raw must carry Gemini's promptTokensDetails exactly as the wire sent it"
    );
    assert_eq!(
        raw.pointer("/candidates/0/finishReason"),
        body.pointer("/candidates/0/finishReason"),
        "raw must keep Gemini's own finishReason spelling"
    );
    assert_eq!(
        raw.pointer("/usageMetadata/totalTokenCount"),
        body.pointer("/usageMetadata/totalTokenCount"),
        "raw must carry the wire's total token count untouched"
    );
}

// ---------------------------------------------------------------------------
// 3: a forced tool call keeps the wire's functionCall and finishReason
// ---------------------------------------------------------------------------

#[tokio::test]
async fn raw_exposes_forced_function_call() {
    const SCENARIO: &str = "raw_capture_matrix/raw_exposes_forced_function_call";
    let observed: Arc<Mutex<Option<Value>>> = Arc::new(Mutex::new(None));
    let sink = Arc::clone(&observed);
    with_gemini_cassette(
        "raw_capture_matrix/raw_exposes_forced_function_call",
        |client| async move {
            let model = client.completion_model(MODEL);
            let response = model
                .completion(forced_tool_request(&model))
                .await
                .expect("forced tool completion should succeed");

            let raw = &response.raw;
            *sink.lock().expect("observation lock") = Some(raw.clone());

            // The typed round trip holds for a functionCall turn too.
            let typed = GenerateContentResponse::deserialize(raw)
                .expect("raw must deserialize into Gemini's GenerateContentResponse");
            assert_eq!(
                serde_json::to_value(&typed).expect("typed raw re-serializes"),
                *raw,
                "GenerateContentResponse must round-trip a functionCall turn through its own \
                 Serialize/Deserialize"
            );
            let renormalized: RigCompletionResponse =
                typed.try_into().expect("typed raw should normalize");
            // Gemini `functionCall` parts carry no id, so rig mints one per
            // normalization: compare the choice by what the wire carried
            // (name + arguments), not by the minted id.
            assert_eq!(
                tool_functions(&renormalized.choice),
                tool_functions(&response.choice)
            );
            assert_eq!(renormalized.finish_reason(), response.finish_reason());
            assert_eq!(renormalized.usage, response.usage);

            // The normalized response says ToolCalls and carries the call as
            // a typed ToolCall …
            assert_eq!(response.finish_reason(), Some(FinishReason::ToolCalls));
            let call = response
                .choice
                .iter()
                .find_map(|content| match content {
                    AssistantContent::ToolCall(call) => Some(call),
                    _ => None,
                })
                .expect("the normalized choice carries the forced tool call");
            assert_eq!(call.function.name, Adder::NAME);
            assert_eq!(
                call.function.arguments,
                serde_json::json!({ "x": 2, "y": 3 })
            );

            // … while raw keeps Gemini's own spelling of both.
            assert_eq!(
                raw.pointer("/candidates/0/finishReason"),
                Some(&Value::String("STOP".to_string())),
                "raw keeps Gemini's finishReason spelling on a call-only turn"
            );
            assert_ne!(
                normalized_without_raw(&response).get("finish_reason"),
                Some(&Value::String("STOP".to_string())),
                "the normalized finish reason is rig's vocabulary, not Gemini's"
            );
            assert!(
                !contains_key(&normalized_without_raw(&response), "functionCall"),
                "functionCall is Gemini's wire spelling; the normalized choice carries a ToolCall"
            );
        },
    )
    .await;

    let raw = observed
        .lock()
        .expect("observation lock")
        .take()
        .expect("the test body observed a raw payload");
    let body = assert_recorded_function_call_body(SCENARIO);
    assert_eq!(
        raw.pointer("/candidates/0/content/parts"),
        body.pointer("/candidates/0/content/parts"),
        "raw must carry the wire's functionCall parts exactly as Gemini sent them"
    );
    assert_eq!(
        raw.pointer("/candidates/0/finishReason"),
        body.pointer("/candidates/0/finishReason"),
        "raw must keep the wire's finishReason on the tool turn"
    );
    // Gemini annotates a call-only STOP with a `finishMessage`; the normalized
    // response has no home for it, so it too is only reachable through raw.
    assert!(
        body.pointer("/candidates/0/finishMessage")
            .and_then(Value::as_str)
            .is_some_and(|message| !message.is_empty()),
        "{SCENARIO}: the recorded call-only turn should carry Gemini's finishMessage"
    );
    assert_eq!(
        raw.pointer("/candidates/0/finishMessage"),
        body.pointer("/candidates/0/finishMessage"),
        "raw must carry the wire's finishMessage untouched"
    );
}

// ---------------------------------------------------------------------------
// 4: a structured-output turn keeps the provider-only fields
// ---------------------------------------------------------------------------

#[tokio::test]
async fn raw_exposes_structured_output_turn() {
    const SCENARIO: &str = "raw_capture_matrix/raw_exposes_structured_output_turn";
    let observed: Arc<Mutex<Option<Value>>> = Arc::new(Mutex::new(None));
    let sink = Arc::clone(&observed);
    with_gemini_cassette(
        "raw_capture_matrix/raw_exposes_structured_output_turn",
        |client| async move {
            let model = client.completion_model(MODEL);
            let response = model
                .completion(structured_output_request(&model))
                .await
                .expect("structured output completion should succeed");

            let raw = &response.raw;
            *sink.lock().expect("observation lock") = Some(raw.clone());

            let typed = GenerateContentResponse::deserialize(raw)
                .expect("raw must deserialize into Gemini's GenerateContentResponse");
            assert_eq!(
                serde_json::to_value(&typed).expect("typed raw re-serializes"),
                *raw,
                "GenerateContentResponse must round-trip a structured-output turn"
            );
            let renormalized: RigCompletionResponse =
                typed.try_into().expect("typed raw should normalize");
            assert_eq!(renormalized.choice, response.choice);
            assert_eq!(renormalized.finish_reason(), response.finish_reason());
            assert_eq!(renormalized.usage, response.usage);

            // The normalized choice is the schema JSON as text …
            assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
            let text = match response.choice.first() {
                Some(AssistantContent::Text(text)) => text.text.clone(),
                other => panic!("structured output should arrive as text, got {other:?}"),
            };
            serde_json::from_str::<SmokeStructuredOutput>(&text)
                .expect("the normalized text should be schema JSON");

            // … and provably carries neither Gemini's finishReason spelling nor
            // the per-modality breakdown: both are only reachable through raw.
            let normalized = normalized_without_raw(&response);
            assert!(!contains_key(&normalized, "promptTokensDetails"));
            assert_ne!(
                normalized.get("finish_reason"),
                Some(&Value::String("STOP".to_string()))
            );
        },
    )
    .await;

    let raw = observed
        .lock()
        .expect("observation lock")
        .take()
        .expect("the test body observed a raw payload");
    let body = assert_recorded_structured_output_turn(SCENARIO);
    assert_eq!(
        raw.pointer("/candidates/0/finishReason"),
        Some(&Value::String("STOP".to_string())),
        "raw keeps Gemini's own finishReason spelling on the structured-output turn"
    );
    assert_eq!(
        raw.pointer("/usageMetadata/promptTokensDetails"),
        body.pointer("/usageMetadata/promptTokensDetails"),
        "raw must carry the wire's promptTokensDetails on the structured-output turn"
    );
    assert_eq!(
        raw.pointer("/candidates/0/content/parts/0/text"),
        body.pointer("/candidates/0/content/parts/0/text"),
        "raw must carry the schema JSON exactly as the wire sent it"
    );
}
