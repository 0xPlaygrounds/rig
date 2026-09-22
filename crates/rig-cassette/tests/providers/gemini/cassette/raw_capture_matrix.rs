//! Feature matrix for raw provider response capture on the Gemini REST
//! (`generateContent`) unary seam.
//!
//! # The feature
//!
//! Raw capture is always on: the driver puts the provider's reply body,
//! parsed as JSON, onto [`rig::completion::CompletionResponse::raw`] — here
//! Gemini's own `generateContent` document, verbatim, which
//! [`GenerateContentResponse`] reads back. There is no opt-in and nothing
//! about it reaches the wire; `raw` is required at construction, so there is
//! no response without the document that produced it and no way for
//! capture to be "not requested".
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
//! | 1 | `raw_roundtrips_generate_content_response` | typed access | `GenerateContentResponse::deserialize(&raw)` reads the document back, and its provider-native fields agree with the normalized response | recorded |
//! | 2 | `raw_exposes_prompt_tokens_details` | un-normalized field | `usageMetadata.promptTokensDetails` == fixture, absent from the normalized response | recorded |
//! | 3 | `raw_exposes_forced_function_call` | forced tool call (`ToolChoice::Specific`) | `raw` reads back; `candidates[0].content.parts[].functionCall` and `finishMessage` == fixture; raw `finishReason` spelled `"STOP"` while `finish_reason() == ToolCalls` | recorded |
//! | 4 | `raw_exposes_structured_output_turn` | structured output (`responseMimeType: application/json` + `responseJsonSchema`) | `raw` reads back; raw `finishReason` spelled `"STOP"` and `usageMetadata.promptTokensDetails` == fixture while the normalized response carries neither | recorded |
//!
//! Every cell is recorded: `GEMINI_API_KEY` was available and the seam under
//! test is the plain `generateContent` route, so nothing needs a unit stand-in.
//!
//! Cell 1 also carries the "one story" contract: the normalized response was
//! folded by one decoder out of these very bytes, so the identity, finish
//! reason, model and usage read off `raw` are the ones the response reports —
//! `raw` and the normalized response can never disagree about the turn they
//! describe.
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
    AssistantContent, CompletionModel, CompletionResponse as RigCompletionResponse, FinishReason,
};
use rig::message::ToolChoice;
use rig::providers::gemini::completion::gemini_api_types::{
    ContentCandidate, GenerateContentResponse, PartKind,
};
use rig::tool::Tool;
use serde::Deserialize;
use serde_json::Value;

use super::super::support::{recorded_request_generation_configs, with_gemini_cassette};
use crate::raw_capture::capture_completion;
use crate::support::{
    Adder, Observed, STRUCTURED_OUTPUT_PROMPT, SmokeStructuredOutput, assistant_text,
    json_contains_key, normalized_without_raw,
};

const PROVIDER: &str = "gemini";

/// Cheap, non-thinking, so the recorded body stays small.
const MODEL: &str = "gemini-2.5-flash-lite";

const PROMPT: &str = "Reply with exactly this one word and nothing else: captured";

/// A prompt the forced-tool cell can only satisfy by calling `add`.
const TOOL_PROMPT: &str = "Use the add tool to add 2 and 3.";

fn request(model: &(impl CompletionModel + Clone)) -> rig::completion::CompletionRequest {
    model.completion_request(PROMPT).temperature(0.0).build()
}

/// The forced-tool request: `add` is offered and `ToolChoice::Specific` pins
/// the turn to it (Gemini `functionCallingConfig.mode: ANY` with
/// `allowedFunctionNames`), so the recorded turn is a `functionCall` part.
fn forced_tool_request(
    model: &(impl CompletionModel + Clone),
) -> rig::completion::CompletionRequest {
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
    model: &(impl CompletionModel + Clone),
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

/// The visible (non-`thought`) text of a candidate, joined the way the
/// decoder folds text blocks.
fn visible_text(candidate: &ContentCandidate) -> String {
    candidate
        .content
        .as_ref()
        .into_iter()
        .flat_map(|content| content.parts.iter())
        .filter(|part| !part.thought.unwrap_or(false))
        .filter_map(|part| match &part.part {
            PartKind::Text(text) => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// 1: typed access is recoverable, and tells the same story
// ---------------------------------------------------------------------------

#[tokio::test]
async fn raw_roundtrips_generate_content_response() {
    const SCENARIO: &str = "raw_capture_matrix/raw_roundtrips_generate_content_response";
    let observed: Observed<RigCompletionResponse> = Observed::default();
    let sink = observed.clone();
    with_gemini_cassette(
        "raw_capture_matrix/raw_roundtrips_generate_content_response",
        |client| async move {
            capture_completion(client.completion(MODEL), request, sink)
                .await
                .expect("completion should succeed");
        },
    )
    .await;

    let response = observed.take();
    let raw = &response.raw;

    // `raw` is Gemini's reply document as it arrived: its own type reads it
    // back, so the escape hatch is typed rather than stringly.
    let typed = GenerateContentResponse::deserialize(raw)
        .expect("raw must deserialize into Gemini's GenerateContentResponse");

    // One decoder folded the normalized response out of these very bytes, so
    // every field it kept must be the one the document carries — `raw` is
    // additive, never a divergent second view.
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
        response.usage.input_tokens
    );
    assert_eq!(
        typed
            .usage_metadata
            .as_ref()
            .map(|usage| usage.total_token_count as u64),
        response.usage.total_tokens
    );
    let candidate = typed
        .candidates
        .first()
        .expect("the recorded turn carries a candidate");
    assert_eq!(
        raw.pointer("/candidates/0/finishReason"),
        Some(&Value::String("STOP".to_string())),
        "Gemini's own finish spelling stays on the document"
    );
    assert_eq!(
        response.finish_reason(),
        Some(FinishReason::Stop),
        "and the normalized response reports rig's vocabulary for it"
    );
    assert_eq!(
        visible_text(candidate),
        assistant_text(&response.choice),
        "the normalized text is exactly the document's visible text parts"
    );

    assert_recorded_generate_content_body(SCENARIO);
}

// ---------------------------------------------------------------------------
// 2: an un-normalized field is readable and matches the wire
// ---------------------------------------------------------------------------

#[tokio::test]
async fn raw_exposes_prompt_tokens_details() {
    const SCENARIO: &str = "raw_capture_matrix/raw_exposes_prompt_tokens_details";
    // The observed response is compared against the fixture bytes only after
    // the wrapper returns, so it is carried out of the test body.
    let observed: Observed<RigCompletionResponse> = Observed::default();
    let sink = observed.clone();
    with_gemini_cassette(
        "raw_capture_matrix/raw_exposes_prompt_tokens_details",
        |client| async move {
            capture_completion(client.completion(MODEL), request, sink)
                .await
                .expect("completion should succeed");
        },
    )
    .await;

    let response = observed.take();

    // The normalized response provably lacks the field: it is only reachable
    // through `raw`.
    assert!(
        !json_contains_key(
            &normalized_without_raw(response.clone()),
            "promptTokensDetails"
        ),
        "promptTokensDetails is not part of rig's normalized response — it is exactly \
         the kind of provider detail `raw` exists to expose"
    );

    let raw = &response.raw;
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
    let observed: Observed<RigCompletionResponse> = Observed::default();
    let sink = observed.clone();
    with_gemini_cassette(
        "raw_capture_matrix/raw_exposes_forced_function_call",
        |client| async move {
            capture_completion(client.completion(MODEL), forced_tool_request, sink)
                .await
                .expect("forced tool completion should succeed");
        },
    )
    .await;

    let response = observed.take();
    let raw = &response.raw;

    // The typed read-back holds for a functionCall turn too.
    let typed = GenerateContentResponse::deserialize(raw)
        .expect("raw must deserialize into Gemini's GenerateContentResponse");
    let candidate = typed
        .candidates
        .first()
        .expect("the recorded turn carries a candidate");
    // Gemini `functionCall` parts carry no id, so the decoder mints one: the
    // document's call is compared by what the wire carried (name +
    // arguments), not by the minted id.
    let wire_calls: Vec<(String, Value)> = candidate
        .content
        .as_ref()
        .into_iter()
        .flat_map(|content| content.parts.iter())
        .filter_map(|part| match &part.part {
            PartKind::FunctionCall(call) => Some((call.name.clone(), call.args.clone())),
            _ => None,
        })
        .collect();
    assert_eq!(wire_calls, tool_functions(&response.choice));
    assert_eq!(
        typed
            .usage_metadata
            .as_ref()
            .map(|usage| usage.total_token_count as u64),
        response.usage.total_tokens
    );

    // The normalized response says ToolCalls and carries the call as a typed
    // ToolCall …
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
    let normalized = normalized_without_raw(response.clone());
    assert_ne!(
        normalized.get("finish_reason"),
        Some(&Value::String("STOP".to_string())),
        "the normalized finish reason is rig's vocabulary, not Gemini's"
    );
    assert!(
        !json_contains_key(&normalized, "functionCall"),
        "functionCall is Gemini's wire spelling; the normalized choice carries a ToolCall"
    );

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
    let observed: Observed<RigCompletionResponse> = Observed::default();
    let sink = observed.clone();
    with_gemini_cassette(
        "raw_capture_matrix/raw_exposes_structured_output_turn",
        |client| async move {
            capture_completion(client.completion(MODEL), structured_output_request, sink)
                .await
                .expect("structured output completion should succeed");
        },
    )
    .await;

    let response = observed.take();
    let raw = &response.raw;

    let typed = GenerateContentResponse::deserialize(raw)
        .expect("raw must deserialize into Gemini's GenerateContentResponse");
    let candidate = typed
        .candidates
        .first()
        .expect("the recorded turn carries a candidate");
    assert_eq!(
        visible_text(candidate),
        assistant_text(&response.choice),
        "the schema JSON reaches the caller as the turn's visible text"
    );
    assert_eq!(
        typed
            .usage_metadata
            .as_ref()
            .map(|usage| usage.total_token_count as u64),
        response.usage.total_tokens
    );

    // The normalized choice is the schema JSON as text …
    assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
    let text = match response.choice.first() {
        Some(AssistantContent::Text(text)) => text.text.clone(),
        other => panic!("structured output should arrive as text, got {other:?}"),
    };
    serde_json::from_str::<SmokeStructuredOutput>(&text)
        .expect("the normalized text should be schema JSON");

    // … and provably carries neither Gemini's finishReason spelling nor the
    // per-modality breakdown: both are only reachable through raw.
    let normalized = normalized_without_raw(response.clone());
    assert!(!json_contains_key(&normalized, "promptTokensDetails"));
    assert_ne!(
        normalized.get("finish_reason"),
        Some(&Value::String("STOP".to_string()))
    );

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
