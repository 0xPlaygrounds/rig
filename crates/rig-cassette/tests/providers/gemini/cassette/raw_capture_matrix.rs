//! Feature matrix for raw provider response capture on the Gemini REST
//! (`generateContent`) unary seam.
//!
//! # The feature
//!
//! Raw capture is always on: the driver puts the provider's reply body,
//! parsed as JSON, onto [`rig::completion::CompletionResponse::raw`] — here
//! Gemini's own `generateContent` document, verbatim, read back as JSON.
//! There is no opt-in and nothing
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
//! | 1 | `raw_roundtrips_generate_content_response` | document access | `raw` reads back as the `generateContent` document, and its provider-native fields agree with the normalized response | recorded |
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
    AssistantContent, CompletionResponse as RigCompletionResponse, FinishReason,
};
use rig::message::ToolChoice;
use rig::tool::Tool;
use serde_json::Value;

use super::super::support::{recorded_request_generation_configs, with_gemini_cassette};
use crate::raw_capture::capture_completion;
use crate::support::{
    Adder, Observed, STRUCTURED_OUTPUT_PROMPT, SmokeStructuredOutput, assistant_text,
    json_contains_key, normalized_without_raw,
};
use rig::completion::CompletionRequest;

const PROVIDER: &str = "gemini";

/// Cheap, non-thinking, so the recorded body stays small.
const MODEL: &str = "gemini-2.5-flash-lite";

const PROMPT: &str = "Reply with exactly this one word and nothing else: captured";

/// A prompt the forced-tool cell can only satisfy by calling `add`.
const TOOL_PROMPT: &str = "Use the add tool to add 2 and 3.";

fn request() -> rig::completion::CompletionRequest {
    CompletionRequest::new(PROMPT).temperature(0.0)
}

/// The forced-tool request: `add` is offered and `ToolChoice::Specific` pins
/// the turn to it (Gemini `functionCallingConfig.mode: ANY` with
/// `allowedFunctionNames`), so the recorded turn is a `functionCall` part.
fn forced_tool_request() -> rig::completion::CompletionRequest {
    CompletionRequest::new(TOOL_PROMPT)
        .temperature(0.0)
        .tool(rig::tool::tool_definition(&Adder))
        .tool_choice(ToolChoice::Specific {
            function_names: vec![rig_core::message::ToolName::new(Adder::NAME).expect("tool name")],
        })
}

/// The structured-output request: rig maps `output_schema` onto
/// `generationConfig.responseMimeType: application/json` +
/// `responseJsonSchema`, Gemini's native structured-output controls.
fn structured_output_request() -> rig::completion::CompletionRequest {
    CompletionRequest::new(STRUCTURED_OUTPUT_PROMPT)
        .temperature(0.0)
        .output_schema(schemars::schema_for!(SmokeStructuredOutput))
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
            AssistantContent::ToolCall(call) => Some((
                call.function.name.to_string(),
                call.function.arguments_value(),
            )),
            _ => None,
        })
        .collect()
}

/// The visible (non-`thought`) text of a candidate, joined the way the
/// decoder folds text blocks.
fn visible_text(candidate: &Value) -> String {
    candidate_parts(candidate)
        .filter(|part| part.get("thought").and_then(Value::as_bool) != Some(true))
        .filter_map(|part| part.get("text").and_then(Value::as_str))
        .collect()
}

/// The parts of a candidate's content, in wire order.
fn candidate_parts(candidate: &Value) -> impl Iterator<Item = &Value> {
    candidate
        .pointer("/content/parts")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
}

/// The first candidate of a `generateContent` document.
fn first_candidate(raw: &Value) -> &Value {
    assert!(
        raw.is_object(),
        "raw must be Gemini's generateContent document, got {raw}"
    );
    raw.pointer("/candidates/0")
        .expect("the recorded turn carries a candidate")
}

/// A `usageMetadata` token count, which Gemini omits when it is zero.
fn usage_count(raw: &Value, field: &str) -> Option<u64> {
    raw.get("usageMetadata")
        .map(|usage| usage.get(field).and_then(Value::as_u64).unwrap_or_default())
}

// ---------------------------------------------------------------------------
// 1: the document is recoverable, and tells the same story
// ---------------------------------------------------------------------------

#[tokio::test]
async fn raw_roundtrips_generate_content_response() {
    const SCENARIO: &str = "raw_capture_matrix/raw_roundtrips_generate_content_response";
    let observed: Observed<RigCompletionResponse> = Observed::default();
    let sink = observed.clone();
    with_gemini_cassette(
        "raw_capture_matrix/raw_roundtrips_generate_content_response",
        |client| async move {
            capture_completion(client.completion(MODEL), request(), sink)
                .await
                .expect("completion should succeed");
        },
    )
    .await;

    let response = observed.take();
    let raw = &response.raw;

    // `raw` is Gemini's reply document as it arrived.
    let candidate = first_candidate(raw);

    // One decoder folded the normalized response out of these very bytes, so
    // every field it kept must be the one the document carries — `raw` is
    // additive, never a divergent second view.
    assert_eq!(
        raw.get("modelVersion").and_then(Value::as_str),
        response.model()
    );
    assert_eq!(
        Some(
            raw.get("responseId")
                .and_then(Value::as_str)
                .unwrap_or_default()
        ),
        response.response_id()
    );
    assert_eq!(
        usage_count(raw, "promptTokenCount"),
        response.usage.input_tokens
    );
    assert_eq!(
        usage_count(raw, "totalTokenCount"),
        response.usage.total_tokens
    );
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
            capture_completion(client.completion(MODEL), request(), sink)
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
            capture_completion(client.completion(MODEL), forced_tool_request(), sink)
                .await
                .expect("forced tool completion should succeed");
        },
    )
    .await;

    let response = observed.take();
    let raw = &response.raw;

    // The read-back holds for a functionCall turn too.
    let candidate = first_candidate(raw);
    // Gemini `functionCall` parts carry no id, so the decoder mints one: the
    // document's call is compared by what the wire carried (name +
    // arguments), not by the minted id.
    let wire_calls: Vec<(String, Value)> = candidate_parts(candidate)
        .filter_map(|part| part.get("functionCall"))
        .map(|call| {
            let name = call
                .get("name")
                .and_then(Value::as_str)
                .expect("a functionCall part names its function");
            let args = call.get("args").cloned().unwrap_or(Value::Null);
            (name.to_owned(), args)
        })
        .collect();
    assert_eq!(wire_calls, tool_functions(&response.choice));
    assert_eq!(
        usage_count(raw, "totalTokenCount"),
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
        call.function.arguments_value(),
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
    let canonical: Vec<AssistantContent> = response
        .choice
        .iter()
        .map(AssistantContent::canonical)
        .collect();
    assert!(
        !json_contains_key(&serde_json::json!(canonical), "functionCall"),
        "functionCall is Gemini's wire spelling, kept only in the call's provider item; \
         the canonical choice carries a ToolCall"
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
            capture_completion(client.completion(MODEL), structured_output_request(), sink)
                .await
                .expect("structured output completion should succeed");
        },
    )
    .await;

    let response = observed.take();
    let raw = &response.raw;

    let candidate = first_candidate(raw);
    assert_eq!(
        visible_text(candidate),
        assistant_text(&response.choice),
        "the schema JSON reaches the caller as the turn's visible text"
    );
    assert_eq!(
        usage_count(raw, "totalTokenCount"),
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
