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
//! | 3 | `raw_exposes_forced_function_call` | forced tool call (`ToolChoice::Specific`) | `raw` reads back; `candidates[0].content.parts[].functionCall` and `finishMessage` == fixture; raw `finishReason` spelled `"STOP"` while `finish_reason() == ToolCalls` | recorded |
//!
//! Every cell is recorded: `GEMINI_API_KEY` was available and the seam under
//! test is the plain `generateContent` route, so nothing needs a unit stand-in.
//!
//! Cell 3 covers a wire shape a text turn never produces. A forced
//! `functionCall` turn is where Gemini's own `finishReason` (`"STOP"`, even
//! on a call-only turn) and rig's normalized `ToolCalls` visibly disagree,
//! so `raw` must keep the wire spelling while the normalized response reports
//! the upgraded reason.

use rig::completion::{
    AssistantContent, CompletionResponse as RigCompletionResponse, FinishReason,
};
use rig::message::ToolChoice;
use rig::providers::gemini::extension::GeminiExt;
use rig::tool::Tool;
use serde_json::Value;

use super::super::support::with_gemini_cassette;
use crate::raw_capture::capture_completion;
use crate::support::{Adder, Observed, json_contains_key, normalized_without_raw};
use rig::completion::CompletionRequest;

const PROVIDER: &str = "gemini";

/// Cheap, non-thinking, so the recorded body stays small.
const MODEL: &str = "gemini-2.5-flash-lite";

/// A prompt the forced-tool cell can only satisfy by calling `add`.
const TOOL_PROMPT: &str = "Use the add tool to add 2 and 3.";

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

// ---------------------------------------------------------------------------
// 2: an un-normalized field is readable and matches the wire
// ---------------------------------------------------------------------------

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

    // The typed extras read the same recorded reply.
    let extras = response
        .extras::<GeminiExt>()
        .expect("a Gemini API reply has Gemini extras")
        .expect("the recorded reply holds the extras' shape");
    assert_eq!(
        extras.model_version.as_deref(),
        Some("gemini-2.5-flash-lite")
    );
    assert_eq!(
        extras.response_id.as_deref(),
        Some("Ag7BauSXOKisz7IPlZS6iAg")
    );
    assert_eq!(extras.service_tier.as_deref(), Some("standard"));
    assert_eq!(
        extras.finish_message.as_deref(),
        Some("Model generated function call(s).")
    );
    let prompt = extras.prompt_tokens_details.unwrap_or_default();
    assert_eq!(
        prompt
            .iter()
            .map(|detail| (detail.modality.as_deref(), detail.token_count))
            .collect::<Vec<_>>(),
        [(Some("TEXT"), Some(67))]
    );
    assert_eq!(
        extras.id, None,
        "an Interactions field on a GenerateContent reply"
    );
}

// ---------------------------------------------------------------------------
// 4: a structured-output turn keeps the provider-only fields
// ---------------------------------------------------------------------------
