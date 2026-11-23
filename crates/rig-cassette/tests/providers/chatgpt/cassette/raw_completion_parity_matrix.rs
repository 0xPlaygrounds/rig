//! Matrix for the two views of one ChatGPT reply: the normalized
//! [`CompletionResponse`](rig::completion::CompletionResponse) the driver returns and the
//! provider envelope it captured on [`raw`](rig::completion::CompletionResponse::raw). Both must
//! tell the same story about `identity()`, `finish_reason()`, `model` and `usage`.
//!
//! The driver reassembles the Responses wire type from the SSE body's terminal
//! `response.completed` event, folds it into the normalized response, and serializes the same
//! value onto `raw`. ChatGPT reads no transport request-id header, so the whole identity lives
//! in the body and `provider_request_id` is `None`. The cells are a text turn
//! (`raw_normalize_reproduces_completion`), a tool-call turn
//! (`raw_normalize_reproduces_completion_with_tool_call`), and an empty-`output` terminal event
//! (`empty_output_fallback_still_carries_raw`), where the content folds from the preceding
//! events while `raw` still carries `output: []`.
//!
//! The cells are ignored until recorded with ChatGPT credentials; a fixture is never
//! fabricated. The empty-`output` event is a backend behavior, not a request shape, so its cell
//! asserts the premise and records only when the backend emits it; the provider's unit test
//! `test_completion_response_from_sse_body_falls_back_to_streamed_text` pins the branch.

use super::GPT_5_4;
use rig::completion::{CompletionResponse as RigCompletionResponse, FinishReason, ToolDefinition};
use rig::message::AssistantContent;
use serde_json::{Value, json};

use super::super::support::with_chatgpt_cassette;
use crate::cassettes::{CassetteMode, recorded_interaction_bodies};
use crate::raw_capture::{assert_no_request_id, capture_completion, responses};
use crate::support::{Observed, assert_matches_recorded_token};
use rig::completion::CompletionRequest;

const CHATGPT_PROVIDER: &str = "chatgpt";
const MODEL: &str = GPT_5_4;
const PROMPT: &str = "Reply with exactly the single word: pong";
const TOOL_PROMPT: &str = "Use the get_weather tool for the city Lisbon.";

fn weather_tool() -> ToolDefinition {
    ToolDefinition {
        name: rig_core::message::ToolName::new("get_weather").expect("tool name"),
        description: "Get the current weather for a city.".to_string(),
        parameters: json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"]
        }),
        annotations: Default::default(),
    }
}

fn request() -> rig::completion::CompletionRequest {
    CompletionRequest::new(PROMPT).max_tokens(64)
}

fn tool_request() -> rig::completion::CompletionRequest {
    CompletionRequest::new(TOOL_PROMPT)
        .tool(weather_tool())
        .max_tokens(128)
}

/// The `msg_…` id of the envelope's output message, when it issued one.
fn wire_message_id(envelope: &Value) -> Option<String> {
    envelope["output"]
        .as_array()?
        .iter()
        .find(|item| item["type"] == "message")
        .and_then(|message| message["id"].as_str())
        .map(str::to_owned)
}

/// The recorded terminal `response.completed` frame's `response` for each
/// interaction of a scenario, in wire order — the premise: every interaction
/// completed with a usage-bearing terminal envelope.
fn recorded_terminal_responses(scenario: &str) -> Vec<Value> {
    recorded_interaction_bodies(CHATGPT_PROVIDER, scenario)
        .into_iter()
        .map(|(_, body)| {
            let terminal = body
                .lines()
                .filter_map(|line| line.trim().strip_prefix("data:"))
                .map(str::trim)
                .filter(|payload| *payload != "[DONE]")
                .filter_map(|payload| serde_json::from_str::<Value>(payload).ok())
                .rev()
                .find(|frame| {
                    frame.get("type").and_then(Value::as_str) == Some("response.completed")
                })
                .map_or_else(
                    || panic!("{scenario}: each interaction must end with response.completed"),
                    |frame| frame["response"].clone(),
                );
            assert!(
                terminal.pointer("/usage/total_tokens").is_some(),
                "{scenario}: the terminal envelope must report usage"
            );
            assert!(
                terminal.get("id").and_then(Value::as_str).is_some(),
                "{scenario}: the terminal envelope must carry a response id"
            );
            terminal
        })
        .collect()
}

/// The one identity claim a live recording pass can make: the fixture holds a
/// placeholder id, so only the *shape* of the freshly minted one is knowable
/// there. Replay compares the id itself, through the shared token comparator.
fn assert_response_id_shape(response: &RigCompletionResponse) {
    if matches!(CassetteMode::current(), CassetteMode::Record) {
        assert!(
            response
                .response_id()
                .is_some_and(|id| id.starts_with("resp_")),
            "response id should be a resp_ id, got {:?}",
            response.response_id()
        );
    }
}

// ---------------------------------------------------------------------------
// 1: text turn — one reply, both views
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "unrecorded (no CHATGPT credentials in this environment)"]
async fn raw_normalize_reproduces_completion() {
    let scenario = "raw_completion_parity_matrix/raw_normalize_reproduces_completion";
    let captured = Observed::default();
    let sink = captured.clone();
    with_chatgpt_cassette(
        "raw_completion_parity_matrix/raw_normalize_reproduces_completion",
        |client| async move {
            capture_completion(client.completion(MODEL), request(), sink)
                .await
                .expect("completion should succeed");
        },
    )
    .await;

    let response = captured.take();
    let envelope = &response.raw;
    assert!(
        envelope["output"]
            .as_array()
            .is_some_and(|items| !items.is_empty()),
        "premise: the terminal envelope carried items"
    );
    assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
    assert_eq!(
        envelope["status"], "completed",
        "the envelope's own verdict is what the finish reason folded from"
    );
    assert!(
        response
            .choice
            .iter()
            .any(|content| matches!(content, AssistantContent::Text(_))),
        "the text turn must fold to assistant text"
    );

    let terminals = recorded_terminal_responses(scenario);
    assert_eq!(terminals.len(), 1, "{scenario}: expected one turn");
    // A completed text turn is exactly the Responses body contract: provider,
    // both ids, model, the status's finish reason, the three counters and the
    // message text, all read out of the recorded terminal envelope.
    responses::assert_reproduces_body(
        &response,
        CHATGPT_PROVIDER,
        &terminals[0],
        "the recorded terminal envelope",
    );
    assert_no_request_id(response.provider_request_id.as_deref(), CHATGPT_PROVIDER);
    assert_response_id_shape(&response);
}

// ---------------------------------------------------------------------------
// 2: tool-call turn
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "unrecorded (no CHATGPT credentials in this environment)"]
async fn raw_normalize_reproduces_completion_with_tool_call() {
    let scenario =
        "raw_completion_parity_matrix/raw_normalize_reproduces_completion_with_tool_call";
    let captured = Observed::default();
    let sink = captured.clone();
    with_chatgpt_cassette(
        "raw_completion_parity_matrix/raw_normalize_reproduces_completion_with_tool_call",
        |client| async move {
            capture_completion(client.completion(MODEL), tool_request(), sink)
                .await
                .expect("completion should succeed");
        },
    )
    .await;

    let response = captured.take();
    assert_eq!(
        response.finish_reason(),
        Some(FinishReason::ToolCalls),
        "a tool-call turn normalizes to ToolCalls"
    );
    assert!(
        response.choice.iter().any(|content| matches!(
            content,
            AssistantContent::ToolCall(call) if call.function.name == "get_weather"
        )),
        "the get_weather call must be on the choice"
    );

    let envelope = &response.raw;
    assert!(
        envelope["output"].as_array().is_some_and(|items| items
            .iter()
            .any(|item| item["type"] == "function_call" && item["name"] == "get_weather")),
        "the captured envelope must carry the provider's own function_call item"
    );

    let terminals = recorded_terminal_responses(scenario);
    assert_eq!(terminals.len(), 1, "{scenario}: expected one turn");
    assert!(
        terminals[0]["output"]
            .as_array()
            .is_some_and(|items| items.iter().any(|item| item["type"] == "function_call")),
        "{scenario}: premise — the recorded terminal envelope carries a function_call item"
    );
    // Not the shared Responses body contract: this turn's envelope completes
    // with a `function_call` item and need issue no message item at all, so
    // the shared "status ⇒ Stop, and the choice is the message's text" claims
    // do not hold. What does hold is stated field by field, with the shared
    // token comparator owning the replay/record split on the ids.
    let from_wire = &terminals[0];
    assert_eq!(response.provider(), CHATGPT_PROVIDER, "provider");
    assert_eq!(response.model(), from_wire["model"].as_str(), "model");
    let usage = &from_wire["usage"];
    assert!(usage.is_object(), "the terminal envelope must report usage");
    assert_eq!(
        (
            response.usage.input_tokens,
            response.usage.output_tokens,
            response.usage.total_tokens
        ),
        (
            usage["input_tokens"].as_u64(),
            usage["output_tokens"].as_u64(),
            usage["total_tokens"].as_u64()
        ),
        "usage"
    );
    assert_no_request_id(response.provider_request_id.as_deref(), CHATGPT_PROVIDER);
    assert_matches_recorded_token(
        response.response_id(),
        from_wire["id"].as_str(),
        "response id",
    );
    let wire_message = wire_message_id(from_wire);
    assert_matches_recorded_token(
        response.response_id(),
        wire_message.as_deref(),
        "message id",
    );
    assert_response_id_shape(&response);
}

// ---------------------------------------------------------------------------
// 3: the empty-output fallback branch also carries raw
// ---------------------------------------------------------------------------

/// Cannot be produced on demand (see the module docs); the body asserts its
/// premise from the fixture so a recording that did not hit the fallback
/// fails instead of passing vacuously.
#[tokio::test]
#[ignore = "unrecorded (no CHATGPT credentials in this environment)"]
async fn empty_output_fallback_still_carries_raw() {
    let scenario = "raw_completion_parity_matrix/empty_output_fallback_still_carries_raw";
    let captured = Observed::default();
    let sink = captured.clone();
    with_chatgpt_cassette(
        "raw_completion_parity_matrix/empty_output_fallback_still_carries_raw",
        |client| async move {
            capture_completion(client.completion(MODEL), request(), sink)
                .await
                .expect("the fallback rebuilds the response from the event stream");
        },
    )
    .await;

    let response = captured.take();
    // The normalized content folded from the events…
    assert!(
        !response.choice.is_empty(),
        "the content must fold from the events the terminal event lacks"
    );
    // …and raw is still the terminal envelope, whose output is empty.
    let typed = &response.raw;
    assert_eq!(
        typed["output"],
        json!([]),
        "premise: this cell exists for the empty-output terminal event"
    );
    assert_eq!(
        typed["status"], "completed",
        "the contentless terminal event still reported completion"
    );

    let terminals = recorded_terminal_responses(scenario);
    assert_eq!(
        terminals[0]["output"],
        json!([]),
        "{scenario}: premise — the recorded terminal envelope carried no output items; \
         a recording where it did does not exercise the fallback"
    );
}
