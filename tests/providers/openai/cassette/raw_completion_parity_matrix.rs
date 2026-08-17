//! Typed-route parity for OpenAI: the provider-native `raw_completion` route
//! reproduces what `CompletionModel::completion` returns.
//!
//! # What this pins
//!
//! On the Chat Completions route the wire type (`openai::CompletionResponse`)
//! is substitutable across every OpenAI-compatible provider, so the transport
//! request id from the `x-request-id` header cannot live on it. Until
//! `GenericCompletionModel::raw_completion_with_request_id` became public, a
//! caller on the typed route had no way to obtain that id: `raw_completion`
//! followed by `normalize` silently produced a response whose
//! `provider_request_id` was `None` while `completion()` reported one. Cell 3
//! pins exactly that asymmetry — the plain route lacks the id, the
//! `_with_request_id` route restores it — so the documented contract is tested
//! rather than asserted in prose.
//!
//! On the Responses route the wire type carries `provider_request_id` itself
//! (stamped by the request driver), so `raw_completion(req).normalize(..)`
//! already reproduces `completion(req)`.
//!
//! Every parity cell issues the same request twice — once through the raw
//! route, once through `completion()` — as two interactions of one scenario;
//! the harness replays them in order. The two responses are distinct provider
//! turns, so each side is first checked against *its own* fixture interaction
//! (id, request-id header, usage, model, finish reason), and then the two
//! sides are compared on the fields the contract names: `finish_reason()`,
//! `model`, `usage`, and `provider_request_id.is_some()`. Identity is
//! compared structurally (`response_id` present with the route's prefix,
//! `provider_request_id` present) — two live turns cannot share ids.
//!
//! # Matrix
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `chat_text_turn_parity` | chat, text turn | raw+id ≡ completion (`Stop`) | recorded |
//! | 2 | `chat_tool_turn_parity` | chat, forced tool call | raw+id ≡ completion (`ToolCalls`) | recorded |
//! | 3 | `chat_plain_raw_completion_lacks_request_id` | chat, `raw_completion` without the id | `provider_request_id` `None` vs `Some` | recorded |
//! | 4 | `responses_text_turn_parity` | Responses, text turn | raw ≡ completion (`Stop`) | recorded |
//! | 5 | `responses_tool_turn_parity` | Responses, forced tool call | raw ≡ completion (`ToolCalls`) | recorded |
//!
//! Every cell is recorded; none is unit-only. Premise, re-derived from each
//! cell's fixture bytes after the wrapper returns: every recorded response
//! carries an `x-request-id` header — without that the cells would prove
//! nothing about the transport id.

use std::future::Future;
use std::pin::Pin;

use rig::completion::{
    AssistantContent, CompletionModel as _, CompletionRequest, CompletionResponse, FinishReason,
    NormalizeCompletionResponse as _, ToolDefinition,
};
use rig::message::ToolChoice;
use rig::prelude::*;
use rig::providers::openai;
use serde_json::{Value, json};

use super::super::support::{
    assert_matches_recorded_token, recorded_request_id_headers, with_openai_cassette,
};

const PROVIDER: &str = "openai";
const MODEL: &str = openai::GPT_4_1_NANO;
const TEXT_PROMPT: &str = "Reply with exactly the single word: pong";
const TOOL_PROMPT: &str = "Call ping exactly once with no arguments.";

fn ping_tool() -> ToolDefinition {
    ToolDefinition {
        name: "ping".to_owned(),
        description: "Matrix tool ping".to_owned(),
        parameters: json!({ "type": "object", "properties": {}, "additionalProperties": false }),
    }
}

/// The text-turn request, identical for both routes: `temperature: 0` keeps
/// the two live turns of a cell as alike as the provider allows.
fn text_request(model: &(impl rig::completion::CompletionModel + Clone)) -> CompletionRequest {
    model
        .completion_request(TEXT_PROMPT)
        .temperature(0.0)
        .max_tokens(16)
        .build()
}

fn tool_request(model: &(impl rig::completion::CompletionModel + Clone)) -> CompletionRequest {
    model
        .completion_request(TOOL_PROMPT)
        .tool(ping_tool())
        .tool_choice(ToolChoice::Required)
        .temperature(0.0)
        .max_tokens(64)
        .build()
}

fn tool_call_names(response: &CompletionResponse) -> Vec<&str> {
    response
        .choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::ToolCall(call) => Some(call.function.name.as_str()),
            _ => None,
        })
        .collect()
}

/// The premise every cell rests on: each recorded interaction's response
/// carried an `x-request-id` header. Returns the recorded (scrubbed) ids in
/// wire order.
fn recorded_request_ids(scenario: &str, expected_interactions: usize) -> Vec<String> {
    let headers = recorded_request_id_headers(scenario);
    assert_eq!(
        headers.len(),
        expected_interactions,
        "{scenario}: interaction count"
    );
    headers
        .into_iter()
        .enumerate()
        .map(|(index, id)| {
            id.unwrap_or_else(|| {
                panic!(
                    "{scenario}: interaction {index} must carry an x-request-id response \
                     header — without it this cell proves nothing about the transport id"
                )
            })
        })
        .collect()
}

/// The two wire shapes: how each names its response and spells its usage.
#[derive(Clone, Copy)]
enum Wire {
    Chat,
    Responses,
}

impl Wire {
    fn id_prefix(self) -> &'static str {
        match self {
            Wire::Chat => "chatcmpl",
            Wire::Responses => "resp_",
        }
    }

    /// `(input, output)` usage keys of the wire body.
    fn usage_keys(self) -> (&'static str, &'static str) {
        match self {
            Wire::Chat => ("prompt_tokens", "completion_tokens"),
            Wire::Responses => ("input_tokens", "output_tokens"),
        }
    }
}

/// One normalized response against the fixture interaction it came from.
fn assert_side_matches_fixture(
    scenario: &str,
    side: &str,
    response: &CompletionResponse,
    body: &Value,
    recorded_request_id: &str,
    wire: Wire,
) {
    let context = format!("{scenario}/{side}");
    let id_prefix = wire.id_prefix();
    let (usage_input_key, usage_output_key) = wire.usage_keys();
    assert_matches_recorded_token(
        response.provider_request_id.as_deref(),
        Some(recorded_request_id),
        &format!("{context}: provider_request_id vs the fixture's x-request-id header"),
    );
    assert_matches_recorded_token(
        response.response_id.as_deref(),
        body["id"].as_str(),
        &format!("{context}: response_id vs the fixture body id"),
    );
    assert!(
        response
            .response_id
            .as_deref()
            .is_some_and(|id| id.starts_with(id_prefix)),
        "{context}: response_id should be a {id_prefix} id, got {:?}",
        response.response_id
    );
    assert_eq!(
        response.model.as_deref(),
        body["model"].as_str(),
        "{context}: model"
    );
    assert_eq!(
        Some(response.usage.input_tokens),
        body["usage"][usage_input_key].as_u64(),
        "{context}: input tokens"
    );
    assert_eq!(
        Some(response.usage.output_tokens),
        body["usage"][usage_output_key].as_u64(),
        "{context}: output tokens"
    );
    assert_eq!(response.provider, PROVIDER, "{context}: provider");
}

/// The contract: the two routes agree on identity shape, finish reason,
/// model, and usage.
fn assert_parity(
    scenario: &str,
    typed: &CompletionResponse,
    normalized: &CompletionResponse,
    expected_finish: FinishReason,
) {
    assert_eq!(
        typed.finish_reason(),
        Some(expected_finish.clone()),
        "{scenario}: typed route finish reason"
    );
    assert_eq!(
        normalized.finish_reason(),
        Some(expected_finish),
        "{scenario}: completion() finish reason"
    );
    assert_eq!(typed.model, normalized.model, "{scenario}: model");
    assert_eq!(typed.usage, normalized.usage, "{scenario}: usage");
    assert_eq!(typed.provider, normalized.provider, "{scenario}: provider");
    let typed_identity = typed.identity();
    let normalized_identity = normalized.identity();
    assert!(
        typed_identity.provider_request_id.is_some()
            && normalized_identity.provider_request_id.is_some(),
        "{scenario}: both routes carry the transport request id"
    );
    assert!(
        typed_identity.response_id.is_some() && normalized_identity.response_id.is_some(),
        "{scenario}: both routes carry the response id"
    );
    assert_eq!(
        typed_identity.message_id.is_some(),
        normalized_identity.message_id.is_some(),
        "{scenario}: both routes agree on whether a message id exists"
    );
    assert_ne!(
        typed_identity.response_id, normalized_identity.response_id,
        "{scenario}: two live turns are two provider responses"
    );
}

// ---------------------------------------------------------------------------
// Chat Completions
// ---------------------------------------------------------------------------

type Observed = std::sync::Arc<std::sync::Mutex<Option<(CompletionResponse, CompletionResponse)>>>;

/// A cassette test body: boxed so the cell can build it in a helper while the
/// wrapper call — and its string-literal scenario, which the cassette safety
/// scan reads — stays in the test itself.
type Body = Box<dyn FnOnce(openai::Client) -> Pin<Box<dyn Future<Output = ()>>>>;

fn take(observed: &Observed) -> (CompletionResponse, CompletionResponse) {
    observed
        .lock()
        .expect("observation mutex")
        .take()
        .expect("test body should save its observation")
}

/// Chat route: the typed route (`raw_completion_with_request_id` → normalize
/// → `with_optional_provider_request_id`), then `completion()`, on the same
/// request — two interactions.
fn chat_parity_body(
    sink: Observed,
    request_for: fn(&openai::CompletionModel) -> CompletionRequest,
) -> Body {
    Box::new(move |client| {
        Box::pin(async move {
            let model = client.completions_api().completion_model(MODEL);
            let (raw, request_id) = model
                .raw_completion_with_request_id(request_for(&model))
                .await
                .expect("raw route should succeed");
            let typed = raw
                .normalize(PROVIDER)
                .expect("raw response should normalize")
                .with_optional_provider_request_id(request_id);
            let normalized = model
                .completion(request_for(&model))
                .await
                .expect("completion() should succeed");
            *sink.lock().expect("observation mutex") = Some((typed, normalized));
        })
    })
}

fn assert_chat_parity(
    scenario: &str,
    observed: &Observed,
    expected_finish: FinishReason,
    expect_tool_call: bool,
) {
    let (typed, normalized) = take(observed);
    let request_ids = recorded_request_ids(scenario, 2);
    let bodies = crate::cassettes::recorded_interaction_bodies(PROVIDER, scenario);
    let body = |index: usize| -> Value {
        serde_json::from_str(&bodies[index].1).expect("recorded body should be JSON")
    };
    assert_eq!(
        bodies[0].0, bodies[1].0,
        "{scenario}: both routes send the identical request"
    );
    let first = body(0);
    let second = body(1);
    let expected_wire_finish = if expect_tool_call {
        "tool_calls"
    } else {
        "stop"
    };
    for (index, body) in [&first, &second].into_iter().enumerate() {
        assert_eq!(
            body["choices"][0]["finish_reason"], expected_wire_finish,
            "{scenario}: interaction {index} wire finish reason"
        );
    }
    assert_side_matches_fixture(
        scenario,
        "raw_completion_with_request_id",
        &typed,
        &first,
        &request_ids[0],
        Wire::Chat,
    );
    assert_side_matches_fixture(
        scenario,
        "completion",
        &normalized,
        &second,
        &request_ids[1],
        Wire::Chat,
    );
    if expect_tool_call {
        assert_eq!(
            tool_call_names(&typed),
            ["ping"],
            "{scenario}: typed route tool call"
        );
        assert_eq!(
            tool_call_names(&normalized),
            ["ping"],
            "{scenario}: completion() tool call"
        );
    } else {
        assert_eq!(typed.choice, normalized.choice, "{scenario}: text choice");
    }
    assert_parity(scenario, &typed, &normalized, expected_finish);
}

#[tokio::test]
async fn chat_text_turn_parity() {
    const SCENARIO: &str = "raw_completion_parity_matrix/chat_text_turn_parity";
    let observed = Observed::default();
    with_openai_cassette(
        "raw_completion_parity_matrix/chat_text_turn_parity",
        chat_parity_body(observed.clone(), text_request),
    )
    .await;
    assert_chat_parity(SCENARIO, &observed, FinishReason::Stop, false);
}

#[tokio::test]
async fn chat_tool_turn_parity() {
    const SCENARIO: &str = "raw_completion_parity_matrix/chat_tool_turn_parity";
    let observed = Observed::default();
    with_openai_cassette(
        "raw_completion_parity_matrix/chat_tool_turn_parity",
        chat_parity_body(observed.clone(), tool_request),
    )
    .await;
    assert_chat_parity(SCENARIO, &observed, FinishReason::ToolCalls, true);
}

/// The asymmetry the public `raw_completion_with_request_id` exists to close:
/// the plain typed route normalizes into a response with no transport id even
/// though the wire carried the `x-request-id` header, while `completion()` on
/// the same model reports it.
#[tokio::test]
async fn chat_plain_raw_completion_lacks_request_id() {
    const SCENARIO: &str =
        "raw_completion_parity_matrix/chat_plain_raw_completion_lacks_request_id";
    let observed = Observed::default();
    let sink = observed.clone();
    with_openai_cassette(
        "raw_completion_parity_matrix/chat_plain_raw_completion_lacks_request_id",
        |client| async move {
            let model = client.completions_api().completion_model(MODEL);
            let plain = model
                .raw_completion(text_request(&model))
                .await
                .expect("raw_completion should succeed")
                .normalize(PROVIDER)
                .expect("raw response should normalize");
            let normalized = model
                .completion(text_request(&model))
                .await
                .expect("completion() should succeed");
            *sink.lock().expect("observation mutex") = Some((plain, normalized));
        },
    )
    .await;

    let (plain, normalized) = take(&observed);
    // Premise: the wire reported a request id on *both* interactions — so the
    // plain route's `None` is a property of the route, not of the recording.
    let request_ids = recorded_request_ids(SCENARIO, 2);
    assert_eq!(
        plain.provider_request_id, None,
        "{SCENARIO}: `raw_completion(..).normalize(..)` has no slot for the transport id"
    );
    assert_matches_recorded_token(
        normalized.provider_request_id.as_deref(),
        Some(&request_ids[1]),
        &format!("{SCENARIO}: completion() reports the fixture's x-request-id"),
    );
    // Everything else the plain route normalizes still matches its fixture.
    let bodies = crate::cassettes::recorded_interaction_bodies(PROVIDER, SCENARIO);
    let first: Value = serde_json::from_str(&bodies[0].1).expect("recorded body should be JSON");
    assert_matches_recorded_token(
        plain.response_id.as_deref(),
        first["id"].as_str(),
        &format!("{SCENARIO}: plain route response_id"),
    );
    assert_eq!(plain.model.as_deref(), first["model"].as_str());
    assert_eq!(
        Some(plain.usage.input_tokens),
        first["usage"]["prompt_tokens"].as_u64()
    );
    assert_eq!(plain.finish_reason(), Some(FinishReason::Stop));
    assert_eq!(normalized.finish_reason(), Some(FinishReason::Stop));
    assert_eq!(plain.model, normalized.model);
}

// ---------------------------------------------------------------------------
// Responses
// ---------------------------------------------------------------------------

/// Responses route: `raw_completion` → normalize (the wire type carries the
/// transport id itself), then `completion()`, on the same request.
fn responses_parity_body(
    sink: Observed,
    request_for: fn(&openai::responses_api::ResponsesCompletionModel) -> CompletionRequest,
) -> Body {
    Box::new(move |client| {
        Box::pin(async move {
            let model = client.completion_model(MODEL);
            let typed = model
                .raw_completion(request_for(&model))
                .await
                .expect("raw route should succeed")
                .normalize(PROVIDER)
                .expect("raw response should normalize");
            let normalized = model
                .completion(request_for(&model))
                .await
                .expect("completion() should succeed");
            *sink.lock().expect("observation mutex") = Some((typed, normalized));
        })
    })
}

fn assert_responses_parity(
    scenario: &str,
    observed: &Observed,
    expected_finish: FinishReason,
    expect_tool_call: bool,
) {
    let (typed, normalized) = take(observed);
    let request_ids = recorded_request_ids(scenario, 2);
    let bodies = crate::cassettes::recorded_interaction_bodies(PROVIDER, scenario);
    assert_eq!(
        bodies[0].0, bodies[1].0,
        "{scenario}: both routes send the identical request"
    );
    let first: Value = serde_json::from_str(&bodies[0].1).expect("recorded body should be JSON");
    let second: Value = serde_json::from_str(&bodies[1].1).expect("recorded body should be JSON");
    for (index, body) in [&first, &second].into_iter().enumerate() {
        assert_eq!(
            body["status"], "completed",
            "{scenario}: interaction {index} completed"
        );
        let has_function_call = body["output"]
            .as_array()
            .is_some_and(|items| items.iter().any(|item| item["type"] == "function_call"));
        assert_eq!(
            has_function_call, expect_tool_call,
            "{scenario}: interaction {index} tool-call premise"
        );
    }
    assert_side_matches_fixture(
        scenario,
        "raw_completion",
        &typed,
        &first,
        &request_ids[0],
        Wire::Responses,
    );
    assert_side_matches_fixture(
        scenario,
        "completion",
        &normalized,
        &second,
        &request_ids[1],
        Wire::Responses,
    );
    if expect_tool_call {
        assert_eq!(
            tool_call_names(&typed),
            ["ping"],
            "{scenario}: typed route tool call"
        );
        assert_eq!(
            tool_call_names(&normalized),
            ["ping"],
            "{scenario}: completion() tool call"
        );
    } else {
        assert_eq!(typed.choice, normalized.choice, "{scenario}: text choice");
        // The Responses route names the assistant message; both sides do.
        assert!(
            typed
                .message_id
                .as_deref()
                .is_some_and(|id| id.starts_with("msg_"))
                && normalized
                    .message_id
                    .as_deref()
                    .is_some_and(|id| id.starts_with("msg_")),
            "{scenario}: both routes carry the msg_ id"
        );
    }
    assert_parity(scenario, &typed, &normalized, expected_finish);
}

#[tokio::test]
async fn responses_text_turn_parity() {
    const SCENARIO: &str = "raw_completion_parity_matrix/responses_text_turn_parity";
    let observed = Observed::default();
    with_openai_cassette(
        "raw_completion_parity_matrix/responses_text_turn_parity",
        responses_parity_body(observed.clone(), text_request),
    )
    .await;
    assert_responses_parity(SCENARIO, &observed, FinishReason::Stop, false);
}

/// A completed Responses turn reports `status: completed` — mapped to `Stop`
/// — and it is `with_optional_finish_reason`'s reconciliation that upgrades
/// it to `ToolCalls` for a turn carrying a function call. Both routes go
/// through the same normalize, so both must agree on the reconciled reason.
#[tokio::test]
async fn responses_tool_turn_parity() {
    const SCENARIO: &str = "raw_completion_parity_matrix/responses_tool_turn_parity";
    let observed = Observed::default();
    with_openai_cassette(
        "raw_completion_parity_matrix/responses_tool_turn_parity",
        responses_parity_body(observed.clone(), tool_request),
    )
    .await;
    assert_responses_parity(SCENARIO, &observed, FinishReason::ToolCalls, true);
}
