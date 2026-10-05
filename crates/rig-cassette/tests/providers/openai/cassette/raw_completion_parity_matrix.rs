//! Raw parity for OpenAI: the provider-native view of a reply reproduces
//! what `Model::call` returns.
//!
//! # What this pins
//!
//! One call yields both views of one reply. The normalized
//! [`CompletionResponse`] is what `completion()` returns, and
//! [`CompletionResponse::raw`] holds the provider's reply document verbatim.
//! Reading `raw` as JSON, under the provider's own field names, must
//! reproduce the response the call already handed back.
//!
//! On the Chat Completions route the transport request id from the
//! `x-request-id` header is not in the reply document. It is a header, not a
//! body field, and `raw` is the body. So the provider-native
//! view reproduces `completion()` only once the call's own transport id is
//! attached to it.
//!
//! On the Responses route the reply document carries no transport id
//! either, for the same reason, so the same rule holds.
//!
//! Every parity cell issues the same request twice — two interactions of one
//! scenario, replayed in order by the harness. The first call supplies the
//! provider-native view, the second the `completion()` view. The two
//! responses are distinct provider turns, so each side is first checked
//! against *its own* fixture interaction (id, request-id header, usage,
//! model, finish reason), and then the two sides are compared on the fields
//! the contract names: `finish_reason()`, `model`, `usage`, and
//! `provider_request_id.is_some()`. Identity is compared structurally
//! (`response_id` present with the route's prefix, `provider_request_id`
//! present) — two live turns cannot share ids.
//!
//! # Matrix
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 5 | `responses_tool_turn_parity` | Responses, forced tool call | raw ≡ completion (`ToolCalls`) | recorded |
//!
//! Every cell is recorded; none is unit-only. Premise, re-derived from each
//! cell's fixture bytes after the wrapper returns: every recorded response
//! carries an `x-request-id` header — without that the cells would prove
//! nothing about the transport id.

use rig::completion::{
    AssistantContent, CompletionRequest, CompletionResponse, FinishReason, ToolDefinition,
};
use rig::message::ToolChoice;
use rig::providers::openai;
use serde_json::{Value, json};

use super::super::support::{recorded_request_id_headers, with_openai_cassette_result};
use crate::raw_capture::{assert_contracted_request_id, capture_completion_pair};
use crate::support::{Observed, assert_matches_recorded_token};

const PROVIDER: &str = "openai";
const MODEL: &str = openai::GPT_4_1_NANO;
const TOOL_PROMPT: &str = "Call ping exactly once with no arguments.";
/// The response header OpenAI contracts as the transport request id, on both
/// routes — the datum `recorded_request_ids` reads back out of the fixture.
const REQUEST_ID_HEADER: &str = "x-request-id";

fn ping_tool() -> ToolDefinition {
    ToolDefinition {
        name: rig_core::message::ToolName::new("ping").expect("tool name"),
        description: "Matrix tool ping".to_owned(),
        parameters: json!({ "type": "object", "properties": {}, "additionalProperties": false }),
    }
}

fn tool_request() -> CompletionRequest {
    CompletionRequest::new(TOOL_PROMPT)
        .tool(ping_tool())
        .tool_choice(ToolChoice::Required)
        .temperature(0.0)
        .max_tokens(64)
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
    Responses,
}

impl Wire {
    fn id_prefix(self) -> &'static str {
        match self {
            Wire::Responses => "resp_",
        }
    }

    /// `(input, output)` usage keys of the wire body.
    fn usage_keys(self) -> (&'static str, &'static str) {
        match self {
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
    assert_contracted_request_id(
        response.provider_request_id.as_deref(),
        Some(recorded_request_id),
        REQUEST_ID_HEADER,
    );
    assert_matches_recorded_token(
        response.response_id(),
        body["id"].as_str(),
        &format!("{context}: response_id vs the fixture body id"),
    );
    assert!(
        response
            .response_id()
            .is_some_and(|id| id.starts_with(id_prefix)),
        "{context}: response_id should be a {id_prefix} id, got {:?}",
        response.response_id()
    );
    assert_eq!(response.model(), body["model"].as_str(), "{context}: model");
    assert_eq!(
        response.usage.input_tokens,
        body["usage"][usage_input_key].as_u64(),
        "{context}: input tokens"
    );
    assert_eq!(
        response.usage.output_tokens,
        body["usage"][usage_output_key].as_u64(),
        "{context}: output tokens"
    );
    assert_eq!(response.provider(), PROVIDER, "{context}: provider");
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
    assert_eq!(typed.model(), normalized.model(), "{scenario}: model");
    assert_eq!(typed.usage, normalized.usage, "{scenario}: usage");
    assert_eq!(
        typed.provider(),
        normalized.provider(),
        "{scenario}: provider"
    );
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
    assert_ne!(
        typed_identity.response_id, normalized_identity.response_id,
        "{scenario}: two live turns are two provider responses"
    );
}

// ---------------------------------------------------------------------------
// Chat Completions
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Responses
// ---------------------------------------------------------------------------

/// [`assert_chat_views_agree`] for the Responses route, whose body names its
/// usage counters differently and whose own `provider_request_id` field is
/// never part of the document the wire sends.
fn assert_responses_views_agree(scenario: &str, reply: &Value, response: &CompletionResponse) {
    assert_eq!(
        response.response_id(),
        reply["id"].as_str(),
        "{scenario}: the response id is the provider's `id`"
    );
    assert_eq!(
        response.model(),
        reply["model"].as_str(),
        "{scenario}: model"
    );
    let usage = &reply["usage"];
    assert!(
        usage.is_object(),
        "{scenario}: the recorded Responses body reports usage"
    );
    assert_eq!(
        response.usage.input_tokens,
        usage["input_tokens"].as_u64(),
        "{scenario}: input tokens"
    );
    assert_eq!(
        response.usage.output_tokens,
        usage["output_tokens"].as_u64(),
        "{scenario}: output tokens"
    );
    assert!(
        response.raw.get("provider_request_id").is_none(),
        "{scenario}: the transport id is a header, so the reply document has none"
    );
}

fn assert_responses_parity(
    scenario: &str,
    observed: &Observed<(CompletionResponse, CompletionResponse)>,
    expected_finish: FinishReason,
    expect_tool_call: bool,
) {
    let (typed, normalized) = observed.take();
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
    // The first reply, read both ways: the provider's own document in `raw`,
    // then rig's normalized view of the same reply.
    assert_responses_views_agree(scenario, &typed.raw, &typed);
    assert_side_matches_fixture(
        scenario,
        "raw view",
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
        // Two live turns: the same text, each block holding its own item.
        let canonical = |choice: &[rig::message::AssistantContent]| {
            choice
                .iter()
                .map(rig::message::AssistantContent::canonical)
                .collect::<Vec<_>>()
        };
        assert_eq!(
            canonical(&typed.choice),
            canonical(&normalized.choice),
            "{scenario}: text choice"
        );
        // The Responses route names the assistant message; both sides do.
        let msg = |response: &rig::completion::CompletionResponse| {
            crate::raw_capture::responses::message_item_id(response)
                .is_some_and(|id| id.starts_with("msg_"))
        };
        assert!(
            msg(&typed) && msg(&normalized),
            "{scenario}: both routes carry the msg_ id"
        );
    }
    assert_parity(scenario, &typed, &normalized, expected_finish);
}

/// A completed Responses turn reports `status: completed` — mapped to `Stop`
/// — and it is `with_optional_finish_reason`'s reconciliation that upgrades
/// it to `ToolCalls` for a turn carrying a function call. Both routes go
/// through the same normalize, so both must agree on the reconciled reason.
#[tokio::test]
async fn responses_tool_turn_parity() {
    const SCENARIO: &str = "raw_completion_parity_matrix/responses_tool_turn_parity";
    let observed = Observed::default();
    with_openai_cassette_result(
        "raw_completion_parity_matrix/responses_tool_turn_parity",
        |client| {
            capture_completion_pair(
                client.openai.completion(MODEL),
                tool_request(),
                observed.clone(),
            )
        },
    )
    .await
    .expect("responses_tool_turn_parity should replay from its cassette");
    assert_responses_parity(SCENARIO, &observed, FinishReason::ToolCalls, true);
}
