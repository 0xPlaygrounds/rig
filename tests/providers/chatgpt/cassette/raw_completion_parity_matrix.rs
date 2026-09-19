//! Matrix for the two views of one ChatGPT reply: the normalized
//! [`CompletionResponse`](rig::completion::CompletionResponse) the driver
//! returns and the provider envelope it captured on
//! [`raw`](rig::completion::CompletionResponse::raw). Both must tell the same
//! story about `identity()`, `finish_reason()`, `model` and `usage`.
//!
//! # The contract
//!
//! One call yields both views. The driver reassembles the Responses wire type
//! from the SSE body's terminal `response.completed` event, folds it into the
//! normalized response, and serializes the same value onto `raw`. ChatGPT
//! reads no transport request-id header, so the whole identity — the
//! `resp_…` response id and the `msg_…` message id — lives in the body:
//! `provider_request_id` is `None`.
//!
//! The state worth its own cell is the empty-`output` terminal event: the
//! assistant content is folded from the preceding stream events while the
//! captured envelope still carries `output: []`, so the two views disagree on
//! where the content came from and agree on everything the envelope states.
//!
//! # Matrix
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `raw_normalize_reproduces_completion` | text turn | the normalized view's identity/finish_reason/model/usage equal the recorded terminal frame's, and `raw` reads back as the same envelope | unrecorded (no CHATGPT credentials in this environment) |
//! | 2 | `raw_normalize_reproduces_completion_with_tool_call` | tool-call turn | same, with `finish_reason == ToolCalls` and the call on the choice matching the frame's `function_call` item | unrecorded (no CHATGPT credentials in this environment) |
//! | 3 | `empty_output_fallback_still_carries_raw` | empty-`output` terminal event | the content folds from the events **and** `raw` carries the envelope whose `output` is empty | unrecorded (no CHATGPT credentials in this environment) |
//!
//! Every cell is unrecorded: neither `CHATGPT_ACCESS_TOKEN`/`CHATGPT_ACCOUNT_ID`
//! nor a usable ChatGPT OAuth cache was present when this matrix was written,
//! and a fixture is never fabricated. Cell 3 additionally cannot be produced
//! on demand: the empty-`output` terminal event is a backend behavior, not a
//! request shape, so even with credentials the cell records only when the
//! backend happens to emit it — its body asserts the premise (`output: []`
//! on the recorded terminal frame) and fails loudly otherwise; the branch is
//! pinned by construction in the provider's own unit test
//! (`test_completion_response_from_sse_body_falls_back_to_streamed_text` in
//! `providers/chatgpt`), and this
//! cell exists so the live-traffic proof has a home when the state shows up.
//!
//! To record cells 1–2: export `CHATGPT_ACCESS_TOKEN` and `CHATGPT_ACCOUNT_ID`,
//! remove the `#[ignore]` attributes, flip the table to `recorded`, then run
//! `RIG_PROVIDER_TEST_MODE=record cargo test -p rig --all-features --test chatgpt chatgpt::cassette::raw_completion_parity_matrix -- --nocapture --test-threads=1`
//! and review `crates/rig-cassette/fixtures/cassettes/chatgpt/raw_completion_parity_matrix/`.

use rig::completion::{
    CompletionModel as _, CompletionResponse as RigCompletionResponse, FinishReason, ToolDefinition,
};
use rig::driver::Bound;
use rig::message::AssistantContent;
use rig::providers::chatgpt;
use rig::providers::openai::responses_api;
use rig::providers::openai::wire::OpenAiWire;
use serde::Deserialize;
use serde_json::{Value, json};

use super::super::support::with_chatgpt_cassette;
use crate::cassettes::{CassetteMode, recorded_interaction_bodies};
use crate::raw_capture::{assert_no_request_id, capture_completion, responses};
use crate::support::{Observed, assert_matches_recorded_token};

const CHATGPT_PROVIDER: &str = "chatgpt";
const MODEL: &str = chatgpt::GPT_5_4;
const PROMPT: &str = "Reply with exactly the single word: pong";
const TOOL_PROMPT: &str = "Use the get_weather tool for the city Lisbon.";

fn weather_tool() -> ToolDefinition {
    ToolDefinition {
        name: "get_weather".to_string(),
        description: "Get the current weather for a city.".to_string(),
        parameters: json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"]
        }),
    }
}

type ChatGptModel = Bound<OpenAiWire>;

fn request(model: &ChatGptModel) -> rig::completion::CompletionRequest {
    model.completion_request(PROMPT).max_tokens(64).build()
}

fn tool_request(model: &ChatGptModel) -> rig::completion::CompletionRequest {
    model
        .completion_request(TOOL_PROMPT)
        .tool(weather_tool())
        .max_tokens(128)
        .build()
}

/// The `msg_…` id of the envelope's output message, when it issued one.
fn wire_message_id(envelope: &responses_api::CompletionResponse) -> Option<String> {
    envelope.output.iter().find_map(|item| match item {
        responses_api::Output::Message(message) => Some(message.id.clone()),
        _ => None,
    })
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
                .response_id
                .as_deref()
                .is_some_and(|id| id.starts_with("resp_")),
            "response id should be a resp_ id, got {:?}",
            response.response_id
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
            capture_completion(client.completion(MODEL), request, sink)
                .await
                .expect("completion should succeed");
        },
    )
    .await;

    let response = captured.take();
    let envelope = responses_api::CompletionResponse::deserialize(&response.raw)
        .expect("`raw` is the serialized responses_api::CompletionResponse");
    assert!(
        !envelope.output.is_empty(),
        "premise: the terminal envelope carried items"
    );
    assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
    assert_eq!(
        envelope.status,
        responses_api::ResponseStatus::Completed,
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
            capture_completion(client.completion(MODEL), tool_request, sink)
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

    let envelope = responses_api::CompletionResponse::deserialize(&response.raw)
        .expect("`raw` is the serialized responses_api::CompletionResponse");
    assert!(
        envelope.output.iter().any(|item| matches!(
            item,
            responses_api::Output::FunctionCall(call) if call.name == "get_weather"
        )),
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
    let from_wire = responses_api::CompletionResponse::deserialize(&terminals[0])
        .expect("recorded terminal envelope must be a Responses response");
    assert_eq!(response.provider, CHATGPT_PROVIDER, "provider");
    assert_eq!(
        response.model.as_deref(),
        Some(from_wire.model.as_str()),
        "model"
    );
    let usage = from_wire
        .usage
        .as_ref()
        .expect("the terminal envelope must report usage");
    assert_eq!(
        (
            response.usage.input_tokens,
            response.usage.output_tokens,
            response.usage.total_tokens
        ),
        (
            Some(usage.input_tokens),
            Some(usage.output_tokens),
            Some(usage.total_tokens)
        ),
        "usage"
    );
    assert_no_request_id(response.provider_request_id.as_deref(), CHATGPT_PROVIDER);
    assert_matches_recorded_token(
        response.response_id.as_deref(),
        Some(from_wire.id.as_str()),
        "response id",
    );
    let wire_message = wire_message_id(&from_wire);
    assert_matches_recorded_token(
        response.message_id.as_deref(),
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
            capture_completion(client.completion(MODEL), request, sink)
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
    let typed = responses_api::CompletionResponse::deserialize(&response.raw)
        .expect("raw must deserialize into responses_api::CompletionResponse");
    assert!(
        typed.output.is_empty(),
        "premise: this cell exists for the empty-output terminal event"
    );
    assert_eq!(
        typed.status,
        responses_api::ResponseStatus::Completed,
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
