//! Matrix for the typed escape hatch's parity with the normalized path on
//! ChatGPT: `raw_completion(req).normalize("chatgpt")` must reproduce what
//! `completion(req)` reports for `identity()`, `finish_reason()`, `model` and
//! `usage`.
//!
//! # The contract
//!
//! [`ResponsesCompletionModel::raw_completion`](rig::providers::chatgpt::ResponsesCompletionModel::raw_completion)
//! and [`CompletionModel::completion`](rig::completion::CompletionModel::completion)
//! share one transport (`send_completion`): both reassemble the Responses
//! wire type from the SSE body's terminal `response.completed` event, and the
//! normalized path is `raw.normalize(PROVIDER_NAME)` plus the captured `raw`.
//! ChatGPT reads no transport request-id header, so the whole identity — the
//! `resp_…` response id and the `msg_…` message id — lives in the body and
//! the typed route needs no `with_optional_provider_request_id` reassembly:
//! `provider_request_id` is `None` on both routes.
//!
//! The one place the two routes diverge is the empty-`output` fallback: when
//! the terminal event carries no items, `completion` rebuilds the content from
//! the preceding events (`completion_response_from_sse_body`) while
//! `raw_completion` returns the terminal envelope as-is with `output: []`, and
//! `raw.normalize(..)` returns the empty-response error. `normalized_completion`
//! captures `raw` on that branch too — the same `raw_response` value.
//!
//! # Matrix
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `raw_normalize_reproduces_completion` | typed route vs `completion` | identity/finish_reason/model/usage each equal their own recorded terminal frame; response-independent fields equal across routes | unrecorded (no CHATGPT credentials in this environment) |
//! | 2 | `raw_normalize_reproduces_completion_with_tool_call` | tool-call turn | same, with `finish_reason == ToolCalls` and the call id from the frame | unrecorded (no CHATGPT credentials in this environment) |
//! | 3 | `empty_output_fallback_still_carries_raw` | empty-`output` terminal event | `completion` rebuilds content from events **and** carries `raw` whose `output` is empty | unrecorded (no CHATGPT credentials in this environment) |
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
//! and review `tests/cassettes/chatgpt/raw_completion_parity_matrix/`.

use rig::completion::NormalizeCompletionResponse as _;
use rig::completion::{
    CompletionModel as _, CompletionResponse as RigCompletionResponse, FinishReason, ToolDefinition,
};
use rig::message::AssistantContent;
use rig::prelude::*;
use rig::providers::chatgpt;
use rig::providers::openai::responses_api;
use serde::Deserialize;
use serde_json::{Value, json};

use super::super::support::with_chatgpt_cassette;
use crate::cassettes::{CassetteMode, recorded_interaction_bodies};

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

fn request(model: &chatgpt::ResponsesCompletionModel) -> rig::completion::CompletionRequest {
    model.completion_request(PROMPT).max_tokens(64).build()
}

fn tool_request(model: &chatgpt::ResponsesCompletionModel) -> rig::completion::CompletionRequest {
    model
        .completion_request(TOOL_PROMPT)
        .tool(weather_tool())
        .max_tokens(128)
        .build()
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
                .map(|frame| frame["response"].clone())
                .unwrap_or_else(|| {
                    panic!("{scenario}: each interaction must end with response.completed")
                });
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

/// A route's normalized response must be exactly the normalization of its own
/// recorded terminal envelope — the on-disk proof that the route did not add,
/// drop or reshape anything the wire said.
fn assert_matches_own_wire(response: &RigCompletionResponse, terminal: &Value, route: &str) {
    let from_wire = responses_api::CompletionResponse::deserialize(terminal)
        .expect("recorded terminal envelope must be a Responses response")
        .normalize(CHATGPT_PROVIDER)
        .expect("recorded terminal envelope must normalize");

    assert_eq!(
        response.finish_reason(),
        from_wire.finish_reason(),
        "{route}: finish_reason"
    );
    assert_eq!(response.model, from_wire.model, "{route}: model");
    assert_eq!(response.usage, from_wire.usage, "{route}: usage");
    assert_eq!(response.provider, CHATGPT_PROVIDER, "{route}: provider");
    // The transport id is `None` on both routes: ChatGPT reads no header.
    assert_eq!(
        response.identity().provider_request_id,
        None,
        "{route}: transport id"
    );
    match CassetteMode::current() {
        // Replay reads the scrubbed ids back, so identity compares exactly.
        CassetteMode::Replay => assert_eq!(response.identity(), from_wire.identity(), "{route}"),
        // Live, the fixture holds placeholders; the shape claim is that the
        // route populated the same identity axes the wire populated.
        CassetteMode::Record => {
            let live = response.identity();
            let wire = from_wire.identity();
            assert_eq!(
                live.response_id.is_some(),
                wire.response_id.is_some(),
                "{route}"
            );
            assert_eq!(
                live.message_id.is_some(),
                wire.message_id.is_some(),
                "{route}"
            );
            assert!(
                live.response_id
                    .as_deref()
                    .is_some_and(|id| id.starts_with("resp_")),
                "{route}: response id should be a resp_ id, got {:?}",
                live.response_id
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 1: text turn — one scenario, two interactions: raw route, then completion
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "unrecorded (no CHATGPT credentials in this environment)"]
async fn raw_normalize_reproduces_completion() {
    let scenario = "raw_completion_parity_matrix/raw_normalize_reproduces_completion";
    let captured = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = std::sync::Arc::clone(&captured);
    with_chatgpt_cassette(
        "raw_completion_parity_matrix/raw_normalize_reproduces_completion",
        |client| async move {
            let model = client.completion_model(MODEL);

            let raw = model
                .raw_completion(request(&model))
                .await
                .expect("raw completion should succeed");
            assert!(
                !raw.output.is_empty(),
                "premise: the terminal envelope carried items"
            );
            let via_raw = raw
                .normalize(CHATGPT_PROVIDER)
                .expect("raw route must normalize");

            let via_completion = model
                .completion(request(&model))
                .await
                .expect("completion should succeed");

            // Response-independent parity across the two routes.
            assert_eq!(via_raw.finish_reason(), via_completion.finish_reason());
            assert_eq!(via_raw.finish_reason(), Some(FinishReason::Stop));
            assert_eq!(via_raw.model, via_completion.model);
            assert_eq!(via_raw.provider, via_completion.provider);
            assert_eq!(
                via_raw.identity().provider_request_id,
                via_completion.identity().provider_request_id
            );
            assert_eq!(
                via_raw.identity().response_id.is_some(),
                via_completion.identity().response_id.is_some()
            );
            assert_eq!(
                via_raw.identity().message_id.is_some(),
                via_completion.identity().message_id.is_some()
            );
            assert_eq!(
                via_raw.usage.input_tokens,
                via_completion.usage.input_tokens
            );
            // The typed route carries no `raw` (it *is* the raw); the
            // normalized route without opt-in carries none either.
            assert!(via_raw.raw.is_none());
            assert!(via_completion.raw.is_none());

            *sink.lock().expect("capture mutex") = vec![via_raw, via_completion];
        },
    )
    .await;

    let responses = std::mem::take(&mut *captured.lock().expect("capture mutex"));
    let terminals = recorded_terminal_responses(scenario);
    assert_eq!(
        terminals.len(),
        2,
        "{scenario}: expected the raw and the completion turns"
    );
    assert_matches_own_wire(&responses[0], &terminals[0], "raw_completion + normalize");
    assert_matches_own_wire(&responses[1], &terminals[1], "completion");
}

// ---------------------------------------------------------------------------
// 2: tool-call turn
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "unrecorded (no CHATGPT credentials in this environment)"]
async fn raw_normalize_reproduces_completion_with_tool_call() {
    let scenario =
        "raw_completion_parity_matrix/raw_normalize_reproduces_completion_with_tool_call";
    let captured = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = std::sync::Arc::clone(&captured);
    with_chatgpt_cassette(
        "raw_completion_parity_matrix/raw_normalize_reproduces_completion_with_tool_call",
        |client| async move {
            let model = client.completion_model(MODEL);

            let raw = model
                .raw_completion(tool_request(&model))
                .await
                .expect("raw completion should succeed");
            let via_raw = raw
                .normalize(CHATGPT_PROVIDER)
                .expect("raw route must normalize");
            let via_completion = model
                .completion(tool_request(&model))
                .await
                .expect("completion should succeed");

            for (route, response) in [("raw", &via_raw), ("completion", &via_completion)] {
                assert_eq!(
                    response.finish_reason(),
                    Some(FinishReason::ToolCalls),
                    "{route}: a tool-call turn normalizes to ToolCalls"
                );
                assert!(
                    response
                        .choice
                        .iter()
                        .any(|content| matches!(content, AssistantContent::ToolCall(call) if call.function.name == "get_weather")),
                    "{route}: the get_weather call must be on the choice"
                );
            }
            assert_eq!(via_raw.model, via_completion.model);
            assert_eq!(via_raw.provider, via_completion.provider);
            *sink.lock().expect("capture mutex") = vec![via_raw, via_completion];
        },
    )
    .await;

    let responses = std::mem::take(&mut *captured.lock().expect("capture mutex"));
    let terminals = recorded_terminal_responses(scenario);
    assert_eq!(
        terminals.len(),
        2,
        "{scenario}: expected the raw and the completion turns"
    );
    for (terminal, (response, route)) in terminals.iter().zip([
        (&responses[0], "raw_completion + normalize"),
        (&responses[1], "completion"),
    ]) {
        assert!(
            terminal["output"]
                .as_array()
                .is_some_and(|items| items.iter().any(|item| item["type"] == "function_call")),
            "{scenario}: premise — the recorded terminal envelope carries a function_call item"
        );
        assert_matches_own_wire(response, terminal, route);
    }
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
    with_chatgpt_cassette(
        "raw_completion_parity_matrix/empty_output_fallback_still_carries_raw",
        |client| async move {
            let model = client.completion_model(MODEL);
            let response = model
                .completion(
                    model
                        .completion_request(PROMPT)
                        .max_tokens(64)
                        .capture_raw_response(true)
                        .build(),
                )
                .await
                .expect("the fallback rebuilds the response from the event stream");

            // The normalized content came from the events…
            assert!(
                !response.choice.is_empty(),
                "the fallback must rebuild the assistant content from the events"
            );
            // …and raw is still the terminal envelope, whose output is empty.
            let raw = response
                .raw
                .as_deref()
                .expect("the fallback branch must carry raw when capture was requested");
            let typed = responses_api::CompletionResponse::deserialize(raw)
                .expect("raw must deserialize into responses_api::CompletionResponse");
            assert!(
                typed.output.is_empty(),
                "premise: this cell exists for the empty-output terminal event"
            );
            assert_eq!(
                serde_json::to_value(&typed).expect("provider type should serialize"),
                *raw
            );
        },
    )
    .await;

    let terminals = recorded_terminal_responses(scenario);
    assert_eq!(
        terminals[0]["output"],
        json!([]),
        "{scenario}: premise — the recorded terminal envelope carried no output items; \
         a recording where it did does not exercise the fallback"
    );
}
