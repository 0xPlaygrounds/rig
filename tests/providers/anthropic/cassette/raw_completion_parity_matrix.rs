//! Parity matrix for the typed escape hatch: `raw_completion` / `raw_stream`
//! followed by rig's own normalization must reproduce what `completion` /
//! `stream` report.
//!
//! # The contract
//!
//! Anthropic's raw type carries the transport id itself
//! (`anthropic::completion::CompletionResponse::provider_request_id`, stamped
//! from the `request-id` response header by the request driver), so the
//! typed route is one step: `raw_completion(req)?.normalize("anthropic")`.
//! That value must agree with `completion(req)` on `identity()`,
//! `finish_reason()`, `model`, and `usage` — otherwise a caller who reaches for
//! the provider type to read a field rig does not normalize silently loses the
//! metadata the normalized route would have given them. Streaming has the same
//! shape: `raw_stream`'s terminal mapped through the public
//! `StreamFinal::from((&str, StreamingCompletionResponse))` must agree with
//! `stream()`'s terminal record.
//!
//! Each cell makes two live requests (one per route) in one scenario, so the
//! fixture holds two interactions in wire order: `completion` / `stream` first,
//! the raw route second. Two requests are two responses, so `identity()` cannot
//! be *literally* equal across them — each attempt reports its own message id
//! and request id. Parity is therefore asserted the only honest way: every
//! identity field each route reports equals what *its own* recorded
//! interaction says (message id from the body, transport id from the
//! `request-id` header), both routes populate the same identity fields, and
//! `finish_reason`, `model`, and `usage.input_tokens` (deterministic for an
//! identical prompt) are equal outright, with each route's `output_tokens`
//! pinned to its own recorded usage.
//!
//! # Matrix
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `text_turn_parity` | `raw_completion` + `normalize` vs `completion`, `end_turn` | equal `Stop`, model, usage; identity per fixture | recorded |
//! | 2 | `tool_call_turn_parity` | same, `tool_use` terminal | equal `ToolCalls` (reconciled), model, usage; identity per fixture | recorded |
//! | 3 | `streamed_text_turn_parity` | `raw_stream` + `StreamFinal::from` vs `stream()` terminal | equal `Stop`, model, usage; identity per fixture | recorded |
//! | 4 | `streamed_tool_call_turn_parity` | same, `tool_use` terminal | equal `ToolCalls`, model, usage; identity per fixture | recorded |
//!
//! Every cell is recorded; the premise each re-derives from its fixture is that
//! both interactions' responses carry a `request-id` header and a `msg_…` id,
//! and that the recorded stop reason is the one the cell is about — a
//! recording that lost either would make the parity claim vacuous.

use futures::StreamExt;
use rig::completion::{
    CompletionModel as _, CompletionResponse as RigCompletionResponse, FinishReason,
    NormalizeCompletionResponse, ResponseIdentity, Usage,
};
use rig::message::ToolChoice;
use rig::prelude::*;
use rig::providers::anthropic;
use rig::streaming::{RawStreamingChoice, StreamFinal, StreamedAssistantContent};
use rig::tool::Tool;

use super::super::support::{
    assert_ids_match_recording, recorded_request_id_headers, sse_json_frames,
    with_anthropic_cassette,
};
use crate::support::{Adder, TOOLS_PREAMBLE};

const ANTHROPIC_PROVIDER: &str = "anthropic";
const TEXT_PROMPT: &str = "Reply with exactly: parity probe";
const TOOL_PROMPT: &str = "What is 2 + 3? Use the tool.";

type AnthropicModel = anthropic::completion::CompletionModel;

fn text_request(model: &AnthropicModel) -> rig::completion::CompletionRequest {
    model.completion_request(TEXT_PROMPT).max_tokens(32).build()
}

/// `tool_choice: required` (Anthropic `any`) so the turn is a `tool_use`
/// terminal by construction, not by the model's mood.
fn tool_request(model: &AnthropicModel) -> rig::completion::CompletionRequest {
    model
        .completion_request(TOOL_PROMPT)
        .preamble(TOOLS_PREAMBLE.to_string())
        .max_tokens(256)
        .tool(rig::tool::tool_definition(&Adder))
        .tool_choice(ToolChoice::Required)
        .build()
}

/// The comparable part of one route's result: what both routes must agree on.
#[derive(Debug, Clone, PartialEq)]
struct Reported {
    identity: ResponseIdentity,
    finish_reason: Option<FinishReason>,
    model: Option<String>,
    usage: Usage,
}

impl Reported {
    fn from_completion(response: &RigCompletionResponse) -> Self {
        Self {
            identity: response.identity(),
            finish_reason: response.finish_reason(),
            model: response.model.clone(),
            usage: response.usage,
        }
    }

    fn from_terminal(terminal: &StreamFinal) -> Self {
        Self {
            identity: terminal.identity(),
            finish_reason: terminal.finish_reason.clone(),
            model: terminal.model.clone(),
            usage: terminal.usage,
        }
    }
}

/// Parity between the normalized route (`first`) and the typed route
/// (`second`), for the fields that do not depend on which HTTP exchange
/// produced them.
fn assert_route_parity(first: &Reported, second: &Reported, expected: FinishReason) {
    assert_eq!(first.finish_reason, Some(expected));
    assert_eq!(
        second.finish_reason, first.finish_reason,
        "the typed route must map the stop reason exactly as `completion` does"
    );
    assert!(first.model.is_some(), "the wire names its model");
    assert_eq!(second.model, first.model);
    assert_eq!(
        second.usage.input_tokens, first.usage.input_tokens,
        "the same prompt costs the same input tokens on both routes"
    );
    // Both routes populate the same identity *fields*; the values are
    // per-exchange and pinned against each route's own fixture below.
    assert!(first.identity.message_id.is_some());
    assert!(second.identity.message_id.is_some());
    assert!(first.identity.provider_request_id.is_some());
    assert!(second.identity.provider_request_id.is_some());
    assert_eq!(first.identity.response_id, None);
    assert_eq!(second.identity.response_id, None);
}

/// Pin each route's identity and output usage to *its own* recorded
/// interaction: message id from the body (interaction `i`), transport id from
/// the `request-id` header, `output_tokens` from the recorded usage.
fn assert_identity_matches_fixture(
    scenario: &str,
    reported: &[Reported],
    recorded_message_ids: Vec<Option<String>>,
    recorded_output_tokens: Vec<u64>,
    expected_stop_reasons: &[&str],
    recorded_stop_reasons: Vec<Option<String>>,
) {
    assert_eq!(
        reported.len(),
        2,
        "{scenario}: one result per route (normalized first, typed second)"
    );
    let request_ids = recorded_request_id_headers(scenario);
    assert_eq!(
        request_ids.len(),
        2,
        "{scenario}: two recorded interactions"
    );
    assert!(
        request_ids.iter().all(Option::is_some),
        "{scenario}: premise — every recorded response carries a `request-id` header, got {request_ids:?}"
    );
    assert!(
        recorded_message_ids
            .iter()
            .all(|id| id.as_deref().is_some_and(|id| id.starts_with("msg_"))),
        "{scenario}: premise — every recorded response names a `msg_…` id, got {recorded_message_ids:?}"
    );
    assert_eq!(
        recorded_stop_reasons,
        expected_stop_reasons
            .iter()
            .map(|reason| Some((*reason).to_string()))
            .collect::<Vec<_>>(),
        "{scenario}: premise — both recorded turns stopped for the reason this cell is about"
    );

    let observed_request_ids: Vec<_> = reported
        .iter()
        .map(|reported| reported.identity.provider_request_id.clone())
        .collect();
    assert_ids_match_recording(&observed_request_ids, &request_ids, scenario);

    let observed_message_ids: Vec<_> = reported
        .iter()
        .map(|reported| reported.identity.message_id.clone())
        .collect();
    assert_ids_match_recording(&observed_message_ids, &recorded_message_ids, scenario);

    let observed_output_tokens: Vec<_> = reported
        .iter()
        .map(|reported| reported.usage.output_tokens)
        .collect();
    assert_eq!(
        observed_output_tokens, recorded_output_tokens,
        "{scenario}: each route reports the output tokens its own exchange recorded"
    );
}

fn recorded_blocking_premise(
    scenario: &str,
) -> (Vec<Option<String>>, Vec<u64>, Vec<Option<String>>) {
    let bodies = crate::cassettes::recorded_interaction_bodies(ANTHROPIC_PROVIDER, scenario);
    let responses: Vec<serde_json::Value> = bodies
        .iter()
        .map(|(_, response)| {
            serde_json::from_str(response).expect("recorded blocking body should be JSON")
        })
        .collect();
    let ids = responses
        .iter()
        .map(|body| body["id"].as_str().map(str::to_string))
        .collect();
    let output_tokens = responses
        .iter()
        .map(|body| {
            body["usage"]["output_tokens"]
                .as_u64()
                .expect("recorded usage.output_tokens")
        })
        .collect();
    let stop_reasons = responses
        .iter()
        .map(|body| body["stop_reason"].as_str().map(str::to_string))
        .collect();
    (ids, output_tokens, stop_reasons)
}

/// Streamed premise from the frames: `message_start` names the id, the
/// terminal `message_delta` names the stop reason and carries the final usage.
fn recorded_streamed_premise(
    scenario: &str,
) -> (Vec<Option<String>>, Vec<u64>, Vec<Option<String>>) {
    let bodies = crate::cassettes::recorded_interaction_bodies(ANTHROPIC_PROVIDER, scenario);
    let mut ids = Vec::new();
    let mut output_tokens = Vec::new();
    let mut stop_reasons = Vec::new();
    for (_, response) in &bodies {
        let frames = sse_json_frames(response);
        let start = frames
            .iter()
            .find(|frame| frame["type"] == "message_start")
            .expect("recorded stream should open with message_start");
        ids.push(start["message"]["id"].as_str().map(str::to_string));
        let delta = frames
            .iter()
            .find(|frame| frame["type"] == "message_delta")
            .expect("recorded stream should carry a terminal message_delta");
        output_tokens.push(
            delta["usage"]["output_tokens"]
                .as_u64()
                .expect("terminal message_delta carries usage.output_tokens"),
        );
        stop_reasons.push(delta["delta"]["stop_reason"].as_str().map(str::to_string));
    }
    (ids, output_tokens, stop_reasons)
}

async fn drain_normalized_terminal(
    mut stream: rig::streaming::StreamingCompletionResponse,
) -> StreamFinal {
    let mut terminal = None;
    while let Some(item) = stream.next().await {
        if let StreamedAssistantContent::Final(final_record) =
            item.expect("normalized stream item should succeed")
        {
            terminal = Some(final_record);
        }
    }
    terminal.expect("normalized stream should yield a terminal record")
}

async fn drain_raw_terminal(
    mut stream: rig::streaming::RawStreamingResult<
        anthropic::streaming::StreamingCompletionResponse,
    >,
) -> anthropic::streaming::StreamingCompletionResponse {
    let mut terminal = None;
    while let Some(item) = stream.next().await {
        if let RawStreamingChoice::FinalResponse(response) =
            item.expect("raw stream item should succeed")
        {
            terminal = Some(response);
        }
    }
    terminal.expect("raw stream should yield a terminal record")
}

type ReportedSink = std::sync::Arc<std::sync::Mutex<Vec<Reported>>>;

/// Body of a blocking cell: both routes, one request each, parity asserted
/// inside; what each reported is kept for the fixture-pinning that runs
/// after the wrapper has written the fixture.
async fn blocking_body(
    client: anthropic::Client,
    build: fn(&AnthropicModel) -> rig::completion::CompletionRequest,
    expected: FinishReason,
    sink: ReportedSink,
) {
    let model = client.completion_model(anthropic::completion::CLAUDE_HAIKU_4_5);

    let normalized = model
        .completion(build(&model))
        .await
        .expect("`completion` should succeed");
    let typed = model
        .raw_completion(build(&model))
        .await
        .expect("`raw_completion` should succeed");
    assert!(
        typed.provider_request_id.is_some(),
        "the raw type carries the transport id itself"
    );
    let via_raw: RigCompletionResponse = typed
        .normalize(ANTHROPIC_PROVIDER)
        .expect("the raw response should normalize");

    let first = Reported::from_completion(&normalized);
    let second = Reported::from_completion(&via_raw);
    assert_route_parity(&first, &second, expected);
    *sink.lock().expect("sink") = vec![first, second];
}

/// Streamed twin of [`blocking_body`].
async fn streamed_body(
    client: anthropic::Client,
    build: fn(&AnthropicModel) -> rig::completion::CompletionRequest,
    expected: FinishReason,
    sink: ReportedSink,
) {
    let model = client.completion_model(anthropic::completion::CLAUDE_HAIKU_4_5);

    let normalized = drain_normalized_terminal(
        model
            .stream(build(&model))
            .await
            .expect("`stream` should open"),
    )
    .await;
    let typed = drain_raw_terminal(
        model
            .raw_stream(build(&model))
            .await
            .expect("`raw_stream` should open"),
    )
    .await;
    assert!(
        typed.provider_request_id.is_some(),
        "the raw terminal carries the transport id itself"
    );
    let via_raw = StreamFinal::from((ANTHROPIC_PROVIDER, typed));

    let first = Reported::from_terminal(&normalized);
    let second = Reported::from_terminal(&via_raw);
    assert_route_parity(&first, &second, expected);
    *sink.lock().expect("sink") = vec![first, second];
}

fn finish_blocking_cell(scenario: &str, sink: &ReportedSink, stop_reason: &str) {
    let (ids, output_tokens, stop_reasons) = recorded_blocking_premise(scenario);
    assert_identity_matches_fixture(
        scenario,
        &sink.lock().expect("sink"),
        ids,
        output_tokens,
        &[stop_reason, stop_reason],
        stop_reasons,
    );
}

fn finish_streamed_cell(scenario: &str, sink: &ReportedSink, stop_reason: &str) {
    let (ids, output_tokens, stop_reasons) = recorded_streamed_premise(scenario);
    assert_identity_matches_fixture(
        scenario,
        &sink.lock().expect("sink"),
        ids,
        output_tokens,
        &[stop_reason, stop_reason],
        stop_reasons,
    );
}

#[tokio::test]
async fn text_turn_parity() {
    let sink = ReportedSink::default();
    with_anthropic_cassette("raw_completion_parity_matrix/text_turn_parity", {
        let sink = sink.clone();
        move |client| blocking_body(client, text_request, FinishReason::Stop, sink)
    })
    .await;
    finish_blocking_cell(
        "raw_completion_parity_matrix/text_turn_parity",
        &sink,
        "end_turn",
    );
}

#[tokio::test]
async fn tool_call_turn_parity() {
    let sink = ReportedSink::default();
    with_anthropic_cassette("raw_completion_parity_matrix/tool_call_turn_parity", {
        let sink = sink.clone();
        move |client| blocking_body(client, tool_request, FinishReason::ToolCalls, sink)
    })
    .await;
    finish_blocking_cell(
        "raw_completion_parity_matrix/tool_call_turn_parity",
        &sink,
        "tool_use",
    );
}

#[tokio::test]
async fn streamed_text_turn_parity() {
    let sink = ReportedSink::default();
    with_anthropic_cassette("raw_completion_parity_matrix/streamed_text_turn_parity", {
        let sink = sink.clone();
        move |client| streamed_body(client, text_request, FinishReason::Stop, sink)
    })
    .await;
    finish_streamed_cell(
        "raw_completion_parity_matrix/streamed_text_turn_parity",
        &sink,
        "end_turn",
    );
}

#[tokio::test]
async fn streamed_tool_call_turn_parity() {
    let sink = ReportedSink::default();
    with_anthropic_cassette(
        "raw_completion_parity_matrix/streamed_tool_call_turn_parity",
        {
            let sink = sink.clone();
            move |client| streamed_body(client, tool_request, FinishReason::ToolCalls, sink)
        },
    )
    .await;
    finish_streamed_cell(
        "raw_completion_parity_matrix/streamed_tool_call_turn_parity",
        &sink,
        "tool_use",
    );
}

// Keeps `Tool` in scope for `Adder`'s definition even if a future edit stops
// naming it directly.
#[allow(dead_code)]
fn _tool_trait_in_scope() -> &'static str {
    Adder::NAME
}
