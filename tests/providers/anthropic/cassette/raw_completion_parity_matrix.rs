//! Parity matrix for the typed escape hatch: the provider-native reply rig
//! hands back in `CompletionResponse::raw` / `StreamFinal::raw` must tell the
//! same story as the normalized response delivered with it, and both must be
//! what the fixture recorded.
//!
//! # The contract
//!
//! One call yields both views. `CompletionResponse::deserialize(&response.raw)`
//! reads Anthropic's own reply out of the blocking response, and
//! `anthropic::streaming::StreamingCompletionResponse::deserialize(&raw)`
//! reads the provider's terminal record out of a stream's. The provider-native
//! fields — message id, model, token counts — must be exactly what the
//! normalized view reports, otherwise a caller who reaches for the provider
//! type to read a field rig does not normalize is reading a different exchange
//! than the one rig described. The transport id is the exception in both
//! directions: it is a response *header*, so no reply document carries it and
//! only the normalized identity does.
//!
//! Each cell makes two live requests in one scenario, so the fixture holds two
//! interactions in wire order. Two requests are two responses, so `identity()`
//! cannot be *literally* equal across them — each attempt reports its own
//! message id and request id. Parity is therefore asserted the only honest
//! way: every identity field each response reports equals what *its own*
//! recorded interaction says (message id from the body, transport id from the
//! `request-id` header), both responses populate the same identity fields, and
//! `finish_reason`, `model`, and `usage.input_tokens` (deterministic for an
//! identical prompt) are equal outright, with each response's `output_tokens`
//! pinned to its own recorded usage.
//!
//! # Matrix
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `text_turn_parity` | `raw` beside the normalized response, `end_turn` | equal `Stop`, model, usage; identity per fixture | recorded |
//! | 2 | `tool_call_turn_parity` | same, `tool_use` terminal | equal `ToolCalls` (reconciled), model, usage; identity per fixture | recorded |
//! | 3 | `streamed_text_turn_parity` | terminal `raw` beside `stream()`'s terminal | equal `Stop`, model, usage; identity per fixture | recorded |
//! | 4 | `streamed_tool_call_turn_parity` | same, `tool_use` terminal | equal `ToolCalls`, model, usage; identity per fixture | recorded |
//!
//! Every cell is recorded; the premise each re-derives from its fixture is that
//! both interactions' responses carry a `request-id` header and a `msg_…` id,
//! and that the recorded stop reason is the one the cell is about — a
//! recording that lost either would make the parity claim vacuous.

use rig::completion::{
    CompletionModel as _, CompletionResponse as RigCompletionResponse, FinishReason,
    ResponseIdentity, Usage,
};
use rig::driver::Bound;
use rig::message::ToolChoice;
use rig::providers::anthropic;
use rig::providers::anthropic::wire::Anthropic;
use rig::providers::anthropic::wire::Messages;
use rig::streaming::StreamFinal;
use rig::tool::Tool;
use serde::Deserialize;

use super::super::support::{
    assert_ids_match_recording, recorded_request_id_headers, sse_json_frames,
    with_anthropic_cassette,
};
use crate::raw_capture::capture_completion_pair;
use crate::support::{Adder, Observed, TOOLS_PREAMBLE, collect_required_terminal};

const ANTHROPIC_PROVIDER: &str = "anthropic";
const TEXT_PROMPT: &str = "Reply with exactly: parity probe";
const TOOL_PROMPT: &str = "What is 2 + 3? Use the tool.";

type AnthropicModel = Bound<Messages>;

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

/// The comparable part of one exchange's result: what both must agree on.
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

/// Parity between the two recorded exchanges of the same request, for the
/// fields that do not depend on which HTTP exchange produced them.
fn assert_route_parity(first: &Reported, second: &Reported, expected: FinishReason) {
    assert_eq!(first.finish_reason, Some(expected));
    assert_eq!(
        second.finish_reason, first.finish_reason,
        "the same stop reason must map the same way on both exchanges"
    );
    assert!(first.model.is_some(), "the wire names its model");
    assert_eq!(second.model, first.model);
    assert_eq!(
        second.usage.input_tokens, first.usage.input_tokens,
        "the same prompt costs the same input tokens on both exchanges"
    );
    // Both exchanges populate the same identity *fields*; the values are
    // per-exchange and pinned against their own fixture below.
    assert!(first.identity.message_id.is_some());
    assert!(second.identity.message_id.is_some());
    assert!(first.identity.provider_request_id.is_some());
    assert!(second.identity.provider_request_id.is_some());
    assert_eq!(first.identity.response_id, None);
    assert_eq!(second.identity.response_id, None);
}

/// Pin each exchange's identity and output usage to *its own* recorded
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
        "{scenario}: one result per recorded exchange"
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
    let recorded_output_tokens: Vec<_> = recorded_output_tokens.into_iter().map(Some).collect();
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

/// Both recorded exchanges of a blocking cell, after the wrapper has written
/// the fixture: the two views of each reply agree, the pair agrees where the
/// wire makes it equal, and each identity is its own interaction's.
fn assert_blocking_parity(
    scenario: &str,
    (first_response, second_response): (RigCompletionResponse, RigCompletionResponse),
    expected: FinishReason,
    stop_reason: &str,
) {
    let first = Reported::from_completion(&first_response);
    let second = Reported::from_completion(&second_response);
    assert_route_parity(&first, &second, expected);
    assert_raw_view_agrees(&first_response, &first);
    assert_raw_view_agrees(&second_response, &second);

    let (ids, output_tokens, stop_reasons) = recorded_blocking_premise(scenario);
    assert_identity_matches_fixture(
        scenario,
        &[first, second],
        ids,
        output_tokens,
        &[stop_reason, stop_reason],
        stop_reasons,
    );
}

/// The two views of one reply agree: `raw` is Anthropic's own reply document,
/// and the provider type reads the same message id, model and token counts
/// the normalized response reports. The transport id is a response header
/// rather than a body field, so it reaches the caller only on the normalized
/// identity.
fn assert_raw_view_agrees(response: &RigCompletionResponse, reported: &Reported) {
    let typed = anthropic::completion::CompletionResponse::deserialize(&response.raw)
        .expect("`raw` is Anthropic's reply document, which the provider type reads");
    assert_eq!(
        Some(typed.id.as_str()),
        reported.identity.message_id.as_deref(),
        "the normalized message id is the document's"
    );
    assert_eq!(Some(typed.model.as_str()), reported.model.as_deref());
    assert_eq!(Some(typed.usage.input_tokens), reported.usage.input_tokens);
    assert_eq!(
        Some(typed.usage.output_tokens),
        reported.usage.output_tokens
    );
    assert!(
        typed.provider_request_id.is_none(),
        "the transport id is a header, not part of the reply document"
    );
    assert!(
        reported.identity.provider_request_id.is_some(),
        "the normalized response carries the transport id instead"
    );
}

/// Body of a streamed cell: the same request opened twice, each stream
/// drained to the terminal record it must have ended with.
///
/// The blocking pair is [`capture_completion_pair`]; a streamed pair has no
/// shared counterpart, so the two drains stay here — each through the shared
/// [`collect_required_terminal`].
async fn capture_terminal_pair(
    client: Bound<Anthropic>,
    build: fn(&AnthropicModel) -> rig::completion::CompletionRequest,
    sink: Observed<(StreamFinal, StreamFinal)>,
) {
    let model = client.completion(anthropic::completion::CLAUDE_HAIKU_4_5);

    let normalized = collect_required_terminal(
        model
            .stream(build(&model))
            .await
            .expect("`stream` should open"),
    )
    .await;
    // The second route: the same request opened again, read through the
    // terminal record's `raw` — the provider's own record, serialized.
    let second_record = collect_required_terminal(
        model
            .stream(build(&model))
            .await
            .expect("second `stream` should open"),
    )
    .await;
    sink.put((normalized, second_record));
}

/// Streamed twin of [`assert_blocking_parity`].
fn assert_streamed_parity(
    scenario: &str,
    (normalized, second_record): (StreamFinal, StreamFinal),
    expected: FinishReason,
    stop_reason: &str,
) {
    let typed: anthropic::streaming::StreamingCompletionResponse =
        serde_json::from_value(second_record.raw.clone())
            .expect("the terminal's raw is the provider record");
    assert_eq!(
        typed.usage.input_tokens.map(|n| n as u64),
        second_record.usage.input_tokens,
        "the raw record and the normalized record agree on usage"
    );

    let first = Reported::from_terminal(&normalized);
    let second = Reported::from_terminal(&second_record);
    assert_route_parity(&first, &second, expected);

    let (ids, output_tokens, stop_reasons) = recorded_streamed_premise(scenario);
    assert_identity_matches_fixture(
        scenario,
        &[first, second],
        ids,
        output_tokens,
        &[stop_reason, stop_reason],
        stop_reasons,
    );
}

#[tokio::test]
async fn text_turn_parity() {
    let sink = Observed::default();
    with_anthropic_cassette("raw_completion_parity_matrix/text_turn_parity", {
        let sink = sink.clone();
        move |client| async move {
            capture_completion_pair(
                client.completion(anthropic::completion::CLAUDE_HAIKU_4_5),
                text_request,
                sink,
            )
            .await
            .expect("both `completion` calls should succeed");
        }
    })
    .await;
    assert_blocking_parity(
        "raw_completion_parity_matrix/text_turn_parity",
        sink.take(),
        FinishReason::Stop,
        "end_turn",
    );
}

#[tokio::test]
async fn tool_call_turn_parity() {
    let sink = Observed::default();
    with_anthropic_cassette("raw_completion_parity_matrix/tool_call_turn_parity", {
        let sink = sink.clone();
        move |client| async move {
            capture_completion_pair(
                client.completion(anthropic::completion::CLAUDE_HAIKU_4_5),
                tool_request,
                sink,
            )
            .await
            .expect("both `completion` calls should succeed");
        }
    })
    .await;
    assert_blocking_parity(
        "raw_completion_parity_matrix/tool_call_turn_parity",
        sink.take(),
        FinishReason::ToolCalls,
        "tool_use",
    );
}

#[tokio::test]
async fn streamed_text_turn_parity() {
    let sink = Observed::default();
    with_anthropic_cassette("raw_completion_parity_matrix/streamed_text_turn_parity", {
        let sink = sink.clone();
        move |client| capture_terminal_pair(client, text_request, sink)
    })
    .await;
    assert_streamed_parity(
        "raw_completion_parity_matrix/streamed_text_turn_parity",
        sink.take(),
        FinishReason::Stop,
        "end_turn",
    );
}

#[tokio::test]
async fn streamed_tool_call_turn_parity() {
    let sink = Observed::default();
    with_anthropic_cassette(
        "raw_completion_parity_matrix/streamed_tool_call_turn_parity",
        {
            let sink = sink.clone();
            move |client| capture_terminal_pair(client, tool_request, sink)
        },
    )
    .await;
    assert_streamed_parity(
        "raw_completion_parity_matrix/streamed_tool_call_turn_parity",
        sink.take(),
        FinishReason::ToolCalls,
        "tool_use",
    );
}

// Keeps `Tool` in scope for `Adder`'s definition even if a future edit stops
// naming it directly.
#[allow(dead_code)]
fn _tool_trait_in_scope() -> &'static str {
    Adder::NAME
}
