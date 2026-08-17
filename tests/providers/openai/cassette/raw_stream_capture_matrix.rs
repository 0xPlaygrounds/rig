//! Raw provider response capture on OpenAI's streaming seams
//! (`StreamFinal::raw`).
//!
//! # What this pins
//!
//! The streamed twin of `raw_capture_matrix`: every terminal
//! `StreamedAssistantContent::Final` carries `raw` — the route's
//! provider-native terminal record, the `R` of that route's `raw_stream`,
//! serialized at the shared `normalize_stream` seam. There is no switch
//! behind it; a terminal `raw` is `Value::Null` only on a record built by
//! hand, never on one a stream yielded. It round-trips into that terminal type and
//! re-serializes equal, it exposes a terminal-only field the normalized
//! `StreamFinal` does not model, and — because capture is unconditional and
//! must stay an escape hatch — re-normalizing the typed terminal reproduces
//! the `usage`, `finish_reason`, `model` and identity the stream reported.
//!
//! Terminal types: Chat Completions'
//! `openai::completion::streaming::StreamingCompletionResponse` (whose
//! `additional_params` accumulates the unmodeled top-level chunk fields —
//! `service_tier`, `system_fingerprint`), and the Responses API's
//! `openai::responses_api::streaming::StreamingCompletionResponse` (whose
//! `status` and `message_id` come from the terminal `response.completed`
//! event alone).
//!
//! # Matrix
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `chat_stream_raw_round_trips_typed` | chat, streamed | chat terminal type round trip; re-normalized `raw` ≡ terminal | recorded |
//! | 2 | `chat_stream_raw_exposes_service_tier` | chat, terminal-only field | `raw["additional_params"]["service_tier"]` = last chunk | recorded |
//! | 3 | `responses_stream_raw_round_trips_typed` | Responses, streamed | Responses terminal type round trip; re-normalized `raw` ≡ terminal | recorded |
//! | 4 | `responses_stream_raw_exposes_status` | Responses, terminal-only field | `raw["status"]` = `response.completed` status | recorded |
//!
//! Every cell is recorded; none is unit-only. Premise, re-derived from each
//! cell's fixture after the wrapper returns: the recorded stream ends with a
//! terminal frame carrying usage — Chat because the request the provider
//! sends already asks for `stream_options.include_usage`, Responses because
//! `response.completed` carries the whole response object.

use std::future::Future;
use std::pin::Pin;

use futures::StreamExt as _;
use rig::completion::{CompletionModel, CompletionRequest, FinishReason};
use rig::prelude::*;
use rig::providers::openai;
use rig::streaming::{StreamFinal, StreamedAssistantContent};
use serde::Deserialize as _;
use serde_json::Value;

use super::super::support::{assert_matches_recorded_token, sse_json_frames, with_openai_cassette};

const PROVIDER: &str = "openai";
const MODEL: &str = openai::GPT_4_1_NANO;
const PROMPT: &str = "Reply with exactly the single word: pong";

type ChatTerminal = openai::completion::streaming::StreamingCompletionResponse;
type ResponsesTerminal = openai::responses_api::streaming::StreamingCompletionResponse;

fn request(model: &(impl CompletionModel + Clone)) -> CompletionRequest {
    model
        .completion_request(PROMPT)
        .temperature(0.0)
        .max_tokens(16)
        .build()
}

/// Drain one normalized stream and return its terminal record.
async fn drain_to_terminal(
    mut stream: rig::streaming::StreamingCompletionResponse,
    context: &str,
) -> StreamFinal {
    let mut terminal = None;
    while let Some(item) = stream.next().await {
        if let StreamedAssistantContent::Final(final_record) =
            item.unwrap_or_else(|err| panic!("{context}: stream item should succeed: {err}"))
        {
            terminal = Some(final_record);
        }
    }
    terminal.unwrap_or_else(|| panic!("{context}: stream should yield a terminal record"))
}

type Observed = std::sync::Arc<std::sync::Mutex<Option<StreamFinal>>>;

/// A cassette test body: boxed so the cell can build it in a helper while the
/// wrapper call — and its string-literal scenario, which the cassette safety
/// scan reads — stays in the test itself.
type Body = Box<dyn FnOnce(openai::Client) -> Pin<Box<dyn Future<Output = ()>>>>;

/// One stream on the chat route; its terminal record is saved onto `sink`.
fn chat_body(sink: Observed) -> Body {
    Box::new(move |client| {
        Box::pin(async move {
            let model = client.completions_api().completion_model(MODEL);
            let stream = model
                .stream(request(&model))
                .await
                .expect("chat stream should open");
            let terminal = drain_to_terminal(stream, "chat stream").await;
            *sink.lock().expect("observation mutex") = Some(terminal);
        })
    })
}

fn responses_body(sink: Observed) -> Body {
    Box::new(move |client| {
        Box::pin(async move {
            let model = client.completion_model(MODEL);
            let stream = model
                .stream(request(&model))
                .await
                .expect("responses stream should open");
            let terminal = drain_to_terminal(stream, "responses stream").await;
            *sink.lock().expect("observation mutex") = Some(terminal);
        })
    })
}

fn take(observed: &Observed) -> StreamFinal {
    observed
        .lock()
        .expect("observation mutex")
        .take()
        .expect("test body should save its observation")
}

/// Chat premise: the request asked for usage on the stream and the last
/// recorded frame carries it. Returns the frames.
fn chat_frames_with_terminal_usage(scenario: &str, request: &str, body: &str) -> Vec<Value> {
    let request: Value = serde_json::from_str(request).expect("recorded request should be JSON");
    assert_eq!(
        request["stream_options"]["include_usage"], true,
        "{scenario}: the chat stream request asks for terminal usage"
    );
    let frames = sse_json_frames(body);
    let last = frames
        .last()
        .unwrap_or_else(|| panic!("{scenario}: the recorded stream must carry frames"));
    assert!(
        last["usage"].is_object(),
        "{scenario}: the last recorded chat frame must carry usage — without a terminal \
         frame this cell asserts nothing about the terminal record"
    );
    frames
}

/// Responses premise: the last recorded frame is `response.completed` with
/// usage on its response object. Returns that response object.
fn responses_completed_frame(scenario: &str, body: &str) -> Value {
    let frames = sse_json_frames(body);
    let last = frames
        .last()
        .unwrap_or_else(|| panic!("{scenario}: the recorded stream must carry frames"));
    assert_eq!(
        last["type"], "response.completed",
        "{scenario}: the recorded stream must end with response.completed"
    );
    assert!(
        last["response"]["usage"].is_object(),
        "{scenario}: the terminal response object must carry usage"
    );
    last["response"].clone()
}

fn last_chunk_field(frames: &[Value], field: &str) -> Value {
    frames
        .iter()
        .filter_map(|frame| frame.get(field))
        .next_back()
        .cloned()
        .unwrap_or(Value::Null)
}

/// The `raw` a streamed terminal must carry — `Value::Null` is reserved for
/// records built by hand, which a terminal off a live stream never is.
fn captured_raw<'a>(scenario: &str, terminal: &'a StreamFinal) -> &'a Value {
    assert!(
        !terminal.raw.is_null(),
        "{scenario}: a streamed terminal always carries `raw`"
    );
    &terminal.raw
}

/// `raw` and the normalized terminal tell one story: mapping the typed
/// terminal through the route's own `From<(&str, R)> for StreamFinal` — the
/// mapper `normalize_stream` ran — reproduces every normalized field. A text
/// turn emits no tool call, so the reconciliation `normalize_stream` layers
/// on top leaves `finish_reason` untouched and the comparison is exact.
fn assert_raw_renormalizes_to(scenario: &str, terminal: &StreamFinal, renormalized: &StreamFinal) {
    assert_eq!(terminal.usage, renormalized.usage, "{scenario}: usage");
    assert_eq!(
        terminal.finish_reason, renormalized.finish_reason,
        "{scenario}: finish reason"
    );
    assert_eq!(terminal.model, renormalized.model, "{scenario}: model");
    assert_eq!(
        terminal.provider, renormalized.provider,
        "{scenario}: provider"
    );
    assert_eq!(
        terminal.identity(),
        renormalized.identity(),
        "{scenario}: identity (message, response and transport ids)"
    );
}

fn assert_normalized_lacks_key(scenario: &str, terminal: &StreamFinal, key: &str) {
    let normalized = serde_json::to_value(terminal).expect("terminal record serializes");
    assert!(
        normalized.get(key).is_none(),
        "{scenario}: the normalized terminal must not model `{key}` — that is what makes \
         it a terminal-only provider field"
    );
}

// ---------------------------------------------------------------------------
// Chat Completions
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chat_stream_raw_round_trips_typed() {
    const SCENARIO: &str = "raw_stream_capture_matrix/chat_stream_raw_round_trips_typed";
    let observed = Observed::default();
    with_openai_cassette(
        "raw_stream_capture_matrix/chat_stream_raw_round_trips_typed",
        chat_body(observed.clone()),
    )
    .await;
    let terminal = take(&observed);
    let bodies = crate::cassettes::recorded_interaction_bodies(PROVIDER, SCENARIO);
    let frames = chat_frames_with_terminal_usage(SCENARIO, &bodies[0].0, &bodies[0].1);

    let raw = captured_raw(SCENARIO, &terminal);
    let typed = ChatTerminal::deserialize(raw)
        .unwrap_or_else(|err| panic!("{SCENARIO}: raw must be the chat terminal type: {err}"));
    assert_eq!(
        serde_json::to_value(&typed).expect("typed terminal serializes"),
        *raw,
        "{SCENARIO}: the typed round trip must re-serialize to the captured value"
    );
    // The captured value is *this* stream's terminal.
    assert_matches_recorded_token(
        typed.response_id.as_deref(),
        last_chunk_field(&frames, "id").as_str(),
        &format!("{SCENARIO}: terminal response id"),
    );
    assert_eq!(
        typed.model.as_deref(),
        last_chunk_field(&frames, "model").as_str(),
        "{SCENARIO}: terminal model"
    );
    assert_eq!(typed.finish_reason, Some(FinishReason::Stop));
    let usage = last_chunk_field(&frames, "usage");
    assert_eq!(
        Some(typed.usage.prompt_tokens as u64),
        usage["prompt_tokens"].as_u64(),
        "{SCENARIO}: terminal prompt tokens"
    );
    assert_eq!(
        typed.usage.completion_tokens.map(|tokens| tokens as u64),
        usage["completion_tokens"].as_u64(),
        "{SCENARIO}: terminal completion tokens"
    );
    // The transport id is stamped on the terminal record and is the same one
    // the normalized record reports.
    assert_eq!(
        typed.provider_request_id, terminal.provider_request_id,
        "{SCENARIO}: the captured terminal carries the stream's request id"
    );
    // One story: the typed terminal re-normalizes to what the stream yielded.
    let renormalized = StreamFinal::from((PROVIDER, typed));
    assert_raw_renormalizes_to(SCENARIO, &terminal, &renormalized);
}

#[tokio::test]
async fn chat_stream_raw_exposes_service_tier() {
    const SCENARIO: &str = "raw_stream_capture_matrix/chat_stream_raw_exposes_service_tier";
    let observed = Observed::default();
    with_openai_cassette(
        "raw_stream_capture_matrix/chat_stream_raw_exposes_service_tier",
        chat_body(observed.clone()),
    )
    .await;
    let terminal = take(&observed);
    let bodies = crate::cassettes::recorded_interaction_bodies(PROVIDER, SCENARIO);
    let frames = chat_frames_with_terminal_usage(SCENARIO, &bodies[0].0, &bodies[0].1);
    let recorded_tier = last_chunk_field(&frames, "service_tier");
    assert!(
        recorded_tier.is_string(),
        "{SCENARIO}: the recorded chunks must report a `service_tier`"
    );

    let raw = captured_raw(SCENARIO, &terminal);
    assert_eq!(
        raw["additional_params"]["service_tier"], recorded_tier,
        "{SCENARIO}: `service_tier` is readable off the captured terminal and equals the \
         last chunk's"
    );
    assert_matches_recorded_token(
        raw["additional_params"]["system_fingerprint"].as_str(),
        last_chunk_field(&frames, "system_fingerprint").as_str(),
        &format!("{SCENARIO}: `system_fingerprint` off the captured terminal vs the fixture"),
    );
    assert_normalized_lacks_key(SCENARIO, &terminal, "additional_params");
    assert_normalized_lacks_key(SCENARIO, &terminal, "service_tier");
}

// ---------------------------------------------------------------------------
// Responses
// ---------------------------------------------------------------------------

#[tokio::test]
async fn responses_stream_raw_round_trips_typed() {
    const SCENARIO: &str = "raw_stream_capture_matrix/responses_stream_raw_round_trips_typed";
    let observed = Observed::default();
    with_openai_cassette(
        "raw_stream_capture_matrix/responses_stream_raw_round_trips_typed",
        responses_body(observed.clone()),
    )
    .await;
    let terminal = take(&observed);
    let bodies = crate::cassettes::recorded_interaction_bodies(PROVIDER, SCENARIO);
    let completed = responses_completed_frame(SCENARIO, &bodies[0].1);

    let raw = captured_raw(SCENARIO, &terminal);
    let typed = ResponsesTerminal::deserialize(raw)
        .unwrap_or_else(|err| panic!("{SCENARIO}: raw must be the Responses terminal type: {err}"));
    assert_eq!(
        serde_json::to_value(&typed).expect("typed terminal serializes"),
        *raw,
        "{SCENARIO}: the typed round trip must re-serialize to the captured value"
    );
    assert_matches_recorded_token(
        typed.response_id.as_deref(),
        completed["id"].as_str(),
        &format!("{SCENARIO}: terminal response id"),
    );
    assert_eq!(
        typed.model.as_deref(),
        completed["model"].as_str(),
        "{SCENARIO}: terminal model"
    );
    assert_eq!(
        Some(typed.usage.input_tokens),
        completed["usage"]["input_tokens"].as_u64(),
        "{SCENARIO}: terminal input tokens"
    );
    assert_eq!(
        Some(typed.usage.output_tokens),
        completed["usage"]["output_tokens"].as_u64(),
        "{SCENARIO}: terminal output tokens"
    );
    assert_eq!(
        typed.provider_request_id, terminal.provider_request_id,
        "{SCENARIO}: the captured terminal carries the stream's request id"
    );
    // One story: the typed terminal re-normalizes to what the stream yielded.
    let renormalized = StreamFinal::from((PROVIDER, typed));
    assert_raw_renormalizes_to(SCENARIO, &terminal, &renormalized);
}

#[tokio::test]
async fn responses_stream_raw_exposes_status() {
    const SCENARIO: &str = "raw_stream_capture_matrix/responses_stream_raw_exposes_status";
    let observed = Observed::default();
    with_openai_cassette(
        "raw_stream_capture_matrix/responses_stream_raw_exposes_status",
        responses_body(observed.clone()),
    )
    .await;
    let terminal = take(&observed);
    let bodies = crate::cassettes::recorded_interaction_bodies(PROVIDER, SCENARIO);
    let completed = responses_completed_frame(SCENARIO, &bodies[0].1);
    let recorded_status = completed["status"].as_str().unwrap_or_else(|| {
        panic!("{SCENARIO}: the terminal response object must carry a string `status`")
    });
    let recorded_message_id = completed["output"]
        .as_array()
        .and_then(|items| items.iter().find(|item| item["type"] == "message"))
        .and_then(|item| item["id"].as_str())
        .unwrap_or_else(|| {
            panic!("{SCENARIO}: the terminal response object must carry a message output item")
        });

    let raw = captured_raw(SCENARIO, &terminal);
    assert_eq!(
        raw["status"].as_str(),
        Some(recorded_status),
        "{SCENARIO}: `status` is readable off the captured terminal and equals the \
         response.completed status"
    );
    assert_matches_recorded_token(
        raw["message_id"].as_str(),
        Some(recorded_message_id),
        &format!("{SCENARIO}: `message_id` off the captured terminal vs the fixture"),
    );
    // `message_id` *is* normalized (`StreamFinal::message_id`); the captured
    // terminal and the normalized record must agree on it.
    assert_eq!(
        raw["message_id"].as_str(),
        terminal.message_id.as_deref(),
        "{SCENARIO}: captured and normalized message ids agree"
    );
    assert_normalized_lacks_key(SCENARIO, &terminal, "status");
}
