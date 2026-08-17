//! Matrix for opt-in raw provider response capture on the streaming path:
//! `CompletionRequest::capture_raw_response` → `StreamFinal::raw`.
//!
//! # The feature
//!
//! `stream()` reads the flag before `raw_stream` consumes the request and
//! passes it to `normalize_stream`, which serializes the provider-native
//! terminal — `anthropic::streaming::StreamingCompletionResponse`, the `R` of
//! `raw_stream` — onto the terminal `StreamFinal::raw` before mapping it. So
//! `raw` is the **terminal record only**: what `raw_stream` would have yielded
//! as its `FinalResponse`, not the stream's frames. Anthropic's terminal is
//! assembled from `message_start` (id, model) and the closing `message_delta`
//! (`stop_reason`, `stop_sequence`, usage), plus the transport `request-id`
//! header the driver stamps.
//!
//! # Matrix
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `flag_off_terminal_raw_is_none` | default request | terminal `raw.is_none()` | recorded |
//! | 2 | `flag_on_terminal_raw_round_trips` | `capture_raw_response(true)` | `raw` deserializes into the Anthropic terminal type and re-serializes equal; terminal-only shape (no frames) | recorded |
//! | 3 | `flag_on_exposes_stop_sequence` | streamed twin of the `stop_sequences: ["alpha"]` request | `raw["stop_sequence"] == "alpha"`, `raw["stop_reason"] == "stop_sequence"` (verbatim spelling) | recorded |
//! | 4 | `request_invariant_off_vs_on` | recorded request bodies of 1 and 2 | byte-identical | recorded (derived from cells 1 and 2) |
//!
//! Cell 4 makes no request of its own: it reads the fixtures of cells 1 and 2,
//! which send the same prompt for exactly that comparison. Every recorded cell
//! re-derives its premise from its own SSE frames: the stream opens with a
//! `message_start` naming a `msg_…` id, closes with a `message_delta` carrying
//! `usage`, and the response carries a `request-id` header.

use futures::StreamExt;
use rig::completion::{CompletionModel as _, FinishReason};
use rig::prelude::*;
use rig::providers::anthropic;
use rig::providers::anthropic::streaming::StreamingCompletionResponse;
use rig::streaming::{StreamFinal, StreamedAssistantContent};
use serde::Deserialize;
use serde_json::{Value, json};

use super::super::support::{
    assert_ids_match_recording, recorded_request_id_headers, with_anthropic_cassette,
};

const ANTHROPIC_PROVIDER: &str = "anthropic";
const PROMPT: &str = "Reply with exactly: raw stream capture probe";
/// From `empty_stop_sequence_matrix.rs`: one word, so the `alpha` sequence
/// matches and Anthropic names it on the terminal `message_delta`.
const IMMEDIATE_PROMPT: &str = "Reply with exactly this one word and nothing else: alpha";

const OFF_SCENARIO: &str = "raw_stream_capture_matrix/flag_off_terminal_raw_is_none";
const ON_SCENARIO: &str = "raw_stream_capture_matrix/flag_on_terminal_raw_round_trips";
const STOP_SEQUENCE_SCENARIO: &str = "raw_stream_capture_matrix/flag_on_exposes_stop_sequence";

type AnthropicModel = anthropic::completion::CompletionModel;

fn probe_request(model: &AnthropicModel, capture: bool) -> rig::completion::CompletionRequest {
    model
        .completion_request(PROMPT)
        .max_tokens(32)
        .capture_raw_response(capture)
        .build()
}

type TerminalSink = std::sync::Arc<std::sync::Mutex<Option<StreamFinal>>>;

async fn drain_terminal(mut stream: rig::streaming::StreamingCompletionResponse) -> StreamFinal {
    let mut terminal = None;
    while let Some(item) = stream.next().await {
        if let StreamedAssistantContent::Final(final_record) =
            item.expect("stream item should succeed")
        {
            terminal = Some(final_record);
        }
    }
    terminal.expect("the stream should yield a terminal record")
}

/// The body of every recorded cell: open the stream the cell's request
/// describes and keep its terminal record for the assertions that run after
/// the wrapper has written the fixture.
async fn probe_body(
    client: anthropic::Client,
    build: impl FnOnce(&AnthropicModel) -> rig::completion::CompletionRequest,
    sink: TerminalSink,
) {
    let model = client.completion_model(anthropic::completion::CLAUDE_HAIKU_4_5);
    let stream = model
        .stream(build(&model))
        .await
        .expect("stream should open");
    *sink.lock().expect("sink") = Some(drain_terminal(stream).await);
}

fn take_terminal(sink: &TerminalSink) -> StreamFinal {
    let terminal = sink.lock().expect("sink").take();
    terminal.expect("the cell body ran")
}

/// The premise every cell rests on, read from its own frames: `message_start`
/// names the id and model, the terminal `message_delta` names the stop reason
/// and carries the final usage, and the response carried a `request-id`.
struct RecordedStream {
    message_id: Option<String>,
    model: Option<String>,
    stop_reason: Option<String>,
    stop_sequence: Option<String>,
    output_tokens: u64,
    input_tokens: u64,
    request_id: Option<String>,
    frame_types: Vec<String>,
}

fn recorded_stream(scenario: &str) -> RecordedStream {
    let frames = crate::cassettes::recorded_sse_json_frames(ANTHROPIC_PROVIDER, scenario);
    let start = frames
        .iter()
        .find(|frame| frame["type"] == "message_start")
        .unwrap_or_else(|| panic!("{scenario}: premise — the stream opens with message_start"));
    let delta = frames
        .iter()
        .find(|frame| frame["type"] == "message_delta")
        .unwrap_or_else(|| panic!("{scenario}: premise — the stream closes with message_delta"));
    let request_ids = recorded_request_id_headers(scenario);
    assert_eq!(request_ids.len(), 1, "{scenario}: one recorded interaction");
    RecordedStream {
        message_id: start["message"]["id"].as_str().map(str::to_string),
        model: start["message"]["model"].as_str().map(str::to_string),
        stop_reason: delta["delta"]["stop_reason"].as_str().map(str::to_string),
        stop_sequence: delta["delta"]["stop_sequence"].as_str().map(str::to_string),
        output_tokens: delta["usage"]["output_tokens"]
            .as_u64()
            .unwrap_or_else(|| panic!("{scenario}: premise — terminal frame carries usage")),
        input_tokens: delta["usage"]["input_tokens"]
            .as_u64()
            .unwrap_or_else(|| panic!("{scenario}: premise — terminal frame carries usage")),
        request_id: request_ids[0].clone(),
        frame_types: frames
            .iter()
            .filter_map(|frame| frame["type"].as_str().map(str::to_string))
            .collect(),
    }
}

/// The normalized terminal reports what its own recording says, capture or
/// not.
fn assert_terminal_matches_fixture(
    scenario: &str,
    terminal: &StreamFinal,
    recorded: &RecordedStream,
) {
    assert!(
        recorded
            .message_id
            .as_deref()
            .is_some_and(|id| id.starts_with("msg_")),
        "{scenario}: premise — message_start names a msg_ id"
    );
    assert!(
        recorded.request_id.is_some(),
        "{scenario}: premise — the response carries a request-id header"
    );
    assert_ids_match_recording(
        std::slice::from_ref(&terminal.message_id),
        std::slice::from_ref(&recorded.message_id),
        scenario,
    );
    assert_ids_match_recording(
        std::slice::from_ref(&terminal.provider_request_id),
        std::slice::from_ref(&recorded.request_id),
        scenario,
    );
    assert_eq!(terminal.model, recorded.model);
    assert_eq!(terminal.usage.input_tokens, recorded.input_tokens);
    assert_eq!(terminal.usage.output_tokens, recorded.output_tokens);
    assert_eq!(terminal.provider, ANTHROPIC_PROVIDER);
}

// ---------------------------------------------------------------------------
// 1: default off
// ---------------------------------------------------------------------------

#[tokio::test]
async fn flag_off_terminal_raw_is_none() {
    let sink = TerminalSink::default();
    with_anthropic_cassette("raw_stream_capture_matrix/flag_off_terminal_raw_is_none", {
        let sink = sink.clone();
        move |client| probe_body(client, |model| probe_request(model, false), sink)
    })
    .await;
    let terminal = take_terminal(&sink);
    assert!(
        terminal.raw.is_none(),
        "capture is opt-in: a stream that did not ask must not pay for it"
    );
    let recorded = recorded_stream(OFF_SCENARIO);
    assert_eq!(recorded.stop_reason.as_deref(), Some("end_turn"));
    assert_eq!(terminal.finish_reason, Some(FinishReason::Stop));
    assert_terminal_matches_fixture(OFF_SCENARIO, &terminal, &recorded);
}

// ---------------------------------------------------------------------------
// 2: on → typed round trip, terminal-only shape
// ---------------------------------------------------------------------------

#[tokio::test]
async fn flag_on_terminal_raw_round_trips() {
    let sink = TerminalSink::default();
    with_anthropic_cassette(
        "raw_stream_capture_matrix/flag_on_terminal_raw_round_trips",
        {
            let sink = sink.clone();
            move |client| probe_body(client, |model| probe_request(model, true), sink)
        },
    )
    .await;
    let terminal = take_terminal(&sink);
    let raw: &Value = terminal
        .raw
        .as_deref()
        .expect("the request opted in, so the terminal `raw` must be populated");

    // Typed access is recoverable and lossless.
    let typed = StreamingCompletionResponse::deserialize(raw)
        .expect("`raw` is the serialized anthropic::streaming::StreamingCompletionResponse");
    assert_eq!(
        serde_json::to_value(&typed).expect("re-serialize"),
        *raw,
        "raw ↔ typed must round-trip without loss"
    );

    // Terminal record only. Pin the exact key set: the fields the Anthropic
    // terminal type carries for an `end_turn` text stream (no
    // `stop_sequence`, which is skipped when absent) — and nothing frame-shaped.
    let mut keys: Vec<&str> = raw
        .as_object()
        .expect("terminal raw is an object")
        .keys()
        .map(String::as_str)
        .collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        [
            "message_id",
            "model",
            "provider_request_id",
            "stop_reason",
            "usage"
        ],
        "the terminal record's own fields, and only those"
    );
    for frame_key in [
        "content_block_delta",
        "content_block_start",
        "content",
        "delta",
        "type",
    ] {
        assert!(
            raw.get(frame_key).is_none(),
            "`raw` is the terminal record, not the frames: found `{frame_key}`"
        );
    }

    // Wire-derived fields equal what the recorded frames say; the transport
    // id is the header the driver stamped.
    let recorded = recorded_stream(ON_SCENARIO);
    assert!(
        recorded
            .frame_types
            .iter()
            .any(|kind| kind == "content_block_delta"),
        "premise: the recorded stream did carry frames `raw` must not contain"
    );
    assert_ids_match_recording(
        &[raw["message_id"].as_str().map(str::to_string)],
        std::slice::from_ref(&recorded.message_id),
        ON_SCENARIO,
    );
    assert_ids_match_recording(
        &[raw["provider_request_id"].as_str().map(str::to_string)],
        std::slice::from_ref(&recorded.request_id),
        ON_SCENARIO,
    );
    assert_eq!(raw["model"].as_str(), recorded.model.as_deref());
    assert_eq!(raw["stop_reason"].as_str(), recorded.stop_reason.as_deref());
    assert_eq!(recorded.stop_reason.as_deref(), Some("end_turn"));
    assert_eq!(raw["usage"]["output_tokens"], json!(recorded.output_tokens));
    assert_eq!(raw["usage"]["input_tokens"], json!(recorded.input_tokens));
    assert_eq!(typed.stop_reason.as_deref(), Some("end_turn"));
    assert_eq!(typed.usage.output_tokens as u64, recorded.output_tokens);

    // The normalized view beside it is unchanged by capture.
    assert_eq!(terminal.finish_reason, Some(FinishReason::Stop));
    assert_terminal_matches_fixture(ON_SCENARIO, &terminal, &recorded);
}

// ---------------------------------------------------------------------------
// 3: a terminal-only field, verbatim
// ---------------------------------------------------------------------------

#[tokio::test]
async fn flag_on_exposes_stop_sequence() {
    let sink = TerminalSink::default();
    with_anthropic_cassette("raw_stream_capture_matrix/flag_on_exposes_stop_sequence", {
        let sink = sink.clone();
        move |client| {
            probe_body(
                client,
                |model| {
                    model
                        .completion_request(IMMEDIATE_PROMPT)
                        .max_tokens(32)
                        .additional_params(json!({ "stop_sequences": ["alpha"] }))
                        .capture_raw_response(true)
                        .build()
                },
                sink,
            )
        }
    })
    .await;
    let terminal = take_terminal(&sink);
    let raw: &Value = terminal.raw.as_deref().expect("the request opted in");

    // Premise from the frames: the terminal `message_delta` stopped on the
    // sequence and named it.
    let recorded = recorded_stream(STOP_SEQUENCE_SCENARIO);
    assert_eq!(
        recorded.stop_reason.as_deref(),
        Some("stop_sequence"),
        "premise: the recorded stream stopped on a sequence"
    );
    assert_eq!(
        recorded.stop_sequence.as_deref(),
        Some("alpha"),
        "premise: the recorded terminal names the sequence"
    );

    // Normalized: folded into `Stop`; the provider's spelling and the
    // sequence itself are only on `raw`.
    assert_eq!(terminal.finish_reason, Some(FinishReason::Stop));
    let normalized = serde_json::to_value(&terminal).expect("terminal serializes");
    let normalized_keys: Vec<&str> = normalized
        .as_object()
        .expect("object")
        .keys()
        .map(String::as_str)
        .filter(|key| *key != "raw")
        .collect();
    assert!(
        !normalized_keys.contains(&"stop_sequence") && !normalized_keys.contains(&"stop_reason"),
        "the normalized terminal has neither `stop_sequence` nor a verbatim `stop_reason` \
         ({normalized_keys:?}) — `raw` is the only way to read them"
    );
    assert_eq!(raw["stop_reason"], "stop_sequence");
    assert_eq!(raw["stop_sequence"], "alpha");
    let typed = StreamingCompletionResponse::deserialize(raw).expect("typed access");
    assert_eq!(typed.stop_sequence.as_deref(), Some("alpha"));
    assert_eq!(typed.stop_reason.as_deref(), Some("stop_sequence"));
    assert_terminal_matches_fixture(STOP_SEQUENCE_SCENARIO, &terminal, &recorded);
}

// ---------------------------------------------------------------------------
// 4: request invariant
// ---------------------------------------------------------------------------

/// The flag never reaches the provider: the streamed request bodies the off
/// and on cells recorded are byte-identical.
#[test]
fn request_invariant_off_vs_on() {
    let off = crate::cassettes::recorded_interaction_bodies(ANTHROPIC_PROVIDER, OFF_SCENARIO);
    let on = crate::cassettes::recorded_interaction_bodies(ANTHROPIC_PROVIDER, ON_SCENARIO);
    assert_eq!(off.len(), 1, "{OFF_SCENARIO}: one recorded interaction");
    assert_eq!(on.len(), 1, "{ON_SCENARIO}: one recorded interaction");
    assert_eq!(
        off[0].0, on[0].0,
        "the recorded request bodies must be byte-identical: capture is local policy"
    );
    let request = crate::cassettes::recorded_json_request(ANTHROPIC_PROVIDER, ON_SCENARIO);
    assert_eq!(
        request["stream"], true,
        "premise: these are the streamed twins"
    );
    for (body, _) in off.iter().chain(on.iter()) {
        assert!(
            !body.contains("capture_raw") && !body.contains("captureRaw"),
            "the flag is `#[serde(skip)]`; it must never serialize: {body}"
        );
    }
}
