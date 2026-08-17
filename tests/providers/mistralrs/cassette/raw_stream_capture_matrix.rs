//! Matrix for opt-in raw terminal-record capture on mistral.rs's streaming
//! `/v1/chat/completions` route (`CompletionRequest::capture_raw_response` →
//! `StreamFinal::raw`).
//!
//! # The feature
//!
//! mistral.rs streams through rig's OpenAI chat-completions client, whose
//! [`raw_stream`](rig::providers::openai::GenericCompletionModel::raw_stream)
//! yields [`openai::StreamingCompletionResponse`] as its terminal record: the
//! usage from the stream's final `data:` frame plus the envelope fields the
//! chunks carried (`object`, `created`, `system_fingerprint`) accumulated under
//! `additional_params`. `StreamFinal::raw` is that record serialized — the
//! terminal record only — populated only when the request opted in, and never
//! on the wire.
//!
//! # Matrix
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `stream_capture_off_raw_is_none` | flag off (default) | terminal `raw == None` | unrecorded (no mistral.rs server in this environment) |
//! | 2 | `stream_capture_on_terminal_round_trips_provider_type` | flag on | `openai::StreamingCompletionResponse::deserialize(&*raw)` re-serializes equal | unrecorded (no mistral.rs server in this environment) |
//! | 3 | `stream_capture_on_exposes_envelope_fields` | terminal-only fields | `additional_params.system_fingerprint`/`object` in `raw` equal the recorded chunks; usage equals the terminal frame | unrecorded (no mistral.rs server in this environment) |
//! | 4 | `stream_request_invariant_off_vs_on` | on-wire request | flag-off and flag-on request bodies byte-identical | unrecorded (no mistral.rs server in this environment) |
//!
//! Every cell is unrecorded: no mistral.rs server was listening on
//! `127.0.0.1:1234` when this matrix was written, and a fixture is never
//! fabricated. To record: start `mistralrs-server` on that port serving
//! `Qwen/Qwen3-4B` (or export `MISTRALRS_BASE_URL`/`MISTRALRS_MODEL`), remove
//! the `#[ignore]` attributes, flip the table to `recorded`, then run
//! `RIG_PROVIDER_TEST_MODE=record cargo test -p rig --all-features --test mistralrs mistralrs::cassette::raw_stream_capture_matrix -- --nocapture --test-threads=1`
//! and review `tests/cassettes/mistralrs/raw_stream_capture_matrix/`.

use futures::StreamExt;
use rig::completion::CompletionModel as _;
use rig::prelude::*;
use rig::providers::openai;
use rig::streaming::{StreamFinal, StreamedAssistantContent};
use serde::Deserialize;
use serde_json::Value;

use super::super::support::{model_name, with_mistralrs_completions_cassette};
use crate::cassettes::{CassetteMode, recorded_interaction_bodies, recorded_sse_json_frames};

const MISTRALRS_PROVIDER: &str = "mistralrs";
const NORMALIZED_PROVIDER: &str = "openai";
const PROMPT: &str = "/no_think Reply with exactly the single word: pong";

fn request(
    model: &openai::CompletionModel,
    capture_raw: bool,
) -> rig::completion::CompletionRequest {
    model
        .completion_request(PROMPT)
        .max_tokens(64)
        .capture_raw_response(capture_raw)
        .build()
}

async fn terminal_of(mut stream: rig::streaming::StreamingCompletionResponse) -> StreamFinal {
    let mut finals = Vec::new();
    while let Some(item) = stream.next().await {
        if let StreamedAssistantContent::Final(record) = item.expect("stream item should be ok") {
            finals.push(record);
        }
    }
    assert_eq!(
        finals.len(),
        1,
        "stream should yield exactly one terminal record"
    );
    finals.remove(0)
}

/// The premise every streaming cell rests on: the recorded SSE stream's last
/// JSON frame carries `usage`. Returns `(all frames, terminal frame)`.
fn recorded_frames_with_terminal(scenario: &str) -> (Vec<Value>, Value) {
    let frames = recorded_sse_json_frames(MISTRALRS_PROVIDER, scenario);
    let terminal = frames
        .last()
        .cloned()
        .unwrap_or_else(|| panic!("{scenario}: the recorded stream should carry frames"));
    assert!(
        terminal.get("usage").is_some_and(Value::is_object),
        "{scenario}: the recorded stream must end with a usage-bearing frame — \
         without it the terminal record carries no usage and this cell proves nothing"
    );
    (frames, terminal)
}

/// Every recorded chunk stamps the same envelope value for `key`; returns it.
fn recorded_envelope_field(frames: &[Value], key: &str, scenario: &str) -> Value {
    let mut values = frames.iter().filter_map(|frame| frame.get(key)).cloned();
    let first = values
        .next()
        .unwrap_or_else(|| panic!("{scenario}: recorded chunks must carry `{key}`"));
    assert!(
        values.all(|value| value == first),
        "{scenario}: every recorded chunk must agree on `{key}`"
    );
    first
}

fn recorded_request_bodies(scenario: &str) -> Vec<String> {
    recorded_interaction_bodies(MISTRALRS_PROVIDER, scenario)
        .into_iter()
        .map(|(request, _)| request)
        .collect()
}

// ---------------------------------------------------------------------------
// 1: off → None
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "unrecorded (no mistral.rs server in this environment)"]
async fn stream_capture_off_raw_is_none() {
    let scenario = "raw_stream_capture_matrix/stream_capture_off_raw_is_none";
    with_mistralrs_completions_cassette(
        "raw_stream_capture_matrix/stream_capture_off_raw_is_none",
        |client| async move {
            let model = client.completion_model(model_name());
            let request = request(&model, false);
            assert!(!request.capture_raw_response, "premise: default is off");
            let terminal =
                terminal_of(model.stream(request).await.expect("stream should start")).await;
            assert!(
                terminal.raw.is_none(),
                "terminal raw must stay None when capture was not requested, got {:?}",
                terminal.raw
            );
            assert!(terminal.usage.total_tokens > 0, "usage is unaffected");
            assert_eq!(terminal.provider, NORMALIZED_PROVIDER);
        },
    )
    .await;

    recorded_frames_with_terminal(scenario);
}

// ---------------------------------------------------------------------------
// 2: on → raw is the raw_stream FinalResponse, serialized
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "unrecorded (no mistral.rs server in this environment)"]
async fn stream_capture_on_terminal_round_trips_provider_type() {
    let scenario = "raw_stream_capture_matrix/stream_capture_on_terminal_round_trips_provider_type";
    let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = std::sync::Arc::clone(&captured);
    with_mistralrs_completions_cassette(
        "raw_stream_capture_matrix/stream_capture_on_terminal_round_trips_provider_type",
        |client| async move {
            let model = client.completion_model(model_name());
            let terminal = terminal_of(
                model
                    .stream(request(&model, true))
                    .await
                    .expect("stream should start"),
            )
            .await;
            let raw = terminal
                .raw
                .as_deref()
                .expect("terminal raw must be populated when capture was requested");
            let typed = openai::StreamingCompletionResponse::<openai::Usage>::deserialize(raw)
                .expect("raw must deserialize into openai::StreamingCompletionResponse");
            assert_eq!(
                serde_json::to_value(&typed).expect("terminal type should serialize"),
                *raw,
                "openai::StreamingCompletionResponse must round-trip through its own serde"
            );
            assert_eq!(
                typed.usage.prompt_tokens as u64,
                terminal.usage.input_tokens
            );
            assert_eq!(
                typed.usage.completion_tokens.map(|tokens| tokens as u64),
                Some(terminal.usage.output_tokens)
            );
            assert_eq!(typed.response_id, terminal.response_id);
            assert_eq!(typed.model, terminal.model);
            *sink.lock().expect("capture mutex") = Some(raw.clone());
        },
    )
    .await;

    let (_, terminal_frame) = recorded_frames_with_terminal(scenario);
    let raw = captured
        .lock()
        .expect("capture mutex")
        .take()
        .expect("the test body must have captured raw");
    assert_eq!(
        raw["usage"]["prompt_tokens"], terminal_frame["usage"]["prompt_tokens"],
        "raw usage must be the terminal frame's usage"
    );
    assert_eq!(
        raw["usage"]["completion_tokens"],
        terminal_frame["usage"]["completion_tokens"]
    );
}

// ---------------------------------------------------------------------------
// 3: terminal-only fields
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "unrecorded (no mistral.rs server in this environment)"]
async fn stream_capture_on_exposes_envelope_fields() {
    let scenario = "raw_stream_capture_matrix/stream_capture_on_exposes_envelope_fields";
    let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = std::sync::Arc::clone(&captured);
    with_mistralrs_completions_cassette(
        "raw_stream_capture_matrix/stream_capture_on_exposes_envelope_fields",
        |client| async move {
            let model = client.completion_model(model_name());
            let terminal = terminal_of(
                model
                    .stream(request(&model, true))
                    .await
                    .expect("stream should start"),
            )
            .await;
            let mut without_raw = terminal.clone();
            without_raw.raw = None;
            let normalized =
                serde_json::to_value(&without_raw).expect("StreamFinal should serialize");
            for field in [
                "system_fingerprint",
                "object",
                "created",
                "additional_params",
            ] {
                assert!(
                    normalized.get(field).is_none(),
                    "normalized StreamFinal must not grow a `{field}` field"
                );
            }
            let raw = terminal
                .raw
                .as_deref()
                .expect("terminal raw must be populated when capture was requested")
                .clone();
            *sink.lock().expect("capture mutex") = Some(raw);
        },
    )
    .await;

    let raw = captured
        .lock()
        .expect("capture mutex")
        .take()
        .expect("the test body must have captured raw");
    let (frames, terminal_frame) = recorded_frames_with_terminal(scenario);
    let params = raw
        .get("additional_params")
        .expect("raw terminal must carry the accumulated envelope under additional_params");
    for key in ["system_fingerprint", "object"] {
        assert_eq!(
            params.get(key),
            Some(&recorded_envelope_field(&frames, key, scenario)),
            "raw.additional_params.{key} must equal the recorded chunk envelope"
        );
    }
    // `created` is volatile: the scrubber placeholders it on disk, so only a
    // replay compares it exactly.
    let created = recorded_envelope_field(&frames, "created", scenario);
    match CassetteMode::current() {
        CassetteMode::Replay => assert_eq!(params.get("created"), Some(&created)),
        CassetteMode::Record => assert!(
            params.get("created").is_some_and(Value::is_u64) && created.is_u64(),
            "raw.additional_params.created must carry the chunk envelope's integer"
        ),
    }
    assert_eq!(raw["usage"], terminal_frame["usage"]);
    let typed = openai::StreamingCompletionResponse::<openai::Usage>::deserialize(&raw)
        .expect("raw must deserialize into openai::StreamingCompletionResponse");
    let typed_params = typed
        .additional_params
        .expect("typed terminal must carry additional_params");
    assert_eq!(
        typed_params.get("system_fingerprint"),
        frames[0].get("system_fingerprint")
    );
}

// ---------------------------------------------------------------------------
// 4: the flag never reaches the provider
// ---------------------------------------------------------------------------

/// One scenario, two interactions in wire order — off then on.
#[tokio::test]
#[ignore = "unrecorded (no mistral.rs server in this environment)"]
async fn stream_request_invariant_off_vs_on() {
    let scenario = "raw_stream_capture_matrix/stream_request_invariant_off_vs_on";
    with_mistralrs_completions_cassette(
        "raw_stream_capture_matrix/stream_request_invariant_off_vs_on",
        |client| async move {
            let model = client.completion_model(model_name());
            let off = terminal_of(
                model
                    .stream(request(&model, false))
                    .await
                    .expect("flag-off stream should start"),
            )
            .await;
            let on = terminal_of(
                model
                    .stream(request(&model, true))
                    .await
                    .expect("flag-on stream should start"),
            )
            .await;
            assert!(off.raw.is_none());
            assert!(on.raw.is_some());
            assert_eq!(off.provider, on.provider);
        },
    )
    .await;

    let requests = recorded_request_bodies(scenario);
    assert_eq!(
        requests.len(),
        2,
        "{scenario}: expected the off and on requests"
    );
    assert_eq!(
        requests[0], requests[1],
        "the flag-on streaming request body must be byte-identical to the \
         flag-off one — capture_raw_response must never reach the server"
    );
    assert!(!requests[0].contains("capture_raw"));
    let body: Value = serde_json::from_str(&requests[0]).expect("recorded request should be JSON");
    assert_eq!(body["stream"], Value::Bool(true));
    for (_, response) in recorded_interaction_bodies(MISTRALRS_PROVIDER, scenario) {
        let last_frame = response
            .lines()
            .filter_map(|line| line.trim().strip_prefix("data:"))
            .map(str::trim)
            .rfind(|payload| *payload != "[DONE]")
            .expect("each recorded stream should carry frames");
        let frame: Value = serde_json::from_str(last_frame).expect("terminal frame should be JSON");
        assert!(frame.get("usage").is_some_and(Value::is_object));
    }
}
