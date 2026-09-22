//! Matrix for raw response capture on ChatGPT's blocking `/responses` path
//! ([`CompletionResponse::raw`](rig::completion::CompletionResponse::raw)).
//!
//! # The feature
//!
//! Capture is always on. Every completion the driver returns carries `raw`:
//! the provider's own reply — the Responses API's
//! [`CompletionResponse`](rig::providers::openai::responses_api::CompletionResponse),
//! reassembled from the terminal `response.completed` event of the SSE body
//! ChatGPT answers even a non-streaming request with — serialized with
//! `serde_json::to_value`. Nothing about it is sent to
//! ChatGPT. `raw == Value::Null` means only that a `CompletionResponse` was
//! built by hand without a provider response behind it, which no cell here can
//! produce. A terminal event carrying no items captures the same envelope with
//! an empty `output` while the assistant content is folded from the preceding
//! events; see `raw_completion_parity_matrix` for that state.
//!
//! The Responses envelope carries fields the normalized
//! [`rig::completion::CompletionResponse`] has no home for — `object`,
//! `status`, `created_at` — and cell 2 reads them back through `raw`.
//!
//! # Matrix
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `raw_round_trips_provider_type` | typed access | `responses_api::CompletionResponse::deserialize(&*raw)` re-serializes equal | unrecorded (no CHATGPT credentials in this environment) |
//! | 2 | `raw_exposes_response_envelope` | provider-only fields | `object`/`status`/`created_at` in `raw` equal the terminal `response.completed` frame | unrecorded (no CHATGPT credentials in this environment) |
//! | 3 | `normalized_fields_equal_raw_renormalized` | normalized view | every normalized field equals the field the envelope on `raw` carries, and that envelope equals the recorded terminal frame's | unrecorded (no CHATGPT credentials in this environment) |
//!
//! Every cell is unrecorded: neither `CHATGPT_ACCESS_TOKEN`/`CHATGPT_ACCOUNT_ID`
//! nor a usable ChatGPT OAuth cache was present when this matrix was written,
//! and a fixture is never fabricated. The bodies are complete and would pass
//! once recorded. To record: export `CHATGPT_ACCESS_TOKEN` and
//! `CHATGPT_ACCOUNT_ID` (the harness placeholders both on disk), remove the
//! `#[ignore]` attributes, flip the table to `recorded`, then run
//! `RIG_PROVIDER_TEST_MODE=record cargo test -p rig --all-features --test chatgpt chatgpt::cassette::raw_capture_matrix -- --nocapture --test-threads=1`
//! and review `crates/rig-cassette/fixtures/cassettes/chatgpt/raw_capture_matrix/`.

use rig::completion::CompletionModel as _;
use rig::driver::Bound;
use rig::providers::chatgpt;
use rig::providers::openai::responses_api;
use rig::providers::openai::wire::OpenAiWire;
use serde::Deserialize;
use serde_json::Value;

use super::super::support::with_chatgpt_cassette;
use crate::cassettes::{CassetteMode, recorded_interaction_bodies, recorded_sse_json_frames};
use crate::raw_capture::{
    assert_no_request_id, assert_normalized_lacks, capture_completion, responses,
};
use crate::support::{Observed, assert_wire_value_matches, normalized_without_raw};

const CHATGPT_PROVIDER: &str = "chatgpt";
const MODEL: &str = chatgpt::GPT_5_4;
const PROMPT: &str = "Reply with exactly the single word: pong";

type ChatGptModel = Bound<OpenAiWire>;

fn request(model: &ChatGptModel) -> rig::completion::CompletionRequest {
    model.completion_request(PROMPT).max_tokens(64).build()
}

/// The premise every cell rests on: the scenario recorded exactly one
/// interaction whose SSE body ends with a `response.completed` frame whose
/// `response` carries usage. Returns that terminal `response` object.
///
/// Stays local: a Responses reply arrives as SSE even on the blocking path, so
/// the "body" the shared contract compares against is a frame's `response`
/// envelope rather than a plain JSON body, and usage hangs under it — neither
/// [`crate::raw_capture::chat::recorded_sole_usage_frame`]'s rule nor its
/// sibling's describes that shape.
fn recorded_terminal_response(scenario: &str) -> Value {
    assert_eq!(
        recorded_interaction_bodies(CHATGPT_PROVIDER, scenario).len(),
        1,
        "{scenario}: the scenario must record exactly one interaction"
    );
    let frames = recorded_sse_json_frames(CHATGPT_PROVIDER, scenario);
    let terminal = frames
        .iter()
        .rev()
        .find(|frame| frame.get("type").and_then(Value::as_str) == Some("response.completed"))
        .unwrap_or_else(|| {
            panic!("{scenario}: the recorded body must carry a response.completed frame")
        });
    let response = terminal["response"].clone();
    assert!(
        response.pointer("/usage/total_tokens").is_some(),
        "{scenario}: the terminal response must report usage"
    );
    assert_eq!(
        response.get("object").and_then(Value::as_str),
        Some("response"),
        "{scenario}: the terminal response must be a Responses envelope"
    );
    assert_eq!(
        response.get("status").and_then(Value::as_str),
        Some("completed"),
        "{scenario}: the terminal response must be completed"
    );
    response
}

// ---------------------------------------------------------------------------
// 1: raw reads back as the provider's own response type
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "unrecorded (no CHATGPT credentials in this environment)"]
async fn raw_round_trips_provider_type() {
    let scenario = "raw_capture_matrix/raw_round_trips_provider_type";
    let captured = Observed::default();
    let sink = captured.clone();
    with_chatgpt_cassette(
        "raw_capture_matrix/raw_round_trips_provider_type",
        |client| async move {
            capture_completion(client.completion(MODEL), request, sink)
                .await
                .expect("completion should succeed");
        },
    )
    .await;

    let response = captured.take();
    let raw = &response.raw;
    let typed = responses_api::CompletionResponse::deserialize(raw)
        .expect("raw must deserialize into responses_api::CompletionResponse");
    // `raw` is the provider's document, so it may carry more than
    // the wire type models — never less, and never a different
    // value for a field the type does parse.
    let reserialized = serde_json::to_value(&typed).expect("provider type should serialize");
    for (field, value) in reserialized
        .as_object()
        .expect("a Responses envelope is a JSON object")
    {
        assert_eq!(
            raw.get(field),
            Some(value),
            "raw.{field} must be the value responses_api::CompletionResponse parsed"
        );
    }

    // The typed view agrees with the normalized one on what the model
    // said, so raw is a superset, not a divergent copy.
    assert_eq!(Some(typed.model.as_str()), response.model.as_deref());
    assert_eq!(response.provider, CHATGPT_PROVIDER);
    assert!(!response.choice.is_empty());

    let terminal = recorded_terminal_response(scenario);
    responses_api::CompletionResponse::deserialize(&terminal)
        .expect("recorded terminal response must be a Responses envelope");
}

// ---------------------------------------------------------------------------
// 2: envelope fields rig does not normalize are readable from raw
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "unrecorded (no CHATGPT credentials in this environment)"]
async fn raw_exposes_response_envelope() {
    let scenario = "raw_capture_matrix/raw_exposes_response_envelope";
    let captured = Observed::default();
    let sink = captured.clone();
    with_chatgpt_cassette(
        "raw_capture_matrix/raw_exposes_response_envelope",
        |client| async move {
            capture_completion(client.completion(MODEL), request, sink)
                .await
                .expect("completion should succeed");
        },
    )
    .await;

    let response = captured.take();
    assert_normalized_lacks(
        &normalized_without_raw(response.clone()),
        &["object", "status", "created_at"],
    );

    let raw = response.raw;
    let terminal = recorded_terminal_response(scenario);
    for field in ["object", "status", "model"] {
        assert_eq!(
            raw.get(field),
            terminal.get(field),
            "raw.{field} must equal the recorded terminal frame"
        );
    }
    assert_wire_value_matches(&raw, &terminal, "created_at");
    assert_wire_value_matches(&raw, &terminal, "id");
    let typed = responses_api::CompletionResponse::deserialize(&raw).expect("raw must deserialize");
    assert_eq!(typed.status, responses_api::ResponseStatus::Completed);
    assert!(matches!(
        typed.object,
        responses_api::ResponseObject::Response
    ));
}

// ---------------------------------------------------------------------------
// 3: the normalized view and raw tell one story
// ---------------------------------------------------------------------------

/// Every field the normalized response carries must be the field the envelope
/// on `raw` carries — and that envelope must be the recorded terminal
/// `response.completed` frame. Capture is a pure serialization of the reply
/// the fold consumed: it neither alters a normalized field nor diverges from
/// the bytes ChatGPT sent.
#[tokio::test]
#[ignore = "unrecorded (no CHATGPT credentials in this environment)"]
async fn normalized_fields_equal_raw_renormalized() {
    let scenario = "raw_capture_matrix/normalized_fields_equal_raw_renormalized";
    let captured = Observed::default();
    let sink = captured.clone();
    with_chatgpt_cassette(
        "raw_capture_matrix/normalized_fields_equal_raw_renormalized",
        |client| async move {
            capture_completion(client.completion(MODEL), request, sink)
                .await
                .expect("completion should succeed");
        },
    )
    .await;

    let response = captured.take();
    let from_raw = responses_api::CompletionResponse::deserialize(&response.raw)
        .expect("raw must deserialize into responses_api::CompletionResponse");
    assert_eq!(response.provider, CHATGPT_PROVIDER);
    responses::assert_native_matches_normalized(&response, &from_raw, "the envelope on raw");
    // Both views here come from the *same* reply, so their ids agree
    // verbatim: the shared contract's token comparison exists for a live
    // value against a scrubbed fixture, and that relaxation does not apply.
    assert_eq!(response.response_id.as_deref(), Some(from_raw.id.as_str()));
    assert_eq!(
        response.message_id,
        from_raw.output.iter().find_map(|item| match item {
            responses_api::Output::Message(message) => Some(message.id.clone()),
            _ => None,
        }),
        "the normalized message id is the envelope's output-message id"
    );
    // ChatGPT reads no transport request-id header, so the whole identity
    // lives in the body and needs no reassembly here.
    assert_no_request_id(response.provider_request_id.as_deref(), CHATGPT_PROVIDER);
    assert!(!response.choice.is_empty());

    let terminal = recorded_terminal_response(scenario);
    // Both sides are read through the same wire type, so the comparison is of
    // the facts the envelope models rather than of incidental JSON shape.
    let from_wire = responses_api::CompletionResponse::deserialize(&terminal)
        .expect("recorded terminal response must be a Responses envelope");
    let mut live = serde_json::to_value(&from_raw).expect("provider type should serialize");
    let mut from_wire = serde_json::to_value(&from_wire).expect("provider type should serialize");
    // A live recording mints fresh ids and stamps; only a replay compares them
    // exactly, a live recording checks presence and shape.
    for field in ["id", "created_at"] {
        assert_wire_value_matches(&live, &from_wire, field);
        if matches!(CassetteMode::current(), CassetteMode::Record) {
            live[field] = Value::Null;
            from_wire[field] = Value::Null;
        }
    }
    assert_eq!(
        live, from_wire,
        "`raw` must be the terminal response.completed frame the recording holds"
    );
}
