//! Raw provider response capture on Doubleword's blocking chat-completions
//! path.
//!
//! **The feature.** Every blocking completion attaches Doubleword's verbatim
//! reply document onto the normalized
//! [`rig::completion::CompletionResponse::raw`]. Doubleword speaks the shared
//! chat-completions wire, so that document reads back as
//! [`openai::CompletionResponse`] — but it is the bytes the provider sent,
//! not a re-serialization of that parse, so it also retains fields the shared
//! type does not model. Capture is always on: there is no flag to request it,
//! nothing about it reaches the wire, and a `Value::Null` only ever means a
//! response built by hand with no provider payload behind it. `raw` is a
//! second view of the same response, never a substitute for a normalized
//! field.
//!
//! Two kinds of field are reachable only through `raw`, and cells 1 and 2
//! pin one each: the `object` tag, which the shared type models and the
//! normalized response has no slot for, and Doubleword's backend usage
//! extras (`cache_creation`, `cache_creation_input_tokens`,
//! `cache_read_input_tokens`), which no Rust type here models and which
//! survive precisely because `raw` is the document.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `raw_round_trips_openai_type` | typed read-back | `raw` deserializes into `openai::CompletionResponse`, and additionally carries usage extras that type does not model | recorded |
//! | 2 | `raw_exposes_object` | provider-only field | `raw.object` equals the fixture body; the unmodeled usage extras reach the caller through `raw` | recorded |
//! | 3 | `normalized_fields_match_raw_renormalized` | normalized view | the normalized fields reproduce the fixture bytes and equal the provider-native fields of the captured payload | recorded |
//!
//! Every cell is recorded. Each re-derives its premise from its own fixture
//! after the wrapper returns: cell 2 reads the tag out of the recorded body
//! rather than trusting the typed view, and cell 3 checks the normalized
//! fields against the recorded body before comparing them with the captured
//! payload's own fields, so a recording that stopped carrying a usage block
//! or a finish reason fails loudly instead of covering nothing. Doubleword
//! contracts no request-id header (`response_identity_edge` documents why),
//! so `provider_request_id` is `None` on every turn here — a documented
//! outcome, pinned as such. `DEFAULT_MODEL` is a reasoning model, so the
//! token budget leaves room for its hidden thinking before the one-word
//! answer.

use rig::completion::{CompletionModel, CompletionRequest};
use rig::providers::openai;
use serde::Deserialize as _;
use serde_json::json;

use super::super::DEFAULT_MODEL;
use super::super::support::with_doubleword_cassette_result;
use crate::cassettes::recorded_json_turn;
use crate::raw_capture::{assert_no_request_id, capture_completion, chat};
use crate::support::Observed;

const PROVIDER: &str = "doubleword";
const PROMPT: &str = "Reply with the single word: pong";

fn request(model: &(impl CompletionModel + Clone)) -> CompletionRequest {
    model.completion_request(PROMPT).max_tokens(256).build()
}

/// The backend usage fields Doubleword sends that no type here models.
const UNMODELLED_USAGE: [&str; 3] = [
    "cache_creation",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
];

// ================================================================
// 1. raw reads back as the shared OpenAI type — and carries more
// ================================================================

#[tokio::test]
async fn raw_round_trips_openai_type() {
    const SCENARIO: &str = "raw_capture_matrix/raw_round_trips_openai_type";
    let sink = Observed::default();
    with_doubleword_cassette_result("raw_capture_matrix/raw_round_trips_openai_type", |client| {
        capture_completion(client.completion(DEFAULT_MODEL), request, sink.clone())
    })
    .await
    .expect("raw_round_trips_openai_type should replay from its cassette");
    let response = sink.take();

    let raw = &response.raw;
    let typed = openai::CompletionResponse::deserialize(raw)
        .expect("raw is the shared OpenAI CompletionResponse Doubleword parses into");
    assert_eq!(Some(typed.id.as_str()), response.response_id.as_deref());

    // `raw` is the document Doubleword sent, not a re-serialization of
    // `typed`: these usage fields have no home on the shared type and reach
    // the caller only because capture keeps the payload whole.
    for field in UNMODELLED_USAGE {
        assert!(
            raw["usage"].get(field).is_some(),
            "raw keeps Doubleword's `usage.{field}`, which the shared type does not model"
        );
    }

    let (_, response_body) = recorded_json_turn(PROVIDER, SCENARIO);
    assert!(
        response_body["choices"][0]["message"]["content"].is_string(),
        "the recorded turn should be a plain text answer"
    );
}

// ================================================================
// 2. A field the normalized response provably lacks
// ================================================================

#[tokio::test]
async fn raw_exposes_object() {
    const SCENARIO: &str = "raw_capture_matrix/raw_exposes_object";
    let sink = Observed::default();
    with_doubleword_cassette_result("raw_capture_matrix/raw_exposes_object", |client| {
        capture_completion(client.completion(DEFAULT_MODEL), request, sink.clone())
    })
    .await
    .expect("raw_exposes_object should replay from its cassette");
    let response = sink.take();

    let (_, body) = recorded_json_turn(PROVIDER, SCENARIO);
    let recorded_object = body["object"]
        .as_str()
        .expect("Doubleword tags every completion with an object");

    let raw = &response.raw;
    assert_eq!(raw["object"], json!(recorded_object));
    // The normalized view has no slot for the tag.
    let normalized = serde_json::to_value(&response).expect("response serializes");
    assert!(normalized.get("object").is_none());
    // Nor for Doubleword's backend usage extras — which is why `raw` is the
    // document rather than the parse: they are in the recording, and they are
    // in `raw`, with the same values.
    for field in UNMODELLED_USAGE {
        assert!(
            body["usage"].get(field).is_some(),
            "the recorded usage carries Doubleword's `{field}`: {}",
            body["usage"]
        );
        assert_eq!(
            raw["usage"][field], body["usage"][field],
            "`usage.{field}` reaches the caller through raw"
        );
        assert!(
            normalized.get(field).is_none(),
            "`{field}` has no normalized slot"
        );
    }
}

// ================================================================
// 3. The normalized view and raw tell one story
// ================================================================

#[tokio::test]
async fn normalized_fields_match_raw_renormalized() {
    const SCENARIO: &str = "raw_capture_matrix/normalized_fields_match_raw_renormalized";
    let sink = Observed::default();
    with_doubleword_cassette_result(
        "raw_capture_matrix/normalized_fields_match_raw_renormalized",
        |client| capture_completion(client.completion(DEFAULT_MODEL), request, sink.clone()),
    )
    .await
    .expect("normalized_fields_match_raw_renormalized should replay from its cassette");
    let response = sink.take();

    let (_, body) = recorded_json_turn(PROVIDER, SCENARIO);
    chat::assert_reproduces_body(&response, PROVIDER, &body, "the recorded body");
    // Doubleword contracts no request-id header, so `None` is the documented
    // outcome rather than an id to compare.
    assert_no_request_id(response.provider_request_id.as_deref(), PROVIDER);

    // The other half: the provider-native fields of the captured payload are
    // the ones the decoder normalized. There is one mapping now, so this pins
    // it against the wire's own vocabulary rather than against a copy of
    // itself.
    let typed = openai::CompletionResponse::deserialize(&response.raw)
        .expect("raw is the shared OpenAI type");
    chat::assert_native_matches_normalized(&response, &typed, "the typed view of raw");
}
