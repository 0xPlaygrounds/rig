//! Parity between Groq's captured reply document and the normalized response
//! it rode on.
//!
//! **The contract.** There is exactly one unary seam: `Chat::encode` builds
//! the request, the driver carries it, and `ChatDecoder` folds the reply into
//! a [`rig::completion::CompletionResponse`] whose `raw` holds Groq's own
//! reply *document* verbatim. Two things follow, and these cells pin both.
//!
//! 1. **`encode` is deterministic.** The same built request produces the same
//!    request bytes every time, so a caller can replay it — and the two
//!    recorded turns of cell 1 must therefore be byte-identical on the
//!    request side.
//! 2. **`raw` is a faithful second view, not a summary.** Every field the
//!    normalized response reports is the field the document carries, for the
//!    very reply it came attached to.
//!
//! The transport request id is the interesting case here and is what cell 2
//! is about. It arrives on a *header* — Groq contracts `x-request-id`, which
//! is the dialect datum `GROQ.request_id_header`, and the driver stamps
//! `provider_request_id` from it. The reply document has no field for it in
//! the shared chat-completions shape; Groq happens to mirror it in its own
//! `x_groq.id` envelope, which no shared type models and which reaches a
//! caller only because `raw` is the document. So the header and the body
//! agree, and the cells assert that they do rather than assuming it.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `encode_is_deterministic_and_raw_is_faithful` | two turns, one seam | both turns send identical request bytes; each response reproduces *its own* interaction's body and `x-request-id`, and its `raw["x_groq"]["id"]` is that same id | recorded |
//! | 2 | `the_transport_id_comes_from_the_header_not_the_body` | header vs document | `provider_request_id` is populated from the header; the shared typed view of `raw` has no slot for it, and the document carries it only in Groq's own envelope | recorded |
//!
//! The scenario literals — and therefore the fixture filenames — keep the
//! names they were recorded under; the cell names describe what the cells now
//! assert.
//!
//! Both cells are recorded. Cell 1's scenario holds **two** interactions,
//! because the cell needs two independent replies to separate "the same bytes
//! went out twice" from "one reply agreed with itself"; the harness replays
//! interactions in order, and two live turns carry two different ids, so each
//! side is compared with its own interaction's recorded body and header. The
//! premise both cells re-derive from their fixture is that Groq's recorded
//! responses carry the `x-request-id` header at all.

use rig::completion::{CompletionModel, CompletionRequest};
use rig::providers::openai;
use rig::providers::openai::wire::GROQ;
use serde::Deserialize;
use serde_json::Value;

use super::RAW_CAPTURE_MODEL;
use super::support::with_groq_cassette_result;
use crate::cassettes::{recorded_json_turns, recorded_response_header};
use crate::raw_capture::{capture_completion, capture_completion_pair, chat};
use crate::support::{Observed, assert_matches_recorded_token};

const PROVIDER: &str = "groq";
const PROMPT: &str = "Reply with the single word: pong";
const REQUEST_ID_HEADER: &str = "x-request-id";

fn request(model: &(impl CompletionModel + Clone)) -> CompletionRequest {
    model.completion_request(PROMPT).max_tokens(16).build()
}

/// The `x-request-id` the recorded interaction at `index` carried — the
/// premise of every cell here.
fn recorded_request_id(scenario: &str, index: usize) -> String {
    recorded_response_header(PROVIDER, scenario, index, REQUEST_ID_HEADER).unwrap_or_else(|| {
        panic!(
            "interaction {index} of {scenario} must carry the x-request-id header Groq contracts"
        )
    })
}

/// One turn's normalized view against its own interaction's recorded bytes:
/// the shared chat-completions contract, plus the two identity claims that
/// are Groq's own — a chat reply names no assistant message, and the
/// transport id came from the header this cell already read.
fn assert_reproduces_fixture(
    response: &rig::completion::CompletionResponse,
    body: &Value,
    request_id: &str,
    context: &str,
) {
    chat::assert_reproduces_body(response, PROVIDER, body, context);
    let identity = response.identity();
    assert_eq!(
        identity.message_id, None,
        "{context}: chat has no message id"
    );
    assert_matches_recorded_token(
        identity.provider_request_id.as_deref(),
        Some(request_id),
        &format!("{context}: request id"),
    );
}

/// The reply document's own view of the same turn, for the fields the
/// document holds: it must be the recorded body, and its `x_groq` envelope
/// must mirror the transport id the header carried.
fn assert_raw_is_the_reply_document(
    response: &rig::completion::CompletionResponse,
    body: &Value,
    request_id: &str,
    context: &str,
) {
    let raw = &response.raw;
    assert_matches_recorded_token(
        raw["id"].as_str(),
        body["id"].as_str(),
        &format!("{context}: raw's own response id"),
    );
    assert_eq!(
        raw.get("usage"),
        body.get("usage"),
        "{context}: raw carries the provider's usage block unchanged"
    );
    assert_eq!(
        raw["choices"][0]["message"]["content"], body["choices"][0]["message"]["content"],
        "{context}: and the answer it reported"
    );
    assert_matches_recorded_token(
        raw["x_groq"]["id"].as_str(),
        Some(request_id),
        &format!("{context}: Groq mirrors the transport id in its own envelope"),
    );
}

// ================================================================
// 1. Two turns, one seam: deterministic bytes, faithful documents
// ================================================================

#[tokio::test]
async fn encode_is_deterministic_and_raw_is_faithful() {
    const SCENARIO: &str = "raw_completion_parity_matrix/raw_with_request_id_reproduces_completion";
    let sink = Observed::default();
    with_groq_cassette_result(
        "raw_completion_parity_matrix/raw_with_request_id_reproduces_completion",
        |client| {
            capture_completion_pair(client.completion(RAW_CAPTURE_MODEL), request, sink.clone())
        },
    )
    .await
    .expect("raw_with_request_id_reproduces_completion should replay from its cassette");

    let (first, second) = sink.take();
    let interactions = recorded_json_turns(PROVIDER, SCENARIO);
    assert_eq!(
        interactions.len(),
        2,
        "the cell records a turn and its twin"
    );
    assert_eq!(
        interactions[0].0, interactions[1].0,
        "`encode` is deterministic, so both turns must send the same request bytes"
    );

    assert_eq!(
        GROQ.request_id_header,
        Some("x-request-id"),
        "Groq contracts x-request-id"
    );
    let first_id = recorded_request_id(SCENARIO, 0);
    let second_id = recorded_request_id(SCENARIO, 1);

    // Each reply is checked against its own interaction: two live turns carry
    // two different ids, so a cross-comparison would prove nothing.
    assert_reproduces_fixture(&first, &interactions[0].1, &first_id, "first turn");
    assert_reproduces_fixture(&second, &interactions[1].1, &second_id, "second turn");
    assert_raw_is_the_reply_document(&first, &interactions[0].1, &first_id, "first turn");
    assert_raw_is_the_reply_document(&second, &interactions[1].1, &second_id, "second turn");

    // Where the wire makes the two turns equal, they are equal.
    assert_eq!(second.provider, first.provider);
    assert_eq!(second.model, first.model);
    assert_eq!(second.finish_reason(), first.finish_reason());
    assert_eq!(second.identity().message_id, first.identity().message_id);
    // Identical request bytes tokenize identically; the output side is the
    // model's to vary.
    assert_eq!(second.usage.input_tokens, first.usage.input_tokens);
}

// ================================================================
// 2. The transport id is a header, not a body field
// ================================================================

#[tokio::test]
async fn the_transport_id_comes_from_the_header_not_the_body() {
    const SCENARIO: &str = "raw_completion_parity_matrix/plain_raw_completion_lacks_request_id";
    let sink = Observed::default();
    with_groq_cassette_result(
        "raw_completion_parity_matrix/plain_raw_completion_lacks_request_id",
        |client| capture_completion(client.completion(RAW_CAPTURE_MODEL), request, sink.clone()),
    )
    .await
    .expect("plain_raw_completion_lacks_request_id should replay from its cassette");

    let response = sink.take();

    // The premise: the header was there to read.
    let request_id = recorded_request_id(SCENARIO, 0);
    assert!(!request_id.trim().is_empty());

    assert_matches_recorded_token(
        response.provider_request_id.as_deref(),
        Some(request_id.as_str()),
        "the driver stamps the transport id from the header Groq contracts",
    );
    assert!(response.response_id.is_some());

    // The shared typed view of the document has no slot for a transport id —
    // which is why reading it off the header is the only way to have it.
    let typed = openai::CompletionResponse::deserialize(&response.raw)
        .expect("raw is the shared OpenAI chat-completions reply Groq sends");
    let typed_document = serde_json::to_value(&typed).expect("typed serializes");
    assert!(
        typed_document.get("x-request-id").is_none()
            && typed_document.get("provider_request_id").is_none()
            && typed_document.get("x_groq").is_none(),
        "no shared chat-completions field carries the transport id: {typed_document}"
    );
    // The document itself carries it only in Groq's own envelope, and that
    // reaches the caller because `raw` is the body rather than the parse.
    assert_matches_recorded_token(
        response.raw["x_groq"]["id"].as_str(),
        Some(request_id.as_str()),
        "Groq's own envelope mirrors the transport id",
    );
}
