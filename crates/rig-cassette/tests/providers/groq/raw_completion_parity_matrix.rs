//! Parity between Groq's captured reply document and the normalized response
//! it rode on.
//!
//! `ChatDecoder` folds the reply into a [`rig::completion::CompletionResponse`]
//! whose `raw` holds Groq's reply document verbatim, so every field the
//! normalized response reports is the field the document carries. The cell
//! here pins the transport request id: it arrives on the `x-request-id`
//! header (`GROQ.request_id_header`), the driver stamps `provider_request_id`
//! from it, and the document carries it only in Groq's own `x_groq.id`
//! envelope, which the shared chat-completions shape does not name. The cell
//! asserts that header and body agree, from a fixture whose responses carry
//! the header.
//!
//! The scenario literal keeps the name it was recorded under; the cell name
//! says what it asserts.

use rig::completion::CompletionRequest;

use super::RAW_CAPTURE_MODEL;
use super::support::with_groq_cassette_result;
use crate::cassettes::recorded_response_header;
use crate::raw_capture::capture_completion;
use crate::support::{Observed, assert_matches_recorded_token};

const PROVIDER: &str = "groq";
const PROMPT: &str = "Reply with the single word: pong";
const REQUEST_ID_HEADER: &str = "x-request-id";

fn request() -> CompletionRequest {
    CompletionRequest::new(PROMPT).max_tokens(16)
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

// ================================================================
// 1. Two turns, one seam: deterministic bytes, faithful documents
// ================================================================

// ================================================================
// 2. The transport id is a header, not a body field
// ================================================================

#[tokio::test]
async fn the_transport_id_comes_from_the_header_not_the_body() {
    const SCENARIO: &str = "raw_completion_parity_matrix/plain_raw_completion_lacks_request_id";
    let sink = Observed::default();
    with_groq_cassette_result(
        "raw_completion_parity_matrix/plain_raw_completion_lacks_request_id",
        |client| {
            capture_completion(
                client.completion(RAW_CAPTURE_MODEL),
                request(),
                sink.clone(),
            )
        },
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
    assert!(response.response_id().is_some());

    // The document has no top-level slot for a transport id, which is why
    // reading it off the header is the only way to have it.
    assert!(
        response.raw.get("x-request-id").is_none()
            && response.raw.get("provider_request_id").is_none(),
        "no chat-completions field carries the transport id: {}",
        response.raw
    );
    // The document itself carries it only in Groq's own envelope, and that
    // reaches the caller because `raw` is the body rather than the parse.
    assert_matches_recorded_token(
        response.raw["x_groq"]["id"].as_str(),
        Some(request_id.as_str()),
        "Groq's own envelope mirrors the transport id",
    );
}
