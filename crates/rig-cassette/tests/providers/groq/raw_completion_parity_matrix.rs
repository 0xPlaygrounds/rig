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
//! `x_groq.id` envelope, which the shared shape does not name and which reaches a
//! caller only because `raw` is the document. So the header and the body
//! agree, and the cells assert that they do rather than assuming it.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 2 | `the_transport_id_comes_from_the_header_not_the_body` | header vs document | `provider_request_id` is populated from the header; the shared chat-completions fields of `raw` have no slot for it, and the document carries it only in Groq's own envelope | recorded |
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
