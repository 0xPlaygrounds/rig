//! Raw provider response capture on xAI's blocking path.
//!
//! **The feature.** Every blocking completion attaches the provider's own
//! reply document — the response body, verbatim — onto the normalized
//! [`rig::completion::CompletionResponse::raw`]. xAI speaks the OpenAI
//! Responses wire, so that document is what the Responses
//! [`CompletionResponse`] reads back; the transport `provider_request_id` is
//! a header rather than a body field, so it is *not* in `raw` (it is on the
//! normalized response instead). Capture is always on: there is
//! no flag to request it, nothing about it reaches the wire, and a
//! `Value::Null` only ever means a response built by hand with no provider
//! payload behind it. `raw` is a second view of the same response, never a
//! substitute for a normalized field. The Responses envelope carries a `status`
//! and, on xAI, a `service_tier` and a `metadata.system_fingerprint` the
//! normalized response has no slot for; those are the fields pinned here as
//! reachable only through `raw`.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 2 | `raw_exposes_status_and_service_tier` | provider-only field | `raw.status`, `raw.service_tier` and `raw.metadata.system_fingerprint` equal the fixture body | recorded |
//!
//! Every cell is recorded. Each re-derives its premise from its own fixture
//! after the wrapper returns: cell 2 reads the status and tier out of the
//! recorded body rather than trusting what the typed view reports, and cell 3
//! checks the normalized fields against the recorded body and the recorded
//! `x-request-id` header before reading the provider-native fields back out
//! of `raw`, so a recording that stopped carrying usage, a status, or the id
//! header fails loudly instead of covering nothing. The Responses format
//! contract those checks are written in — the recorded body's identity,
//! finish reason and accounting, and the provider-native view of the same
//! reply — is [`crate::raw_capture::responses`].

use rig::completion::CompletionRequest;
use rig::providers::xai;
use serde_json::json;

use super::support::with_xai_cassette_result;
use crate::cassettes::recorded_json_turn;
use crate::raw_capture::{assert_normalized_lacks, capture_completion};
use crate::support::{Observed, assert_matches_recorded_token, normalized_without_raw};

const PROVIDER: &str = "xai";
const MODEL: &str = xai::GROK_3_MINI;
const PROMPT: &str = "Reply with the single word: pong";

fn request() -> CompletionRequest {
    CompletionRequest::new(PROMPT)
}

// ================================================================
// 1. raw round-trips the Responses type
// ================================================================

// ================================================================
// 2. Fields the normalized response provably lacks
// ================================================================

#[tokio::test]
async fn raw_exposes_status_and_service_tier() {
    const SCENARIO: &str = "raw_capture_matrix/raw_exposes_status_and_service_tier";
    let sink = Observed::default();
    with_xai_cassette_result(
        "raw_capture_matrix/raw_exposes_status_and_service_tier",
        |client| capture_completion(client.completion(MODEL), request(), sink.clone()),
    )
    .await
    .expect("raw_exposes_status_and_service_tier should replay from its cassette");
    let response = sink.take();

    let (_, body) = recorded_json_turn(PROVIDER, SCENARIO);
    let recorded_status = body["status"]
        .as_str()
        .expect("a Responses body carries a status");
    let recorded_tier = body["service_tier"]
        .as_str()
        .expect("xAI reports service_tier on its Responses body");
    let recorded_fingerprint = body["metadata"]["system_fingerprint"]
        .as_str()
        .expect("xAI reports metadata.system_fingerprint");

    let raw = &response.raw;
    assert_eq!(raw["status"], json!(recorded_status));
    assert_eq!(raw["service_tier"], json!(recorded_tier));
    assert_matches_recorded_token(
        raw["metadata"]["system_fingerprint"].as_str(),
        Some(recorded_fingerprint),
        "system fingerprint",
    );
    // And the normalized view has no slot for any of them: the status is
    // folded into a finish reason, the rest has nowhere to go. The capture is
    // cleared out of the serialized view first, so the fields it carries
    // cannot satisfy the check it is the counterexample to.
    let normalized = normalized_without_raw(response);
    assert_normalized_lacks(&normalized, &["status", "service_tier", "metadata"]);
}

// ================================================================
// 3. The normalized view and raw tell one story
// ================================================================
