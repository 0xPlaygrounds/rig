//! Auth rejections on both Z.AI dialects — the only cells in this suite that
//! are **recorded**.
//!
//! They needed no credential: a deliberately invalid key produces a real 401,
//! so these two fixtures were captured against the live API in an environment
//! with no `ZAI_API_KEY` at all. Every other cell here is still unrecorded.
//!
//! What they settle:
//!
//! * **Base-URL composition on both dialects.** The recorded paths are
//!   `/api/paas/v4/chat/completions` and `/api/anthropic/v1/messages` — no
//!   doubled `/v4`, no dropped prefix.
//! * **The Anthropic dialect's auth header is accepted.** rig sends `x-api-key`
//!   there (`anthropic::client::AnthropicKey`) while Z.AI's own quick-start
//!   documents `Authorization: Bearer`, and Z.AI publishes no
//!   Anthropic-compatibility page — so whether that client can authenticate at
//!   all was open. It can: the replay proves rig sent `x-api-key` (the `zai`
//!   policy requires it, and a missing header fails the match), and the
//!   recorded answer is a *credential* rejection — `token expired or
//!   incorrect`. Z.AI answers a request with no auth header differently, with
//!   `type: "1001"`, `Authentication parameter not received in Header`. So the
//!   header was read.
//! * **The two error envelopes differ.** The OpenAI dialect answers
//!   `{"error":{"code":"401",…}}` and the Anthropic dialect
//!   `{"error":{"message":…,"type":"401"}}` — Z.AI reshapes the envelope per
//!   dialect, and both are asserted below.
//! * **The classification asymmetry is deliberate.** `ZAiExt` leaves
//!   `REQUEST_ID_HEADER` at its `None` default, so an OpenAI-dialect failure
//!   keeps the transport shape `CompletionError::HttpError`, while
//!   `ZAiAnthropicExt` inherits `AnthropicCompatibleProvider`'s
//!   `Some("request-id")` contract and classifies as
//!   `CompletionError::ProviderResponse` (MIGRATING.md names
//!   `zai::AnthropicClient` in that contract set). Asserted per dialect so a
//!   later reader does not mistake it for a bug and "fix" one into the other.

use rig::completion::{CompletionError, CompletionModel};
use rig::prelude::*;

use super::super::support::{
    recorded_request_path, recorded_response_body, with_zai_anthropic_cassette_bogus_key,
    with_zai_general_cassette_bogus_key,
};
use super::super::{ANTHROPIC_MODEL, CHEAP_GENERAL_MODEL};

#[tokio::test]
async fn general_bogus_key_is_an_http_error() {
    with_zai_general_cassette_bogus_key("general/bogus_key_401", |client| async move {
        let model = client.completion_model(CHEAP_GENERAL_MODEL);
        let request = model.completion_request("Say hi.").max_tokens(16).build();

        let error = model
            .completion(request)
            .await
            .expect_err("an invalid key should be rejected");

        assert!(
            matches!(error, CompletionError::HttpError(_)),
            "Z.AI's OpenAI dialect reports no request-id header, so its failures keep the \
             transport shape; got {error:?}"
        );

        let status = error
            .provider_response_status()
            .expect("the rejection should preserve its HTTP status");
        assert_eq!(status.as_u16(), 401, "unexpected status: {status}");

        let body = error
            .provider_response_json()
            .expect("error body should be JSON")
            .expect("the rejection should preserve its body");
        assert_eq!(
            body["error"]["code"], "401",
            "Z.AI's OpenAI dialect keys the envelope on `code`; got {body}"
        );
    })
    .await;

    assert_eq!(
        recorded_request_path("general/bogus_key_401"),
        "/api/paas/v4/chat/completions",
        "the general base URL must compose with the endpoint suffix exactly once"
    );
}

#[tokio::test]
async fn anthropic_bogus_key_is_a_provider_response() {
    with_zai_anthropic_cassette_bogus_key("anthropic/bogus_key_401", |client| async move {
        let model = client.completion_model(ANTHROPIC_MODEL);
        let request = model.completion_request("Say hi.").max_tokens(16).build();

        let error = model
            .completion(request)
            .await
            .expect_err("an invalid key should be rejected");

        assert!(
            matches!(error, CompletionError::ProviderResponse(_)),
            "the Anthropic dialect inherits the request-id contract, so its failures \
             classify as provider responses; got {error:?}"
        );

        let status = error
            .provider_response_status()
            .expect("the rejection should preserve its HTTP status");
        assert_eq!(status.as_u16(), 401, "unexpected status: {status}");

        let body = error
            .provider_response_json()
            .expect("error body should be JSON")
            .expect("the rejection should preserve its body");
        assert_eq!(
            body["error"]["type"], "401",
            "Z.AI's Anthropic dialect keys the envelope on `type`; got {body}"
        );
    })
    .await;

    assert_eq!(
        recorded_request_path("anthropic/bogus_key_401"),
        "/api/anthropic/v1/messages",
        "the Anthropic base URL must compose with the Messages suffix exactly once"
    );

    // The finding this cell exists for: `x-api-key` was read, not ignored. Z.AI
    // spells a *missing* auth header `type: "1001"`; a rejection that instead
    // blames the credential is the evidence that rig's header reached the
    // authenticator.
    let body = recorded_response_body("anthropic/bogus_key_401");
    assert_ne!(
        body["error"]["type"], "1001",
        "1001 is Z.AI's `Authentication parameter not received in Header`; seeing it here \
         would mean the Anthropic dialect's x-api-key is not the header Z.AI reads"
    );
}
