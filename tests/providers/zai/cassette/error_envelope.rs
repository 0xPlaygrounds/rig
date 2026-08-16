//! Auth rejections on both Z.AI dialects.
//!
//! The two clients classify the *same* HTTP status differently, by design:
//! `ZAiExt` leaves `REQUEST_ID_HEADER` at its `None` default, so an OpenAI-
//! dialect failure keeps the transport shape `CompletionError::HttpError`,
//! while `ZAiAnthropicExt` inherits `AnthropicCompatibleProvider`'s
//! `Some("request-id")` contract and so classifies as
//! `CompletionError::ProviderResponse` (MIGRATING.md names
//! `zai::AnthropicClient` in that contract set). These cells assert the
//! asymmetry deliberately so a later reader does not mistake it for a bug and
//! "fix" one side into the other.
//!
//! The Anthropic cell doubles as the answer to a question the code cannot
//! settle: whether `https://api.z.ai/api/anthropic` accepts the `x-api-key`
//! header rig sends there at all (Z.AI documents `Authorization: Bearer` for
//! its own API and publishes no Anthropic-compatibility reference). A 401
//! whose body blames the header shape means rig's Anthropic-dialect client is
//! unusable; a 401 that blames the credential means it is fine.

use rig::completion::{CompletionError, CompletionModel};
use rig::prelude::*;

use super::super::support::{
    with_zai_anthropic_cassette_bogus_key, with_zai_general_cassette_bogus_key,
};
use super::super::{CHEAP_GENERAL_MODEL, CODING_MODEL};

#[tokio::test]
#[ignore = "unrecorded: requires ZAI_API_KEY to record the cassette"]
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
        assert!(
            status.is_client_error(),
            "an invalid key is a client error, got {status}"
        );
        assert!(
            error
                .provider_response_json()
                .expect("error body should be JSON")
                .is_some_and(|body| body.get("error").is_some()),
            "Z.AI answers a rejection with an `error` envelope"
        );
    })
    .await;
}

#[tokio::test]
#[ignore = "unrecorded: requires ZAI_API_KEY to record the cassette"]
async fn anthropic_bogus_key_is_a_provider_response() {
    with_zai_anthropic_cassette_bogus_key("anthropic/bogus_key_401", |client| async move {
        let model = client.completion_model(CODING_MODEL);
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
        assert!(
            status.is_client_error(),
            "an invalid key is a client error, got {status}"
        );
    })
    .await;
}
