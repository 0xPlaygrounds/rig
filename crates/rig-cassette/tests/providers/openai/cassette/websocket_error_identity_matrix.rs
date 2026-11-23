//! Error identity on the OpenAI Responses websocket transport.
//!
//! A websocket upgrade the provider rejects is an ordinary HTTP response,
//! and this endpoint answers it exactly as its HTTP twin does: status,
//! request id and JSON error body. The caller must reach the same
//! `ProviderError` identity whichever transport carried the rejection, so
//! `provider_response_status()`, `provider_response_body()` and
//! `provider_request_id()` stay populated on the websocket path too.
//!
//! Only an auth-class failure is recordable as a handshake: the model is
//! named in the `response.create` message, so a bad model fails in band on
//! an established socket. The rest of the error space is covered by unit
//! tests of `from_tungstenite` in `rig-tungstenite` and
//! `websocket_provider_error_*` in `rig-core`, joined against a real socket
//! by `crates/rig-tungstenite/tests/handshake_rejection.rs`.

use rig::error::ProviderError;

use super::super::support::with_openai_websocket_cassette;
use rig::completion::CompletionRequest;

/// What a caller can learn from a failed connection, through the three
/// error identity accessors.
fn observable(error: &ProviderError) -> (Option<u16>, bool, bool) {
    (
        error
            .provider_response_status()
            .map(|status| status.as_u16()),
        error
            .provider_response_body()
            .is_some_and(|body| body.contains("invalid_api_key")),
        error.provider_request_id().is_some(),
    )
}

/// Both transports run against the same credential and path inside one
/// cassette: the upgrade `GET /v1/responses` and the unary
/// `POST /v1/responses`. What a caller can observe must match between the
/// two errors. Running both means a regression on either side fails the cell.
#[tokio::test]
async fn handshake_rejection_matches_the_http_twin() {
    with_openai_websocket_cassette(
        "websocket_error_identity_matrix/handshake_rejection_matches_the_http_twin",
        |client| async move {
            let model = client.openai.responses("gpt-4o-mini");
            let websocket_error = model
                .responses_websocket()
                .connect()
                .await
                .err()
                .expect("an invalid key must fail the upgrade");

            let http_error = model
                .call(CompletionRequest::new("Never authenticated"))
                .await
                .expect_err("the same key must fail the unary request");

            assert_eq!(
                observable(&websocket_error),
                observable(&http_error),
                "the websocket transport must not carry less than the HTTP one\n  \
                 websocket: {websocket_error}\n  http: {http_error}"
            );
            // Not vacuously equal: both must actually carry the identity.
            assert_eq!(observable(&http_error), (Some(401), true, true));
        },
    )
    .await;
}
