//! Edge matrix for error identity on the OpenAI Responses **websocket**
//! transport.
//!
//! **Bug.** A websocket upgrade the provider *rejects* never becomes a
//! websocket — it is an ordinary HTTP response, and this endpoint answers it
//! exactly as its HTTP twin does. A live handshake against
//! `wss://api.openai.com/v1/responses` with an invalid key returns:
//!
//! ```text
//! HTTP/1.1 401 Unauthorized
//! x-request-id: req_...
//! {"error":{"message":"Incorrect API key provided: …","type":"invalid_request_error",
//!           "code":"invalid_api_key","param":null},"status":401}
//! ```
//!
//! `tungstenite` hands rig all three — its `Error::Http` carries the status,
//! the headers, and a body filled in from the read tail — and
//! `websocket_provider_error` flattened the lot to `error.to_string()`, i.e.
//! `ProviderError("HTTP error: 401 Unauthorized")`. So
//! `provider_response_status()`, `provider_response_body()` and
//! `provider_request_id()` all came back `None`, and a caller could not tell a
//! bad key from a quota stop from a network fault.
//!
//! That is the contract rig#2314 and rig#2315 established for the crate's
//! other two transports — the blocking path keeps it through
//! `send_completion`, the SSE connect path through `sse_transport` — leaving
//! the websocket the one transport that still dropped it.
//!
//! **How these cells fail on `origin/main`.** The recorded handshake replays
//! identically — the fix changes nothing rig sends — and the assertions fail:
//! `main` yields `ProviderError("HTTP error: 401 Unauthorized")` where the cell
//! expects a `ProviderResponse` carrying status, body and request id.
//!
//! **Why this matrix is small, and where its coverage actually lives.** The
//! input is not a request space but an error space: the only failure carrying
//! a provider response is `tungstenite::Error::Http`, and every other variant
//! has none. The unit cells beside the fix
//! (`websocket_provider_error_*` in
//! `crates/rig-core/src/providers/openai/responses_api/websocket.rs`) cover
//! that space — body present/absent, request id present/absent/empty, the
//! rejection's headers present/absent, nine status classes including 2xx and
//! 3xx, and every non-`Http` variant except `Tls`, whose inner error cannot be
//! constructed portably in a test.
//!
//! **These recorded cells do not run on the PR gate.** The module is behind
//! `#[cfg(feature = "websocket")]`, which is not in the facade's default
//! features, and the gate's `rig` test steps do not enable it — so on a pull
//! request this file is type-checked by clippy's `--all-features` job and
//! *executed* by the nightly and merge-queue runs. The unit cells above, which
//! live in `rig-core`, do run on every PR.
//!
//! Only the *auth-class* failure is reachable as a recorded handshake: the
//! model is named in the `response.create` message, not in the upgrade, so a
//! bad model fails **in band** as an `error` event on an established socket
//! rather than as a handshake rejection. There is exactly one recordable
//! handshake failure, and it is cell 1.
//!
//! | # | cell | trigger | asserts | status |
//! |---|------|---------|---------|--------|
//! | 1 | `handshake_rejection_carries_status_body_and_request_id` | invalid key | 401 + body + `x-request-id` | recorded |
//! | 2 | `handshake_rejection_matches_the_http_twin` | invalid key | same identity as the HTTP 401 | recorded |
//!
//! Cell 2 is the parity claim stated as a test: the same credential rejected
//! on the same path must reach the caller the same way whether the transport
//! was HTTP or a websocket upgrade. It replays the recorded websocket
//! handshake **and drives the unary HTTP path against the same credential in
//! the same cassette**, then compares the two errors. Both transports run, so
//! a regression on either side fails the cell — reading the twin's recorded
//! bytes instead, as an earlier version did, could not have detected an HTTP
//! regression at all.

use rig::client::completion::CompletionClient;
use rig::completion::{CompletionError, CompletionModel};
use rig::rig_reqwest::openai_websocket::ResponsesWebSocketExt as _;

use super::super::support::with_openai_websocket_cassette;

/// What a caller can actually learn from a failed connection — the three
/// accessors the rig#2314/#2315 contract is written in terms of.
fn observable(error: &CompletionError) -> (Option<u16>, bool, bool) {
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

#[tokio::test]
async fn handshake_rejection_carries_status_body_and_request_id() {
    with_openai_websocket_cassette(
        "websocket_error_identity_matrix/handshake_rejection_carries_status_body_and_request_id",
        |client| async move {
            let error = client
                .responses_websocket("gpt-4o-mini")
                .await
                .err()
                .expect("an invalid key must fail the upgrade");

            let (status, names_the_cause, has_request_id) = observable(&error);
            assert_eq!(status, Some(401), "the rejection's status must survive");
            assert!(
                names_the_cause,
                "the provider's own error body must survive: {error}"
            );
            assert!(
                has_request_id,
                "the transport request id must survive, as it does on every other transport"
            );
            assert!(
                matches!(error, CompletionError::ProviderResponse(_)),
                "a rejection carrying a provider response classifies as one: {error:?}"
            );
        },
    )
    .await;
}

/// The parity the bug broke, driven rather than described: **both transports
/// run**, against the same credential and the same path, inside one cassette —
/// the upgrade `GET /v1/responses` and the unary `POST /v1/responses` — and
/// what a caller can observe is compared between the two errors.
///
/// An earlier version of this cell read the HTTP twin's recorded *bytes*
/// instead of running the HTTP transport, which a review pointed out could not
/// detect the thing it was named for: had the unary path regressed to
/// `from_http_response` (dropping the request id), this cell would still have
/// passed. Executing both is the only form that cannot.
#[tokio::test]
async fn handshake_rejection_matches_the_http_twin() {
    with_openai_websocket_cassette(
        "websocket_error_identity_matrix/handshake_rejection_matches_the_http_twin",
        |client| async move {
            let websocket_error = client
                .responses_websocket("gpt-4o-mini")
                .await
                .err()
                .expect("an invalid key must fail the upgrade");

            let model = client.completion_model("gpt-4o-mini");
            let http_error = model
                .completion(model.completion_request("Never authenticated").build())
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
