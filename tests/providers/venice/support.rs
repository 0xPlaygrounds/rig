use futures::FutureExt;
use rig::providers::venice;
use std::future::Future;
use std::panic::AssertUnwindSafe;

use crate::cassettes::{CassetteSpec, ProviderCassette};

// The direct-recording client is shared with the other suites whose response
// bodies are binary; it is gated on the same feature that compiles this
// scenario, since the PR gate builds this target with the default feature set
// where an ungated helper would be dead code under `-D warnings`.
#[cfg(feature = "audio")]
use crate::cassettes::DirectRecordingHttpClient;

const VENICE_BASE_URL: &str = venice::VENICE_API_BASE_URL;

async fn venice_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, venice::Client) {
    let cassette = ProviderCassette::start("venice", spec, VENICE_BASE_URL).await;
    let client = venice::Client::builder()
        .api_key(cassette.api_key("VENICE_API_KEY"))
        .base_url(cassette.base_url())
        .build()
        .expect("client should build");

    (cassette, client)
}

pub(super) async fn with_venice_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(venice::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = venice_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

/// Cassette wrapper for scenarios whose response body is binary; see
/// [`DirectRecordingHttpClient`].
#[cfg(feature = "audio")]
pub(super) async fn with_venice_direct_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(venice::Client<DirectRecordingHttpClient>) -> Fut,
    Fut: Future<Output = ()>,
{
    let cassette = ProviderCassette::start_direct_recording("venice", spec, VENICE_BASE_URL).await;
    let http_client = DirectRecordingHttpClient::new(cassette.direct_recorder());
    let client = venice::Client::builder()
        .api_key(cassette.api_key("VENICE_API_KEY"))
        .base_url(cassette.base_url())
        .http_client(http_client)
        .build()
        .expect("client should build");

    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

pub(super) async fn with_venice_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(venice::Client) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let (cassette, client) = venice_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test_result(result).await
}

/// Compare a generated token (response id, system fingerprint, request id)
/// observed by a test with the value its fixture holds.
///
/// Fixtures are placeholder-scrubbed (`chatcmpl-REDACTED_1`, `fp_REDACTED_1`,
/// `req_REDACTED_1`), so on the recording pass the live token cannot equal
/// the fixture's; both are then required to be present and non-empty. On
/// replay the harness serves the scrubbed bytes back, so equality is exact —
/// which is what CI runs. Presence must agree in both modes.
pub(super) fn assert_matches_recorded_token(
    actual: Option<&str>,
    recorded: Option<&str>,
    context: &str,
) {
    match crate::cassettes::CassetteMode::current() {
        crate::cassettes::CassetteMode::Replay => {
            assert_eq!(
                actual, recorded,
                "{context}: replay serves the fixture's token back"
            );
        }
        crate::cassettes::CassetteMode::Record => {
            assert_eq!(
                actual.is_some(),
                recorded.is_some(),
                "{context}: live and recorded token presence must agree"
            );
            if let (Some(actual), Some(recorded)) = (actual, recorded) {
                assert!(
                    !actual.trim().is_empty() && !recorded.trim().is_empty(),
                    "{context}: live and recorded token must both be non-empty"
                );
            }
        }
    }
}
