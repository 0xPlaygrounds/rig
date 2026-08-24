use futures::FutureExt;
use rig::client::DefaultTransportBuilder as _;
use rig::providers::xai;
use std::future::Future;
use std::panic::AssertUnwindSafe;

use crate::cassettes::{CassetteSpec, ProviderCassette};

async fn xai_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, xai::Client) {
    let cassette = ProviderCassette::start("xai", spec, "https://api.x.ai").await;
    let client = xai::Client::builder()
        .api_key(cassette.api_key("XAI_API_KEY"))
        .base_url(cassette.base_url())
        .build()
        .expect("xAI client should build");

    (cassette, client)
}

pub(super) async fn with_xai_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(xai::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = xai_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

pub(super) async fn with_xai_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(xai::Client) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let (cassette, client) = xai_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test_result(result).await
}

/// Bogus-key variant for recording real 401s (rig#2314 error matrix).
pub(super) async fn with_xai_cassette_bogus_key<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(xai::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let cassette = ProviderCassette::start("xai", spec, "https://api.x.ai").await;
    let client = xai::Client::builder()
        .api_key("xai-invalid-edge-matrix-key")
        .base_url(cassette.base_url())
        .build()
        .expect("xAI client should build");
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
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

/// The response headers of every interaction recorded under `scenario`, in
/// cassette order, as lower-cased `(name, value)` pairs.
///
/// The shared body readers deliberately stop at bodies; a matrix whose
/// premise is "the provider sent its request-id header" has to read the
/// header side of the fixture, and the recorder keeps only allowlisted
/// response headers (`x-request-id` among them), so what is here is exactly
/// what replay serves.
pub(super) fn recorded_response_headers(scenario: &str) -> Vec<Vec<(String, String)>> {
    use serde::Deserialize as _;

    let path = crate::cassettes::cassette_path("xai", scenario);
    let contents = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("cassette {} should be readable: {error}", path.display()));

    serde_yaml::Deserializer::from_str(&contents)
        .map(|document| {
            let interaction = serde_yaml::Value::deserialize(document).unwrap_or_else(|error| {
                panic!("cassette {} should deserialize: {error}", path.display())
            });
            interaction["then"]["header"]
                .as_sequence()
                .map(|headers| {
                    headers
                        .iter()
                        .map(|header| {
                            (
                                header["name"]
                                    .as_str()
                                    .expect("header name")
                                    .to_ascii_lowercase(),
                                header["value"].as_str().expect("header value").to_owned(),
                            )
                        })
                        .collect()
                })
                .unwrap_or_default()
        })
        .collect()
}

/// Cassette wrapper for the xai prompt-caching matrix
/// (`tests/cassettes/xai/prompt_caching/`).
///
/// Builds the cassette directly rather than delegating to [`with_xai_cassette`]: this
/// provider's cassette-safety `source_dir` covers `support.rs` itself, and the
/// scan requires every call to a *registered* wrapper to pass a string-literal
/// scenario. A delegating wrapper passes its `spec` variable through, which the
/// scan reports as an unregistered scenario. The duplication is three lines and
/// the alternative is an unscannable suite.
pub(super) async fn with_xai_prompt_caching_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(xai::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = xai_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}
