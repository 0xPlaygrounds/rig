use rig::client::DefaultTransportBuilder as _;
use std::future::Future;
use std::panic::AssertUnwindSafe;

use futures::FutureExt;
use rig::providers::mistral;

use crate::cassettes::{CassetteSpec, ProviderCassette};

const MISTRAL_BASE_URL: &str = "https://api.mistral.ai";

async fn mistral_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, mistral::Client) {
    let cassette = ProviderCassette::start("mistral", spec, MISTRAL_BASE_URL).await;
    let client = mistral::Client::builder()
        .api_key(cassette.api_key("MISTRAL_API_KEY"))
        .base_url(cassette.base_url())
        .build()
        .expect("Mistral cassette client should build");

    (cassette, client)
}

/// Unit-body wrapper for the recorded embedding matrix
/// (`tests/cassettes/mistral/embedding_matrix/`).
pub(super) async fn with_mistral_embedding_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(mistral::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = mistral_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

pub(super) async fn with_mistral_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(mistral::Client) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let (cassette, client) = mistral_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test_result(result).await
}

/// Wrapper for the multimodal-content matrix (#2290).
///
/// Behaviourally identical to [`with_mistral_cassette_result`] — fixture
/// layout comes from the scenario prefix, not from the wrapper. It exists as a
/// named seam for the matrix, so a later change to how those cells build their
/// client (a second base URL, a different key) lands here instead of on every
/// Mistral cassette test at once.
pub(super) async fn with_mistral_multimodal_cassette<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(mistral::Client) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let (cassette, client) = mistral_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test_result(result).await
}

/// Bogus-key variant for recording real 401s. Inlined rather than delegating
/// to another registered wrapper: `cassette_safety`'s source scan covers this
/// whole directory, and a wrapper call whose scenario is a variable fails it.
pub(super) async fn with_mistral_cassette_bogus_key_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(mistral::Client) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let cassette = ProviderCassette::start("mistral", spec, MISTRAL_BASE_URL).await;
    let client = mistral::Client::builder()
        .api_key("invalid-edge-matrix-key")
        .base_url(cassette.base_url())
        .build()
        .expect("Mistral cassette client should build");
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test_result(result).await
}

/// Wrapper for the capability matrix (model listing, embeddings batching and
/// dimensions), kept as its own named seam so those cells' fixtures stay
/// separable from the completion suites.
pub(super) async fn with_mistral_capability_cassette<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(mistral::Client) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let (cassette, client) = mistral_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test_result(result).await
}

/// Live-recorded terminal metadata and primary-choice matrix.
pub(super) async fn with_mistral_terminal_metadata_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(mistral::Client) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    // Kept inlined for the same reason as the bogus-key wrapper above: this
    // directory is source-scanned for cassette registrations, so delegating
    // through another registered wrapper would look like a variable scenario.
    let (cassette, client) = mistral_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test_result(result).await
}

/// Live-recorded tool-call truncation boundary matrix.
pub(super) async fn with_mistral_tool_truncation_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(mistral::Client) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    // Inlined because cassette_safety source-scans this whole directory.
    let (cassette, client) = mistral_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test_result(result).await
}

/// Live-recorded Mistral tool-call lifecycle matrix.
pub(super) async fn with_mistral_tool_lifecycle_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(mistral::Client) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let (cassette, client) = mistral_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test_result(result).await
}

/// Live-recorded caller-history roundtrip matrix.
pub(super) async fn with_mistral_history_roundtrip_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(mistral::Client) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    // Inlined because cassette_safety source-scans this whole directory.
    let (cassette, client) = mistral_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test_result(result).await
}

/// Live-recorded Mistral tool-policy and response-format finalization matrix.
pub(super) async fn with_mistral_request_shape_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(mistral::Client) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    // Inlined because cassette_safety source-scans this whole directory.
    let (cassette, client) = mistral_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test_result(result).await
}

/// Live-recorded evidence for Mistral's model-level logprobs rejection.
pub(super) async fn with_mistral_logprobs_rejection_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(mistral::Client) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    // Inlined because cassette_safety source-scans this whole directory.
    let (cassette, client) = mistral_cassette(spec).await;
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

/// The response headers of every interaction recorded under `scenario`, in
/// cassette order, as lower-cased `(name, value)` pairs.
///
/// The shared body readers deliberately stop at bodies; a matrix whose
/// premise is "the provider sent its request-id header" has to read the
/// header side of the fixture, and the recorder keeps only allowlisted
/// response headers (`mistral-correlation-id` among them), so what is here is exactly
/// what replay serves.
pub(super) fn recorded_response_headers(scenario: &str) -> Vec<Vec<(String, String)>> {
    use serde::Deserialize as _;

    let path = crate::cassettes::cassette_path("mistral", scenario);
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

/// Cassette wrapper for the mistral prompt-caching matrix
/// (`tests/cassettes/mistral/prompt_caching/`).
///
/// Builds the cassette directly rather than delegating to [`with_mistral_cassette_result`]: this
/// provider's cassette-safety `source_dir` covers `support.rs` itself, and the
/// scan requires every call to a *registered* wrapper to pass a string-literal
/// scenario. A delegating wrapper passes its `spec` variable through, which the
/// scan reports as an unregistered scenario. The duplication is three lines and
/// the alternative is an unscannable suite.
pub(super) async fn with_mistral_prompt_caching_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(mistral::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = mistral_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}
