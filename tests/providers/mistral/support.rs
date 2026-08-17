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
