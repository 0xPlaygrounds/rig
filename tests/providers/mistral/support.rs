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

/// Wrapper for the embedding-dimension matrix, kept apart from
/// [`with_mistral_capability_cassette`] so the Codestral width cells are
/// auditable on their own.
pub(super) async fn with_mistral_embedding_cassette<F, Fut, E>(
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
