use std::future::Future;
use std::panic::AssertUnwindSafe;

use futures::FutureExt;
use rig::driver::Bound;
use rig::http_client::BoxedHttpClient;
use rig::prelude::*;
use rig::providers::openai::wire::{MISTRAL, OpenAI};

use crate::cassettes::{CassetteSpec, ProviderCassette};

const MISTRAL_BASE_URL: &str = "https://api.mistral.ai";

/// The Mistral dialect of the OpenAI config bound to the bundled transport —
/// what a cassette test builds its models from, now that a model is a bound
/// wire.
pub(super) type BoundMistral = Bound<OpenAI, BoxedHttpClient>;

/// The Mistral config pointed at `cassette`.
fn mistral_config(cassette: &ProviderCassette) -> OpenAI {
    OpenAI::with_key(&MISTRAL, cassette.api_key("MISTRAL_API_KEY"))
        .with_base_url(cassette.base_url())
}

async fn mistral_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, BoundMistral) {
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "mistral",
        spec,
        MISTRAL_BASE_URL,
    )
    .await;
    let client = mistral_config(&cassette)
        .bound()
        .expect("Mistral cassette client should build");

    (cassette, client)
}

/// Unit-body wrapper for the recorded embedding matrix
/// (`crates/rig-cassette/fixtures/cassettes/mistral/embedding_matrix/`).
pub(super) async fn with_mistral_embedding_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(BoundMistral) -> Fut,
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
    F: FnOnce(BoundMistral) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let spec = spec.into();
    let (cassette, client) = mistral_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    crate::cassettes::checkpoint_attempt(&cassette, "mistral", spec.scenario()).await;
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
    F: FnOnce(BoundMistral) -> Fut,
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
    F: FnOnce(BoundMistral) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "mistral",
        spec,
        MISTRAL_BASE_URL,
    )
    .await;
    // The rejected credential is this wrapper's subject.
    cassette.expect_account_failure(crate::cassettes::AccountFailure::Auth);
    let client = OpenAI::with_key(&MISTRAL, "invalid-edge-matrix-key")
        .with_base_url(cassette.base_url())
        .bound()
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
    F: FnOnce(BoundMistral) -> Fut,
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
    F: FnOnce(BoundMistral) -> Fut,
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
    F: FnOnce(BoundMistral) -> Fut,
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
    F: FnOnce(BoundMistral) -> Fut,
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
    F: FnOnce(BoundMistral) -> Fut,
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
    F: FnOnce(BoundMistral) -> Fut,
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
    F: FnOnce(BoundMistral) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    // Inlined because cassette_safety source-scans this whole directory.
    let (cassette, client) = mistral_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test_result(result).await
}

/// Cassette wrapper for the mistral prompt-caching matrix
/// (`crates/rig-cassette/fixtures/cassettes/mistral/prompt_caching/`).
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
    F: FnOnce(BoundMistral) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = mistral_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}
