use futures::FutureExt;
use rig::driver::Bound;
use rig::prelude::*;
use rig::providers::openai::OpenAI;
use rig::providers::xai;
use std::future::Future;
use std::panic::AssertUnwindSafe;

use crate::cassettes::{CassetteSpec, ProviderCassette};

async fn xai_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, Bound<OpenAI>) {
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "xai",
        spec,
        "https://api.x.ai",
    )
    .await;
    let client = OpenAI::with_key(&xai::DIALECT, cassette.api_key("XAI_API_KEY"))
        .with_base_url(cassette.base_url())
        .bound()
        .expect("xAI client should build");

    (cassette, client)
}

pub(super) async fn with_xai_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(Bound<OpenAI>) -> Fut,
    Fut: Future<Output = ()>,
{
    let spec = spec.into();
    let (cassette, client) = xai_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    crate::cassettes::checkpoint_attempt(&cassette, "xai", spec.scenario()).await;
    cassette.finish_after_test(result).await;
}

pub(super) async fn with_xai_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(Bound<OpenAI>) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let (cassette, client) = xai_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test_result(result).await
}

/// Bogus-key variant for recording real 401s (rig#2314 error matrix).
pub(super) async fn with_xai_cassette_bogus_key<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(Bound<OpenAI>) -> Fut,
    Fut: Future<Output = ()>,
{
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "xai",
        spec,
        "https://api.x.ai",
    )
    .await;
    // The rejected credential is this wrapper's subject.
    cassette.expect_account_failure(crate::cassettes::AccountFailure::Auth);
    let client = OpenAI::with_key(&xai::DIALECT, "xai-invalid-edge-matrix-key")
        .with_base_url(cassette.base_url())
        .bound()
        .expect("xAI client should build");
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

/// Cassette wrapper for the xai prompt-caching matrix
/// (`crates/rig-cassette/fixtures/cassettes/xai/prompt_caching/`).
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
    F: FnOnce(Bound<OpenAI>) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = xai_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}
