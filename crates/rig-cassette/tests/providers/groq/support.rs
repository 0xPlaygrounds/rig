use rig_test_support::cassette_models::OpenAiModels;
use std::future::Future;
use std::panic::AssertUnwindSafe;

use futures::FutureExt;
use rig::providers::openai::wire::{GROQ, OpenAIConfig};

use crate::cassettes::{CassetteSpec, ProviderCassette};

async fn groq_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, OpenAIConfig) {
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "groq",
        spec,
        GROQ.base_url,
    )
    .await;
    let groq = OpenAIConfig::with_key(&GROQ, cassette.api_key("GROQ_API_KEY"))
        .with_base_url(cassette.base_url());

    (cassette, groq)
}

pub(super) async fn with_groq_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(OpenAiModels) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let spec = spec.into();
    let (cassette, groq) = groq_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(OpenAiModels::new(
        groq,
        rig::rig_reqwest::shared(),
    )))
    .catch_unwind()
    .await;
    crate::cassettes::checkpoint_attempt(&cassette, "groq", spec.scenario()).await;
    cassette.finish_after_test_result(result).await
}

/// Bogus-key variant for recording real 401s (rig#2314 error matrix).
pub(super) async fn with_groq_cassette_bogus_key_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(OpenAiModels) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "groq",
        spec,
        GROQ.base_url,
    )
    .await;
    // The rejected credential is this wrapper's subject.
    cassette.expect_account_failure(crate::cassettes::AccountFailure::Auth);
    let groq = OpenAIConfig::with_key(&GROQ, "gsk-invalid-edge-matrix-key")
        .with_base_url(cassette.base_url());
    let result = AssertUnwindSafe(test_body(OpenAiModels::new(
        groq,
        rig::rig_reqwest::shared(),
    )))
    .catch_unwind()
    .await;
    cassette.finish_after_test_result(result).await
}

/// Cassette wrapper for the groq prompt-caching matrix
/// (`crates/rig-cassette/fixtures/cassettes/groq/prompt_caching/`).
///
/// Builds the cassette directly rather than delegating to [`with_groq_cassette_result`]: this
/// provider's cassette-safety `source_dir` covers `support.rs` itself, and the
/// scan requires every call to a *registered* wrapper to pass a string-literal
/// scenario. A delegating wrapper passes its `spec` variable through, which the
/// scan reports as an unregistered scenario. The duplication is three lines and
/// the alternative is an unscannable suite.
pub(super) async fn with_groq_prompt_caching_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(OpenAiModels) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = groq_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(OpenAiModels::new(
        client,
        rig::rig_reqwest::shared(),
    )))
    .catch_unwind()
    .await;
    cassette.finish_after_test(result).await;
}
