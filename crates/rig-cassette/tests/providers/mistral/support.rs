use rig_test_support::cassette_models::OpenAiModels;
use std::future::Future;
use std::panic::AssertUnwindSafe;

use futures::FutureExt;
use rig::providers::openai::wire::{MISTRAL, OpenAIConfig};

use crate::cassettes::{CassetteSpec, ProviderCassette};

const MISTRAL_BASE_URL: &str = "https://api.mistral.ai";

/// The Mistral config pointed at `cassette`.
fn mistral_config(cassette: &ProviderCassette) -> OpenAIConfig {
    OpenAIConfig::with_key(&MISTRAL, cassette.api_key("MISTRAL_API_KEY"))
        .with_base_url(cassette.base_url())
}

async fn mistral_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, OpenAIConfig) {
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "mistral",
        spec,
        MISTRAL_BASE_URL,
    )
    .await;
    let client = mistral_config(&cassette);

    (cassette, client)
}

/// Unit-body cassette wrapper.
pub(super) async fn with_mistral_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(OpenAiModels) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = mistral_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(OpenAiModels::new(
        client,
        rig_test_support::cassettes::local_http(),
    )))
    .catch_unwind()
    .await;
    cassette.finish_after_test(result).await;
}

pub(super) async fn with_mistral_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(OpenAiModels) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let spec = spec.into();
    let (cassette, client) = mistral_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(OpenAiModels::new(
        client,
        rig_test_support::cassettes::local_http(),
    )))
    .catch_unwind()
    .await;
    crate::cassettes::checkpoint_attempt(&cassette, "mistral", spec.scenario()).await;
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
    F: FnOnce(OpenAiModels) -> Fut,
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
    let client = OpenAIConfig::with_key(&MISTRAL, "invalid-edge-matrix-key")
        .with_base_url(cassette.base_url());
    let result = AssertUnwindSafe(test_body(OpenAiModels::new(
        client,
        rig_test_support::cassettes::local_http(),
    )))
    .catch_unwind()
    .await;
    cassette.finish_after_test_result(result).await
}
