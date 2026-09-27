use futures::FutureExt;
use rig::providers::openai::wire::{OpenAIConfig, PERPLEXITY};
use rig_test_support::cassette_models::OpenAiModels;
use std::future::Future;
use std::panic::AssertUnwindSafe;

use crate::cassettes::{CassetteSpec, ProviderCassette};

async fn perplexity_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, OpenAIConfig) {
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "perplexity",
        spec,
        "https://api.perplexity.ai",
    )
    .await;
    let perplexity = OpenAIConfig::with_key(&PERPLEXITY, cassette.api_key("PERPLEXITY_API_KEY"))
        .with_base_url(cassette.base_url());

    (cassette, perplexity)
}

pub(super) async fn with_perplexity_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(OpenAiModels) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = perplexity_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(OpenAiModels::new(
        client,
        rig::rig_reqwest::shared(),
    )))
    .catch_unwind()
    .await;
    cassette.finish_after_test(result).await;
}

/// Cassette wrapper for the perplexity prompt-caching matrix
/// (`crates/rig-cassette/fixtures/cassettes/perplexity/prompt_caching/`).
///
/// Delegates to [`with_perplexity_cassette`] — the behavior is identical, and deliberately shared
/// so the two cannot drift apart when the base wrapper gains policy. What the
/// separate name buys is a per-suite entry in the cassette-safety registry, so
/// the cache fixtures are auditable as one concern's evidence.
pub(super) async fn with_perplexity_prompt_caching_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(OpenAiModels) -> Fut,
    Fut: Future<Output = ()>,
{
    with_perplexity_cassette(spec, test_body).await;
}
