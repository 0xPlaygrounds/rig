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
        rig_test_support::cassettes::local_http(),
    )))
    .catch_unwind()
    .await;
    cassette.finish_after_test(result).await;
}
