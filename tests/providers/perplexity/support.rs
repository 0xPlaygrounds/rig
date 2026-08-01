use futures::FutureExt;
use rig::providers::perplexity;
use std::future::Future;
use std::panic::AssertUnwindSafe;

use crate::cassettes::{CassetteSpec, ProviderCassette};

async fn perplexity_cassette(
    spec: impl Into<CassetteSpec>,
) -> (ProviderCassette, perplexity::Client) {
    let cassette = ProviderCassette::start("perplexity", spec, "https://api.perplexity.ai").await;
    let client = perplexity::Client::builder()
        .api_key(cassette.api_key("PERPLEXITY_API_KEY"))
        .base_url(cassette.base_url())
        .build()
        .expect("Perplexity cassette client should build");

    (cassette, client)
}

pub(super) async fn with_perplexity_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(perplexity::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = perplexity_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}
