use futures::FutureExt;
use rig::providers::cohere;
use std::future::Future;
use std::panic::AssertUnwindSafe;

use crate::cassettes::{CassetteSpec, ProviderCassette};

async fn cohere_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, cohere::Client) {
    let cassette = ProviderCassette::start("cohere", spec, "https://api.cohere.ai").await;
    let client = cohere::Client::builder()
        .api_key(cassette.api_key("COHERE_API_KEY"))
        .base_url(cassette.base_url())
        .build()
        .expect("client should build");

    (cassette, client)
}

pub(super) async fn with_cohere_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(cohere::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = cohere_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}
