use futures::FutureExt;
use rig::providers::xai;
use std::future::Future;
use std::panic::AssertUnwindSafe;

use crate::cassettes::{CassetteSpec, ProviderCassette};

async fn xai_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, xai::Client) {
    let cassette = ProviderCassette::start("xai", spec, "https://api.x.ai").await;
    let client = xai::Client::builder()
        .api_key(cassette.api_key("XAI_API_KEY"))
        .base_url(cassette.base_url())
        .build()
        .expect("xAI client should build");

    (cassette, client)
}

pub(super) async fn with_xai_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(xai::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = xai_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

pub(super) async fn with_xai_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(xai::Client) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let (cassette, client) = xai_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test_result(result).await
}
