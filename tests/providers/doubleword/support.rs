use futures::FutureExt;
use rig::providers::doubleword;
use std::future::Future;
use std::panic::AssertUnwindSafe;

use crate::cassettes::{CassetteSpec, ProviderCassette};

const DOUBLEWORD_BASE_URL: &str = "https://api.doubleword.ai/v1";

async fn doubleword_cassette(
    spec: impl Into<CassetteSpec>,
) -> (ProviderCassette, doubleword::Client) {
    let cassette = ProviderCassette::start("doubleword", spec, DOUBLEWORD_BASE_URL).await;
    let client = doubleword::Client::builder()
        .api_key(cassette.api_key("DOUBLEWORD_API_KEY"))
        .base_url(cassette.base_url())
        .build()
        .expect("client should build");

    (cassette, client)
}

pub(super) async fn with_doubleword_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(doubleword::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = doubleword_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

pub(super) async fn with_doubleword_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(doubleword::Client) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let (cassette, client) = doubleword_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test_result(result).await
}
