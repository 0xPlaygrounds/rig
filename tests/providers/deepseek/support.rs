use std::future::Future;
use std::panic::AssertUnwindSafe;

use futures::FutureExt;
use rig::providers::deepseek;

use crate::cassettes::{CassetteSpec, ProviderCassette};

async fn deepseek_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, deepseek::Client) {
    let cassette = ProviderCassette::start("deepseek", spec, "https://api.deepseek.com").await;
    let client = deepseek::Client::builder()
        .api_key(cassette.api_key("DEEPSEEK_API_KEY"))
        .base_url(cassette.base_url())
        .build()
        .expect("DeepSeek client should build");

    (cassette, client)
}

pub(super) async fn with_deepseek_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(deepseek::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = deepseek_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

pub(super) async fn with_deepseek_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(deepseek::Client) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let (cassette, client) = deepseek_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test_result(result).await
}
