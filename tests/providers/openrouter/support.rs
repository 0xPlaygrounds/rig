use rig::providers::{openai, openrouter};
use std::future::Future;
use std::panic::AssertUnwindSafe;

use crate::cassettes::{CassetteSpec, ProviderCassette};
use futures::FutureExt;

const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";

async fn openrouter_cassette(
    spec: impl Into<CassetteSpec>,
) -> (ProviderCassette, openrouter::Client) {
    let cassette = ProviderCassette::start("openrouter", spec, OPENROUTER_BASE_URL).await;
    let client = openrouter::Client::builder()
        .api_key(cassette.api_key("OPENROUTER_API_KEY"))
        .base_url(cassette.base_url())
        .build()
        .expect("OpenRouter cassette client should build");

    (cassette, client)
}

async fn openrouter_openai_cassette(
    spec: impl Into<CassetteSpec>,
) -> (ProviderCassette, openai::Client) {
    let cassette = ProviderCassette::start("openrouter", spec, OPENROUTER_BASE_URL).await;
    let client = openai::Client::builder()
        .api_key(cassette.api_key("OPENROUTER_API_KEY"))
        .base_url(cassette.base_url())
        .build()
        .expect("OpenRouter OpenAI-compatible cassette client should build");

    (cassette, client)
}

pub(super) async fn with_openrouter_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(openrouter::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = openrouter_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

pub(super) async fn with_openrouter_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(openrouter::Client) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let (cassette, client) = openrouter_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test_result(result).await
}

pub(super) async fn with_openrouter_openai_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(openai::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = openrouter_openai_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}
