use rig::providers::openai;
use std::future::Future;

use crate::cassettes::ProviderCassette;

pub(super) async fn openai_cassette(scenario: &'static str) -> (ProviderCassette, openai::Client) {
    let cassette = ProviderCassette::start("openai", scenario, "https://api.openai.com/v1").await;
    let client = openai::Client::builder()
        .api_key(cassette.api_key("OPENAI_API_KEY"))
        .base_url(cassette.base_url())
        .build()
        .expect("client should build");

    (cassette, client)
}

pub(super) async fn openai_completions_cassette(
    scenario: &'static str,
) -> (ProviderCassette, openai::CompletionsClient) {
    let (cassette, client) = openai_cassette(scenario).await;
    (cassette, client.completions_api())
}

pub(super) async fn with_openai_cassette<F, Fut>(scenario: &'static str, test_body: F)
where
    F: FnOnce(openai::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = openai_cassette(scenario).await;
    test_body(client).await;
    cassette.finish().await;
}

pub(super) async fn with_openai_completions_cassette<F, Fut>(scenario: &'static str, test_body: F)
where
    F: FnOnce(openai::CompletionsClient) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = openai_completions_cassette(scenario).await;
    test_body(client).await;
    cassette.finish().await;
}

pub(super) async fn with_openai_cassette_result<F, Fut, E>(
    scenario: &'static str,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(openai::Client) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let (cassette, client) = openai_cassette(scenario).await;
    let result = test_body(client).await;
    cassette.finish().await;
    result
}

pub(super) async fn with_openai_completions_cassette_result<F, Fut, E>(
    scenario: &'static str,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(openai::CompletionsClient) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let (cassette, client) = openai_completions_cassette(scenario).await;
    let result = test_body(client).await;
    cassette.finish().await;
    result
}
