use std::future::Future;
use std::panic::AssertUnwindSafe;

use futures::FutureExt;
use rig::providers::openai;

use crate::cassettes::{CassetteSpec, ProviderCassette};

pub(super) const DEFAULT_BASE_URL: &str = "http://127.0.0.1:1234/v1";
pub(super) const DEFAULT_API_KEY: &str = "local";
pub(super) const DEFAULT_MODEL: &str = "Qwen/Qwen3-4B";
pub(super) const SYSTEM_PROMPT: &str =
    "You are concise. Include a few details so streaming is visible.";

pub(super) fn model_name() -> String {
    std::env::var("MISTRALRS_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string())
}

async fn mistralrs_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, openai::Client) {
    let real_base_url =
        std::env::var("MISTRALRS_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_string());
    let api_key =
        std::env::var("MISTRALRS_API_KEY").unwrap_or_else(|_| DEFAULT_API_KEY.to_string());
    let cassette = ProviderCassette::start("mistralrs", spec, &real_base_url).await;
    let client = openai::Client::builder()
        .api_key(api_key)
        .base_url(cassette.base_url())
        .build()
        .expect("mistral.rs OpenAI-compatible client should build");

    (cassette, client)
}

async fn mistralrs_raw_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, String) {
    let real_base_url =
        std::env::var("MISTRALRS_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_string());
    let cassette = ProviderCassette::start("mistralrs", spec, &real_base_url).await;
    let base_url = cassette.base_url();
    (cassette, base_url)
}

pub(super) async fn with_mistralrs_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(openai::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = mistralrs_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

pub(super) async fn with_mistralrs_completions_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(openai::CompletionsClient) -> Fut,
    Fut: Future<Output = ()>,
{
    with_mistralrs_cassette(spec, |client| async move {
        test_body(client.completions_api()).await;
    })
    .await;
}

pub(super) async fn with_mistralrs_raw_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(String) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, base_url) = mistralrs_raw_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(base_url)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}
