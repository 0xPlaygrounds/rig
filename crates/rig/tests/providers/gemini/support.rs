use futures::FutureExt;
use rig::providers::gemini;
use std::future::Future;
use std::panic::AssertUnwindSafe;

use crate::cassettes::{CassetteSpec, ProviderCassette};

async fn gemini_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, gemini::Client) {
    let cassette =
        ProviderCassette::start("gemini", spec, "https://generativelanguage.googleapis.com").await;
    let client = gemini::Client::builder()
        .api_key(cassette.api_key("GEMINI_API_KEY"))
        .base_url(cassette.base_url())
        .build()
        .expect("client should build");

    (cassette, client)
}

async fn gemini_interactions_cassette(
    spec: impl Into<CassetteSpec>,
) -> (ProviderCassette, gemini::InteractionsClient) {
    let (cassette, client) = gemini_cassette(spec).await;
    (cassette, client.interactions_api())
}

pub(super) async fn with_gemini_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(gemini::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = gemini_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

pub(super) async fn with_gemini_interactions_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(gemini::InteractionsClient) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = gemini_interactions_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}
