use std::future::Future;
use std::panic::AssertUnwindSafe;

use futures::FutureExt;
use rig::providers::mistral;

use crate::cassettes::{CassetteSpec, ProviderCassette};

const MISTRAL_BASE_URL: &str = "https://api.mistral.ai";

async fn mistral_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, mistral::Client) {
    let cassette = ProviderCassette::start("mistral", spec, MISTRAL_BASE_URL).await;
    let client = mistral::Client::builder()
        .api_key(cassette.api_key("MISTRAL_API_KEY"))
        .base_url(cassette.base_url())
        .build()
        .expect("Mistral cassette client should build");

    (cassette, client)
}

pub(super) async fn with_mistral_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(mistral::Client) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let (cassette, client) = mistral_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test_result(result).await
}
