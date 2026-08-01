use std::future::Future;
use std::panic::AssertUnwindSafe;

use futures::FutureExt;
use rig::providers::groq;

use crate::cassettes::{CassetteSpec, ProviderCassette};

async fn groq_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, groq::Client) {
    let cassette = ProviderCassette::start("groq", spec, "https://api.groq.com/openai/v1").await;
    let client = groq::Client::builder()
        .api_key(cassette.api_key("GROQ_API_KEY"))
        .base_url(cassette.base_url())
        .build()
        .expect("Groq client should build");

    (cassette, client)
}

pub(super) async fn with_groq_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(groq::Client) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let (cassette, client) = groq_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test_result(result).await
}
