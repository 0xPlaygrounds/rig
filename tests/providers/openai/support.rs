use rig::providers::openai;
use std::future::Future;
use std::panic::AssertUnwindSafe;

use crate::cassettes::{CassetteSpec, ProviderCassette};
use futures::FutureExt;

async fn openai_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, openai::Client) {
    let cassette = ProviderCassette::start("openai", spec, "https://api.openai.com/v1").await;
    let client = openai::Client::builder()
        .api_key(cassette.api_key("OPENAI_API_KEY"))
        .base_url(cassette.base_url())
        .build()
        .expect("client should build");

    (cassette, client)
}

async fn openai_completions_cassette(
    spec: impl Into<CassetteSpec>,
) -> (ProviderCassette, openai::CompletionsClient) {
    let (cassette, client) = openai_cassette(spec).await;
    (cassette, client.completions_api())
}

pub(super) async fn with_openai_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(openai::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = openai_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

pub(super) async fn with_openai_completions_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(openai::CompletionsClient) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = openai_completions_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

pub(super) async fn with_openai_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(openai::Client) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let (cassette, client) = openai_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test_result(result).await
}

pub(super) async fn with_openai_completions_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(openai::CompletionsClient) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let (cassette, client) = openai_completions_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test_result(result).await
}

/// Per-bug wrapper for the Chat Completions refusal matrix
/// (`tests/cassettes/openai/refusal_matrix/`).
///
/// Yields the Responses client; cells that drive Chat Completions call
/// [`openai::Client::completions_api`] on it, so one wrapper covers both
/// surfaces of a bug whose logic lives in the shared chat-completions types.
pub(super) async fn with_openai_refusal_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(openai::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    with_openai_cassette(spec, test_body).await;
}

/// Per-bug wrapper for the output-token-cap spelling matrix
/// (`tests/cassettes/openai/max_completion_tokens_matrix/`).
pub(super) async fn with_openai_max_tokens_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(openai::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    with_openai_cassette(spec, test_body).await;
}

/// Like [`with_openai_cassette`], but authenticating with a deliberately
/// invalid API key — for recording real 401s with no secret near the fixture.
pub(super) async fn with_openai_cassette_bogus_key<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(openai::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let cassette = ProviderCassette::start("openai", spec, "https://api.openai.com/v1").await;
    let client = openai::Client::builder()
        .api_key("sk-invalid-edge-matrix-key")
        .base_url(cassette.base_url())
        .build()
        .expect("client should build");
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}
