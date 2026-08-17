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

/// Bogus-key variant for recording real 401s: the shared model-listing fetch
/// must classify a rejected listing with provider, path and status context
/// (rig#2079), and only a real rejection proves it.
pub(super) async fn with_openrouter_cassette_bogus_key_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(openrouter::Client) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let cassette = ProviderCassette::start("openrouter", spec, OPENROUTER_BASE_URL).await;
    let client = openrouter::Client::builder()
        .api_key("sk-invalid-edge-matrix-key")
        .base_url(cassette.base_url())
        .build()
        .expect("OpenRouter client should build");
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test_result(result).await
}

/// Refusal edge matrix (`refusal_matrix/*`): a structured-output refusal
/// arrives as a sibling of `content`, and OpenRouter's own normalize dropped
/// it. Its own wrapper keeps the matrix auditable as one unit.
pub(super) async fn with_openrouter_refusal_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(openrouter::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    with_openrouter_cassette(spec, test_body).await;
}

/// Reasoning-usage edge matrix (`reasoning_usage_matrix/*`): OpenRouter's
/// `usage.completion_tokens_details.reasoning_tokens` had no landing slot and
/// the normalized `Usage.reasoning_tokens` was a hardcoded zero.
pub(super) async fn with_openrouter_usage_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(openrouter::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    with_openrouter_cassette(spec, test_body).await;
}

/// Live-recorded native OpenRouter log-probability transport matrix.
pub(super) async fn with_openrouter_stream_logprobs_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(openrouter::Client) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    with_openrouter_cassette_result(spec, test_body).await
}

/// Live-recorded malformed/truncated tool-call contract matrix.
pub(super) async fn with_openrouter_tool_truncation_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(openrouter::Client) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    with_openrouter_cassette_result(spec, test_body).await
}

/// Live-recorded native OpenRouter tool-call lifecycle matrix.
pub(super) async fn with_openrouter_tool_lifecycle_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(openrouter::Client) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    with_openrouter_cassette_result(spec, test_body).await
}

/// Live-recorded terminal identity, usage, and routed-provider metadata
/// matrix.
pub(super) async fn with_openrouter_terminal_metadata_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(openrouter::Client) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    with_openrouter_cassette_result(spec, test_body).await
}

/// Live-recorded caller-history roundtrip matrix.
pub(super) async fn with_openrouter_history_roundtrip_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(openrouter::Client) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    with_openrouter_cassette_result(spec, test_body).await
}

/// Live-recorded reasoning/tool ordering and signed-history parity matrix.
pub(super) async fn with_openrouter_reasoning_tool_order_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(openrouter::Client) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    with_openrouter_cassette_result(spec, test_body).await
}
