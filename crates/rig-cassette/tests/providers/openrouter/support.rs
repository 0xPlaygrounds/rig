use rig::driver::Bound;
use rig::http_client::BoxedHttpClient;
use rig::prelude::*;
use rig::providers::openai::wire::{OPENROUTER, OpenAI, Route};
use std::future::Future;
use std::panic::AssertUnwindSafe;

use crate::cassettes::{CassetteSpec, ProviderCassette};
use futures::FutureExt;

const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";

/// OpenRouter's chat surface: the `OPENROUTER` dialect of the OpenAI chat
/// wire, bound to the bundled transport.
///
/// Named once here so every `FnOnce(..)` bound below and every suite that
/// needs to spell the provider out agrees on one type.
pub(super) type BoundOpenRouter = Bound<OpenAI, BoxedHttpClient>;

/// The same OpenRouter host on its `/responses` route, for the compatibility
/// suite: OpenRouter serves `/responses` as well, and the point of those
/// cells is that rig's Responses wire drives it once the configuration is
/// routed there.
pub(super) type BoundOpenRouterResponses = Bound<OpenAI, BoxedHttpClient>;

async fn openrouter_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, BoundOpenRouter) {
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "openrouter",
        spec,
        OPENROUTER_BASE_URL,
    )
    .await;
    let bound = OpenAI::with_key(&OPENROUTER, cassette.api_key("OPENROUTER_API_KEY"))
        .with_base_url(cassette.base_url())
        .bound()
        .expect("OpenRouter cassette transport should build");

    (cassette, bound)
}

async fn openrouter_openai_cassette(
    spec: impl Into<CassetteSpec>,
) -> (ProviderCassette, BoundOpenRouterResponses) {
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "openrouter",
        spec,
        OPENROUTER_BASE_URL,
    )
    .await;
    let bound = OpenAI::with_key(&OPENROUTER, cassette.api_key("OPENROUTER_API_KEY"))
        .with_base_url(cassette.base_url())
        .with_route(Route::Responses)
        .bound()
        .expect("OpenRouter Responses cassette transport should build");

    (cassette, bound)
}

pub(super) async fn with_openrouter_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(BoundOpenRouter) -> Fut,
    Fut: Future<Output = ()>,
{
    let spec = spec.into();
    let (cassette, bound) = openrouter_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(bound)).catch_unwind().await;
    crate::cassettes::checkpoint_attempt(&cassette, "openrouter", spec.scenario()).await;
    cassette.finish_after_test(result).await;
}

pub(super) async fn with_openrouter_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(BoundOpenRouter) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let (cassette, bound) = openrouter_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(bound)).catch_unwind().await;
    cassette.finish_after_test_result(result).await
}

pub(super) async fn with_openrouter_openai_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(BoundOpenRouterResponses) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, bound) = openrouter_openai_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(bound)).catch_unwind().await;
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
    F: FnOnce(BoundOpenRouter) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "openrouter",
        spec,
        OPENROUTER_BASE_URL,
    )
    .await;
    // The rejected credential is this wrapper's subject.
    cassette.expect_account_failure(crate::cassettes::AccountFailure::Auth);
    let bound = OpenAI::with_key(&OPENROUTER, "sk-invalid-edge-matrix-key")
        .with_base_url(cassette.base_url())
        .bound()
        .expect("OpenRouter transport should build");
    let result = AssertUnwindSafe(test_body(bound)).catch_unwind().await;
    cassette.finish_after_test_result(result).await
}

/// Refusal edge matrix (`refusal_matrix/*`): a structured-output refusal
/// arrives as a sibling of `content`, and the chat decoder must not drop it.
/// Its own wrapper keeps the matrix auditable as one unit.
pub(super) async fn with_openrouter_refusal_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(BoundOpenRouter) -> Fut,
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
    F: FnOnce(BoundOpenRouter) -> Fut,
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
    F: FnOnce(BoundOpenRouter) -> Fut,
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
    F: FnOnce(BoundOpenRouter) -> Fut,
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
    F: FnOnce(BoundOpenRouter) -> Fut,
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
    F: FnOnce(BoundOpenRouter) -> Fut,
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
    F: FnOnce(BoundOpenRouter) -> Fut,
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
    F: FnOnce(BoundOpenRouter) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    with_openrouter_cassette_result(spec, test_body).await
}

/// Cassette wrapper for the openrouter prompt-caching matrix
/// (`crates/rig-cassette/fixtures/cassettes/openrouter/prompt_caching/`).
///
/// Delegates to [`with_openrouter_cassette`] — the behavior is identical, and deliberately shared
/// so the two cannot drift apart when the base wrapper gains policy. What the
/// separate name buys is a per-suite entry in the cassette-safety registry, so
/// the cache fixtures are auditable as one concern's evidence.
pub(super) async fn with_openrouter_prompt_caching_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(BoundOpenRouter) -> Fut,
    Fut: Future<Output = ()>,
{
    with_openrouter_cassette(spec, test_body).await;
}
