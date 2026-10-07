use rig::providers::openai::wire::{OPENROUTER, OpenAIConfig, Route};
use rig_test_support::cassette_models::OpenAiModels;
use std::future::Future;
use std::panic::AssertUnwindSafe;

use crate::cassettes::{CassetteSpec, ProviderCassette};
use futures::FutureExt;

const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";

async fn openrouter_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, OpenAIConfig) {
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "openrouter",
        spec,
        OPENROUTER_BASE_URL,
    )
    .await;
    let bound = OpenAIConfig::with_key(&OPENROUTER, cassette.api_key("OPENROUTER_API_KEY"))
        .with_base_url(cassette.base_url());

    (cassette, bound)
}

async fn openrouter_openai_cassette(
    spec: impl Into<CassetteSpec>,
) -> (ProviderCassette, OpenAIConfig) {
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "openrouter",
        spec,
        OPENROUTER_BASE_URL,
    )
    .await;
    let bound = OpenAIConfig::with_key(&OPENROUTER, cassette.api_key("OPENROUTER_API_KEY"))
        .with_base_url(cassette.base_url())
        .with_route(Route::Responses);

    (cassette, bound)
}

pub(super) async fn with_openrouter_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(OpenAiModels) -> Fut,
    Fut: Future<Output = ()>,
{
    let spec = spec.into();
    let (cassette, bound) = openrouter_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(OpenAiModels::new(
        bound,
        rig_test_support::cassettes::local_http(),
    )))
    .catch_unwind()
    .await;
    crate::cassettes::checkpoint_attempt(&cassette, "openrouter", spec.scenario()).await;
    cassette.finish_after_test(result).await;
}

/// [`with_openrouter_cassette`], with `check` run on what the session
/// recorded before the recording is written (`cassettes::finish_checked`).
pub(super) async fn with_openrouter_checked_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
    check: impl FnOnce(&std::path::Path, &str),
) where
    F: FnOnce(OpenAiModels) -> Fut,
    Fut: Future<Output = ()>,
{
    let spec = spec.into();
    let (cassette, bound) = openrouter_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(OpenAiModels::new(
        bound,
        rig_test_support::cassettes::local_http(),
    )))
    .catch_unwind()
    .await;
    crate::cassettes::checkpoint_attempt(&cassette, "openrouter", spec.scenario()).await;
    crate::cassettes::finish_checked(cassette, "openrouter", spec.scenario(), result, check).await;
}

pub(super) async fn with_openrouter_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(OpenAiModels) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let (cassette, bound) = openrouter_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(OpenAiModels::new(
        bound,
        rig_test_support::cassettes::local_http(),
    )))
    .catch_unwind()
    .await;
    cassette.finish_after_test_result(result).await
}

pub(super) async fn with_openrouter_openai_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(OpenAiModels) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, bound) = openrouter_openai_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(OpenAiModels::new(
        bound,
        rig_test_support::cassettes::local_http(),
    )))
    .catch_unwind()
    .await;
    cassette.finish_after_test(result).await;
}

/// Refusal edge matrix (`refusal_matrix/*`): a structured-output refusal
/// arrives as a sibling of `content`, and the chat decoder must not drop it.
/// Its own wrapper keeps the matrix auditable as one unit.
pub(super) async fn with_openrouter_refusal_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(OpenAiModels) -> Fut,
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
    F: FnOnce(OpenAiModels) -> Fut,
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
    F: FnOnce(OpenAiModels) -> Fut,
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
    F: FnOnce(OpenAiModels) -> Fut,
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
    F: FnOnce(OpenAiModels) -> Fut,
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
    F: FnOnce(OpenAiModels) -> Fut,
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
    F: FnOnce(OpenAiModels) -> Fut,
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
    F: FnOnce(OpenAiModels) -> Fut,
    Fut: Future<Output = ()>,
{
    with_openrouter_cassette(spec, test_body).await;
}

/// `options` as the typed provider options of an OpenRouter request.
pub(super) fn openrouter_options(
    options: rig::providers::openrouter::extension::OpenRouterOptions,
) -> rig::completion::ProviderOptions {
    rig::completion::ProviderOptions::new().set(options)
}

/// Provider preferences that pin `providers`, in order, with no fallback.
pub(super) fn pinned_order(providers: &[&str]) -> rig::completion::ProviderOptions {
    openrouter_options(
        rig::providers::openrouter::extension::OpenRouterOptions::new().provider(
            rig::providers::openrouter::extension::ProviderPreferences::new()
                .order(providers.iter().copied())
                .allow_fallbacks(false),
        ),
    )
}
