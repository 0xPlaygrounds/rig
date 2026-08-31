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

/// Compare a generated token (response id, system fingerprint, request id)
/// observed by a test with the value its fixture holds.
///
/// Fixtures are placeholder-scrubbed (`chatcmpl-REDACTED_1`, `fp_REDACTED_1`,
/// `req_REDACTED_1`), so on the recording pass the live token cannot equal
/// the fixture's; both are then required to be present and non-empty. On
/// replay the harness serves the scrubbed bytes back, so equality is exact —
/// which is what CI runs. Presence must agree in both modes.
pub(super) fn assert_matches_recorded_token(
    actual: Option<&str>,
    recorded: Option<&str>,
    context: &str,
) {
    match crate::cassettes::CassetteMode::current() {
        crate::cassettes::CassetteMode::Replay => {
            assert_eq!(
                actual, recorded,
                "{context}: replay serves the fixture's token back"
            );
        }
        crate::cassettes::CassetteMode::Record => {
            assert_eq!(
                actual.is_some(),
                recorded.is_some(),
                "{context}: live and recorded token presence must agree"
            );
            if let (Some(actual), Some(recorded)) = (actual, recorded) {
                assert!(
                    !actual.trim().is_empty() && !recorded.trim().is_empty(),
                    "{context}: live and recorded token must both be non-empty"
                );
            }
        }
    }
}

/// Cassette wrapper for the openrouter prompt-caching matrix
/// (`tests/cassettes/openrouter/prompt_caching/`).
///
/// Delegates to [`with_openrouter_cassette`] — the behavior is identical, and deliberately shared
/// so the two cannot drift apart when the base wrapper gains policy. What the
/// separate name buys is a per-suite entry in the cassette-safety registry, so
/// the cache fixtures are auditable as one concern's evidence.
pub(super) async fn with_openrouter_prompt_caching_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(openrouter::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    with_openrouter_cassette(spec, test_body).await;
}
