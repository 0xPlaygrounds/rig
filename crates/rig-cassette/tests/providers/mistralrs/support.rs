use std::future::Future;
use std::panic::AssertUnwindSafe;

use futures::FutureExt;
use rig::driver::Bound;
use rig::http_client::BoxedHttpClient;
use rig::prelude::*;
use rig::providers::openai::wire::{OpenAI, Route};

use crate::cassettes::{CassetteSpec, ProviderCassette};

pub(super) const DEFAULT_BASE_URL: &str = "http://127.0.0.1:1234/v1";
pub(super) const DEFAULT_API_KEY: &str = "local";
pub(super) const DEFAULT_MODEL: &str = "Qwen/Qwen3-4B";
pub(super) const SYSTEM_PROMPT: &str =
    "You are concise. Include a few details so streaming is visible.";

/// mistral.rs's Responses surface, bound to the bundled transport.
///
/// mistral.rs is a local server rather than a hosted provider, so it has no
/// dialect of its own: it speaks the plain OpenAI format at a base URL the
/// caller supplies. Both surfaces are therefore the `OPENAI` dialect with an
/// explicit base URL.
pub(super) type BoundResponses = Bound<OpenAI, BoxedHttpClient>;

/// mistral.rs's chat-completions surface, bound to the bundled transport:
/// the same dialect routed to `/chat/completions` once, so a cell's
/// `client.agent(model)` lands there.
pub(super) type BoundCompletions = Bound<OpenAI, BoxedHttpClient>;

pub(super) fn model_name() -> String {
    std::env::var("MISTRALRS_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string())
}

/// The local server's address and credential, from the environment when the
/// suite is recorded against a running mistral.rs.
fn server() -> (String, String) {
    (
        std::env::var("MISTRALRS_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_string()),
        std::env::var("MISTRALRS_API_KEY").unwrap_or_else(|_| DEFAULT_API_KEY.to_string()),
    )
}

async fn mistralrs_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, BoundResponses) {
    let (real_base_url, api_key) = server();
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "mistralrs",
        spec,
        &real_base_url,
    )
    .await;
    let responses = OpenAI::new(api_key)
        .with_base_url(cassette.base_url())
        .bound()
        .expect("mistral.rs Responses transport should build");

    (cassette, responses)
}

async fn mistralrs_completions_cassette(
    spec: impl Into<CassetteSpec>,
) -> (ProviderCassette, BoundCompletions) {
    let (real_base_url, api_key) = server();
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "mistralrs",
        spec,
        &real_base_url,
    )
    .await;
    let completions = OpenAI::new(api_key)
        .with_base_url(cassette.base_url())
        .with_route(Route::Chat)
        .bound()
        .expect("mistral.rs chat-completions transport should build");

    (cassette, completions)
}

async fn mistralrs_raw_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, String) {
    let real_base_url =
        std::env::var("MISTRALRS_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_string());
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "mistralrs",
        spec,
        &real_base_url,
    )
    .await;
    let base_url = cassette.base_url();
    (cassette, base_url)
}

pub(super) async fn with_mistralrs_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(BoundResponses) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, responses) = mistralrs_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(responses)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

/// The chat-completions surface of the same server.
///
/// The two endpoints are two configurations of the same dialect rather than
/// two faces of one object, so this builds its own instead of crossing over
/// from the Responses one.
pub(super) async fn with_mistralrs_completions_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(BoundCompletions) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, completions) = mistralrs_completions_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(completions))
        .catch_unwind()
        .await;
    cassette.finish_after_test(result).await;
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
