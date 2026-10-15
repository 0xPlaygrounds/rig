use futures::FutureExt;
use rig::providers::openai::OpenAIConfig;
use rig::providers::xai;
use rig_test_support::cassette_models::OpenAiModels;
use std::future::Future;
use std::panic::AssertUnwindSafe;

use crate::cassettes::{CassetteSpec, ProviderCassette};

/// Sends every Responses request with `store: false` unless it sets `store`
/// itself. xAI stores a reply by default and lists none, so a recording would
/// otherwise leave responses nobody can find to delete. The body is rewritten
/// before it is sent, so a fixture holds what xAI received.
struct Unstored;

impl rig::http_client::HttpMiddleware for Unstored {
    fn before_request_body<'a>(
        &'a self,
        exchange: rig::http_client::Exchange<'a>,
        _headers: &'a rig::http_client::HeaderMap,
        body: rig::http_client::Bytes,
    ) -> rig::wasm_compat::WasmBoxedFuture<'a, rig::http_client::Result<rig::http_client::Bytes>>
    {
        Box::pin(async move {
            if exchange.method != rig::http_client::Method::POST
                || !exchange.uri.path().ends_with("/responses")
            {
                return Ok(body);
            }
            let Ok(serde_json::Value::Object(mut fields)) = serde_json::from_slice(&body) else {
                return Ok(body);
            };
            if fields.contains_key("store") {
                return Ok(body);
            }
            fields.insert("store".to_owned(), false.into());
            Ok(serde_json::to_vec(&fields).map_or(body, Into::into))
        })
    }
}

/// The transport every xAI cassette sends through.
fn http() -> rig::http_client::DynHttpClient {
    rig_test_support::cassettes::local_http().with_middleware(Unstored)
}

async fn xai_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, OpenAIConfig) {
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "xai",
        spec,
        "https://api.x.ai",
    )
    .await;
    let client = OpenAIConfig::with_key(&xai::DIALECT, cassette.api_key("XAI_API_KEY"))
        .with_base_url(cassette.base_url());

    (cassette, client)
}

pub(super) async fn with_xai_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(OpenAiModels) -> Fut,
    Fut: Future<Output = ()>,
{
    let spec = spec.into();
    let (cassette, client) = xai_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(OpenAiModels::new(client, http())))
        .catch_unwind()
        .await;
    crate::cassettes::checkpoint_attempt(&cassette, "xai", spec.scenario()).await;
    cassette.finish_after_test(result).await;
}

pub(super) async fn with_xai_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(OpenAiModels) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let (cassette, client) = xai_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(OpenAiModels::new(client, http())))
        .catch_unwind()
        .await;
    cassette.finish_after_test_result(result).await
}

/// Bogus-key variant for recording real 401s (rig#2314 error matrix).
pub(super) async fn with_xai_cassette_bogus_key<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(OpenAiModels) -> Fut,
    Fut: Future<Output = ()>,
{
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "xai",
        spec,
        "https://api.x.ai",
    )
    .await;
    // The rejected credential is this wrapper's subject.
    cassette.expect_account_failure(crate::cassettes::AccountFailure::Auth);
    let client = OpenAIConfig::with_key(&xai::DIALECT, "xai-invalid-edge-matrix-key")
        .with_base_url(cassette.base_url());
    let result = AssertUnwindSafe(test_body(OpenAiModels::new(client, http())))
        .catch_unwind()
        .await;
    cassette.finish_after_test(result).await;
}

/// Cassette wrapper for the xai prompt-caching matrix
/// (`crates/rig-cassette/fixtures/cassettes/xai/prompt_caching/`).
pub(super) async fn with_xai_prompt_caching_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(OpenAiModels) -> Fut,
    Fut: Future<Output = ()>,
{
    with_xai_cassette(spec, test_body).await;
}
