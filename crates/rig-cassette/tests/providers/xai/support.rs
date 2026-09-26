use futures::FutureExt;
use rig::driver::Bound;
use rig::prelude::*;
use rig::providers::openai::OpenAI;
use rig::providers::xai;
use std::future::Future;
use std::panic::AssertUnwindSafe;

use crate::cassettes::{CassetteSpec, ProviderCassette};

/// Sends `"store": false` on every Responses request that does not choose
/// `store` itself. xAI stores a Responses reply by default and offers no way
/// to list what it stored, so a recording would otherwise leave responses
/// nobody can find to delete. A cell that needs stored state (a chain that
/// deletes it again) sets `store` explicitly and is left alone. The body is
/// rewritten before it is sent, so the cassette records what xAI received.
struct StorelessResponses;

impl rig::http_client::HttpMiddleware for StorelessResponses {
    fn before_request_body<'a>(
        &'a self,
        method: &'a rig::http_client::Method,
        uri: &'a rig::http_client::Uri,
        _headers: &'a rig::http_client::HeaderMap,
        body: bytes::Bytes,
    ) -> rig::wasm_compat::WasmBoxedFuture<'a, rig::http_client::Result<bytes::Bytes>> {
        Box::pin(async move {
            if method != rig::http_client::Method::POST || !uri.path().ends_with("/responses") {
                return Ok(body);
            }
            let Ok(serde_json::Value::Object(mut request)) = serde_json::from_slice(&body) else {
                return Ok(body);
            };
            if request.contains_key("store") {
                return Ok(body);
            }
            request.insert("store".to_owned(), serde_json::Value::Bool(false));
            Ok(serde_json::to_vec(&request).map_or(body, bytes::Bytes::from))
        })
    }
}

async fn xai_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, Bound<OpenAI>) {
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "xai",
        spec,
        "https://api.x.ai",
    )
    .await;
    let client = OpenAI::with_key(&xai::DIALECT, cassette.api_key("XAI_API_KEY"))
        .with_base_url(cassette.base_url())
        .bind(
            rig::http_client::ReqwestClient::default()
                .boxed()
                .with_middleware(StorelessResponses),
        );

    (cassette, client)
}

pub(super) async fn with_xai_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(Bound<OpenAI>) -> Fut,
    Fut: Future<Output = ()>,
{
    let spec = spec.into();
    let (cassette, client) = xai_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    crate::cassettes::checkpoint_attempt(&cassette, "xai", spec.scenario()).await;
    cassette.finish_after_test(result).await;
}

pub(super) async fn with_xai_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(Bound<OpenAI>) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let (cassette, client) = xai_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test_result(result).await
}

/// Bogus-key variant for recording real 401s (rig#2314 error matrix).
pub(super) async fn with_xai_cassette_bogus_key<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(Bound<OpenAI>) -> Fut,
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
    let client = OpenAI::with_key(&xai::DIALECT, "xai-invalid-edge-matrix-key")
        .with_base_url(cassette.base_url())
        .bound()
        .expect("xAI client should build");
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

/// Cassette wrapper for the xai prompt-caching matrix
/// (`crates/rig-cassette/fixtures/cassettes/xai/prompt_caching/`).
///
/// Builds the cassette directly rather than delegating to [`with_xai_cassette`]: this
/// provider's cassette-safety `source_dir` covers `support.rs` itself, and the
/// scan requires every call to a *registered* wrapper to pass a string-literal
/// scenario. A delegating wrapper passes its `spec` variable through, which the
/// scan reports as an unregistered scenario. The duplication is three lines and
/// the alternative is an unscannable suite.
pub(super) async fn with_xai_prompt_caching_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(Bound<OpenAI>) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = xai_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}
