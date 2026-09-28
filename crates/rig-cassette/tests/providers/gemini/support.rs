use futures::FutureExt;
use rig::providers::gemini::GeminiConfig;
use rig_test_support::cassette_models::GeminiModels;
use std::future::Future;
use std::panic::AssertUnwindSafe;

use crate::cassettes::{CassetteSpec, ProviderCassette};

async fn gemini_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, GeminiConfig) {
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "gemini",
        spec,
        "https://generativelanguage.googleapis.com",
    )
    .await;
    let gemini =
        GeminiConfig::new(cassette.api_key("GEMINI_API_KEY")).with_base_url(cassette.base_url());

    (cassette, gemini)
}

/// The effect corpus's provider-breadth matrix (Matrix N):
/// `crates/rig-cassette/fixtures/cassettes/gemini/corpus_breadth/`.
pub(super) async fn with_gemini_corpus_breadth_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(GeminiModels) -> Fut,
    Fut: Future<Output = ()>,
{
    with_gemini_cassette(spec, test_body).await;
}

/// The effect corpus's delta-wire matrix (Matrix K):
/// `crates/rig-cassette/fixtures/cassettes/gemini/corpus_delta/`.
pub(super) async fn with_gemini_corpus_delta_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(GeminiModels) -> Fut,
    Fut: Future<Output = ()>,
{
    with_gemini_cassette(spec, test_body).await;
}

/// The effect corpus's retrieval matrix (Matrix A):
/// `crates/rig-cassette/fixtures/cassettes/gemini/corpus_retrieval/`.
pub(super) async fn with_gemini_corpus_retrieval_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(GeminiModels) -> Fut,
    Fut: Future<Output = ()>,
{
    with_gemini_cassette(spec, test_body).await;
}

pub(super) async fn with_gemini_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(GeminiModels) -> Fut,
    Fut: Future<Output = ()>,
{
    let spec = spec.into();
    let (cassette, client) = gemini_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(GeminiModels::new(
        client,
        rig::rig_reqwest::shared(),
    )))
    .catch_unwind()
    .await;
    crate::cassettes::checkpoint_attempt(&cassette, "gemini", spec.scenario()).await;
    cassette.finish_after_test(result).await;
}
