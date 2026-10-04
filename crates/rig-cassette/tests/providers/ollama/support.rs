use futures::FutureExt;
use rig::providers::ollama::wire::OllamaConfig;
use rig_test_support::cassette_models::OllamaModels;
use std::future::Future;
use std::panic::AssertUnwindSafe;

use crate::cassettes::{CassetteSpec, ProviderCassette};

/// Start an Ollama cassette and bind a provider pointed at it.
///
/// Replays by default; set `RIG_PROVIDER_TEST_MODE=record` (with a local Ollama
/// server on http://localhost:11434) to record. Ollama needs no API key.
async fn ollama_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, OllamaConfig) {
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "ollama",
        spec,
        "http://localhost:11434",
    )
    .await;
    let ollama = OllamaConfig::new().with_base_url(cassette.base_url());

    (cassette, ollama)
}

pub(super) async fn with_ollama_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(OllamaModels) -> Fut,
    Fut: Future<Output = ()>,
{
    let spec = spec.into();
    let (cassette, client) = ollama_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(OllamaModels::new(
        client,
        rig_test_support::cassettes::local_http(),
    )))
    .catch_unwind()
    .await;
    crate::cassettes::checkpoint_attempt(&cassette, "ollama", spec.scenario()).await;
    cassette.finish_after_test(result).await;
}
