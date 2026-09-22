use futures::FutureExt;
use rig::driver::Bound;
use rig::http_client::BoxedHttpClient;
use rig::prelude::*;
use rig::providers::ollama::wire::Ollama;
use std::future::Future;
use std::panic::AssertUnwindSafe;

use crate::cassettes::{CassetteSpec, ProviderCassette};

/// The Ollama config bound to the bundled transport — what a cassette test
/// builds its models from, now that a model is a bound wire.
pub(super) type BoundOllama = Bound<Ollama, BoxedHttpClient>;

/// Start an Ollama cassette and bind a provider pointed at it.
///
/// Replays by default; set `RIG_PROVIDER_TEST_MODE=record` (with a local Ollama
/// server on http://localhost:11434) to record. Ollama needs no API key.
async fn ollama_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, BoundOllama) {
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "ollama",
        spec,
        "http://localhost:11434",
    )
    .await;
    let ollama = Ollama::new()
        .with_base_url(cassette.base_url())
        .bound()
        .expect("transport should build");

    (cassette, ollama)
}

pub(super) async fn with_ollama_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(BoundOllama) -> Fut,
    Fut: Future<Output = ()>,
{
    let spec = spec.into();
    let (cassette, client) = ollama_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    crate::cassettes::checkpoint_attempt(&cassette, "ollama", spec.scenario()).await;
    cassette.finish_after_test(result).await;
}
