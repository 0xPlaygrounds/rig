use futures::FutureExt;
use rig::client::Nothing;
use rig::providers::ollama;
use std::future::Future;
use std::panic::AssertUnwindSafe;

use crate::cassettes::{CassetteSpec, ProviderCassette};

/// Start an Ollama cassette and build a client pointed at it.
///
/// Replays by default; set `RIG_PROVIDER_TEST_MODE=record` (with a local Ollama
/// server on http://localhost:11434) to record. Ollama needs no API key.
async fn ollama_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, ollama::Client) {
    let cassette = ProviderCassette::start("ollama", spec, "http://localhost:11434").await;
    let client = ollama::Client::builder()
        .api_key(Nothing)
        .base_url(cassette.base_url())
        .build()
        .expect("client should build");

    (cassette, client)
}

pub(super) async fn with_ollama_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(ollama::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = ollama_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}
