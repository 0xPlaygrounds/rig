//! Cassette helpers for llamafile provider tests.
//!
//! Replays by default. Set `RIG_PROVIDER_TEST_MODE=record` to record against a
//! local OpenAI-compatible llama.cpp-family server. Recording defaults to
//! Ollama's OpenAI-compatible endpoint (`http://localhost:11434`); set
//! `LLAMAFILE_CASSETTE_UPSTREAM` to record against a different server (e.g. an
//! actual llamafile on `http://localhost:8080`).

use futures::FutureExt;
use rig::providers::llamafile;
use std::future::Future;
use std::panic::AssertUnwindSafe;

use crate::cassettes::{CassetteSpec, ProviderCassette};

/// Chat model used by the recorded cassettes.
pub(super) const CASSETTE_CHAT_MODEL: &str = "llama3.2:latest";
/// Embedding model used by the recorded cassettes.
pub(super) const CASSETTE_EMBEDDING_MODEL: &str = "all-minilm:latest";

fn record_upstream() -> String {
    std::env::var("LLAMAFILE_CASSETTE_UPSTREAM")
        .unwrap_or_else(|_| "http://localhost:11434".to_string())
}

async fn llamafile_cassette(
    spec: impl Into<CassetteSpec>,
) -> (ProviderCassette, llamafile::Client) {
    let cassette = ProviderCassette::start("llamafile", spec, &record_upstream()).await;
    let client = llamafile::Client::from_url(&cassette.base_url()).expect("client should build");

    (cassette, client)
}

pub(super) async fn with_llamafile_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(llamafile::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = llamafile_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}
