//! Cassette helpers for the llama.cpp suite.
//!
//! Replays by default. Set `RIG_PROVIDER_TEST_MODE=record` to record against a
//! local `llama-server`; `LLAMACPP_CASSETTE_UPSTREAM` overrides the default
//! `http://localhost:8080`.
//!
//! These tests drive llama.cpp through rig's **generic OpenAI-compatible
//! client** rather than through `providers::llamafile`. That is deliberate and
//! it is the point of the suite: the two paths differ (llamafile overrides
//! `build_uri` to append `/v1` and cannot send an `Authorization` header at
//! all), so a defect in one is invisible to the other. The llamafile suite
//! covers the provider; this one covers the path a caller takes when they point
//! `openai::Client` at their own server.
//!
//! # Determinism
//!
//! A fixture recorded from a local server is reproducible only if the
//! generation is pinned. Every cassette here was recorded against
//! `unsloth/Qwen3-1.7B-GGUF` Q4_K_M, launched as:
//!
//! ```text
//! llama-server -m <Qwen3-1.7B-Q4_K_M.gguf> --host 127.0.0.1 --port 8080 \
//!     --jinja --seed 42 --temp 0 -c 4096
//! ```
//!
//! What a replayed fixture proves is rig's request shape and rig's decoding of
//! a real response — not that this model answers the same way on another
//! machine. Assertions therefore key on structure (a tool call was requested,
//! with this name and these arguments) rather than on prose.

use futures::FutureExt;
use rig::providers::openai;
use std::future::Future;
use std::panic::AssertUnwindSafe;

use crate::cassettes::{CassetteSpec, ProviderCassette};

/// The model the recorded cassettes were made against.
pub(super) const CASSETTE_MODEL: &str = "Qwen3-1.7B-Q4_K_M";

fn record_upstream() -> String {
    std::env::var("LLAMACPP_CASSETTE_UPSTREAM")
        .unwrap_or_else(|_| "http://localhost:8080".to_string())
}

async fn llamacpp_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, openai::Client) {
    let upstream = format!("{}/v1", record_upstream().trim_end_matches('/'));
    let cassette = ProviderCassette::start("llamacpp", spec, &upstream).await;
    // llama.cpp accepts any bearer token (and needs none unless the server was
    // started with `--api-key`), so a literal placeholder keeps record and
    // replay identical and puts no real credential anywhere near a fixture.
    let client = openai::Client::builder()
        .api_key("llamacpp-local")
        .base_url(cassette.base_url())
        .build()
        .expect("client should build");

    (cassette, client)
}

/// Drive a scenario through the Responses-API client.
pub(super) async fn with_llamacpp_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(openai::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = llamacpp_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

/// Drive a scenario through the Chat Completions client, which is what almost
/// every cell here uses — it is the surface llama.cpp is usually driven on.
pub(super) async fn with_llamacpp_completions_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(openai::CompletionsClient) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = llamacpp_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client.completions_api()))
        .catch_unwind()
        .await;
    cassette.finish_after_test(result).await;
}

/// [`with_llamacpp_completions_cassette`] for a cell whose body returns
/// `Result`.
pub(super) async fn with_llamacpp_completions_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(openai::CompletionsClient) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let (cassette, client) = llamacpp_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client.completions_api()))
        .catch_unwind()
        .await;
    cassette.finish_after_test_result(result).await
}
