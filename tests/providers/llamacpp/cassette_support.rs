//! Cassette helpers for the llama.cpp suite.
//!
//! Replays by default. Set `RIG_PROVIDER_TEST_MODE=record` to record against a
//! local `llama-server`; `LLAMACPP_CASSETTE_UPSTREAM` overrides the default
//! `http://localhost:8080`, and the per-configuration variables below point
//! individual matrices at the differently-launched servers they need.
//!
//! # One suite, one provider
//!
//! This suite drives `providers::llamacpp` — the provider a user reaches for.
//! It is the merge of two suites that previously recorded the same scenarios
//! twice, once through a bare `openai::Client` and once through the
//! now-deleted `providers::llamafile`; 19 of the 61 fixtures were duplicates
//! and the provider path is the copy that survived.
//!
//! The bare-`openai::Client`-against-a-local-server path is still covered, by
//! [`with_llamacpp_bare_openai_cassette`] and the cells in
//! `cassette/bare_openai_client.rs` — deliberately small. It exists to pin
//! what genuinely differs between the two paths (base-URL composition, the
//! `Authorization` header, the absence of this provider's associated consts),
//! not to re-record the generation matrix. Keep it that way.
//!
//! # Determinism
//!
//! A fixture recorded from a local server is reproducible only if the
//! generation is pinned. Unless a cell's own doc comment says otherwise, every
//! cassette here was recorded against **`unsloth/Qwen3-1.7B-GGUF` Q4_K_M**
//! served by `llama-server` **b10499 (commit 6d05498)**, launched as:
//!
//! ```text
//! llama-server -m <Qwen3-1.7B-Q4_K_M.gguf> --host 127.0.0.1 --port 8080 \
//!     --jinja --seed 42 --temp 0 -c 4096
//! ```
//!
//! Cells that need a different server configuration or a different model say
//! so in their own module documentation and use the matching wrapper; the
//! table of every server invocation this corpus was recorded under lives in
//! `cassette/mod.rs`.
//!
//! What a replayed fixture proves is rig's request shape and rig's decoding of
//! a real response — not that this model answers the same way on another
//! machine. Assertions therefore key on structure (a tool call was requested,
//! with this name and these arguments) rather than on prose.

use futures::FutureExt;
use rig::providers::{llamacpp, openai};
use std::future::Future;
use std::panic::AssertUnwindSafe;

use crate::cassettes::{CassetteSpec, ProviderCassette};

/// The chat model the recorded cassettes were made against.
pub(super) const CASSETTE_MODEL: &str = "Qwen3-1.7B-Q4_K_M";

/// The embedding model the recorded embeddings cassettes were made against.
///
/// A real embedding model, not a causal LM pooled with `--pooling mean`: the
/// server for these cells loads `Qwen/Qwen3-Embedding-0.6B-GGUF`.
pub(super) const CASSETTE_EMBEDDING_MODEL: &str = "Qwen3-Embedding-0.6B-Q8_0";

/// The reranker the `/v1/rerank` cassettes were made against.
///
/// A cross-encoder is required — a causal LM cannot be served with
/// `--pooling rank` at all.
pub(super) const CASSETTE_RERANK_MODEL: &str = "bge-reranker-v2-m3-Q4_K_M";

/// The vision model the multimodal cassettes were made against.
pub(super) const CASSETTE_VISION_MODEL: &str = "Qwen3-VL-2B-Instruct-Q8_0";

/// The recording upstream for one server configuration.
///
/// Each configuration is a separately launched `llama-server`, so recording is
/// batched per configuration rather than per test file. `var` names the
/// environment variable that overrides it and `port` is the default this
/// repository's recording script binds that configuration to.
fn upstream(var: &str, port: u16) -> String {
    std::env::var(var).unwrap_or_else(|_| format!("http://localhost:{port}"))
}

fn record_upstream() -> String {
    // `LLAMACPP_CASSETTE_UPSTREAM` keeps its historical meaning: the default
    // `--jinja --seed 42 --temp 0 -c 4096` server.
    upstream("LLAMACPP_CASSETTE_UPSTREAM", 8080)
}

async fn llamacpp_cassette_on(
    spec: impl Into<CassetteSpec>,
    upstream: &str,
) -> (ProviderCassette, llamacpp::Client) {
    let cassette = ProviderCassette::start("llamacpp", spec, upstream).await;
    // No credential: `llama-server` needs none unless started with
    // `--api-key`, and the provider's default is a genuinely absent header
    // rather than a placeholder one. The `--api-key` half is pinned by
    // `cassette/error_matrix.rs`, which launches a server that requires it.
    let client = llamacpp::Client::from_url(&cassette.base_url()).expect("client should build");

    (cassette, client)
}

/// Drive a scenario against the default recording server.
pub(super) async fn with_llamacpp_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(llamacpp::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = llamacpp_cassette_on(spec, &record_upstream()).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

/// [`with_llamacpp_cassette`] for a cell whose body returns `Result`.
pub(super) async fn with_llamacpp_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(llamacpp::Client) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let (cassette, client) = llamacpp_cassette_on(spec, &record_upstream()).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test_result(result).await
}

/// Build one wrapper per differently-launched `llama-server`.
///
/// Each configuration is its own recording batch, and giving each its own
/// wrapper name is what makes the cassette-safety registry able to say which
/// fixtures came from which server rather than treating the corpus as one
/// undifferentiated pile.
macro_rules! server_config_wrapper {
    ($(#[$meta:meta])* $name:ident, $var:literal, $port:literal) => {
        $(#[$meta])*
        pub(super) async fn $name<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
        where
            F: FnOnce(llamacpp::Client) -> Fut,
            Fut: Future<Output = ()>,
        {
            let (cassette, client) = llamacpp_cassette_on(spec, &upstream($var, $port)).await;
            let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
            cassette.finish_after_test(result).await;
        }
    };
}

server_config_wrapper!(
    /// `--embeddings --pooling mean` on a real embedding model
    /// (`Qwen/Qwen3-Embedding-0.6B-GGUF`).
    with_llamacpp_embeddings_cassette,
    "LLAMACPP_EMBEDDINGS_UPSTREAM",
    8081
);

server_config_wrapper!(
    /// `--mmproj` vision server (`ggml-org/Qwen3-VL-2B-Instruct-GGUF`).
    with_llamacpp_vision_cassette,
    "LLAMACPP_VISION_UPSTREAM",
    8082
);

server_config_wrapper!(
    /// `-c 512`: the smallest context that still loads, so a modest prompt
    /// overflows it deterministically.
    with_llamacpp_small_context_cassette,
    "LLAMACPP_SMALL_CONTEXT_UPSTREAM",
    8083
);

server_config_wrapper!(
    /// `--no-jinja`: the built-in ChatML template instead of the model's own,
    /// which is what a tool request degrades against.
    with_llamacpp_no_jinja_cassette,
    "LLAMACPP_NO_JINJA_UPSTREAM",
    8084
);

server_config_wrapper!(
    /// `--reranking` on a cross-encoder (`gpustack/bge-reranker-v2-m3-GGUF`).
    with_llamacpp_rerank_cassette,
    "LLAMACPP_RERANK_UPSTREAM",
    8085
);

server_config_wrapper!(
    /// `--embeddings --pooling none`, whose per-token output the OpenAI
    /// embeddings wire cannot express.
    with_llamacpp_pooling_none_cassette,
    "LLAMACPP_POOLING_NONE_UPSTREAM",
    8086
);

server_config_wrapper!(
    /// `--embeddings --pooling mean` on a **causal LM** rather than an
    /// embedding model, which is what the pre-merge embeddings cells were
    /// recorded against.
    with_llamacpp_causal_embeddings_cassette,
    "LLAMACPP_CAUSAL_EMBEDDINGS_UPSTREAM",
    8087
);

server_config_wrapper!(
    /// The competent tier (`unsloth/Qwen3-8B-GGUF` Q4_K_M) for cells whose
    /// claim needs a model that can actually hold a schema or orchestrate
    /// several tools.
    with_llamacpp_competent_cassette,
    "LLAMACPP_COMPETENT_UPSTREAM",
    8088
);

/// A server started with `--api-key`, driven by a client that presents the
/// matching key.
///
/// The key is a literal placeholder rather than a credential: it is what the
/// recording server was started with, so record and replay send identical
/// bytes and nothing secret goes near a fixture. `Authorization` is a
/// sensitive header and is not recorded at all.
pub(super) async fn with_llamacpp_api_key_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(llamacpp::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let cassette = ProviderCassette::start(
        "llamacpp",
        spec,
        &upstream("LLAMACPP_API_KEY_UPSTREAM", 8089),
    )
    .await;
    let client = llamacpp::Client::builder()
        .api_key(CASSETTE_API_KEY)
        .base_url(cassette.base_url())
        .build()
        .expect("client should build");
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

/// The same `--api-key` server, driven by a client that presents **no** key.
pub(super) async fn with_llamacpp_missing_api_key_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(llamacpp::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) =
        llamacpp_cassette_on(spec, &upstream("LLAMACPP_API_KEY_UPSTREAM", 8089)).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

/// The key the `--api-key` recording server was started with.
pub(super) const CASSETTE_API_KEY: &str = "llamacpp-local-test-key";

/// Drive a scenario through a **bare `openai::Client`** pointed at the local
/// server, which is what a caller does with any OpenAI-compatible server rig
/// has no provider for.
///
/// Deliberately narrow — see this module's header. The client is given a
/// literal placeholder key because `openai::Client` has no optional-key form
/// and llama.cpp accepts any bearer token when it was not started with
/// `--api-key`; that difference is itself one of the things this path pins.
pub(super) async fn with_llamacpp_bare_openai_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(openai::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let cassette = ProviderCassette::start("llamacpp", spec, &record_upstream()).await;
    let client = openai::Client::builder()
        .api_key("llamacpp-local")
        // Note the `/v1`: a bare `openai::Client` composes paths straight onto
        // its base URL, so the caller supplies the prefix that
        // `llamacpp::Client` supplies for them.
        .base_url(&format!("{}/v1", cassette.base_url().trim_end_matches('/')))
        .build()
        .expect("client should build");
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}
