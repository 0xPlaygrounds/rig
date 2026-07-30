//! Cassette helpers for llamafile provider tests.
//!
//! Replays by default. Set `RIG_PROVIDER_TEST_MODE=record` to record against a
//! local OpenAI-compatible llama.cpp-family server. Recording defaults to
//! Ollama's OpenAI-compatible endpoint (`http://localhost:11434`); set
//! `LLAMAFILE_CASSETTE_UPSTREAM` to record against a different server. The
//! committed chat cassettes were recorded against an actual llama.cpp
//! `llama-server` (`brew install llama.cpp`, `llama-server -m <gguf> --jinja`,
//! then `LLAMAFILE_CASSETTE_UPSTREAM=http://localhost:<port>`); the embeddings
//! cassette was recorded against Ollama.

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

/// Connection data for a running llamafile cassette server.
///
/// Providers are data now: the cassette hands the test its base URL and each
/// test mints a `functions::Config` for the model it exercises. The classic
/// `llamafile::Client::from_url` appended `/v1` itself, so the recorded paths
/// are `/v1/...`; that suffix is applied here to keep the wire URLs identical.
pub(super) struct LlamafileCassetteEnv {
    base_url: String,
}

impl LlamafileCassetteEnv {
    pub(super) fn config(&self, model: &str) -> llamafile::functions::Config {
        llamafile::functions::Config::new(model).with_base_url(&self.base_url)
    }

    pub(super) fn embedding_config(&self, model: &str) -> llamafile::functions::EmbeddingConfig {
        llamafile::functions::EmbeddingConfig::new(model).with_base_url(&self.base_url)
    }
}

async fn llamafile_cassette(
    spec: impl Into<CassetteSpec>,
) -> (ProviderCassette, LlamafileCassetteEnv) {
    let cassette = ProviderCassette::start("llamafile", spec, &record_upstream()).await;
    let env = LlamafileCassetteEnv {
        base_url: format!("{}/v1", cassette.base_url().trim_end_matches('/')),
    };

    (cassette, env)
}

pub(super) async fn with_llamafile_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(LlamafileCassetteEnv) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, env) = llamafile_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(env)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}
