use futures::FutureExt;
use rig::providers::ollama;
use std::future::Future;
use std::panic::AssertUnwindSafe;

use crate::cassettes::{CassetteSpec, ProviderCassette};

/// Connection data for a running Ollama cassette server.
///
/// Providers are data now: the cassette hands the test its base URL and each
/// test mints a `functions::Config` for the model it exercises. Ollama needs
/// no API key.
pub(super) struct OllamaCassetteEnv {
    base_url: String,
}

impl OllamaCassetteEnv {
    pub(super) fn config(&self, model: &str) -> ollama::functions::Config {
        ollama::functions::Config::new(model).with_base_url(&self.base_url)
    }

    #[allow(dead_code)]
    pub(super) fn embedding_config(&self, model: &str) -> ollama::functions::EmbeddingConfig {
        ollama::functions::EmbeddingConfig::new(model).with_base_url(&self.base_url)
    }
}

/// Start an Ollama cassette and describe the server pointed at it.
///
/// Replays by default; set `RIG_PROVIDER_TEST_MODE=record` (with a local Ollama
/// server on http://localhost:11434) to record.
async fn ollama_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, OllamaCassetteEnv) {
    let cassette = ProviderCassette::start("ollama", spec, "http://localhost:11434").await;
    let env = OllamaCassetteEnv {
        base_url: cassette.base_url(),
    };

    (cassette, env)
}

pub(super) async fn with_ollama_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(OllamaCassetteEnv) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, env) = ollama_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(env)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}
