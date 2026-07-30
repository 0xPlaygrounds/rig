use futures::FutureExt;
use rig::AgentBuilder;
use rig::completion::{CompletionError, CompletionRequest, CompletionResponse};
use rig::http_runtime::HttpRuntime;
use rig::provider::ProviderConfig;
use rig::providers::gemini;
use rig::streaming::StreamingCompletionResponse;
use std::future::Future;
use std::panic::AssertUnwindSafe;

use crate::cassettes::{CassetteSpec, ProviderCassette};

/// Connection details for a running Gemini cassette proxy.
///
/// Replaces the deleted `gemini::Client`: tests mint a plain
/// [`gemini::functions::Config`] (or an [`AgentBuilder`]) per model, pointed at
/// the cassette's base URL and API key.
pub(super) struct GeminiCassette {
    api_key: String,
    base_url: String,
}

#[allow(dead_code)]
impl GeminiCassette {
    /// Completion config for `model` aimed at the cassette proxy.
    pub(crate) fn config(&self, model: impl Into<String>) -> gemini::functions::Config {
        gemini::functions::Config::new(model)
            .with_api_key(self.api_key.clone())
            .with_base_url(self.base_url.clone())
    }

    /// Embedding config for `model` aimed at the cassette proxy.
    ///
    /// The recordings were made through the deleted `gemini::EmbeddingModel`,
    /// which stamped the model's documented default dimensionality into every
    /// entry as `output_dimensionality`. `EmbeddingConfig::new` leaves
    /// `dimensions` unset (letting Gemini apply the same default server-side),
    /// which is equivalent on the wire *response* but not byte-identical on the
    /// *request* — so apply the lookup explicitly to keep replay exact.
    pub(crate) fn embedding_config(
        &self,
        model: impl Into<String>,
    ) -> gemini::functions::EmbeddingConfig {
        let model = model.into();
        let dimensions = gemini::embedding::model_default_ndims(&model);
        let cfg = gemini::functions::EmbeddingConfig::new(model)
            .with_api_key(self.api_key.clone())
            .with_base_url(self.base_url.clone());
        match dimensions {
            Some(dimensions) => cfg.with_dimensions(dimensions),
            None => cfg,
        }
    }

    /// An [`AgentBuilder`] for `model` aimed at the cassette proxy.
    pub(crate) fn agent(&self, model: impl Into<String>) -> AgentBuilder {
        AgentBuilder::new(ProviderConfig::Gemini(self.config(model)))
    }

    /// The agent-facing provider selection for `model`.
    pub(crate) fn provider_config(&self, model: impl Into<String>) -> ProviderConfig {
        ProviderConfig::Gemini(self.config(model))
    }

    /// A real-HTTP runtime — the cassette proxy is a live local server.
    pub(crate) fn http(&self) -> HttpRuntime {
        HttpRuntime::new()
    }

    /// One non-streamed completion against `model` — the raw-model harness
    /// that replaced `client.completion_model(m).completion(req)`.
    pub(crate) async fn complete(
        &self,
        model: &str,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, CompletionError> {
        gemini::functions::complete(&self.config(model), &self.http(), request).await
    }

    /// One streamed completion against `model` — replaced
    /// `client.completion_model(m).stream(req)`.
    pub(crate) async fn stream(
        &self,
        model: &str,
        request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse, CompletionError> {
        gemini::functions::open_stream(&self.config(model), &self.http(), request).await
    }
}

async fn gemini_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, GeminiCassette) {
    let cassette =
        ProviderCassette::start("gemini", spec, "https://generativelanguage.googleapis.com").await;
    let handle = GeminiCassette {
        api_key: cassette.api_key("GEMINI_API_KEY"),
        base_url: cassette.base_url(),
    };

    (cassette, handle)
}

pub(super) async fn with_gemini_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(GeminiCassette) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, handle) = gemini_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(handle)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}
