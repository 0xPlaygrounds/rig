use rig::AgentBuilder;
use rig::http_runtime::HttpRuntime;
use rig::provider::ProviderConfig;
use rig::providers::{openai, openrouter};
use std::future::Future;
use std::panic::AssertUnwindSafe;

use crate::cassettes::{CassetteSpec, ProviderCassette};
use futures::FutureExt;

const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";

/// Connection details for a running OpenRouter cassette proxy.
///
/// Replaces the deleted `openrouter::Client`: tests mint a plain
/// [`openrouter::functions::Config`] (or an [`AgentBuilder`]) per model, pointed
/// at the cassette's base URL and API key.
pub(super) struct OpenRouterCassette {
    api_key: String,
    base_url: String,
}

#[allow(dead_code)]
impl OpenRouterCassette {
    /// Completion config for `model` aimed at the cassette proxy.
    pub(crate) fn config(&self, model: impl Into<String>) -> openrouter::functions::Config {
        openrouter::functions::Config::new(model)
            .with_api_key(self.api_key.clone())
            .with_base_url(self.base_url.clone())
    }

    /// Embedding config for `model` aimed at the cassette proxy.
    pub(crate) fn embedding_config(
        &self,
        model: impl Into<String>,
    ) -> openrouter::functions::EmbeddingConfig {
        openrouter::functions::EmbeddingConfig::new(model)
            .with_api_key(self.api_key.clone())
            .with_base_url(self.base_url.clone())
    }

    /// The agent-facing provider config for `model`.
    pub(crate) fn provider_config(&self, model: impl Into<String>) -> ProviderConfig {
        ProviderConfig::OpenRouter(self.config(model))
    }

    /// An [`AgentBuilder`] for `model` aimed at the cassette proxy.
    pub(crate) fn agent(&self, model: impl Into<String>) -> AgentBuilder {
        AgentBuilder::new(self.provider_config(model))
    }

    /// A real-HTTP runtime — the cassette proxy is a live local server.
    pub(crate) fn http(&self) -> HttpRuntime {
        HttpRuntime::new()
    }
}

/// Connection details for the OpenRouter cassette proxy driven through Rig's
/// OpenAI Responses provider.
pub(super) struct OpenRouterOpenAiCassette {
    api_key: String,
    base_url: String,
}

#[allow(dead_code)]
impl OpenRouterOpenAiCassette {
    /// Responses-API config for `model` aimed at the cassette proxy.
    pub(crate) fn config(
        &self,
        model: impl Into<String>,
    ) -> openai::responses_api::functions::Config {
        openai::responses_api::functions::Config::new(model)
            .with_api_key(self.api_key.clone())
            .with_base_url(self.base_url.clone())
    }

    pub(crate) fn provider_config(&self, model: impl Into<String>) -> ProviderConfig {
        ProviderConfig::OpenAiResponses(self.config(model))
    }

    pub(crate) fn agent(&self, model: impl Into<String>) -> AgentBuilder {
        AgentBuilder::new(self.provider_config(model))
    }

    pub(crate) fn http(&self) -> HttpRuntime {
        HttpRuntime::new()
    }
}

async fn openrouter_cassette(
    spec: impl Into<CassetteSpec>,
) -> (ProviderCassette, OpenRouterCassette) {
    let cassette = ProviderCassette::start("openrouter", spec, OPENROUTER_BASE_URL).await;
    let handle = OpenRouterCassette {
        api_key: cassette.api_key("OPENROUTER_API_KEY"),
        base_url: cassette.base_url(),
    };

    (cassette, handle)
}

async fn openrouter_openai_cassette(
    spec: impl Into<CassetteSpec>,
) -> (ProviderCassette, OpenRouterOpenAiCassette) {
    let cassette = ProviderCassette::start("openrouter", spec, OPENROUTER_BASE_URL).await;
    let handle = OpenRouterOpenAiCassette {
        api_key: cassette.api_key("OPENROUTER_API_KEY"),
        base_url: cassette.base_url(),
    };

    (cassette, handle)
}

pub(super) async fn with_openrouter_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(OpenRouterCassette) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = openrouter_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

pub(super) async fn with_openrouter_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(OpenRouterCassette) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let (cassette, client) = openrouter_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test_result(result).await
}

pub(super) async fn with_openrouter_openai_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(OpenRouterOpenAiCassette) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = openrouter_openai_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}
