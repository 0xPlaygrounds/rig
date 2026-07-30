use rig::AgentBuilder;
use rig::http_runtime::HttpRuntime;
use rig::provider::ProviderConfig;
use rig::providers::openai;
use std::future::Future;
use std::panic::AssertUnwindSafe;

use crate::cassettes::{CassetteSpec, ProviderCassette};
use futures::FutureExt;

/// Connection details for a running OpenAI cassette proxy, Responses-API face.
///
/// Replaces the deleted `openai::Client`: tests mint a plain
/// [`openai::responses_api::functions::Config`] (or an [`AgentBuilder`]) per
/// model, pointed at the cassette's base URL and API key.
pub(super) struct OpenAiCassette {
    api_key: String,
    base_url: String,
    system_instructions_as_messages: bool,
}

#[allow(dead_code)]
impl OpenAiCassette {
    /// Responses-API completion config for `model` aimed at the cassette proxy.
    pub(crate) fn config(
        &self,
        model: impl Into<String>,
    ) -> openai::responses_api::functions::Config {
        let cfg = openai::responses_api::functions::Config::new(model)
            .with_api_key(self.api_key.clone())
            .with_base_url(self.base_url.clone());
        if self.system_instructions_as_messages {
            cfg.with_system_instructions_as_messages()
        } else {
            cfg
        }
    }

    /// Send system instructions as `system` input items instead of the
    /// top-level `instructions` field, for every config this handle mints.
    pub(crate) fn with_system_instructions_as_messages(mut self) -> Self {
        self.system_instructions_as_messages = true;
        self
    }

    /// Alias of [`Self::config`], for call sites that want to be explicit about
    /// which OpenAI face they are exercising.
    pub(crate) fn responses_config(
        &self,
        model: impl Into<String>,
    ) -> openai::responses_api::functions::Config {
        self.config(model)
    }

    /// Chat-Completions completion config for `model` aimed at the cassette proxy.
    pub(crate) fn completions_config(&self, model: impl Into<String>) -> openai::functions::Config {
        openai::functions::Config::new(model)
            .with_api_key(self.api_key.clone())
            .with_base_url(self.base_url.clone())
    }

    /// Embedding config for `model` aimed at the cassette proxy.
    pub(crate) fn embedding_config(
        &self,
        model: impl Into<String>,
    ) -> openai::functions::EmbeddingConfig {
        openai::functions::EmbeddingConfig::new(model)
            .with_api_key(self.api_key.clone())
            .with_base_url(self.base_url.clone())
    }

    /// The Responses-API [`ProviderConfig`] for `model`.
    pub(crate) fn provider_config(&self, model: impl Into<String>) -> ProviderConfig {
        ProviderConfig::OpenAiResponses(self.config(model))
    }

    /// An [`AgentBuilder`] for `model` aimed at the cassette proxy.
    pub(crate) fn agent(&self, model: impl Into<String>) -> AgentBuilder {
        AgentBuilder::new(self.provider_config(model))
    }

    /// The Chat-Completions face of the same cassette proxy.
    pub(crate) fn completions_api(&self) -> OpenAiCompletionsCassette {
        OpenAiCompletionsCassette {
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
        }
    }

    /// A real-HTTP runtime — the cassette proxy is a live local server.
    pub(crate) fn http(&self) -> HttpRuntime {
        HttpRuntime::new()
    }
}

/// Connection details for a running OpenAI cassette proxy, Chat-Completions face.
pub(super) struct OpenAiCompletionsCassette {
    api_key: String,
    base_url: String,
}

#[allow(dead_code)]
impl OpenAiCompletionsCassette {
    /// Chat-Completions config for `model` aimed at the cassette proxy.
    pub(crate) fn config(&self, model: impl Into<String>) -> openai::functions::Config {
        openai::functions::Config::new(model)
            .with_api_key(self.api_key.clone())
            .with_base_url(self.base_url.clone())
    }

    /// Embedding config for `model` aimed at the cassette proxy.
    pub(crate) fn embedding_config(
        &self,
        model: impl Into<String>,
    ) -> openai::functions::EmbeddingConfig {
        openai::functions::EmbeddingConfig::new(model)
            .with_api_key(self.api_key.clone())
            .with_base_url(self.base_url.clone())
    }

    /// The Chat-Completions [`ProviderConfig`] for `model`.
    pub(crate) fn provider_config(&self, model: impl Into<String>) -> ProviderConfig {
        ProviderConfig::OpenAi(self.config(model))
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

async fn openai_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, OpenAiCassette) {
    let cassette = ProviderCassette::start("openai", spec, "https://api.openai.com/v1").await;
    let handle = OpenAiCassette {
        api_key: cassette.api_key("OPENAI_API_KEY"),
        base_url: cassette.base_url(),
        system_instructions_as_messages: false,
    };

    (cassette, handle)
}

async fn openai_completions_cassette(
    spec: impl Into<CassetteSpec>,
) -> (ProviderCassette, OpenAiCompletionsCassette) {
    let (cassette, handle) = openai_cassette(spec).await;
    let handle = handle.completions_api();
    (cassette, handle)
}

pub(super) async fn with_openai_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(OpenAiCassette) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, handle) = openai_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(handle)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

pub(super) async fn with_openai_completions_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(OpenAiCompletionsCassette) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, handle) = openai_completions_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(handle)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

pub(super) async fn with_openai_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(OpenAiCassette) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let (cassette, handle) = openai_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(handle)).catch_unwind().await;
    cassette.finish_after_test_result(result).await
}

pub(super) async fn with_openai_completions_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(OpenAiCompletionsCassette) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let (cassette, handle) = openai_completions_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(handle)).catch_unwind().await;
    cassette.finish_after_test_result(result).await
}
