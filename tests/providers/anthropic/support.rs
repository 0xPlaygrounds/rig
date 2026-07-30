use futures::FutureExt;
use rig::AgentBuilder;
use rig::completion::{CompletionError, CompletionRequest, CompletionResponse};
use rig::http_runtime::HttpRuntime;
use rig::provider::ProviderConfig;
use rig::providers::anthropic;
use rig::streaming::StreamingCompletionResponse;
use std::future::Future;
use std::panic::AssertUnwindSafe;

use crate::cassettes::{CassetteSpec, ProviderCassette};

/// Connection details for a running Anthropic cassette proxy.
///
/// Replaces the deleted `anthropic::Client`: tests mint a plain
/// [`anthropic::functions::Config`] (or an [`AgentBuilder`]) per model,
/// pointed at the cassette's base URL and API key.
#[derive(Clone)]
pub(super) struct AnthropicCassette {
    api_key: String,
    base_url: String,
    betas: Vec<String>,
}

#[allow(dead_code)]
impl AnthropicCassette {
    /// Completion config for `model` aimed at the cassette proxy.
    pub(crate) fn config(&self, model: impl Into<String>) -> anthropic::functions::Config {
        let mut cfg = anthropic::functions::Config::new(model)
            .with_api_key(self.api_key.clone())
            .with_base_url(anthropic::functions::normalize_base_url(&self.base_url));
        cfg.anthropic_betas = self.betas.clone();
        cfg
    }

    /// The provider selection for `model` aimed at the cassette proxy.
    pub(crate) fn provider_config(&self, model: impl Into<String>) -> ProviderConfig {
        ProviderConfig::Anthropic(self.config(model))
    }

    /// An [`AgentBuilder`] for `model` aimed at the cassette proxy.
    pub(crate) fn agent(&self, model: impl Into<String>) -> AgentBuilder {
        AgentBuilder::new(self.provider_config(model))
    }

    /// A real-HTTP runtime — the cassette proxy is a live local server.
    pub(crate) fn http(&self) -> HttpRuntime {
        HttpRuntime::new()
    }

    /// A direct completion handle for `model`, standing in for the deleted
    /// `CompletionModel` trait object: `completion` / `stream` call the
    /// `anthropic::functions` free functions against a real HTTP runtime.
    pub(crate) fn completion_model(&self, model: impl Into<String>) -> AnthropicCompletion {
        AnthropicCompletion {
            cfg: self.config(model),
            http: HttpRuntime::new(),
        }
    }
}

/// Direct (non-agent) completion handle bound to a cassette-backed config.
#[derive(Clone)]
#[allow(dead_code)]
pub(super) struct AnthropicCompletion {
    cfg: anthropic::functions::Config,
    http: HttpRuntime,
}

#[allow(dead_code)]
impl AnthropicCompletion {
    pub(crate) fn config(&self) -> &anthropic::functions::Config {
        &self.cfg
    }

    pub(crate) async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, CompletionError> {
        anthropic::functions::complete(&self.cfg, &self.http, request).await
    }

    pub(crate) async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse, CompletionError> {
        anthropic::functions::open_stream(&self.cfg, &self.http, request).await
    }
}

pub(super) struct AnthropicFilesCassette {
    pub(super) client: AnthropicCassette,
    pub(super) base_url: String,
    pub(super) api_key: String,
}

async fn anthropic_cassette(
    spec: impl Into<CassetteSpec>,
) -> (ProviderCassette, AnthropicCassette) {
    let cassette = ProviderCassette::start("anthropic", spec, "https://api.anthropic.com").await;
    let handle = AnthropicCassette {
        api_key: cassette.api_key("ANTHROPIC_API_KEY"),
        base_url: cassette.base_url(),
        betas: Vec::new(),
    };

    (cassette, handle)
}

pub(super) async fn with_anthropic_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(AnthropicCassette) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = anthropic_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

#[allow(dead_code)]
pub(super) async fn with_anthropic_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(AnthropicCassette) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let (cassette, client) = anthropic_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test_result(result).await
}

pub(super) async fn with_anthropic_files_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    beta_header: &'static str,
    test_body: F,
) where
    F: FnOnce(AnthropicFilesCassette) -> Fut,
    Fut: Future<Output = ()>,
{
    let cassette = ProviderCassette::start("anthropic", spec, "https://api.anthropic.com").await;
    let base_url = anthropic::functions::normalize_base_url(&cassette.base_url());
    let api_key = cassette.api_key("ANTHROPIC_API_KEY");
    let client = AnthropicCassette {
        api_key: api_key.clone(),
        base_url: base_url.clone(),
        betas: vec![beta_header.to_string()],
    };

    let parts = AnthropicFilesCassette {
        client,
        base_url,
        api_key,
    };
    let result = AssertUnwindSafe(test_body(parts)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}
