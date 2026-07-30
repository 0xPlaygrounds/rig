use std::future::Future;
use std::panic::AssertUnwindSafe;

use futures::FutureExt;
use rig::provider::ProviderConfig;
use rig::providers::openai;

use crate::cassettes::{CassetteSpec, ProviderCassette};

pub(super) const DEFAULT_BASE_URL: &str = "http://127.0.0.1:1234/v1";
pub(super) const DEFAULT_API_KEY: &str = "local";
pub(super) const DEFAULT_MODEL: &str = "Qwen/Qwen3-4B";
pub(super) const SYSTEM_PROMPT: &str =
    "You are concise. Include a few details so streaming is visible.";

pub(super) fn model_name() -> String {
    std::env::var("MISTRALRS_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string())
}

/// Connection data for a running mistral.rs cassette server.
///
/// Providers are data now: the cassette hands the test its base URL and API
/// key, and each test mints a `functions::Config` for the model and API face
/// (chat completions or Responses) it exercises.
pub(super) struct MistralRsCassetteEnv {
    base_url: String,
    api_key: String,
}

impl MistralRsCassetteEnv {
    /// The OpenAI chat-completions face (`/chat/completions`).
    pub(super) fn chat_config(&self, model: impl Into<String>) -> openai::functions::Config {
        openai::functions::Config::new(model)
            .with_api_key(&self.api_key)
            .with_base_url(&self.base_url)
    }

    pub(super) fn chat_provider(&self, model: impl Into<String>) -> ProviderConfig {
        ProviderConfig::OpenAi(self.chat_config(model))
    }

    /// The OpenAI Responses face (`/responses`). mistral.rs needs system
    /// instructions sent as input messages, which the deleted
    /// `Client::with_system_instructions_as_messages()` toggled.
    pub(super) fn responses_config(
        &self,
        model: impl Into<String>,
    ) -> openai::responses_api::functions::Config {
        openai::responses_api::functions::Config::new(model)
            .with_api_key(&self.api_key)
            .with_base_url(&self.base_url)
            .with_system_instructions_as_messages()
    }

    pub(super) fn responses_provider(&self, model: impl Into<String>) -> ProviderConfig {
        ProviderConfig::OpenAiResponses(self.responses_config(model))
    }
}

async fn mistralrs_cassette(
    spec: impl Into<CassetteSpec>,
) -> (ProviderCassette, MistralRsCassetteEnv) {
    let real_base_url =
        std::env::var("MISTRALRS_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_string());
    let api_key =
        std::env::var("MISTRALRS_API_KEY").unwrap_or_else(|_| DEFAULT_API_KEY.to_string());
    let cassette = ProviderCassette::start("mistralrs", spec, &real_base_url).await;
    let env = MistralRsCassetteEnv {
        base_url: cassette.base_url(),
        api_key,
    };

    (cassette, env)
}

async fn mistralrs_raw_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, String) {
    let real_base_url =
        std::env::var("MISTRALRS_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_string());
    let cassette = ProviderCassette::start("mistralrs", spec, &real_base_url).await;
    let base_url = cassette.base_url();
    (cassette, base_url)
}

pub(super) async fn with_mistralrs_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(MistralRsCassetteEnv) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, env) = mistralrs_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(env)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

pub(super) async fn with_mistralrs_raw_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(String) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, base_url) = mistralrs_raw_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(base_url)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}
