//! Shared helpers for llama.cpp live tests.
//!
//! Providers are data now: there is no client to build, so these helpers hand
//! back an `openai::functions::Config` (llama.cpp speaks the OpenAI-compatible
//! wire format) or the `ProviderConfig` arm wrapping it.

use rig::provider::ProviderConfig;
use rig::providers::openai;

const DEFAULT_API_BASE_URL: &str = "http://localhost:8080/v1";
const DEFAULT_API_KEY: &str = "none";
const DEFAULT_MODEL: &str = "model";

pub(super) fn api_base_url() -> String {
    std::env::var("LLAMACPP_API_BASE_URL").unwrap_or_else(|_| DEFAULT_API_BASE_URL.to_string())
}

pub(super) fn api_key() -> String {
    std::env::var("LLAMACPP_API_KEY").unwrap_or_else(|_| DEFAULT_API_KEY.to_string())
}

pub(super) fn model_name() -> String {
    std::env::var("LLAMACPP_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string())
}

/// Connection data plus the model under test.
pub(super) fn config(model: impl Into<String>) -> openai::functions::Config {
    openai::functions::Config::new(model)
        .with_api_key(api_key())
        .with_base_url(api_base_url())
}

/// The `ProviderConfig` arm for llama.cpp's OpenAI-compatible endpoint.
pub(super) fn provider(model: impl Into<String>) -> ProviderConfig {
    ProviderConfig::OpenAi(config(model))
}
