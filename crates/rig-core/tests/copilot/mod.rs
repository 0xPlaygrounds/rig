mod agent;
mod auth;
mod embeddings;
mod extractor;
mod extractor_usage;
mod multi_extract;
mod permission_control;
mod reasoning_roundtrip;
mod reasoning_tool_roundtrip;
mod request_hook;
mod routing;
mod streaming;
mod streaming_tools;
mod structured_output;
mod typed_prompt_tools;

use rig_core::providers::copilot;
use std::borrow::Cow;

pub(crate) const LIVE_MODEL: &str = copilot::GPT_4O;
pub(crate) const LIVE_LIGHT_MODEL: &str = copilot::GPT_4O_MINI;

fn first_env_value(keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|name| {
        std::env::var(name)
            .ok()
            .filter(|value| !value.trim().is_empty())
    })
}

pub(crate) fn copilot_api_key() -> Option<String> {
    first_env_value(&["GITHUB_COPILOT_API_KEY", "COPILOT_API_KEY"])
}

pub(crate) fn copilot_github_access_token() -> Option<String> {
    first_env_value(&["COPILOT_GITHUB_ACCESS_TOKEN", "GITHUB_TOKEN"])
}

pub(crate) fn live_responses_model() -> Cow<'static, str> {
    first_env_value(&["GITHUB_COPILOT_RESPONSES_MODEL", "COPILOT_RESPONSES_MODEL"])
        .map(Cow::Owned)
        .unwrap_or_else(|| Cow::Borrowed(copilot::GPT_5_3_CODEX))
}

pub(crate) fn live_embedding_model() -> Cow<'static, str> {
    first_env_value(&["GITHUB_COPILOT_EMBEDDING_MODEL", "COPILOT_EMBEDDING_MODEL"])
        .map(Cow::Owned)
        .unwrap_or_else(|| Cow::Borrowed(copilot::TEXT_EMBEDDING_3_SMALL))
}

fn env_base_url() -> Option<String> {
    first_env_value(&["GITHUB_COPILOT_API_BASE", "COPILOT_BASE_URL"])
}

fn with_base_url(mut builder: copilot::ClientBuilder) -> copilot::ClientBuilder {
    if let Some(base_url) = env_base_url() {
        builder = builder.base_url(base_url);
    }

    builder
}

pub(crate) fn api_key_builder(api_key: impl Into<String>) -> copilot::ClientBuilder {
    with_base_url(copilot::Client::builder().api_key(api_key.into()))
}

pub(crate) fn github_access_token_builder(
    access_token: impl Into<String>,
) -> copilot::ClientBuilder {
    with_base_url(copilot::Client::builder().github_access_token(access_token.into()))
}

pub(crate) fn oauth_builder() -> copilot::ClientBuilder {
    with_base_url(copilot::Client::builder().oauth())
}

pub(crate) fn live_builder() -> copilot::ClientBuilder {
    if let Some(api_key) = copilot_api_key() {
        api_key_builder(api_key)
    } else if let Some(access_token) = copilot_github_access_token() {
        github_access_token_builder(access_token)
    } else {
        oauth_builder()
    }
}

pub(crate) fn live_client() -> copilot::Client {
    live_builder().build().expect("Copilot client should build")
}
