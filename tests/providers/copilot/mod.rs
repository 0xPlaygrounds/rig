mod agent;
mod auth;
mod embeddings;
mod extractor;
mod extractor_usage;
mod models;
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

use rig::providers::copilot;
use std::borrow::Cow;
use std::future::Future;
use std::panic::AssertUnwindSafe;

use crate::cassettes::{CassetteSpec, ProviderCassette};
use futures::FutureExt;

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

fn cassette_base_url() -> String {
    env_base_url().unwrap_or_else(|| "https://api.githubcopilot.com".to_string())
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

async fn copilot_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, copilot::Client) {
    let cassette_base_url = cassette_base_url();
    let cassette = ProviderCassette::start("copilot", spec, &cassette_base_url).await;
    let client = copilot::Client::builder()
        .api_key(cassette.api_key("GITHUB_COPILOT_API_KEY"))
        .base_url(cassette.base_url())
        .build()
        .expect("Copilot cassette client should build");

    (cassette, client)
}

pub(crate) async fn with_copilot_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(copilot::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = copilot_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

pub(crate) async fn with_copilot_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(copilot::Client) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let (cassette, client) = copilot_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test_result(result).await
}
