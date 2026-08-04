//! Cassette/credential plumbing for the Copilot suites.
//!
//! Cassette helpers return the public concrete [`copilot::Client`], pointed at
//! the cassette proxy with its replay credential.

use assert_fs::TempDir;
use futures::FutureExt;
use rig::AgentBuilder;
use rig::provider::ProviderConfig;
use rig::providers::copilot;
use rig::providers::copilot::auth::{AuthSource, Authenticator, DeviceCodePrompter};
use std::borrow::Cow;
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::path::Path;

use crate::cassettes::{CassetteSpec, ProviderCassette};

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

/// An [`Authenticator`] over `source`, optionally rooted at `token_dir`.
///
/// Replaces the deleted `ClientBuilder::{api_key, github_access_token, oauth,
/// token_dir, allow_device_flow}` knobs.
#[allow(dead_code)]
pub(crate) fn authenticator(
    source: AuthSource,
    token_dir: Option<&Path>,
    allow_device_flow: bool,
) -> Authenticator {
    Authenticator::new(
        source,
        token_dir.map(|dir| dir.join("access-token")),
        token_dir.map(|dir| dir.join("api-key.json")),
        DeviceCodePrompter::default(),
        allow_device_flow,
    )
}

/// Resolve `authenticator` into a completion [`Config`](copilot::functions::Config)
/// for `model` — the replacement for `client.authorize()` plus `client.agent(m)`.
#[allow(dead_code)]
pub(crate) async fn config_from_auth(
    model: impl Into<String>,
    authenticator: &Authenticator,
) -> copilot::functions::Config {
    let mut cfg = copilot::functions::config_from_auth(model, authenticator)
        .await
        .expect("Copilot credential resolution should succeed");
    if let Some(base_url) = env_base_url() {
        cfg = cfg.with_base_url(base_url);
    }
    cfg
}

/// Live (non-cassette) config for `model`, using the classic client's
/// credential precedence: API key, then GitHub access token, then cached OAuth.
#[allow(dead_code)]
pub(crate) async fn live_config(model: impl Into<String>) -> copilot::functions::Config {
    copilot::functions::config_from_env(model)
        .await
        .expect("Copilot credentials should resolve")
}

/// Live [`AgentBuilder`] for `model`; the replacement for `live_client().agent(m)`.
#[allow(dead_code)]
pub(crate) async fn live_agent(model: impl Into<String>) -> AgentBuilder {
    AgentBuilder::new(ProviderConfig::Copilot(live_config(model).await))
}

async fn copilot_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, copilot::Client) {
    let cassette_base_url = cassette_base_url();
    let cassette = ProviderCassette::start("copilot", spec, &cassette_base_url).await;
    let client = copilot::Client::builder()
        .api_key(cassette.api_key("GITHUB_COPILOT_API_KEY"))
        .base_url(cassette.base_url())
        .build()
        .expect("cassette client should build");

    (cassette, client)
}

/// A cassette whose credential is resolved through the non-interactive OAuth
/// path (a pre-seeded, unexpired `api-key.json`, device flow disabled).
async fn copilot_noninteractive_oauth_cassette(
    spec: impl Into<CassetteSpec>,
) -> (ProviderCassette, copilot::Client, TempDir) {
    let cassette_base_url = cassette_base_url();
    let cassette = ProviderCassette::start("copilot", spec, &cassette_base_url).await;
    let temp = TempDir::new().expect("temp token directory should be created");
    let api_key_record = serde_json::json!({
        "token": cassette.api_key("GITHUB_COPILOT_API_KEY"),
        "expires_at": i64::MAX,
    });
    std::fs::write(
        temp.path().join("api-key.json"),
        serde_json::to_vec_pretty(&api_key_record).expect("api key record should serialize"),
    )
    .expect("api key record should be written");

    let authenticator = authenticator(AuthSource::OAuth, Some(temp.path()), false);
    let auth = authenticator
        .auth_context()
        .await
        .expect("cached OAuth auth should not require device flow");

    let client = copilot::Client::builder()
        .api_key(auth.api_key)
        .base_url(cassette.base_url())
        .build()
        .expect("non-interactive OAuth cassette client should build");

    (cassette, client, temp)
}

pub(super) async fn with_copilot_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(copilot::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, handle) = copilot_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(handle)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

pub(super) async fn with_copilot_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(copilot::Client) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let (cassette, handle) = copilot_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(handle)).catch_unwind().await;
    cassette.finish_after_test_result(result).await
}

pub(super) async fn with_copilot_noninteractive_oauth_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(copilot::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, handle, _temp) = copilot_noninteractive_oauth_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(handle)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}
