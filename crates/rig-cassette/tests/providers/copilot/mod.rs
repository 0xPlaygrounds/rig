use rig_test_support::cassette_models::CopilotModels;
mod agent;
mod auth;
mod embeddings;
mod models;
mod multi_extract;
mod noninteractive_oauth_cassette;
mod permission_control;
mod raw_capture_matrix;
mod raw_completion_parity_matrix;
mod raw_stream_capture_matrix;
mod reasoning_roundtrip;
mod reasoning_tool_roundtrip;
mod routing;
mod streaming;
mod streaming_tools;
mod structured_output;
mod typed_prompt_tools;

use assert_fs::TempDir;
use rig::providers::copilot;
use rig::providers::copilot::auth::{
    AuthError, AuthSource, Authenticator, DeviceCodeHandler, FileTokenStore, MemoryTokenStore,
    TokenStore,
};
use rig::providers::openai::OpenAIConfig;
use std::borrow::Cow;
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::sync::Arc;

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
        .map_or_else(|| Cow::Borrowed(copilot::GPT_5_3_CODEX), Cow::Owned)
}

pub(crate) fn live_embedding_model() -> Cow<'static, str> {
    first_env_value(&["GITHUB_COPILOT_EMBEDDING_MODEL", "COPILOT_EMBEDDING_MODEL"]).map_or_else(
        || Cow::Borrowed(copilot::TEXT_EMBEDDING_3_SMALL),
        Cow::Owned,
    )
}

fn env_base_url() -> Option<String> {
    first_env_value(&["GITHUB_COPILOT_API_BASE", "COPILOT_BASE_URL"])
}

fn cassette_base_url() -> String {
    env_base_url().unwrap_or_else(|| "https://api.githubcopilot.com".to_string())
}

/// Resolve a Copilot credential and hold it in a provider configuration.
///
/// The exchange is a conversation — a device flow, a refresh, a kept
/// credential — so it is not a wire and never will be: the authenticator
/// runs it over a transport, and the configuration takes the session token
/// and the API base the exchange reported.
///
/// `token_dir` is `None` when the caller wants nothing kept on disk; the
/// cells that assert keeping pass a `TempDir`.
pub(crate) async fn authorize(
    source: AuthSource,
    token_dir: Option<&Path>,
    allow_device_flow: bool,
) -> Result<OpenAIConfig, AuthError> {
    let store: Arc<dyn TokenStore> = match token_dir {
        Some(dir) => Arc::new(FileTokenStore::new(dir)),
        None => Arc::new(MemoryTokenStore::default()),
    };
    let authenticator = Authenticator::new(
        source,
        store,
        DeviceCodeHandler::default(),
        allow_device_flow,
    );
    let context = authenticator
        .auth_context(&rig_test_support::cassettes::local_http())
        .await?;
    let mut provider = OpenAIConfig::with_key(&copilot::DIALECT, context.api_key);
    if let Some(api_base) = context.api_base {
        provider.base_url = api_base;
    }

    Ok(match env_base_url() {
        Some(base_url) => provider.with_base_url(base_url),
        None => provider,
    })
}

/// The credential the environment offers, in the order the client tried:
/// an exchanged session token, then a GitHub access token, then OAuth.
pub(crate) fn live_source() -> AuthSource {
    if let Some(api_key) = copilot_api_key() {
        AuthSource::ApiKey(api_key)
    } else if let Some(access_token) = copilot_github_access_token() {
        AuthSource::GitHubAccessToken(access_token)
    } else {
        AuthSource::OAuth
    }
}

pub(crate) async fn live_client() -> CopilotModels {
    CopilotModels::new(
        authorize(live_source(), None, true)
            .await
            .expect("Copilot credential should resolve"),
        rig_test_support::cassettes::local_http(),
    )
}

async fn copilot_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, OpenAIConfig) {
    let cassette_base_url = cassette_base_url();
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "copilot",
        spec,
        &cassette_base_url,
    )
    .await;
    let bound = OpenAIConfig::with_key(
        &copilot::DIALECT,
        cassette.api_key("GITHUB_COPILOT_API_KEY"),
    )
    .with_base_url(cassette.base_url());

    (cassette, bound)
}

async fn copilot_noninteractive_oauth_cassette(
    spec: impl Into<CassetteSpec>,
) -> (ProviderCassette, OpenAIConfig, TempDir) {
    let cassette_base_url = cassette_base_url();
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "copilot",
        spec,
        &cassette_base_url,
    )
    .await;
    let temp = TempDir::new().expect("temp token directory should be created");
    let api_key_record = serde_json::json!({
        "token": cassette.api_key("GITHUB_COPILOT_API_KEY"),
        "expires_at": i64::MAX,
    });
    std::fs::write(
        FileTokenStore::new(temp.path()).path(copilot::PROVIDER_NAME),
        serde_json::to_vec_pretty(&api_key_record).expect("api key record should serialize"),
    )
    .expect("api key record should be written");

    let bound = authorize(AuthSource::OAuth, Some(temp.path()), false)
        .await
        .expect("the cached Copilot API key should resolve without a device flow")
        .with_base_url(cassette.base_url());

    (cassette, bound, temp)
}

pub(crate) async fn with_copilot_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(CopilotModels) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = copilot_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(CopilotModels::new(
        client,
        rig_test_support::cassettes::local_http(),
    )))
    .catch_unwind()
    .await;
    cassette.finish_after_test(result).await;
}

pub(crate) async fn with_copilot_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(CopilotModels) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let (cassette, client) = copilot_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(CopilotModels::new(
        client,
        rig_test_support::cassettes::local_http(),
    )))
    .catch_unwind()
    .await;
    cassette.finish_after_test_result(result).await
}

pub(crate) async fn with_copilot_noninteractive_oauth_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(CopilotModels) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client, _temp) = copilot_noninteractive_oauth_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(CopilotModels::new(
        client,
        rig_test_support::cassettes::local_http(),
    )))
    .catch_unwind()
    .await;
    cassette.finish_after_test(result).await;
}
