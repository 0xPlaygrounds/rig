mod agent;
mod auth;
#[path = "cassette/ecs_completion.rs"]
mod ecs_completion;
#[path = "cassette/ecs_extractor.rs"]
mod ecs_extractor;
#[path = "cassette/ecs_extractor_usage.rs"]
mod ecs_extractor_usage;
mod embeddings;
mod extractor;
mod extractor_usage;
mod models;
mod multi_extract;
mod noninteractive_oauth_cassette;
mod permission_control;
mod raw_capture_matrix;
mod raw_completion_parity_matrix;
mod raw_stream_capture_matrix;
mod reasoning_roundtrip;
mod reasoning_tool_roundtrip;
mod request_hook;
mod routing;
mod streaming;
mod streaming_tools;
mod structured_output;
mod typed_prompt_tools;

use assert_fs::TempDir;
use rig::driver::{Bind, Bound};
use rig::http_client::{BoxedHttpClient, ReqwestClient};
use rig::providers::copilot;
use rig::providers::copilot::auth::{AuthError, AuthSource, Authenticator, DeviceCodeHandler};
use rig::providers::copilot::wire::Copilot;
use std::borrow::Cow;
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::path::Path;

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

/// The bundled transport, erased: the socket every cell here speaks over,
/// and the one the credential exchange runs on.
fn transport() -> BoxedHttpClient {
    ReqwestClient::default().boxed()
}

/// Resolve a Copilot credential and hold it in a provider configuration.
///
/// The exchange is a conversation — a device flow, a refresh, a shared file
/// cache — so it is not a wire and never will be: `copilot::auth` runs it
/// over the same transport the wire then uses, and `Copilot::from_auth`
/// holds the result, honouring the API base the exchange reported. This is
/// what `ClientBuilder::{api_key, github_access_token, oauth, token_dir,
/// allow_device_flow}` plus `Client::authorize` did between them.
///
/// `token_dir` is `None` when the caller wants no on-disk cache; the cells
/// that assert caching pass a `TempDir`.
pub(crate) async fn authorize(
    source: AuthSource,
    token_dir: Option<&Path>,
    allow_device_flow: bool,
) -> Result<Copilot, AuthError> {
    let (access_token_file, api_key_file) = match token_dir {
        Some(dir) => (
            Some(dir.join("access-token")),
            Some(dir.join("api-key.json")),
        ),
        None => (None, None),
    };
    let context = Authenticator::new(
        source,
        access_token_file,
        api_key_file,
        DeviceCodeHandler::default(),
        allow_device_flow,
    )
    .auth_context(&transport())
    .await?;
    let provider = Copilot::from_auth(&context);

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

pub(crate) async fn live_client() -> Bound<Copilot> {
    authorize(live_source(), None, true)
        .await
        .expect("Copilot credential should resolve")
        .bind(transport())
}

async fn copilot_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, Bound<Copilot>) {
    let cassette_base_url = cassette_base_url();
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "copilot",
        spec,
        &cassette_base_url,
    )
    .await;
    let bound = Copilot::new(cassette.api_key("GITHUB_COPILOT_API_KEY"))
        .with_base_url(cassette.base_url())
        .bind(transport());

    (cassette, bound)
}

async fn copilot_noninteractive_oauth_cassette(
    spec: impl Into<CassetteSpec>,
) -> (ProviderCassette, Bound<Copilot>, TempDir) {
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
        temp.path().join("api-key.json"),
        serde_json::to_vec_pretty(&api_key_record).expect("api key record should serialize"),
    )
    .expect("api key record should be written");

    let bound = authorize(AuthSource::OAuth, Some(temp.path()), false)
        .await
        .expect("the cached Copilot API key should resolve without a device flow")
        .with_base_url(cassette.base_url())
        .bind(transport());

    (cassette, bound, temp)
}

pub(crate) async fn with_copilot_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(Bound<Copilot>) -> Fut,
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
    F: FnOnce(Bound<Copilot>) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let (cassette, client) = copilot_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test_result(result).await
}

pub(crate) async fn with_copilot_noninteractive_oauth_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(Bound<Copilot>) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client, _temp) = copilot_noninteractive_oauth_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}
