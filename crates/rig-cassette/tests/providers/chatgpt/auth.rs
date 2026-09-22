//! ChatGPT OAuth device flow and refresh smoke tests.

use assert_fs::TempDir;
use rig::driver::{Bind as _, Bound};
use rig::http_client::BoxedHttpClient;
use rig::prelude::*;
use rig::providers::chatgpt;
use rig::providers::openai::OpenAI;
use rig::rig_reqwest::client::bundled;
use serde_json::json;
use std::fs;
use std::path::Path;

use crate::chatgpt::LIVE_MODEL;
use crate::support::{
    BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response, collect_stream_final_response,
};

/// Resolve the OAuth credential cached in `path` — refreshing it, or running
/// the device flow, on `http` — and hand back the provider configuration that
/// already holds it.
///
/// The exchange is not a wire: `OpenAI` stores an access token, so the
/// conversation that produces one runs first, on the transport the completion
/// then speaks over.
async fn oauth_provider_with_auth_file(path: &Path, http: &BoxedHttpClient) -> OpenAI {
    let context = chatgpt::auth::Authenticator::new(
        chatgpt::auth::AuthSource::OAuth,
        Some(path.to_path_buf()),
        chatgpt::auth::DeviceCodeHandler::default(),
        true,
    )
    .auth_context(http)
    .await
    .expect("ChatGPT OAuth should resolve an access token");

    let mut provider = OpenAI::with_key(&chatgpt::DIALECT, context.access_token);
    if let Some(account_id) = context.account_id {
        provider = provider.with_account_id(account_id);
    }
    if let Ok(base_url) =
        std::env::var("CHATGPT_API_BASE").or_else(|_| std::env::var("OPENAI_CHATGPT_API_BASE"))
    {
        provider = provider.with_base_url(base_url);
    }

    provider
}

/// [`oauth_provider_with_auth_file`], bound to a fresh bundled transport.
async fn oauth_client_with_auth_file(path: &Path) -> Bound<OpenAI> {
    let http = bundled().expect("the bundled transport should build");
    oauth_provider_with_auth_file(path, &http).await.bind(http)
}

fn seed_refresh_auth_file(path: &Path) {
    let refresh_token =
        std::env::var("CHATGPT_REFRESH_TOKEN").expect("CHATGPT_REFRESH_TOKEN should be set");
    let account_id = std::env::var("CHATGPT_ACCOUNT_ID").ok();
    let id_token = std::env::var("CHATGPT_ID_TOKEN").ok();

    let record = json!({
        "access_token": serde_json::Value::Null,
        "refresh_token": refresh_token,
        "id_token": id_token,
        "expires_at": 0,
        "account_id": account_id,
    });

    fs::write(
        path,
        serde_json::to_vec_pretty(&record).expect("seed auth record"),
    )
    .expect("auth record should be written");
}

#[tokio::test]
#[ignore = "requires interactive ChatGPT OAuth device flow"]
async fn oauth_device_flow_authorize_and_cached_completion_smoke() {
    let temp = TempDir::new().expect("temp dir");
    let auth_file = temp.path().join("auth.json");

    // Building the provider *is* the authorization: the credential has to be
    // in hand before a wire can carry it, so the device flow runs here.
    let client = oauth_client_with_auth_file(&auth_file).await;

    assert!(
        auth_file.is_file(),
        "device authorization should populate the auth cache"
    );

    let agent = client.agent(LIVE_MODEL).preamble(BASIC_PREAMBLE).build();
    let mut stream = agent.prompt(BASIC_PROMPT).stream();
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("authorized streaming completion should succeed");

    assert_nonempty_response(&response);

    let cached_client = oauth_client_with_auth_file(&auth_file).await;

    let cached_agent = cached_client.agent(LIVE_MODEL).build();
    let mut cached_stream = cached_agent
        .prompt("Reply with the single word cached.")
        .stream();
    let cached_response = collect_stream_final_response(&mut cached_stream)
        .await
        .expect("cached streaming completion should succeed");

    assert_nonempty_response(&cached_response);
}

#[tokio::test]
#[ignore = "requires CHATGPT_REFRESH_TOKEN"]
async fn refresh_token_cache_authorize_and_completion_smoke() {
    let temp = TempDir::new().expect("temp dir");
    let auth_file = temp.path().join("auth.json");
    seed_refresh_auth_file(&auth_file);

    let client = oauth_client_with_auth_file(&auth_file).await;

    let record: serde_json::Value =
        serde_json::from_slice(&fs::read(&auth_file).expect("auth file should exist"))
            .expect("auth file should deserialize");
    assert!(
        record
            .get("access_token")
            .and_then(|value| value.as_str())
            .is_some(),
        "refresh should persist an access token"
    );
    assert!(
        record
            .get("refresh_token")
            .and_then(|value| value.as_str())
            .is_some(),
        "refresh should persist a refresh token"
    );

    let agent = client.agent(LIVE_MODEL).build();
    let mut stream = agent
        .prompt("Reply with the single word refreshed.")
        .stream();
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("refreshed streaming completion should succeed");

    assert_nonempty_response(&response);
}
