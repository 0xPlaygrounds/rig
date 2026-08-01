//! ChatGPT OAuth device flow and refresh smoke tests.

use assert_fs::TempDir;
use rig::AgentBuilder;
use rig::providers::chatgpt;
use serde_json::json;
use std::fs;
use std::path::Path;

use crate::chatgpt::LIVE_MODEL;
use crate::support::{
    BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response, collect_stream_final_response,
};

/// The OAuth credential source the deleted builder configured.
///
/// The `CHATGPT_API_BASE` / `OPENAI_CHATGPT_API_BASE` override is applied by
/// [`chatgpt::functions::config_from_auth`].
fn oauth_authenticator_with_auth_file(path: &Path) -> chatgpt::auth::Authenticator {
    chatgpt::auth::Authenticator::new(
        chatgpt::auth::AuthSource::OAuth,
        Some(path.to_path_buf()),
        chatgpt::auth::DeviceCodePrompter::default(),
        true,
    )
}

/// `client.authorize()` plus `client.agent(model)`, as construction-time
/// credential resolution followed by an agent over the resolved config.
async fn authorized_agent(path: &Path, model: &str) -> AgentBuilder {
    let cfg =
        chatgpt::functions::config_from_auth(model, &oauth_authenticator_with_auth_file(path))
            .await
            .expect("authorization should succeed");
    AgentBuilder::new(cfg)
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

    let agent_builder = authorized_agent(&auth_file, LIVE_MODEL).await;

    assert!(
        auth_file.is_file(),
        "device authorization should populate the auth cache"
    );

    let agent = agent_builder.preamble(BASIC_PREAMBLE).build();
    let mut stream = agent.runner(BASIC_PROMPT).stream_run();
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("authorized streaming completion should succeed");

    assert_nonempty_response(&response);

    let cached_agent = authorized_agent(&auth_file, LIVE_MODEL).await.build();
    let mut cached_stream = cached_agent
        .runner("Reply with the single word cached.")
        .stream_run();
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

    let agent_builder = authorized_agent(&auth_file, LIVE_MODEL).await;

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

    let agent = agent_builder.build();
    let mut stream = agent
        .runner("Reply with the single word refreshed.")
        .stream_run();
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("refreshed streaming completion should succeed");

    assert_nonempty_response(&response);
}
