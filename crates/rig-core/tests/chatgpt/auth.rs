//! ChatGPT OAuth device flow and refresh smoke tests.

use assert_fs::TempDir;
use rig_core::client::CompletionClient;
use rig_core::providers::chatgpt;
use rig_core::streaming::StreamingPrompt;
use serde_json::json;
use std::fs;
use std::path::Path;

use crate::chatgpt::LIVE_MODEL;
use crate::support::{
    BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response, collect_stream_final_response,
};

fn oauth_builder_with_auth_file(path: &Path) -> chatgpt::ClientBuilder {
    let mut builder = chatgpt::Client::builder().oauth().auth_file(path);

    if let Ok(base_url) =
        std::env::var("CHATGPT_API_BASE").or_else(|_| std::env::var("OPENAI_CHATGPT_API_BASE"))
    {
        builder = builder.base_url(base_url);
    }

    builder
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

    let client = oauth_builder_with_auth_file(&auth_file)
        .build()
        .expect("ChatGPT OAuth client should build");

    client
        .authorize()
        .await
        .expect("device authorization should succeed");

    assert!(
        auth_file.is_file(),
        "device authorization should populate the auth cache"
    );

    let agent = client.agent(LIVE_MODEL).preamble(BASIC_PREAMBLE).build();
    let mut stream = agent.stream_prompt(BASIC_PROMPT).await;
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("authorized streaming completion should succeed");

    assert_nonempty_response(&response);

    let cached_client = oauth_builder_with_auth_file(&auth_file)
        .build()
        .expect("cached ChatGPT OAuth client should build");

    let cached_agent = cached_client.agent(LIVE_MODEL).build();
    let mut cached_stream = cached_agent
        .stream_prompt("Reply with the single word cached.")
        .await;
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

    let client = oauth_builder_with_auth_file(&auth_file)
        .build()
        .expect("ChatGPT refresh client should build");

    client
        .authorize()
        .await
        .expect("refresh authorization should succeed");

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
        .stream_prompt("Reply with the single word refreshed.")
        .await;
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("refreshed streaming completion should succeed");

    assert_nonempty_response(&response);
}
