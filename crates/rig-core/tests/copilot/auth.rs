//! Copilot OAuth and bootstrap smoke tests.

use assert_fs::TempDir;
use rig_core::client::CompletionClient;
use rig_core::completion::Prompt;
use rig_core::providers::copilot;
use serde_json::json;
use std::fs;
use std::path::Path;

use crate::copilot::{
    LIVE_MODEL, api_key_builder, copilot_api_key, copilot_github_access_token,
    github_access_token_builder, oauth_builder,
};
use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

fn required_copilot_api_key() -> String {
    copilot_api_key().expect("GITHUB_COPILOT_API_KEY or COPILOT_API_KEY should be set")
}

fn required_copilot_github_access_token() -> String {
    copilot_github_access_token()
        .expect("COPILOT_GITHUB_ACCESS_TOKEN or GITHUB_TOKEN should be set")
}

fn oauth_builder_with_token_dir(path: &Path) -> copilot::ClientBuilder {
    oauth_builder().token_dir(path)
}

#[tokio::test]
#[ignore = "requires GITHUB_COPILOT_API_KEY or COPILOT_API_KEY"]
async fn api_key_completion_smoke() {
    let client = api_key_builder(required_copilot_api_key())
        .build()
        .expect("Copilot API key client should build");

    client
        .authorize()
        .await
        .expect("api key auth should succeed");

    let response = client
        .agent(LIVE_MODEL)
        .preamble(BASIC_PREAMBLE)
        .build()
        .prompt(BASIC_PROMPT)
        .await
        .expect("api key-backed completion should succeed");

    assert_nonempty_response(&response);
}

#[tokio::test]
#[ignore = "requires COPILOT_GITHUB_ACCESS_TOKEN or GITHUB_TOKEN"]
async fn github_access_token_completion_smoke() {
    let client = github_access_token_builder(required_copilot_github_access_token())
        .build()
        .expect("Copilot bootstrap-token client should build");

    client
        .authorize()
        .await
        .expect("bootstrap-token auth should succeed");

    let response = client
        .agent(LIVE_MODEL)
        .preamble(BASIC_PREAMBLE)
        .build()
        .prompt(BASIC_PROMPT)
        .await
        .expect("bootstrap-token-backed completion should succeed");

    assert_nonempty_response(&response);
}

#[tokio::test]
#[ignore = "requires interactive GitHub Copilot OAuth device flow"]
async fn oauth_device_flow_authorize_and_cached_completion_smoke() {
    let temp = TempDir::new().expect("temp dir");
    let token_dir = temp.path();

    let client = oauth_builder_with_token_dir(token_dir)
        .build()
        .expect("Copilot OAuth client should build");

    client
        .authorize()
        .await
        .expect("device authorization should succeed");

    assert!(
        token_dir.join("access-token").is_file(),
        "device flow should cache the GitHub access token"
    );
    assert!(
        token_dir.join("api-key.json").is_file(),
        "device flow should cache the Copilot API key"
    );

    client
        .authorize()
        .await
        .expect("cached oauth auth should succeed");

    let response = client
        .agent(LIVE_MODEL)
        .preamble(BASIC_PREAMBLE)
        .build()
        .prompt(BASIC_PROMPT)
        .await
        .expect("authorized completion should succeed");

    assert_nonempty_response(&response);

    let cached_client = oauth_builder_with_token_dir(token_dir)
        .build()
        .expect("cached Copilot client should build");
    cached_client
        .authorize()
        .await
        .expect("cached oauth auth should succeed");
    let cached_response = cached_client
        .agent(LIVE_MODEL)
        .build()
        .prompt("Reply with the single word cached.")
        .await
        .expect("cached completion should succeed");

    assert_nonempty_response(&cached_response);
}

#[tokio::test]
#[ignore = "requires COPILOT_GITHUB_ACCESS_TOKEN or GITHUB_TOKEN"]
async fn access_token_bootstrap_refresh_and_completion_smoke() {
    let temp = TempDir::new().expect("temp dir");
    let token_dir = temp.path();

    fs::write(
        token_dir.join("access-token"),
        required_copilot_github_access_token(),
    )
    .expect("access token should be written");
    fs::write(
        token_dir.join("api-key.json"),
        serde_json::to_vec_pretty(&json!({
            "token": "expired-token",
            "expires_at": 0,
        }))
        .expect("expired api key record"),
    )
    .expect("expired api key record should be written");

    let client = oauth_builder_with_token_dir(token_dir)
        .build()
        .expect("Copilot OAuth client should build");

    client
        .authorize()
        .await
        .expect("bootstrap refresh should succeed");

    let api_key_record: serde_json::Value = serde_json::from_slice(
        &fs::read(token_dir.join("api-key.json")).expect("api key record should exist"),
    )
    .expect("api key record should deserialize");

    assert!(
        api_key_record
            .get("token")
            .and_then(|value| value.as_str())
            .is_some(),
        "bootstrap refresh should persist a Copilot API key"
    );

    if let Some(api_base) = api_key_record
        .get("endpoints")
        .and_then(|value| value.get("api"))
        .and_then(|value| value.as_str())
    {
        assert!(
            !api_base.trim().is_empty(),
            "dynamic Copilot API base should not be empty when present"
        );
    }

    let response = client
        .agent(LIVE_MODEL)
        .preamble(BASIC_PREAMBLE)
        .build()
        .prompt(BASIC_PROMPT)
        .await
        .expect("bootstrap-backed completion should succeed");

    assert_nonempty_response(&response);
}
