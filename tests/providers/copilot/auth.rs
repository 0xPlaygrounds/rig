//! Copilot OAuth and bootstrap smoke tests.
//!
//! The classic `client.authorize()` step is now credential resolution at
//! config-construction time: an [`Authenticator`] resolves the credential and
//! `config_from_auth` folds it (and the token-reported endpoint) into a
//! [`copilot::functions::Config`].

use assert_fs::TempDir;
use rig::AgentBuilder;
use rig::provider::ProviderConfig;
use rig::providers::copilot;
use rig::providers::copilot::auth::{AuthSource, Authenticator};
use serde_json::json;
use std::fs;
use std::path::Path;

use crate::copilot::{
    LIVE_MODEL, authenticator, config_from_auth, copilot_api_key, copilot_github_access_token,
};
use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

fn required_copilot_api_key() -> String {
    copilot_api_key().expect("GITHUB_COPILOT_API_KEY or COPILOT_API_KEY should be set")
}

fn required_copilot_github_access_token() -> String {
    copilot_github_access_token()
        .expect("COPILOT_GITHUB_ACCESS_TOKEN or GITHUB_TOKEN should be set")
}

fn oauth_authenticator_with_token_dir(path: &Path) -> Authenticator {
    authenticator(AuthSource::OAuth, Some(path), true)
}

/// Resolve `auth` into an [`AgentBuilder`] for `model` — the replacement for
/// `client.authorize()` followed by `client.agent(model)`.
async fn agent_for(model: &str, auth: &Authenticator) -> AgentBuilder {
    AgentBuilder::new(ProviderConfig::Copilot(config_from_auth(model, auth).await))
}

#[tokio::test]
#[ignore = "requires GITHUB_COPILOT_API_KEY or COPILOT_API_KEY"]
async fn api_key_completion_smoke() {
    let auth = authenticator(AuthSource::ApiKey(required_copilot_api_key()), None, false);

    let response = agent_for(LIVE_MODEL, &auth)
        .await
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
    let auth = authenticator(
        AuthSource::GitHubAccessToken(required_copilot_github_access_token()),
        None,
        false,
    );

    let response = agent_for(LIVE_MODEL, &auth)
        .await
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

    let auth = oauth_authenticator_with_token_dir(token_dir);
    auth.auth_context()
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

    let response = agent_for(LIVE_MODEL, &auth)
        .await
        .preamble(BASIC_PREAMBLE)
        .build()
        .prompt(BASIC_PROMPT)
        .await
        .expect("authorized completion should succeed");

    assert_nonempty_response(&response);

    let cached_auth = oauth_authenticator_with_token_dir(token_dir);
    let cached_response = agent_for(LIVE_MODEL, &cached_auth)
        .await
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

    let auth = oauth_authenticator_with_token_dir(token_dir);
    // Resolving the config performs the bootstrap refresh the classic
    // `client.authorize()` used to trigger.
    let cfg: copilot::functions::Config = config_from_auth(LIVE_MODEL, &auth).await;

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

    let response = AgentBuilder::new(ProviderConfig::Copilot(cfg))
        .preamble(BASIC_PREAMBLE)
        .build()
        .prompt(BASIC_PROMPT)
        .await
        .expect("bootstrap-backed completion should succeed");

    assert_nonempty_response(&response);
}
