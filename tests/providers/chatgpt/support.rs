use assert_fs::TempDir;
use rig::providers::chatgpt::{self, ChatGPTAuth};
use std::future::Future;
use std::panic::AssertUnwindSafe;

use crate::cassettes::{CassetteSpec, ProviderCassette};
use futures::FutureExt;

async fn chatgpt_cassette_with_default_instructions(
    spec: impl Into<CassetteSpec>,
    default_instructions: impl Into<String>,
) -> (ProviderCassette, chatgpt::Client) {
    let cassette =
        ProviderCassette::start("chatgpt", spec, "https://chatgpt.com/backend-api/codex").await;
    let client = chatgpt::Client::builder()
        .api_key(ChatGPTAuth::AccessToken {
            access_token: cassette.api_key("CHATGPT_ACCESS_TOKEN"),
            account_id: Some(cassette.api_key("CHATGPT_ACCOUNT_ID")),
        })
        .base_url(cassette.base_url())
        .default_instructions(default_instructions)
        .build()
        .expect("client should build");

    (cassette, client)
}

async fn chatgpt_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, chatgpt::Client) {
    chatgpt_cassette_with_default_instructions(spec, "").await
}

async fn chatgpt_noninteractive_oauth_cassette(
    spec: impl Into<CassetteSpec>,
) -> (ProviderCassette, chatgpt::Client, TempDir) {
    let cassette =
        ProviderCassette::start("chatgpt", spec, "https://chatgpt.com/backend-api/codex").await;
    let temp = TempDir::new().expect("temp auth directory should be created");
    let auth_file = temp.path().join("auth.json");
    let record = serde_json::json!({
        "access_token": cassette.api_key("CHATGPT_ACCESS_TOKEN"),
        "refresh_token": serde_json::Value::Null,
        "id_token": serde_json::Value::Null,
        "expires_at": i64::MAX,
        "account_id": cassette.api_key("CHATGPT_ACCOUNT_ID"),
    });
    std::fs::write(
        &auth_file,
        serde_json::to_vec_pretty(&record).expect("auth record should serialize"),
    )
    .expect("auth record should be written");

    let client = chatgpt::Client::builder()
        .oauth()
        .allow_device_flow(false)
        .auth_file(&auth_file)
        .base_url(cassette.base_url())
        .default_instructions("")
        .build()
        .expect("non-interactive ChatGPT OAuth cassette client should build");

    (cassette, client, temp)
}

pub(super) async fn with_chatgpt_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(chatgpt::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = chatgpt_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

pub(super) async fn with_chatgpt_cassette_default_instructions<F, Fut>(
    spec: impl Into<CassetteSpec>,
    default_instructions: impl Into<String>,
    test_body: F,
) where
    F: FnOnce(chatgpt::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) =
        chatgpt_cassette_with_default_instructions(spec, default_instructions).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

pub(super) async fn with_chatgpt_noninteractive_oauth_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(chatgpt::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client, _temp) = chatgpt_noninteractive_oauth_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}
