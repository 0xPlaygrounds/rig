use assert_fs::TempDir;
use rig::driver::{Bind as _, Bound};
use rig::prelude::*;
use rig::providers::chatgpt;
use rig::providers::openai::OpenAI;
use rig::rig_reqwest::client::bundled;
use std::future::Future;
use std::panic::AssertUnwindSafe;

use crate::cassettes::{CassetteSpec, ProviderCassette};
use futures::FutureExt;

async fn chatgpt_cassette_with_default_instructions(
    spec: impl Into<CassetteSpec>,
    default_instructions: impl Into<String>,
) -> (ProviderCassette, Bound<OpenAI>) {
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "chatgpt",
        spec,
        "https://chatgpt.com/backend-api/codex",
    )
    .await;
    let client = OpenAI::with_key(&chatgpt::DIALECT, cassette.api_key("CHATGPT_ACCESS_TOKEN"))
        .with_account_id(cassette.api_key("CHATGPT_ACCOUNT_ID"))
        .with_base_url(cassette.base_url())
        .with_instructions(default_instructions)
        .bound()
        .expect("transport should build");

    (cassette, client)
}

async fn chatgpt_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, Bound<OpenAI>) {
    chatgpt_cassette_with_default_instructions(spec, "").await
}

async fn chatgpt_noninteractive_oauth_cassette(
    spec: impl Into<CassetteSpec>,
) -> (ProviderCassette, Bound<OpenAI>, TempDir) {
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "chatgpt",
        spec,
        "https://chatgpt.com/backend-api/codex",
    )
    .await;
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

    // The credential exchange is not a wire: `OpenAI` holds an
    // already-exchanged token, so the cache the record above seeded is read
    // first — on the very transport the completion then speaks over, and with
    // the device flow refused so a stale cache fails loudly.
    let http = bundled().expect("transport should build");
    let context = chatgpt::auth::Authenticator::new(
        chatgpt::auth::AuthSource::OAuth,
        Some(auth_file),
        chatgpt::auth::DeviceCodeHandler::default(),
        false,
    )
    .auth_context(&http)
    .await
    .expect("non-interactive ChatGPT OAuth cassette credential should resolve");

    let mut provider = OpenAI::with_key(&chatgpt::DIALECT, context.access_token)
        .with_base_url(cassette.base_url())
        .with_instructions("");
    if let Some(account_id) = context.account_id {
        provider = provider.with_account_id(account_id);
    }

    (cassette, provider.bind(http), temp)
}

pub(super) async fn with_chatgpt_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(Bound<OpenAI>) -> Fut,
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
    F: FnOnce(Bound<OpenAI>) -> Fut,
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
    F: FnOnce(Bound<OpenAI>) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client, _temp) = chatgpt_noninteractive_oauth_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}
