use rig::providers::chatgpt::{self, ChatGPTAuth};
use std::future::Future;
use std::panic::AssertUnwindSafe;

use crate::cassettes::{CassetteSpec, ProviderCassette};
use futures::FutureExt;

async fn chatgpt_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, chatgpt::Client) {
    let cassette =
        ProviderCassette::start("chatgpt", spec, "https://chatgpt.com/backend-api/codex").await;
    let client = chatgpt::Client::builder()
        .api_key(ChatGPTAuth::AccessToken {
            access_token: cassette.api_key("CHATGPT_ACCESS_TOKEN"),
            account_id: Some(cassette.api_key("CHATGPT_ACCOUNT_ID")),
        })
        .base_url(cassette.base_url())
        .default_instructions("")
        .build()
        .expect("client should build");

    (cassette, client)
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
