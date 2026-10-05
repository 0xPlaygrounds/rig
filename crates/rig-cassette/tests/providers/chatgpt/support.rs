use rig::providers::chatgpt;
use rig::providers::openai::OpenAIConfig;
use rig_test_support::cassette_models::OpenAiModels;
use std::future::Future;
use std::panic::AssertUnwindSafe;

use crate::cassettes::{CassetteSpec, ProviderCassette};
use futures::FutureExt;

async fn chatgpt_cassette_with_default_instructions(
    spec: impl Into<CassetteSpec>,
    default_instructions: impl Into<String>,
) -> (ProviderCassette, OpenAIConfig) {
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "chatgpt",
        spec,
        "https://chatgpt.com/backend-api/codex",
    )
    .await;
    let client =
        OpenAIConfig::with_key(&chatgpt::DIALECT, cassette.api_key("CHATGPT_ACCESS_TOKEN"))
            .with_account_id(cassette.api_key("CHATGPT_ACCOUNT_ID"))
            .with_base_url(cassette.base_url())
            .with_instructions(default_instructions);

    (cassette, client)
}

async fn chatgpt_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, OpenAIConfig) {
    chatgpt_cassette_with_default_instructions(spec, "").await
}

pub(super) async fn with_chatgpt_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(OpenAiModels) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = chatgpt_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(OpenAiModels::new(
        client,
        rig_test_support::cassettes::local_http(),
    )))
    .catch_unwind()
    .await;
    cassette.finish_after_test(result).await;
}
