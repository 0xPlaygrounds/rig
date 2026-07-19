//! Bedrock Mantle OpenAI-compatible agent smoke tests (HTTP ProviderCassette).

use rig::bedrock::mantle::OPENAI_GPT_OSS_20B;
use rig::client::CompletionClient;
use rig::completion::Prompt;

use super::super::super::support::{
    with_bedrock_mantle_cassette, with_bedrock_mantle_completions_cassette,
};
use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
async fn completion_smoke() {
    with_bedrock_mantle_completions_cassette(
        "mantle/agent/completion_smoke",
        |client| async move {
            let agent = client
                .agent(OPENAI_GPT_OSS_20B)
                .preamble(BASIC_PREAMBLE)
                .build();

            let response = agent
                .prompt(BASIC_PROMPT)
                .await
                .expect("mantle completions should succeed");

            assert_nonempty_response(&response);
        },
    )
    .await;
}

#[tokio::test]
async fn responses_smoke() {
    with_bedrock_mantle_cassette("mantle/agent/responses_smoke", |client| async move {
        let agent = client
            .agent(OPENAI_GPT_OSS_20B)
            .preamble(BASIC_PREAMBLE)
            .additional_params(serde_json::json!({"store": false}))
            .build();

        let response = agent
            .prompt(BASIC_PROMPT)
            .await
            .expect("mantle responses should succeed");

        assert_nonempty_response(&response);
    })
    .await;
}
