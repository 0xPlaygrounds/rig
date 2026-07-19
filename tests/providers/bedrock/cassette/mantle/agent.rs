//! Bedrock Mantle OpenAI-compatible agent smoke tests (HTTP ProviderCassette).

use rig::bedrock::mantle::{
    OPENAI_GPT_5_6_LUNA, OPENAI_GPT_5_6_SOL, OPENAI_GPT_5_6_TERRA, OPENAI_GPT_OSS_20B,
};
use rig::client::CompletionClient;
use rig::completion::Prompt;

use super::super::super::support::{
    with_bedrock_mantle_cassette, with_bedrock_mantle_completions_cassette,
    with_bedrock_mantle_gpt5_cassette,
};
use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

/// Shared agent params for GPT-5.x Mantle Responses (avoid 30-day store default).
fn gpt5_agent_params() -> serde_json::Value {
    serde_json::json!({
        "store": false,
        "reasoning": { "effort": "medium" }
    })
}

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

#[tokio::test]
async fn gpt5_luna_responses_smoke() {
    with_bedrock_mantle_gpt5_cassette(
        "mantle/agent/gpt5_luna_responses_smoke",
        |client| async move {
            let agent = client
                .agent(OPENAI_GPT_5_6_LUNA)
                .preamble(BASIC_PREAMBLE)
                .additional_params(gpt5_agent_params())
                .build();

            let response = agent
                .prompt(BASIC_PROMPT)
                .await
                .expect("mantle GPT-5.6 Luna responses should succeed");

            assert_nonempty_response(&response);
        },
    )
    .await;
}

#[tokio::test]
async fn gpt5_sol_responses_smoke() {
    with_bedrock_mantle_gpt5_cassette(
        "mantle/agent/gpt5_sol_responses_smoke",
        |client| async move {
            let agent = client
                .agent(OPENAI_GPT_5_6_SOL)
                .preamble(BASIC_PREAMBLE)
                .additional_params(gpt5_agent_params())
                .build();

            let response = agent
                .prompt(BASIC_PROMPT)
                .await
                .expect("mantle GPT-5.6 Sol responses should succeed");

            assert_nonempty_response(&response);
        },
    )
    .await;
}

#[tokio::test]
async fn gpt5_terra_responses_smoke() {
    with_bedrock_mantle_gpt5_cassette(
        "mantle/agent/gpt5_terra_responses_smoke",
        |client| async move {
            let agent = client
                .agent(OPENAI_GPT_5_6_TERRA)
                .preamble(BASIC_PREAMBLE)
                .additional_params(gpt5_agent_params())
                .build();

            let response = agent
                .prompt(BASIC_PROMPT)
                .await
                .expect("mantle GPT-5.6 Terra responses should succeed");

            assert_nonempty_response(&response);
        },
    )
    .await;
}
