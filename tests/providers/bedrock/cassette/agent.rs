//! AWS Bedrock agent completion replay smoke test.

use rig::agent::AgentBuilder;
use rig::bedrock;
use rig::client::CompletionClient;
use rig::completion::Prompt;

use super::super::support::with_bedrock_cassette;
use crate::support::{
    Adder, BASIC_PREAMBLE, BASIC_PROMPT, CONTEXT_DOCS, CONTEXT_PROMPT, STREAMING_TOOLS_PREAMBLE,
    STREAMING_TOOLS_PROMPT, Subtract, assert_contains_any_case_insensitive,
    assert_mentions_expected_number, assert_nonempty_response,
};

#[tokio::test]
async fn completion_smoke() {
    with_bedrock_cassette("agent/completion_smoke", |client| async move {
        let agent = client
            .agent(bedrock::completion::AMAZON_NOVA_LITE)
            .preamble(BASIC_PREAMBLE)
            .build();

        let response = agent
            .prompt(BASIC_PROMPT)
            .await
            .expect("completion should succeed");

        assert_nonempty_response(&response);
    })
    .await;
}

#[tokio::test]
async fn completion_with_context_smoke() {
    with_bedrock_cassette("agent/completion_with_context_smoke", |client| async move {
        let agent = client
            .agent(bedrock::completion::AMAZON_NOVA_LITE)
            .preamble("Answer the user using only the supplied context.")
            .context(CONTEXT_DOCS[0])
            .context(CONTEXT_DOCS[1])
            .context(CONTEXT_DOCS[2])
            .build();

        let response = agent
            .prompt(CONTEXT_PROMPT)
            .await
            .expect("context completion should succeed");

        assert_contains_any_case_insensitive(&response, &["ancient tool", "farm"]);
    })
    .await;
}

#[tokio::test]
async fn tool_roundtrip_smoke() {
    with_bedrock_cassette("agent/tool_roundtrip_smoke", |client| async move {
        let agent = client
            .agent(bedrock::completion::AMAZON_NOVA_LITE)
            .preamble(STREAMING_TOOLS_PREAMBLE)
            .max_tokens(1024)
            .tool(Adder)
            .tool(Subtract)
            .default_max_turns(2)
            .build();

        let response = agent
            .prompt(STREAMING_TOOLS_PROMPT)
            .await
            .expect("tool prompt should succeed");

        assert_mentions_expected_number(&response, -3);
    })
    .await;
}

#[tokio::test]
async fn prompt_caching_completion_smoke() {
    with_bedrock_cassette(
        "agent/prompt_caching_completion_smoke",
        |client| async move {
            let model = client
                .completion_model(bedrock::completion::AMAZON_NOVA_LITE)
                .with_prompt_caching();
            let agent = AgentBuilder::new(model).preamble(BASIC_PREAMBLE).build();

            let response = agent
                .prompt(BASIC_PROMPT)
                .await
                .expect("prompt-caching completion should succeed");

            assert_nonempty_response(&response);
        },
    )
    .await;
}
