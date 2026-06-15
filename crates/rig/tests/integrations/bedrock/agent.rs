//! AWS Bedrock completion smoke tests inspired by the OpenAI and Anthropic provider tests.

use rig::agent::AgentBuilder;
use rig::client::CompletionClient;
use rig::completion::Prompt;

use super::{
    BEDROCK_COMPLETION_MODEL, client,
    support::{
        Adder, BASIC_PREAMBLE, BASIC_PROMPT, CONTEXT_DOCS, CONTEXT_PROMPT,
        STREAMING_TOOLS_PREAMBLE, STREAMING_TOOLS_PROMPT, Subtract,
        assert_contains_any_case_insensitive, assert_mentions_expected_number,
        assert_nonempty_response,
    },
};

#[tokio::test]
#[ignore = "requires AWS credentials and Bedrock model access"]
async fn completion_smoke() {
    let agent = client()
        .agent(BEDROCK_COMPLETION_MODEL)
        .preamble(BASIC_PREAMBLE)
        .build();

    let response = agent
        .prompt(BASIC_PROMPT)
        .await
        .expect("completion should succeed");

    assert_nonempty_response(&response);
}

#[tokio::test]
#[ignore = "requires AWS credentials and Bedrock model access"]
async fn completion_with_context_smoke() {
    let agent = client()
        .agent(BEDROCK_COMPLETION_MODEL)
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
}

#[tokio::test]
#[ignore = "requires AWS credentials and Bedrock model access"]
async fn tool_roundtrip_smoke() {
    let agent = client()
        .agent(BEDROCK_COMPLETION_MODEL)
        .preamble(STREAMING_TOOLS_PREAMBLE)
        .max_tokens(1024)
        .tool(Adder)
        .tool(Subtract)
        .build();

    let response = agent
        .prompt(STREAMING_TOOLS_PROMPT)
        .await
        .expect("tool prompt should succeed");

    assert_mentions_expected_number(&response, -3);
}

#[tokio::test]
#[ignore = "requires AWS credentials and Bedrock model access"]
async fn prompt_caching_completion_smoke() {
    let bedrock_client = client();
    let model = bedrock_client
        .completion_model(BEDROCK_COMPLETION_MODEL)
        .with_prompt_caching();
    let agent = AgentBuilder::new(model).preamble(BASIC_PREAMBLE).build();

    let response = agent
        .prompt(BASIC_PROMPT)
        .await
        .expect("prompt-caching completion should succeed");

    assert_nonempty_response(&response);
}
