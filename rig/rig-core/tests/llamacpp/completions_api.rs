//! Migrated from `examples/openai_agent_completions_api.rs` against a local llama.cpp server.

use rig::client::CompletionClient;
use rig::completion::Prompt;

use crate::support::assert_nonempty_response;

use super::support;

#[tokio::test]
#[ignore = "requires a local llama.cpp OpenAI-compatible server"]
async fn completions_api_agent_prompt() {
    let agent = support::client()
        .completion_model(support::model_name())
        .completions_api()
        .into_agent_builder()
        .preamble("You are a helpful assistant.")
        .build();

    let response = agent
        .prompt("Hello world!")
        .await
        .expect("completions api prompt should succeed");

    assert_nonempty_response(&response);
}
