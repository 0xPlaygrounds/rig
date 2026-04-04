//! Migrated from `examples/agent.rs`.

use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers::openai;

use crate::support::assert_nonempty_response;

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn completion_smoke() {
    let client = openai::Client::from_env();
    let agent = client
        .agent(openai::GPT_4O)
        .preamble("You are a comedian here to entertain the user using humour and jokes.")
        .build();

    let response = agent
        .prompt("Entertain me!")
        .await
        .expect("prompt should succeed");

    assert_nonempty_response(&response);
}
