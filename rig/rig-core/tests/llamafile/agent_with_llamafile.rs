//! Migrated from `examples/agent_with_llamafile.rs`.

use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::providers::llamafile;

use crate::support::assert_nonempty_response;

#[tokio::test]
#[ignore = "requires a local llamafile server at http://localhost:8080"]
async fn completion_smoke() {
    let client = llamafile::Client::from_url("http://localhost:8080");
    let agent = client
        .agent(llamafile::LLAMA_CPP)
        .preamble("You are a helpful assistant.")
        .build();

    let response = agent
        .prompt("Explain what llamafile is in two sentences.")
        .await
        .expect("prompt should succeed");

    assert_nonempty_response(&response);
}
