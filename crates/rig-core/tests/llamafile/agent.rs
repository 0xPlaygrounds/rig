//! Migrated from `examples/agent_with_llamafile.rs`.

use rig_core::client::CompletionClient;
use rig_core::completion::Prompt;

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

use super::support;

#[tokio::test]
#[ignore = "requires a local llamafile server at http://localhost:8080"]
async fn completion_smoke() {
    if support::skip_if_server_unavailable() {
        return;
    }

    let client = support::client();
    let agent = client
        .agent(support::model_name())
        .preamble(BASIC_PREAMBLE)
        .build();

    let response = agent
        .prompt(BASIC_PROMPT)
        .await
        .expect("prompt should succeed");

    assert_nonempty_response(&response);
}
