//! Live instruction-routing checks for a local llamafile server.

use rig::client::CompletionClient;
use rig::completion::Prompt;

use crate::support::{
    INSTRUCTIONS_CONFLICT_PROMPT, INSTRUCTIONS_EXPECTED_MARKER, INSTRUCTIONS_PREAMBLE,
    assert_contains_any_case_insensitive,
};

use super::support;

#[tokio::test]
#[ignore = "requires a local llamafile server at http://localhost:8080"]
async fn preamble_is_honored() {
    if support::skip_if_server_unavailable() {
        return;
    }

    let response = support::client()
        .agent(support::model_name())
        .preamble(INSTRUCTIONS_PREAMBLE)
        .build()
        .prompt(INSTRUCTIONS_CONFLICT_PROMPT)
        .await
        .expect("llamafile completion should succeed");

    assert_contains_any_case_insensitive(&response, &[INSTRUCTIONS_EXPECTED_MARKER]);
}
