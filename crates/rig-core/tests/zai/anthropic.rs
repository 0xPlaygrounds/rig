//! Z.AI Anthropic-compatible completion smoke test.

use rig_core::client::CompletionClient;
use rig_core::completion::Prompt;
use rig_core::providers::zai;

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};
use crate::zai::anthropic_client;

#[tokio::test]
#[ignore = "requires ZAI_API_KEY"]
async fn anthropic_compatible_completion_smoke() {
    let response = anthropic_client()
        .agent(zai::GLM_4_6)
        .preamble(BASIC_PREAMBLE)
        .build()
        .prompt(BASIC_PROMPT)
        .await
        .expect("Z.AI Anthropic-compatible completion should succeed");

    assert_nonempty_response(&response);
}
