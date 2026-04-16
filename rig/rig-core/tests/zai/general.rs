//! Z.AI general OpenAI-compatible completion smoke test.

use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::providers::zai;

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};
use crate::zai::general_client;

#[tokio::test]
#[ignore = "requires ZAI_API_KEY"]
async fn general_openai_compatible_completion_smoke() {
    let response = general_client()
        .agent(zai::GLM_4_6)
        .preamble(BASIC_PREAMBLE)
        .build()
        .prompt(BASIC_PROMPT)
        .await
        .expect("Z.AI general completion should succeed");

    assert_nonempty_response(&response);
}
