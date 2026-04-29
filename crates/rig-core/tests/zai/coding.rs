//! Z.AI coding OpenAI-compatible completion smoke test.

use rig_core::client::CompletionClient;
use rig_core::completion::Prompt;
use rig_core::providers::zai;

use crate::support::assert_nonempty_response;
use crate::zai::coding_client;

#[tokio::test]
#[ignore = "requires ZAI_API_KEY"]
async fn coding_openai_compatible_completion_smoke() {
    let response = coding_client()
        .agent(zai::GLM_4_6)
        .preamble("You are a concise coding assistant.")
        .build()
        .prompt("In one short sentence, explain what a unit test is.")
        .await
        .expect("Z.AI coding completion should succeed");

    assert_nonempty_response(&response);
}
