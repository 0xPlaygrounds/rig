//! Doubleword agent completion smoke test.

use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers::doubleword;

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires DOUBLEWORD_API_KEY"]
async fn completion_smoke() {
    let client = doubleword::Client::from_env().expect("client should build");
    let agent = client
        .agent(doubleword::QWEN3_5_9B)
        .preamble(BASIC_PREAMBLE)
        .build();

    let response = agent
        .prompt(BASIC_PROMPT)
        .await
        .expect("completion should succeed");

    assert_nonempty_response(&response);
}
