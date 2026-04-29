//! DeepSeek agent completion smoke test.

use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::completion::Prompt;
use rig_core::providers::deepseek;

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY"]
async fn completion_smoke() {
    let client = deepseek::Client::from_env().expect("client should build");
    let agent = client
        .agent(deepseek::DEEPSEEK_V4_FLASH)
        .preamble(BASIC_PREAMBLE)
        .build();

    let response = agent
        .prompt(BASIC_PROMPT)
        .await
        .expect("completion should succeed");

    assert_nonempty_response(&response);
}
