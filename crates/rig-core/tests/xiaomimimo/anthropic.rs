//! Xiaomi MiMo Anthropic-compatible completion smoke test.

use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::completion::Prompt;
use rig_core::providers::xiaomimimo;

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires XIAOMI_MIMO_API_KEY"]
async fn anthropic_compatible_completion_smoke() {
    let response = xiaomimimo::AnthropicClient::from_env()
        .expect("client should build")
        .agent(xiaomimimo::MIMO_V2_5_PRO)
        .preamble(BASIC_PREAMBLE)
        .build()
        .prompt(BASIC_PROMPT)
        .await
        .expect("Xiaomi MiMo Anthropic-compatible completion should succeed");

    assert_nonempty_response(&response);
}
