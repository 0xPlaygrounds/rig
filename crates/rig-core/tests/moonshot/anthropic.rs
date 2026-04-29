//! Moonshot Anthropic-compatible completion smoke test.

use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::completion::Prompt;
use rig_core::providers::moonshot;

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires MOONSHOT_API_KEY"]
async fn anthropic_compatible_completion_smoke() {
    let response = moonshot::AnthropicClient::from_env()
        .expect("moonshot anthropic client should build")
        .agent(moonshot::KIMI_K2_5)
        .preamble(BASIC_PREAMBLE)
        .build()
        .prompt(BASIC_PROMPT)
        .await
        .expect("Moonshot Anthropic-compatible completion should succeed");

    assert_nonempty_response(&response);
}
