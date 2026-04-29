//! Moonshot agent completion smoke test.

use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::completion::Prompt;
use rig_core::providers::moonshot;

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires MOONSHOT_API_KEY"]
async fn completion_smoke() {
    let client = moonshot::Client::from_env().expect("moonshot client should build");
    let agent = client
        .agent(moonshot::MOONSHOT_CHAT)
        .preamble(BASIC_PREAMBLE)
        .temperature(0.5)
        .max_tokens(1024)
        .build();

    let response = agent
        .prompt(BASIC_PROMPT)
        .await
        .expect("completion should succeed");

    assert_nonempty_response(&response);
}
