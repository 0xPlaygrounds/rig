//! OpenRouter agent completion smoke test.

use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers::gemini::completion::GEMINI_2_5_PRO_EXP_03_25;
use rig::providers::openrouter;

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY"]
async fn completion_smoke() {
    let client = openrouter::Client::from_env();
    let agent = client
        .agent(GEMINI_2_5_PRO_EXP_03_25)
        .preamble(BASIC_PREAMBLE)
        .build();

    let response = agent
        .prompt(BASIC_PROMPT)
        .await
        .expect("completion should succeed");

    assert_nonempty_response(&response);
}
