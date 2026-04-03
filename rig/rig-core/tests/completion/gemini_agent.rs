//! Gemini agent completion smoke test.

use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers::gemini;

use crate::support::{PREAMBLE, PROMPT, assert_nontrivial_response};

#[tokio::test]
#[ignore = "requires GEMINI_API_KEY"]
async fn completion_smoke() {
    let client = gemini::Client::from_env();
    let agent = client
        .agent(gemini::completion::GEMINI_2_5_FLASH)
        .preamble(PREAMBLE)
        .build();

    let response = agent
        .prompt(PROMPT)
        .await
        .expect("completion should succeed");

    assert_nontrivial_response(&response);
}
