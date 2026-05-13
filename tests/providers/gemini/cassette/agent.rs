//! Gemini agent completion smoke test.

use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::providers::gemini;

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
async fn completion_smoke() {
    let (cassette, client) = super::super::support::gemini_cassette("agent/completion_smoke").await;
    let agent = client
        .agent(gemini::completion::GEMINI_2_5_FLASH)
        .preamble(BASIC_PREAMBLE)
        .build();

    let response = agent
        .prompt(BASIC_PROMPT)
        .await
        .expect("completion should succeed");

    assert_nonempty_response(&response);

    cassette.finish().await;
}
