//! Copilot route-specific completion smoke tests.

use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::providers::copilot;

use crate::copilot::live_client;
use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires Copilot credentials or existing OAuth cache"]
async fn chat_models_route_through_chat_completions() {
    let response = live_client()
        .agent(copilot::GPT_4O)
        .preamble(BASIC_PREAMBLE)
        .build()
        .prompt(BASIC_PROMPT)
        .await
        .expect("chat-completions route should succeed");

    assert_nonempty_response(&response);
}

#[tokio::test]
#[ignore = "requires Copilot credentials or existing OAuth cache"]
async fn codex_models_route_through_responses() {
    let response = live_client()
        .agent(copilot::GPT_5_1_CODEX)
        .preamble(BASIC_PREAMBLE)
        .build()
        .prompt("In one short sentence, explain what refactoring is.")
        .await
        .expect("responses route should succeed");

    assert_nonempty_response(&response);
}
