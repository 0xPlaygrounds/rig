//! Live instruction-routing checks for Copilot Chat and Responses routes.

use rig::client::CompletionClient;
use rig::completion::Prompt;

use crate::copilot::{LIVE_MODEL, live_client, live_responses_model};
use crate::support::{
    INSTRUCTIONS_CONFLICT_PROMPT, INSTRUCTIONS_EXPECTED_MARKER, INSTRUCTIONS_PREAMBLE,
    assert_contains_any_case_insensitive,
};

#[tokio::test]
#[ignore = "requires Copilot credentials or existing OAuth cache"]
async fn chat_route_preamble_is_honored() {
    let response = live_client()
        .agent(LIVE_MODEL)
        .preamble(INSTRUCTIONS_PREAMBLE)
        .build()
        .prompt(INSTRUCTIONS_CONFLICT_PROMPT)
        .await
        .expect("Copilot chat-completions route should succeed");

    assert_contains_any_case_insensitive(&response, &[INSTRUCTIONS_EXPECTED_MARKER]);
}

#[tokio::test]
#[ignore = "requires Copilot credentials or existing OAuth cache"]
async fn responses_route_preamble_compat_shim_is_honored() {
    let response = live_client()
        .agent(live_responses_model())
        .preamble(INSTRUCTIONS_PREAMBLE)
        .build()
        .prompt(INSTRUCTIONS_CONFLICT_PROMPT)
        .await
        .expect("Copilot responses route should succeed");

    assert_contains_any_case_insensitive(&response, &[INSTRUCTIONS_EXPECTED_MARKER]);
}
