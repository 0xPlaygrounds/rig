//! Live instruction-routing checks for a local llama.cpp OpenAI-compatible server.

use rig::client::CompletionClient;
use rig::completion::Prompt;

use crate::support::{
    INSTRUCTIONS_CONFLICT_PROMPT, INSTRUCTIONS_EXPECTED_MARKER, INSTRUCTIONS_PREAMBLE,
    assert_contains_any_case_insensitive,
};

use super::support;

#[tokio::test]
#[ignore = "requires a local llama.cpp OpenAI-compatible server"]
async fn chat_completions_preamble_is_honored() {
    let response = support::completions_client()
        .agent(support::model_name())
        .preamble(INSTRUCTIONS_PREAMBLE)
        .build()
        .prompt(INSTRUCTIONS_CONFLICT_PROMPT)
        .await
        .expect("llama.cpp Chat Completions-compatible completion should succeed");

    assert_contains_any_case_insensitive(&response, &[INSTRUCTIONS_EXPECTED_MARKER]);
}

#[tokio::test]
#[ignore = "requires a local llama.cpp OpenAI-compatible server with /responses support"]
async fn responses_api_preamble_is_honored_or_reveals_need_for_compat_shim() {
    let response = support::client()
        .agent(support::model_name())
        .preamble(INSTRUCTIONS_PREAMBLE)
        .build()
        .prompt(INSTRUCTIONS_CONFLICT_PROMPT)
        .await
        .expect("llama.cpp Responses-compatible completion should succeed");

    assert_contains_any_case_insensitive(&response, &[INSTRUCTIONS_EXPECTED_MARKER]);
}
