//! Live instruction-routing checks for Hyperbolic.

use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers::hyperbolic;

use crate::support::{
    INSTRUCTIONS_CONFLICT_PROMPT, INSTRUCTIONS_EXPECTED_MARKER, INSTRUCTIONS_PREAMBLE,
    assert_contains_any_case_insensitive,
};

#[tokio::test]
#[ignore = "requires HYPERBOLIC_API_KEY"]
async fn preamble_is_honored() {
    let response = hyperbolic::Client::from_env()
        .expect("client should build")
        .agent(hyperbolic::LLAMA_3_1_8B)
        .preamble(INSTRUCTIONS_PREAMBLE)
        .build()
        .prompt(INSTRUCTIONS_CONFLICT_PROMPT)
        .await
        .expect("Hyperbolic completion should succeed");

    assert_contains_any_case_insensitive(&response, &[INSTRUCTIONS_EXPECTED_MARKER]);
}
