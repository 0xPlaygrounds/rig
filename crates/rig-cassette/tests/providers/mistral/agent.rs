//! Mistral agent completion smoke test.

use rig::providers::openai::wire::MISTRAL;
use rig_test_support::cassette_models::OpenAiModels;

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

use super::DEFAULT_MODEL;

#[tokio::test]
#[ignore = "requires MISTRAL_API_KEY"]
async fn completion_smoke() {
    let client = OpenAiModels::from_env_for(&MISTRAL).expect("MISTRAL_API_KEY should be set");
    let agent = rig::AgentBuilder::new(client.completion(DEFAULT_MODEL))
        .preamble(BASIC_PREAMBLE)
        .build();

    let response = agent
        .prompt(BASIC_PROMPT)
        .await
        .expect("completion should succeed");

    assert_nonempty_response(&response.output());
}
