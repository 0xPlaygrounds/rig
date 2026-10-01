//! Groq agent completion smoke test.

use rig::providers::openai::wire::GROQ;
use rig_test_support::cassette_models::OpenAiModels;

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

use super::AGENT_MODEL;

#[tokio::test]
#[ignore = "requires GROQ_API_KEY"]
async fn completion_smoke() {
    let groq = OpenAiModels::from_env_for(&GROQ).expect("GROQ_API_KEY should be set");
    let agent = rig::AgentBuilder::new(groq.completion(AGENT_MODEL))
        .preamble(BASIC_PREAMBLE)
        .build();

    let response = agent
        .prompt(BASIC_PROMPT)
        .await
        .expect("completion should succeed");

    assert_nonempty_response(&response.output());
}
