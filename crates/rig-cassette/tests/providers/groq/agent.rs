//! Groq agent completion smoke test.

use rig::providers::openai::wire::{GROQ, OpenAI};
use rig_test_support::endpoint::Endpoint;

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

use super::AGENT_MODEL;

#[tokio::test]
#[ignore = "requires GROQ_API_KEY"]
async fn completion_smoke() {
    let groq = Endpoint::new(
        OpenAI::from_env_with(&GROQ).expect("GROQ_API_KEY should be set"),
        rig::rig_reqwest::shared(),
    );
    let agent = groq.agent(AGENT_MODEL).preamble(BASIC_PREAMBLE).build();

    let response = agent
        .prompt(BASIC_PROMPT)
        .await
        .expect("completion should succeed");

    assert_nonempty_response(&response.output);
}
