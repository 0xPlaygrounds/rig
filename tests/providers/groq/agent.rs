//! Groq agent completion smoke test.

use rig::prelude::*;
use rig::providers::openai::wire::{GROQ, OpenAI};

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

use super::AGENT_MODEL;

#[tokio::test]
#[ignore = "requires GROQ_API_KEY"]
async fn completion_smoke() {
    let groq = OpenAI::from_env_with(&GROQ)
        .expect("GROQ_API_KEY should be set")
        .bound()
        .expect("transport should build");
    let agent = groq.agent(AGENT_MODEL).preamble(BASIC_PREAMBLE).build();

    let response = agent
        .prompt(BASIC_PROMPT)
        .await
        .expect("completion should succeed");

    assert_nonempty_response(&response.output);
}
