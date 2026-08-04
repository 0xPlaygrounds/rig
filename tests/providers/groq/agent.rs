//! Groq agent completion smoke test.

use rig::prelude::*;

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

use super::AGENT_MODEL;

#[tokio::test]
#[ignore = "requires GROQ_API_KEY"]
async fn completion_smoke() {
    let agent = rig::providers::groq::Client::from_env()
        .expect("client should build")
        .agent(AGENT_MODEL)
        .preamble(BASIC_PREAMBLE)
        .build();

    let response = agent
        .prompt(BASIC_PROMPT)
        .await
        .expect("completion should succeed");

    assert_nonempty_response(&response);
}
