//! OrcaRouter agent completion smoke test.

use rig::completion::Prompt;
use rig::prelude::*;
use rig::providers::orcarouter;

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires ORCAROUTER_API_KEY"]
async fn completion_smoke() {
    let client = orcarouter::Client::from_env().expect("client should build");
    let agent = client
        .agent(orcarouter::ORCAROUTER_AUTO)
        .preamble(BASIC_PREAMBLE)
        .build();

    let response = agent
        .prompt(BASIC_PROMPT)
        .await
        .expect("completion should succeed");

    assert_nonempty_response(&response);
}
