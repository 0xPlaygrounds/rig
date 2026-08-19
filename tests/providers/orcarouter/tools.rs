//! OrcaRouter tools smoke test.

use rig::completion::Prompt;
use rig::prelude::*;
use rig::providers::orcarouter;

use crate::support::{TOOLS_PREAMBLE, TOOLS_PROMPT, assert_mentions_expected_number};

#[tokio::test]
#[ignore = "requires ORCAROUTER_API_KEY"]
async fn tools_smoke() {
    let client = orcarouter::Client::from_env().expect("client should build");
    let agent = client
        .agent(orcarouter::ORCAROUTER_AUTO)
        .preamble(TOOLS_PREAMBLE)
        .default_max_turns(2)
        .tool(crate::support::Adder)
        .tool(crate::support::Subtract)
        .build();

    let response = agent
        .prompt(TOOLS_PROMPT)
        .await
        .expect("tool prompt should succeed");

    assert_mentions_expected_number(&response, -3);
}
