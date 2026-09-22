//! Cohere tools smoke test.

use rig::prelude::*;
use rig::providers::cohere::{self, wire::Cohere};

use crate::support::{
    Adder, Subtract, TOOLS_PREAMBLE, TOOLS_PROMPT, assert_mentions_expected_number,
};

#[tokio::test]
#[ignore = "requires COHERE_API_KEY"]
async fn tools_smoke() {
    let cohere = Cohere::from_env()
        .expect("config should build from env")
        .bound()
        .expect("transport should build");
    let agent = cohere
        .agent(cohere::COMMAND_A_03_2025)
        .preamble(TOOLS_PREAMBLE)
        .tool(Adder)
        .tool(Subtract)
        .build();

    let response = agent
        .prompt(TOOLS_PROMPT)
        .await
        .expect("tool prompt should succeed");

    assert_mentions_expected_number(&response.output, -3);
}
