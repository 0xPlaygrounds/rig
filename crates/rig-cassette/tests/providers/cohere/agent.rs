//! Cohere agent completion smoke test.

use rig::prelude::*;
use rig::providers::cohere::{self, wire::Cohere};
use rig_test_support::endpoint::Endpoint;

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires COHERE_API_KEY"]
async fn completion_smoke() {
    let cohere = Endpoint::new(
        Cohere::from_env().expect("config should build from env"),
        rig::rig_reqwest::bundled().expect("transport should build"),
    );
    let agent = cohere
        .agent(cohere::COMMAND_A_03_2025)
        .preamble(BASIC_PREAMBLE)
        .build();

    let response = agent
        .prompt(BASIC_PROMPT)
        .await
        .expect("completion should succeed");

    assert_nonempty_response(&response.output);
}
