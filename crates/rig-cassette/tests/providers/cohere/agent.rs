//! Cohere agent completion smoke test.

use rig::providers::cohere::{self, wire::Cohere};
use rig::wire::Wire as _;

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires COHERE_API_KEY"]
async fn completion_smoke() {
    let cohere = Cohere::from_env().expect("config should build from env");
    let agent = rig::AgentBuilder::new(
        cohere
            .completion(cohere::COMMAND_A_03_2025)
            .on(rig::transport()),
    )
    .preamble(BASIC_PREAMBLE)
    .build();

    let response = agent
        .prompt(BASIC_PROMPT)
        .await
        .expect("completion should succeed");

    assert_nonempty_response(&response.output);
}
