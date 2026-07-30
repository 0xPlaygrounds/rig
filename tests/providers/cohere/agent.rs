//! Cohere agent completion smoke test.

use rig::prelude::*;
use rig::providers::cohere;

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires COHERE_API_KEY"]
async fn completion_smoke() {
    let cfg = cohere::functions::Config::from_env(cohere::COMMAND_R).expect("config should build");
    let agent = AgentBuilder::new(ProviderConfig::Cohere(cfg))
        .preamble(BASIC_PREAMBLE)
        .build();

    let response = agent
        .prompt(BASIC_PROMPT)
        .await
        .expect("completion should succeed");

    assert_nonempty_response(&response);
}
