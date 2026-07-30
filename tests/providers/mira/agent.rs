//! Mira agent completion smoke test.

use rig::prelude::*;
use rig::providers::{mira, openai};

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires MIRA_API_KEY"]
async fn completion_smoke() {
    let cfg = mira::functions::Config::from_env(openai::GPT_4O).expect("config should build");
    let agent = AgentBuilder::new(ProviderConfig::Mira(cfg))
        .preamble(BASIC_PREAMBLE)
        .build();

    let response = agent
        .prompt(BASIC_PROMPT)
        .await
        .expect("completion should succeed");

    assert_nonempty_response(&response);
}
