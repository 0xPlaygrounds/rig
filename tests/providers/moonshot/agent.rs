//! Moonshot agent completion smoke test.

use rig::prelude::*;
use rig::providers::moonshot;

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires MOONSHOT_API_KEY"]
async fn completion_smoke() {
    let cfg = moonshot::functions::Config::from_env(moonshot::MOONSHOT_CHAT)
        .expect("moonshot config should build");
    let agent = AgentBuilder::new(ProviderConfig::Moonshot(cfg))
        .preamble(BASIC_PREAMBLE)
        .temperature(0.5)
        .max_tokens(1024)
        .build();

    let response = agent
        .prompt(BASIC_PROMPT)
        .await
        .expect("completion should succeed");

    assert_nonempty_response(&response);
}
