//! Moonshot Anthropic-compatible completion smoke test.

use rig::prelude::*;
use rig::providers::moonshot;

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires MOONSHOT_API_KEY"]
async fn anthropic_compatible_completion_smoke() {
    let cfg = moonshot::functions::anthropic_config_from_env(moonshot::KIMI_K2_5)
        .expect("moonshot anthropic config should build");
    let response = AgentBuilder::new(cfg)
        .preamble(BASIC_PREAMBLE)
        .build()
        .prompt(BASIC_PROMPT)
        .await
        .expect("Moonshot Anthropic-compatible completion should succeed");

    assert_nonempty_response(&response);
}
