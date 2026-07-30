//! MiniMax Anthropic-compatible completion smoke test.

use rig::prelude::*;
use rig::providers::minimax;

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires MINIMAX_API_KEY"]
async fn anthropic_compatible_completion_smoke() {
    let cfg = minimax::functions::anthropic_config_from_env(minimax::MINIMAX_M2)
        .expect("config should build");
    let response = AgentBuilder::new(ProviderConfig::Anthropic(cfg))
        .preamble(BASIC_PREAMBLE)
        .build()
        .prompt(BASIC_PROMPT)
        .await
        .expect("MiniMax Anthropic-compatible completion should succeed");

    assert_nonempty_response(&response);
}
