//! MiniMax OpenAI-compatible completion smoke test.

use rig::prelude::*;
use rig::providers::minimax;

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires MINIMAX_API_KEY"]
async fn openai_compatible_completion_smoke() {
    let cfg =
        minimax::functions::Config::from_env(minimax::MINIMAX_M2_7).expect("config should build");
    let response = AgentBuilder::new(cfg)
        .preamble(BASIC_PREAMBLE)
        .build()
        .prompt(BASIC_PROMPT)
        .await
        .expect("MiniMax OpenAI-compatible completion should succeed");

    assert_nonempty_response(&response);
}
