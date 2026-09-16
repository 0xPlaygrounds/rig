//! MiniMax Anthropic-compatible completion smoke test.

use rig::prelude::*;
use rig::providers::anthropic::wire::{self as anthropic_wire, Anthropic};
use rig::providers::minimax;

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires MINIMAX_API_KEY"]
async fn anthropic_compatible_completion_smoke() {
    let response = Anthropic::from_env_with(&anthropic_wire::MINIMAX)
        .expect("MINIMAX_API_KEY should be set")
        .bound()
        .expect("client should build")
        .agent(minimax::MINIMAX_M2)
        .preamble(BASIC_PREAMBLE)
        .build()
        .prompt(BASIC_PROMPT)
        .await
        .expect("MiniMax Anthropic-compatible completion should succeed")
        .output;

    assert_nonempty_response(&response);
}
