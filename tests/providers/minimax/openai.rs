//! MiniMax OpenAI-compatible completion smoke test.

use rig::prelude::*;
use rig::providers::minimax;
use rig::providers::openai::wire::{self as openai_wire, OpenAI};

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires MINIMAX_API_KEY"]
async fn openai_compatible_completion_smoke() {
    let response = OpenAI::from_env_with(&openai_wire::MINIMAX)
        .expect("MINIMAX_API_KEY should be set")
        .bound()
        .expect("client should build")
        .agent(minimax::MINIMAX_M2_7)
        .preamble(BASIC_PREAMBLE)
        .build()
        .prompt(BASIC_PROMPT)
        .await
        .expect("MiniMax OpenAI-compatible completion should succeed")
        .output;

    assert_nonempty_response(&response);
}
