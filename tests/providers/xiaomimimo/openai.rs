//! Xiaomi MiMo OpenAI-compatible completion smoke test.

use rig::prelude::*;
use rig::providers::openai::wire::{self as openai_wire, OpenAI};
use rig::providers::xiaomimimo;

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires XIAOMI_MIMO_API_KEY"]
async fn openai_compatible_completion_smoke() {
    let response = OpenAI::from_env_with(&openai_wire::XIAOMIMIMO)
        .expect("XIAOMI_MIMO_API_KEY should be set")
        .bound()
        .expect("client should build")
        .agent(xiaomimimo::MIMO_V2_5_PRO)
        .preamble(BASIC_PREAMBLE)
        .build()
        .prompt(BASIC_PROMPT)
        .await
        .expect("Xiaomi MiMo OpenAI-compatible completion should succeed")
        .output;

    assert_nonempty_response(&response);
}
