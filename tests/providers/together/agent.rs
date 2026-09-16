//! Together agent completion smoke test.

use rig::prelude::*;
use rig::providers::openai::wire::{OpenAI, TOGETHER};
use rig::providers::together;

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires TOGETHER_API_KEY"]
async fn completion_smoke() {
    let provider = OpenAI::from_env_with(&TOGETHER)
        .expect("config should build from env")
        .bound()
        .expect("transport should build");
    let agent = provider
        .agent(together::MIXTRAL_8X7B_INSTRUCT_V0_1)
        .preamble(BASIC_PREAMBLE)
        .build();

    let response = agent
        .prompt(BASIC_PROMPT)
        .await
        .expect("completion should succeed");

    assert_nonempty_response(&response.output);
}
