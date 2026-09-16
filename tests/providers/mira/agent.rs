//! Mira agent completion smoke test.

use rig::prelude::*;
use rig::providers::openai::{
    self,
    wire::{MIRA, OpenAI},
};

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires MIRA_API_KEY"]
async fn completion_smoke() {
    let provider = OpenAI::from_env_with(&MIRA)
        .expect("config should build from env")
        .bound()
        .expect("transport should build");
    let agent = provider
        .agent(openai::GPT_4O)
        .preamble(BASIC_PREAMBLE)
        .build();

    let response = agent
        .prompt(BASIC_PROMPT)
        .await
        .expect("completion should succeed")
        .output;

    assert_nonempty_response(&response);
}
