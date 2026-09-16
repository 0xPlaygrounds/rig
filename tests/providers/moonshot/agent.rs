//! Moonshot agent completion smoke test.

use rig::prelude::*;
use rig::providers::moonshot;
use rig::providers::openai::wire::{self as openai_wire, OpenAI};

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires MOONSHOT_API_KEY"]
async fn completion_smoke() {
    let client = OpenAI::from_env_with(&openai_wire::MOONSHOT)
        .expect("MOONSHOT_API_KEY should be set")
        .bound()
        .expect("moonshot client should build");
    let agent = client
        .agent(moonshot::KIMI_K3)
        .preamble(BASIC_PREAMBLE)
        .temperature(0.5)
        .max_tokens(1024)
        .build();

    let response = agent
        .prompt(BASIC_PROMPT)
        .await
        .expect("completion should succeed")
        .output;

    assert_nonempty_response(&response);
}
