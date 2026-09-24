//! Hyperbolic agent completion smoke test.

use rig::prelude::*;
use rig::providers::hyperbolic;
use rig::providers::openai::wire::{HYPERBOLIC, OpenAI};

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires HYPERBOLIC_API_KEY"]
async fn completion_smoke() {
    let provider = OpenAI::from_env_with(&HYPERBOLIC)
        .expect("config should build from env")
        .bound()
        .expect("transport should build");
    let agent = provider
        .endpoint(|provider_config| provider_config.completion(hyperbolic::DEEPSEEK_R1))
        .into_agent_builder()
        .preamble(BASIC_PREAMBLE)
        .build();

    let response = agent
        .prompt(BASIC_PROMPT)
        .await
        .expect("completion should succeed")
        .output;

    assert_nonempty_response(&response);
}
