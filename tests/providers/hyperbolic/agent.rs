//! Hyperbolic agent completion smoke test.

use rig::providers::hyperbolic;
use rig::providers::openai::wire::{HYPERBOLIC, OpenAI};
use rig::wire::Wire as _;

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires HYPERBOLIC_API_KEY"]
async fn completion_smoke() {
    let provider = OpenAI::from_env_with(&HYPERBOLIC).expect("config should build from env");
    let agent = rig::AgentBuilder::new(
        provider
            .completion(hyperbolic::DEEPSEEK_R1)
            .on(rig::transport()),
    )
    .preamble(BASIC_PREAMBLE)
    .build();

    let response = agent
        .prompt(BASIC_PROMPT)
        .await
        .expect("completion should succeed")
        .output;

    assert_nonempty_response(&response);
}
