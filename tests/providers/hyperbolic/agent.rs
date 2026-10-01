//! Hyperbolic agent completion smoke test.

use rig::providers::hyperbolic;

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires HYPERBOLIC_API_KEY"]
async fn completion_smoke() {
    let provider = hyperbolic::from_env().expect("config should build from env");
    let agent = rig::AgentBuilder::new(provider.completion(hyperbolic::DEEPSEEK_R1))
        .preamble(BASIC_PREAMBLE)
        .build();

    let response = agent
        .prompt(BASIC_PROMPT)
        .await
        .expect("completion should succeed")
        .output();

    assert_nonempty_response(&response);
}
