//! Together agent completion smoke test.

use rig::providers::together;

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires TOGETHER_API_KEY"]
async fn completion_smoke() {
    let provider = together::from_env().expect("config should build from env");
    let agent = rig::AgentBuilder::new(provider.completion(together::MIXTRAL_8X7B_INSTRUCT_V0_1))
        .preamble(BASIC_PREAMBLE)
        .build();

    let response = agent
        .prompt(BASIC_PROMPT)
        .await
        .expect("completion should succeed");

    assert_nonempty_response(&response.output());
}
