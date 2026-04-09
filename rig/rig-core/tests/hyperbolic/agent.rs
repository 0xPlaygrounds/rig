//! Hyperbolic agent completion smoke test.

use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers::hyperbolic;

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires HYPERBOLIC_API_KEY"]
async fn completion_smoke() {
    let client = hyperbolic::Client::from_env();
    let agent = client
        .agent(hyperbolic::DEEPSEEK_R1)
        .preamble(BASIC_PREAMBLE)
        .build();

    let response = agent
        .prompt(BASIC_PROMPT)
        .await
        .expect("completion should succeed");

    assert_nonempty_response(&response);
}
