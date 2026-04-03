//! Hyperbolic agent completion smoke test.

use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers::hyperbolic;

use crate::support::{PREAMBLE, PROMPT, assert_nontrivial_response};

#[tokio::test]
#[ignore = "requires HYPERBOLIC_API_KEY"]
async fn completion_smoke() {
    let client = hyperbolic::Client::from_env();
    let agent = client
        .agent(hyperbolic::DEEPSEEK_R1)
        .preamble(PREAMBLE)
        .build();

    let response = agent
        .prompt(PROMPT)
        .await
        .expect("completion should succeed");

    assert_nontrivial_response(&response);
}
