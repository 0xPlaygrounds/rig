//! Hyperbolic agent completion smoke test.

use rig::prelude::*;
use rig::providers::hyperbolic;

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires HYPERBOLIC_API_KEY"]
async fn completion_smoke() {
    let cfg = hyperbolic::functions::Config::from_env(hyperbolic::DEEPSEEK_R1)
        .expect("config should build");
    let agent = AgentBuilder::new(ProviderConfig::Hyperbolic(cfg))
        .preamble(BASIC_PREAMBLE)
        .build();

    let response = agent
        .prompt(BASIC_PROMPT)
        .await
        .expect("completion should succeed");

    assert_nonempty_response(&response);
}
