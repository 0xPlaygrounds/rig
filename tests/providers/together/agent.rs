//! Together agent completion smoke test.

use rig::prelude::*;
use rig::providers::together;

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires TOGETHER_API_KEY"]
async fn completion_smoke() {
    let cfg = together::functions::Config::from_env(together::MIXTRAL_8X7B_INSTRUCT_V0_1)
        .expect("config should build");
    let agent = AgentBuilder::new(ProviderConfig::Together(cfg))
        .preamble(BASIC_PREAMBLE)
        .build();

    let response = agent
        .prompt(BASIC_PROMPT)
        .await
        .expect("completion should succeed");

    assert_nonempty_response(&response);
}
