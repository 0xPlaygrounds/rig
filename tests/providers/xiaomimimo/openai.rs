//! Xiaomi MiMo OpenAI-compatible completion smoke test.

use rig::prelude::*;
use rig::providers::xiaomimimo;

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires XIAOMI_MIMO_API_KEY"]
async fn openai_compatible_completion_smoke() {
    let cfg = xiaomimimo::functions::Config::from_env(xiaomimimo::MIMO_V2_5_PRO)
        .expect("config should build");
    let response = AgentBuilder::new(ProviderConfig::XiaomiMimo(cfg))
        .preamble(BASIC_PREAMBLE)
        .build()
        .prompt(BASIC_PROMPT)
        .await
        .expect("Xiaomi MiMo OpenAI-compatible completion should succeed");

    assert_nonempty_response(&response);
}
