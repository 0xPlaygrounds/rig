//! Xiaomi MiMo Anthropic-compatible completion smoke test.

use rig::providers::xiaomimimo;

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires XIAOMIMIMO_API_KEY"]
async fn anthropic_compatible_completion_smoke() {
    let response = rig::AgentBuilder::new(
        xiaomimimo::anthropic_from_env()
            .expect("XIAOMIMIMO_API_KEY should be set")
            .completion(xiaomimimo::MIMO_V2_5_PRO),
    )
    .preamble(BASIC_PREAMBLE)
    .build()
    .prompt(BASIC_PROMPT)
    .await
    .expect("Xiaomi MiMo Anthropic-compatible completion should succeed")
    .output;

    assert_nonempty_response(&response);
}
