//! Moonshot Anthropic-compatible completion smoke test.

use rig::providers::anthropic::wire::{self as anthropic_wire, Anthropic};
use rig::providers::moonshot;
use rig::wire::Wire as _;

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires MOONSHOT_API_KEY"]
async fn anthropic_compatible_completion_smoke() {
    let response = rig::AgentBuilder::new(
        Anthropic::from_env_with(&anthropic_wire::MOONSHOT)
            .expect("MOONSHOT_API_KEY should be set")
            .completion(moonshot::KIMI_K3)
            .on(rig::transport()),
    )
    .preamble(BASIC_PREAMBLE)
    .build()
    .prompt(BASIC_PROMPT)
    .await
    .expect("Moonshot Anthropic-compatible completion should succeed")
    .output;

    assert_nonempty_response(&response);
}
