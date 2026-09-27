//! MiniMax Anthropic-compatible completion smoke test.

use rig::providers::minimax;

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires MINIMAX_API_KEY"]
async fn anthropic_compatible_completion_smoke() {
    let response = rig::AgentBuilder::new(
        minimax::anthropic_from_env()
            .expect("MINIMAX_API_KEY should be set")
            .completion(minimax::MINIMAX_M2),
    )
    .preamble(BASIC_PREAMBLE)
    .build()
    .prompt(BASIC_PROMPT)
    .await
    .expect("MiniMax Anthropic-compatible completion should succeed")
    .output;

    assert_nonempty_response(&response);
}
