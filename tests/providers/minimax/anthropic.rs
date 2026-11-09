//! MiniMax Anthropic-compatible completion smoke test.

use rig::providers::minimax;

use crate::support::assert_completion_smoke;

#[tokio::test]
#[ignore = "requires MINIMAX_API_KEY"]
async fn anthropic_compatible_completion_smoke() {
    assert_completion_smoke(
        minimax::anthropic_from_env()
            .expect("MINIMAX_API_KEY should be set")
            .completion(minimax::MINIMAX_M3),
    )
    .await;
}
