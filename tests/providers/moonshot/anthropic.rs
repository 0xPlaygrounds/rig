//! Moonshot Anthropic-compatible completion smoke test.

use rig::providers::moonshot;

use crate::support::assert_completion_smoke;

#[tokio::test]
#[ignore = "requires MOONSHOT_API_KEY"]
async fn anthropic_compatible_completion_smoke() {
    assert_completion_smoke(
        moonshot::anthropic_from_env()
            .expect("MOONSHOT_API_KEY should be set")
            .completion(moonshot::KIMI_K3),
    )
    .await;
}
