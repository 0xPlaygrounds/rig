//! Xiaomi MiMo Anthropic-compatible completion smoke test.

use rig::providers::xiaomimimo;

use crate::support::assert_completion_smoke;

#[tokio::test]
#[ignore = "requires XIAOMI_MIMO_API_KEY"]
async fn anthropic_compatible_completion_smoke() {
    assert_completion_smoke(
        xiaomimimo::anthropic_from_env()
            .expect("XIAOMI_MIMO_API_KEY should be set")
            .completion(xiaomimimo::MIMO_V2_5_PRO),
    )
    .await;
}
