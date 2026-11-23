//! Hyperbolic agent completion smoke test.

use rig::providers::hyperbolic;

use crate::support::assert_completion_smoke;

#[tokio::test]
#[ignore = "requires HYPERBOLIC_API_KEY"]
async fn completion_smoke() {
    assert_completion_smoke(
        hyperbolic::from_env()
            .expect("config should build from env")
            .completion(hyperbolic::DEEPSEEK_R1),
    )
    .await;
}
