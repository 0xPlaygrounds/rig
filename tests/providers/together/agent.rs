//! Together agent completion smoke test.

use rig::providers::together;

use crate::support::assert_completion_smoke;

#[tokio::test]
#[ignore = "requires TOGETHER_API_KEY"]
async fn completion_smoke() {
    assert_completion_smoke(
        together::from_env()
            .expect("config should build from env")
            .completion(together::MIXTRAL_8X7B_INSTRUCT_V0_1),
    )
    .await;
}
