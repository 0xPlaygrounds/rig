//! Mira agent completion smoke test.

use rig::providers::mira;
use rig::providers::openai;

use crate::support::assert_completion_smoke;

#[tokio::test]
#[ignore = "requires MIRA_API_KEY"]
async fn completion_smoke() {
    assert_completion_smoke(
        mira::from_env()
            .expect("config should build from env")
            .completion(openai::GPT_4O),
    )
    .await;
}
