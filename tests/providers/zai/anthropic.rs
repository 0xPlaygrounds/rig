//! Z.AI Anthropic-compatible completion smoke test.

use rig::providers::zai;

use crate::support::assert_completion_smoke;
use crate::zai::anthropic_client;

#[tokio::test]
#[ignore = "requires ZAI_API_KEY"]
async fn anthropic_compatible_completion_smoke() {
    assert_completion_smoke(anthropic_client().completion(zai::GLM_4_6)).await;
}
