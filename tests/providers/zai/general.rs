//! Z.AI general OpenAI-compatible completion smoke test.

use rig::providers::zai;

use crate::support::assert_completion_smoke;
use crate::zai::general_client;

#[tokio::test]
#[ignore = "requires ZAI_API_KEY"]
async fn general_openai_compatible_completion_smoke() {
    assert_completion_smoke(general_client().completion(zai::GLM_4_6)).await;
}
