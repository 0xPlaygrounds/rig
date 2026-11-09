//! Hugging Face agent completion smoke test.

use rig::providers::huggingface;

use crate::support::assert_completion_smoke;

#[tokio::test]
#[ignore = "requires HUGGINGFACE_API_KEY"]
async fn completion_smoke() {
    assert_completion_smoke(
        huggingface::from_env()
            .expect("config should build from env")
            .completion("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"),
    )
    .await;
}
