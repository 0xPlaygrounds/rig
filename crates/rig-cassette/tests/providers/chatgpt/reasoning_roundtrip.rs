//! ChatGPT reasoning roundtrip tests.

use crate::chatgpt::{LIVE_MODEL, live_client};
use crate::reasoning::{self, ReasoningRoundtripAgent};

#[tokio::test]
#[ignore = "requires ChatGPT credentials or existing OAuth cache"]
async fn streaming() {
    reasoning::run_reasoning_roundtrip_streaming(
        ReasoningRoundtripAgent::new(live_client().await.completion(LIVE_MODEL), None)
            .with_options(
                rig::completion::GenerationOptions::default()
                    .reasoning(rig::completion::Effort::Medium),
            ),
    )
    .await;
}
