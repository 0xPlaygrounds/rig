//! xAI reasoning roundtrip tests.
//!
//! Run only these cases with:
//! `cargo test -p rig --test xai xai::reasoning_roundtrip::streaming -- --ignored --nocapture`

use rig::client::CompletionClient;
use rig::providers::xai;

use super::support::with_xai_cassette;
use crate::reasoning::{self, ReasoningRoundtripAgent};

#[tokio::test]
async fn streaming() {
    with_xai_cassette("reasoning_roundtrip/streaming", |client| async move {
        reasoning::run_reasoning_roundtrip_streaming(ReasoningRoundtripAgent::new(
            client.completion_model(xai::GROK_3_MINI),
            None,
        ))
        .await;
    })
    .await;
}

#[tokio::test]
async fn nonstreaming() {
    with_xai_cassette("reasoning_roundtrip/nonstreaming", |client| async move {
        reasoning::run_reasoning_roundtrip_nonstreaming(ReasoningRoundtripAgent::new(
            client.completion_model(xai::GROK_3_MINI),
            None,
        ))
        .await;
    })
    .await;
}
