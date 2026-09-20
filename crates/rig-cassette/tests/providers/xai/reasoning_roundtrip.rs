//! xAI reasoning roundtrip tests.
//!
//! Run only these cases with:
//! `cargo test -p rig --test xai xai::reasoning_roundtrip::streaming -- --ignored --nocapture`

use rig::providers::xai;

use super::support::with_xai_cassette;
use crate::reasoning::{self, ReasoningRoundtripAgent};

#[rig_test_support::cassette(rig_test_support::recording::Scenario::live(
    "xai/reasoning_roundtrip/streaming"
))]
#[tokio::test]
async fn streaming() {
    with_xai_cassette("reasoning_roundtrip/streaming", |client| async move {
        reasoning::run_reasoning_roundtrip_streaming(ReasoningRoundtripAgent::new(
            client.completion(xai::GROK_3_MINI),
            None,
        ))
        .await;
    })
    .await;
}

#[rig_test_support::cassette(rig_test_support::recording::Scenario::live(
    "xai/reasoning_roundtrip/nonstreaming"
))]
#[tokio::test]
async fn nonstreaming() {
    with_xai_cassette("reasoning_roundtrip/nonstreaming", |client| async move {
        reasoning::run_reasoning_roundtrip_nonstreaming(ReasoningRoundtripAgent::new(
            client.completion(xai::GROK_3_MINI),
            None,
        ))
        .await;
    })
    .await;
}
