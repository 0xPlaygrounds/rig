//! OpenAI reasoning roundtrip tests.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use rig::providers::openai;

use rig::completion::Effort;
use rig::providers::openai::extension::{OpenAiOptions, ReasoningSummary};

use super::super::support::{effort, stateless, with_openai_cassette};
use crate::reasoning::{self, ReasoningRoundtripAgent};

#[tokio::test]
async fn nonstreaming() {
    with_openai_cassette("reasoning_roundtrip/nonstreaming", |client| async move {
        reasoning::run_reasoning_roundtrip_nonstreaming(
            ReasoningRoundtripAgent::new(client.openai.completion("gpt-5.2"), None)
                .with_options(effort(Effort::Medium))
                .with_provider_options(stateless()),
        )
        .await;
    })
    .await;
}

#[tokio::test]
async fn reasoning_delta_hook_streaming() {
    with_openai_cassette("reasoning_delta_hook/streaming", |client| async move {
        let summary = OpenAiOptions::new().reasoning_summary(ReasoningSummary::Detailed);
        reasoning::run_reasoning_delta_hook_streaming_with(
            client.openai.completion(openai::GPT_5_6),
            |builder| builder.reasoning(Effort::High).provider_option(summary),
            "openai",
        )
        .await;
    })
    .await;
}
