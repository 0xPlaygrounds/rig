//! Mistral's reasoning trace survives a two-turn conversation: a
//! `reasoning_effort` turn answers with a thinking chunk, which comes back as
//! a `Reasoning` block and is replayed in the next request.

use rig::completion::AssistantContent;
use rig::providers::mistral;

use super::support::with_mistral_cassette_result;
use crate::reasoning::{self, ReasoningRoundtripAgent};

fn agent(client: &rig_test_support::cassette_models::OpenAiModels) -> ReasoningRoundtripAgent {
    ReasoningRoundtripAgent::new(
        client.completion(mistral::MISTRAL_SMALL),
        Some(serde_json::json!({ "reasoning_effort": "high" })),
    )
}

#[tokio::test]
async fn nonstreaming() -> anyhow::Result<()> {
    with_mistral_cassette_result("reasoning_roundtrip/nonstreaming", |client| async move {
        reasoning::run_reasoning_roundtrip_nonstreaming(agent(&client)).await;
        Ok::<(), anyhow::Error>(())
    })
    .await
}

#[tokio::test]
async fn streaming() -> anyhow::Result<()> {
    with_mistral_cassette_result("reasoning_roundtrip/streaming", |client| async move {
        let mut traced = false;
        reasoning::run_reasoning_roundtrip_streaming_with_final(agent(&client), |response| {
            traced = response
                .choice
                .iter()
                .any(|content| matches!(content, AssistantContent::Reasoning(_)));
        })
        .await;
        anyhow::ensure!(traced, "the thinking chunk did not reach the turn");
        Ok(())
    })
    .await
}
