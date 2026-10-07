//! Copilot reasoning roundtrip tests.

use crate::copilot::{live_responses_model, with_copilot_cassette};
use crate::reasoning::{self, ReasoningRoundtripAgent};

#[tokio::test]
async fn streaming() {
    with_copilot_cassette("reasoning_roundtrip/streaming", |client| async move {
        let expected = serde_json::json!({
            "context": "current_turn",
            "effort": "medium",
            "summary": null
        });
        // Copilot's terminal record carries reasoning metadata that rig's
        // normalized `CompletionResponse` does not model; its `raw` keeps it.
        let mut finals = Vec::new();
        reasoning::run_reasoning_roundtrip_streaming_with_final(
            ReasoningRoundtripAgent::new(client.completion(live_responses_model()), None)
                .with_options(
                    rig::completion::GenerationOptions::default()
                        .reasoning(rig::completion::Effort::Medium),
                ),
            |response| finals.push(response.clone()),
        )
        .await;

        let response = finals
            .first()
            .expect("Copilot reasoning stream should yield a provider final response");
        let reasoning = &response.raw["reasoning"];
        assert_eq!(reasoning["context"].as_str(), Some("current_turn"));
        assert_eq!(reasoning.as_object(), expected.as_object());
    })
    .await;
}

#[tokio::test]
async fn nonstreaming() {
    with_copilot_cassette("reasoning_roundtrip/nonstreaming", |client| async move {
        reasoning::run_reasoning_roundtrip_nonstreaming(
            ReasoningRoundtripAgent::new(client.completion(live_responses_model()), None)
                .with_options(
                    rig::completion::GenerationOptions::default()
                        .reasoning(rig::completion::Effort::Medium),
                ),
        )
        .await;
    })
    .await;
}
