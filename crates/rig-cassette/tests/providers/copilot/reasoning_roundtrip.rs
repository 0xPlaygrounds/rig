//! Copilot reasoning roundtrip tests.

use crate::copilot::{live_responses_model, with_copilot_cassette};
use crate::reasoning::{self, ReasoningRoundtripAgent};
use rig::wire::Wire as _;

#[tokio::test]
async fn streaming() {
    with_copilot_cassette("reasoning_roundtrip/streaming", |client| async move {
        let expected = serde_json::json!({
            "context": "current_turn",
            "effort": "medium",
            "summary": null
        });
        // Copilot's terminal record carries reasoning metadata that rig's
        // normalized `StreamFinal` does not model; its `raw` keeps it.
        let mut finals = Vec::new();
        reasoning::run_reasoning_roundtrip_streaming_with_final(
            ReasoningRoundtripAgent::new(
                client.completion(live_responses_model()).on(rig::transport()),
                Some(serde_json::json!({
                    "reasoning": { "effort": "medium" }
                })),
            ),
            |response| finals.push(response.clone()),
        )
        .await;

        let response = finals
            .first()
            .expect("Copilot reasoning stream should yield a provider final response");
        let response: rig::providers::openai::responses_api::streaming::StreamingCompletionResponse =
            serde_json::from_value(response.raw.clone())
                .expect("Copilot reasoning stream should use the Responses route");
        assert_eq!(response.reasoning_context.as_deref(), Some("current_turn"));
        assert_eq!(response.reasoning_metadata.as_ref(), expected.as_object());
    })
    .await;
}

#[tokio::test]
async fn nonstreaming() {
    with_copilot_cassette("reasoning_roundtrip/nonstreaming", |client| async move {
        reasoning::run_reasoning_roundtrip_nonstreaming(ReasoningRoundtripAgent::new(
            client
                .completion(live_responses_model())
                .on(rig::transport()),
            Some(serde_json::json!({
                "reasoning": { "effort": "medium" }
            })),
        ))
        .await;
    })
    .await;
}
