//! Copilot reasoning roundtrip tests.

use rig::prelude::*;

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
        reasoning::run_reasoning_roundtrip_streaming_with_final(
            ReasoningRoundtripAgent::new(
                client.completion_model(live_responses_model()),
                Some(serde_json::json!({
                    "reasoning": { "effort": "medium" }
                })),
            ),
            |final_response| {
                // The normalized StreamFinal no longer carries the raw
                // Responses payload; the route is still visible via the
                // normalized provider tag, and the reasoning shape is checked
                // against the recorded SSE bodies below.
                assert!(
                    !final_response.provider.is_empty(),
                    "stream final should carry a provider tag"
                );
            },
        )
        .await;

        // Wire-level reasoning assertions, re-checked against the recorded
        // SSE `response.completed` events (previously read from the typed
        // Copilot Responses stream final).
        let bodies = crate::cassettes::recorded_response_bodies("copilot", "reasoning_roundtrip/streaming");
        let mut completed_events = 0usize;
        for body in &bodies {
            for line in body.lines() {
                let Some(payload) = line.strip_prefix("data:") else {
                    continue;
                };
                let Ok(event) = serde_json::from_str::<serde_json::Value>(payload.trim()) else {
                    continue;
                };
                if event["type"] != "response.completed" {
                    continue;
                }
                completed_events += 1;
                let reasoning = &event["response"]["reasoning"];
                assert_eq!(
                    reasoning["context"].as_str(),
                    Some("current_turn"),
                    "reasoning context should be current_turn"
                );
                assert_eq!(
                    reasoning, &expected,
                    "reasoning metadata should match the expected object"
                );
            }
        }
        assert!(
            completed_events > 0,
            "recorded stream should contain response.completed events"
        );
    })
    .await;
}

#[tokio::test]
async fn nonstreaming() {
    with_copilot_cassette("reasoning_roundtrip/nonstreaming", |client| async move {
        reasoning::run_reasoning_roundtrip_nonstreaming(ReasoningRoundtripAgent::new(
            client.completion_model(live_responses_model()),
            Some(serde_json::json!({
                "reasoning": { "effort": "medium" }
            })),
        ))
        .await;
    })
    .await;
}
