//! Adversarial handle round-trips on OpenAI: see
//! `rig_test_support::history_survival::adversarial`.

use serde_json::json;

use super::super::support::with_openai_cassette;
use crate::history_survival::adversarial::{self, Hop};

fn reasoning(effort: &str) -> Option<serde_json::Value> {
    Some(json!({
        "reasoning": { "effort": effort },
        "include": ["reasoning.encrypted_content"],
        "store": false
    }))
}

#[tokio::test]
async fn colliding_ids_responses() {
    const SCENARIO: &str = "adversarial/colliding_ids_responses";
    with_openai_cassette("adversarial/colliding_ids_responses", |client| async move {
        adversarial::colliding_ids(
            client.openai.responses("gpt-4.1-mini"),
            "call_dup",
            Some(json!({ "store": false })),
        )
        .await;
    })
    .await;
    adversarial::assert_colliding_recorded("openai", SCENARIO);
}

/// First foreign hop: the Anthropic source continues on Responses.
#[tokio::test]
async fn three_provider_round_trip() {
    with_openai_cassette(
        "adversarial/three_provider_round_trip",
        |client| async move {
            adversarial::round_trip_hop(
                client.openai.responses("gpt-5-mini"),
                Hop::OpenAiResponses,
                reasoning("low"),
            )
            .await;
        },
    )
    .await;
    adversarial::assert_round_trip_recorded(Hop::OpenAiResponses);
}
