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
async fn colliding_ids_chat() {
    const SCENARIO: &str = "adversarial/colliding_ids_chat";
    with_openai_cassette("adversarial/colliding_ids_chat", |client| async move {
        adversarial::colliding_ids(&client.openai.chat("gpt-4.1-mini"), "call_dup", None).await;
    })
    .await;
    adversarial::assert_colliding_recorded("openai", SCENARIO);
}

#[tokio::test]
async fn colliding_ids_responses() {
    const SCENARIO: &str = "adversarial/colliding_ids_responses";
    with_openai_cassette("adversarial/colliding_ids_responses", |client| async move {
        adversarial::colliding_ids(
            &client.openai.responses("gpt-4.1-mini"),
            "call_dup",
            Some(json!({ "store": false })),
        )
        .await;
    })
    .await;
    adversarial::assert_colliding_recorded("openai", SCENARIO);
}

#[tokio::test]
async fn out_of_order_results_chat() {
    const SCENARIO: &str = "adversarial/out_of_order_results_chat";
    with_openai_cassette(
        "adversarial/out_of_order_results_chat",
        |client| async move {
            adversarial::out_of_order_results(&client.openai.chat("gpt-4.1-mini"), None).await;
        },
    )
    .await;
    adversarial::assert_carried("openai", SCENARIO, "tool_call_id", 1);
}

#[tokio::test]
async fn out_of_order_results_responses() {
    const SCENARIO: &str = "adversarial/out_of_order_results_responses";
    with_openai_cassette(
        "adversarial/out_of_order_results_responses",
        |client| async move {
            adversarial::out_of_order_results(
                &client.openai.responses("gpt-5-mini"),
                reasoning("low"),
            )
            .await;
        },
    )
    .await;
    adversarial::assert_carried("openai", SCENARIO, "encrypted_content", 1);
}

/// High effort on a multi-step puzzle yields a long ciphertext, which must
/// come back whole.
#[tokio::test]
async fn long_encrypted_payload() {
    const SCENARIO: &str = "adversarial/long_encrypted_payload";
    with_openai_cassette("adversarial/long_encrypted_payload", |client| async move {
        adversarial::reasoning_round_trip(
            &client.openai.responses("gpt-5-mini"),
            "Work this out carefully before acting. Let n be the number of primes below 600 \
             and s the sum of the decimal digits of the 90th prime. Count the primes in \
             blocks of 100 and verify each block. If n + s is even, the record is alpha, \
             otherwise beta. Then call lookup_code for that record and report its code.",
            reasoning("high"),
            32768,
            false,
        )
        .await;
    })
    .await;
    adversarial::assert_carried("openai", SCENARIO, "encrypted_content", 8192);
}

/// First foreign hop: the Anthropic source continues on Responses.
#[tokio::test]
async fn three_provider_round_trip() {
    with_openai_cassette(
        "adversarial/three_provider_round_trip",
        |client| async move {
            adversarial::round_trip_hop(
                &client.openai.responses("gpt-5-mini"),
                Hop::OpenAiResponses,
                reasoning("low"),
            )
            .await;
        },
    )
    .await;
    adversarial::assert_round_trip_recorded(Hop::OpenAiResponses);
}
