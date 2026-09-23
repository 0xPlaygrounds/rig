//! Adversarial handle round-trips on Gemini: see
//! `rig_test_support::history_survival::adversarial`.

use rig::providers::gemini::completion::GEMINI_3_FLASH_PREVIEW;

use super::super::support::with_gemini_cassette;
use crate::history_survival::adversarial::{self, Hop};

#[tokio::test]
async fn colliding_ids() {
    const SCENARIO: &str = "adversarial/colliding_ids";
    with_gemini_cassette("adversarial/colliding_ids", |client| async move {
        adversarial::colliding_ids(&client.completion(GEMINI_3_FLASH_PREVIEW), "call_dup", None)
            .await;
    })
    .await;
    adversarial::assert_colliding_recorded("gemini", SCENARIO);
}

#[tokio::test]
async fn out_of_order_results() {
    const SCENARIO: &str = "adversarial/out_of_order_results";
    with_gemini_cassette("adversarial/out_of_order_results", |client| async move {
        adversarial::out_of_order_results(&client.completion(GEMINI_3_FLASH_PREVIEW), None).await;
    })
    .await;
    adversarial::assert_carried("gemini", SCENARIO, "thought_signature", 1);
}

/// Second foreign hop: Anthropic then Responses history continues on Gemini.
#[tokio::test]
async fn three_provider_round_trip() {
    with_gemini_cassette(
        "adversarial/three_provider_round_trip",
        |client| async move {
            adversarial::round_trip_hop(
                &client.completion(GEMINI_3_FLASH_PREVIEW),
                Hop::Gemini,
                None,
            )
            .await;
        },
    )
    .await;
    adversarial::assert_round_trip_recorded(Hop::Gemini);
}
