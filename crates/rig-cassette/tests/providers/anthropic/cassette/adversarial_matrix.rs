//! Adversarial handle round-trips on Anthropic: see
//! `rig_test_support::history_survival::adversarial`.

use serde_json::json;

use super::super::support::with_anthropic_cassette;
use crate::history_survival::adversarial::{self, Hop};

fn thinking() -> Option<serde_json::Value> {
    Some(json!({ "thinking": { "type": "enabled", "budget_tokens": 1024 } }))
}

/// `display: omitted` returns thinking blocks with no text and a signature.
fn omitted_thinking() -> Option<serde_json::Value> {
    Some(json!({
        "thinking": { "type": "enabled", "budget_tokens": 1024, "display": "omitted" }
    }))
}

const LOOKUP: &str =
    "Think it through, then call lookup_code for record alpha and report its code.";

#[tokio::test]
async fn colliding_ids() {
    const SCENARIO: &str = "adversarial/colliding_ids";
    with_anthropic_cassette("adversarial/colliding_ids", |client| async move {
        adversarial::colliding_ids(&client.completion("claude-haiku-4-5"), "toolu_dup", None).await;
    })
    .await;
    adversarial::assert_colliding_recorded("anthropic", SCENARIO);
}

#[tokio::test]
async fn out_of_order_results() {
    const SCENARIO: &str = "adversarial/out_of_order_results";
    with_anthropic_cassette("adversarial/out_of_order_results", |client| async move {
        adversarial::out_of_order_results(&client.completion("claude-haiku-4-5"), thinking()).await;
    })
    .await;
    adversarial::assert_carried("anthropic", SCENARIO, "signature", 1);
}

#[tokio::test]
async fn empty_signed_reasoning() {
    const SCENARIO: &str = "adversarial/empty_signed_reasoning";
    with_anthropic_cassette("adversarial/empty_signed_reasoning", |client| async move {
        let first = adversarial::reasoning_round_trip(
            &client.completion("claude-sonnet-4-6"),
            LOOKUP,
            omitted_thinking(),
            4096,
            false,
        )
        .await;
        assert!(
            adversarial::has_empty_signed_reasoning(&first),
            "{:?}",
            first.choice
        );
    })
    .await;
    adversarial::assert_carried("anthropic", SCENARIO, "signature", 1);
}

#[tokio::test]
async fn empty_signed_reasoning_streamed() {
    const SCENARIO: &str = "adversarial/empty_signed_reasoning_streamed";
    with_anthropic_cassette(
        "adversarial/empty_signed_reasoning_streamed",
        |client| async move {
            let first = adversarial::reasoning_round_trip(
                &client.completion("claude-sonnet-4-6"),
                LOOKUP,
                omitted_thinking(),
                4096,
                true,
            )
            .await;
            assert!(
                adversarial::has_empty_signed_reasoning(&first),
                "{:?}",
                first.choice
            );
        },
    )
    .await;
    let frames = crate::cassettes::recorded_sse_json_frames("anthropic", SCENARIO);
    let signature = frames
        .iter()
        .find_map(|frame| frame["delta"]["signature"].as_str())
        .expect("turn one streamed a signature");
    let bodies = crate::cassettes::recorded_interaction_bodies("anthropic", SCENARIO);
    let next: serde_json::Value = serde_json::from_str(&bodies[1].0).expect("JSON");
    let carried = next["messages"][1]["content"]
        .as_array()
        .into_iter()
        .flatten()
        .any(|block| block["type"] == "thinking" && block["signature"] == signature);
    assert!(
        carried,
        "the streamed signature returns on its thinking block"
    );
}

/// The round trip returns home: Anthropic, Responses, Gemini, Anthropic.
#[tokio::test]
async fn three_provider_round_trip() {
    with_anthropic_cassette(
        "adversarial/three_provider_round_trip",
        |client| async move {
            adversarial::round_trip_hop(
                &client.completion("claude-sonnet-4-6"),
                Hop::Anthropic,
                thinking(),
            )
            .await;
        },
    )
    .await;
    adversarial::assert_round_trip_recorded(Hop::Anthropic);
}
