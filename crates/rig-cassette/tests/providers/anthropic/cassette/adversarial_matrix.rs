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
async fn empty_signed_reasoning() {
    const SCENARIO: &str = "adversarial/empty_signed_reasoning";
    with_anthropic_cassette("adversarial/empty_signed_reasoning", |client| async move {
        let first = adversarial::reasoning_round_trip(
            client.completion("claude-sonnet-4-6"),
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

/// The round trip returns home: AnthropicModels, Responses, Gemini, Anthropic.
#[tokio::test]
async fn three_provider_round_trip() {
    with_anthropic_cassette(
        "adversarial/three_provider_round_trip",
        |client| async move {
            adversarial::round_trip_hop(
                client.completion("claude-sonnet-4-6"),
                Hop::Anthropic,
                thinking(),
            )
            .await;
        },
    )
    .await;
    adversarial::assert_round_trip_recorded(Hop::Anthropic);
}
