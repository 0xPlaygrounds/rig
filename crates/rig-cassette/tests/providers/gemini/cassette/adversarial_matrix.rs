//! Adversarial handle round-trips on Gemini: see
//! `rig_test_support::history_survival::adversarial`.

use rig::providers::gemini::completion::GEMINI_3_FLASH_PREVIEW;

use super::super::support::with_gemini_cassette;
use crate::history_survival::adversarial;
use crate::history_survival::adversarial::Hop;

/// Second foreign hop: Anthropic then Responses history continues on Gemini.
#[tokio::test]
async fn three_provider_round_trip() {
    with_gemini_cassette(
        "adversarial/three_provider_round_trip",
        |client| async move {
            adversarial::round_trip_hop(
                client.completion(GEMINI_3_FLASH_PREVIEW),
                Hop::Gemini,
                None,
            )
            .await;
        },
    )
    .await;
    adversarial::assert_round_trip_recorded(Hop::Gemini);
}
