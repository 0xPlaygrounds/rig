//! Adversarial handle round-trips on OpenAI: see
//! `rig_test_support::history_survival::adversarial`.

use rig::completion::Effort;
use rig::providers::openai::extension::{Include, OpenAiOptions};

use super::super::support::{effort, openai_options, with_openai_cassette};
use crate::history_survival::Options;
use crate::history_survival::adversarial::{self, Hop};

fn reasoning(level: Effort) -> Options {
    Options::new(
        effort(level),
        openai_options(
            OpenAiOptions::new()
                .store(false)
                .include([Include::ReasoningEncryptedContent]),
        ),
    )
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
                reasoning(Effort::Low),
            )
            .await;
        },
    )
    .await;
    adversarial::assert_round_trip_recorded(Hop::OpenAiResponses);
}
