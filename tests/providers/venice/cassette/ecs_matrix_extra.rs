//! The ECS contract matrix's world-only rows on the Venice wire: the batch
//! holds of §8.1 (row 11), a recorded 4xx through `spawn_run` with the
//! witness's facts (row 13) and `despawn_run` on a run cancelled with
//! its stream in flight (row 15). The drivers are in
//! `tests/common/ecs_matrix/extra.rs`; this file holds the scenario
//! literals and the wire's models.

use rig::completion::CompletionModel;
use rig::prelude::*;
use rig::providers::venice::MISTRAL_SMALL_3_2_24B;

use super::super::support::with_venice_cassette;
use crate::ecs_matrix::{
    Wire, cells,
    extra::{Approval, ErrorProbe, batch_hold, despawn_waits_for_the_stream, error_facts},
};

fn wire(client: &rig::providers::venice::Client) -> Wire<impl CompletionModel + Clone + 'static> {
    Wire {
        thinking: crate::ecs_matrix::cells::ThinkingWire::Venice,
        model: client.completion_model(MISTRAL_SMALL_3_2_24B),
        route: Some(client.completion_model(MISTRAL_SMALL_3_2_24B)),
        temperature: Some(0.0),
        additional_params: None,
    }
}

/// A batch-held call approved by removing `Held` dispatches, lands in call
/// order, and the scene saved afterwards loads.
#[tokio::test]
async fn batch_held_call_approved_by_removing_held() {
    with_venice_cassette(
        "corpus_matrix/serving_concurrent_concurrency_one",
        |client| async move {
            batch_hold(&wire(&client), Approval::RemoveHeld).await;
        },
    )
    .await;
}

/// The same, approved by `release_hold("rig-ecs/batch")`.
#[tokio::test]
async fn batch_held_call_approved_by_releasing_the_batch_owner() {
    with_venice_cassette(
        "corpus_matrix/serving_concurrent_concurrency_one",
        |client| async move {
            batch_hold(&wire(&client), Approval::ReleaseBatchOwner).await;
        },
    )
    .await;
}

/// The recorded 4xx, unary: the run fails as the provider's response and
/// the record and the witness carry the same facts.
#[tokio::test]
async fn error_facts_unary() {
    with_venice_cassette(
        "error_envelope/nonexistent_model_error_preserves_status_and_body",
        |client| async move {
            error_facts(
                client.completion_model("venice-nonexistent-rig-test"),
                ErrorProbe {
                    prompt: "Say hi.",
                    max_tokens: Some(16),
                    additional_params: None,
                    streamed: false,
                    status: 404,
                    code: None,
                },
            )
            .await;
        },
    )
    .await;
}

/// The recorded 4xx on the streaming surface.
#[tokio::test]
async fn error_facts_streamed() {
    with_venice_cassette(
        "error_envelope/nonexistent_model_streaming_error_preserves_status_and_body",
        |client| async move {
            error_facts(
                client.completion_model("venice-nonexistent-rig-test"),
                ErrorProbe {
                    prompt: "Say hi.",
                    max_tokens: Some(16),
                    additional_params: None,
                    streamed: true,
                    status: 404,
                    code: None,
                },
            )
            .await;
        },
    )
    .await;
}

/// A run cancelled while its completion still streams refuses
/// `despawn_run` until the stream drains.
#[tokio::test]
async fn despawn_run_waits_for_an_in_flight_stream() {
    with_venice_cassette(
        "corpus_matrix/endings_text_delta_stop",
        |client| async move {
            despawn_waits_for_the_stream(&wire(&client), &cells::ENDINGS_TEXT_DELTA_STOP).await;
        },
    )
    .await;
}
