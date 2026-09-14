//! The ECS contract matrix's world-only rows on the Gemini REST wire: the batch
//! holds of §8.1 (row 11), a recorded 4xx through `spawn_run` with the
//! witness's facts (row 13), the id-less wire's minted ids (row 14, owed from #2499) and `despawn_run` on a run cancelled with
//! its stream in flight (row 15). The drivers are in
//! `tests/common/ecs_matrix/extra.rs`; this file holds the scenario
//! literals and the wire's models.

use rig::completion::CompletionModel;
use rig::prelude::*;
use rig::providers::gemini::completion::{
    GEMINI_2_5_FLASH, GEMINI_3_1_FLASH_LITE_PREVIEW, GEMINI_3_FLASH_PREVIEW,
};

use super::super::support::with_gemini_cassette;
use crate::ecs_matrix::{
    Wire, cells,
    extra::{
        Approval, ErrorProbe, batch_hold, despawn_waits_for_the_stream, error_facts, minted_ids,
    },
};

fn wire(client: &rig::providers::gemini::Client) -> Wire<impl CompletionModel + Clone + 'static> {
    Wire {
        thinking: crate::ecs_matrix::cells::ThinkingWire::Gemini,
        model: client.completion_model(GEMINI_3_FLASH_PREVIEW),
        route: Some(client.completion_model(GEMINI_3_1_FLASH_LITE_PREVIEW)),
        temperature: Some(0.0),
        additional_params: None,
    }
}

/// The recording's own model, for the cell over a breadth recording.
fn legacy(client: &rig::providers::gemini::Client) -> Wire<impl CompletionModel + Clone + 'static> {
    Wire {
        thinking: crate::ecs_matrix::cells::ThinkingWire::Gemini,
        model: client.completion_model(GEMINI_2_5_FLASH),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
    }
}

/// A batch-held call approved by removing `Held` dispatches, lands in call
/// order, and the scene saved afterwards loads.
#[tokio::test]
async fn batch_held_call_approved_by_removing_held() {
    with_gemini_cassette(
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
    with_gemini_cassette(
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
    with_gemini_cassette(
        "error_envelope/nonexistent_model_error_preserves_status_and_body",
        |client| async move {
            error_facts(
                client.completion_model("gemini-nonexistent-rig-test"),
                ErrorProbe {
                    prompt: "Say hi.",
                    max_tokens: Some(16),
                    additional_params: None,
                    streamed: false,
                    status: 404,
                    code: Some("NOT_FOUND"),
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
    with_gemini_cassette(
        "error_envelope/nonexistent_model_streaming_error_preserves_status_and_body",
        |client| async move {
            error_facts(
                client.completion_model("gemini-nonexistent-rig-test"),
                ErrorProbe {
                    prompt: "Say hi.",
                    max_tokens: Some(16),
                    additional_params: None,
                    streamed: true,
                    status: 404,
                    code: Some("NOT_FOUND"),
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
    with_gemini_cassette("corpus_breadth/text_delta_stop", |client| async move {
        despawn_waits_for_the_stream(&legacy(&client), &cells::BREADTH_TEXT_DELTA_STOP).await;
    })
    .await;
}

/// The id-less wire's two calls in one turn carry distinct minted ids,
/// both answered in the adjacent utterance, and a re-run mints the same.
#[tokio::test]
async fn id_less_calls_get_distinct_stable_minted_ids() {
    let first = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let again = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = first.clone();
    with_gemini_cassette(
        "corpus_matrix/serving_concurrent_concurrency_two",
        |client| async move {
            *sink.lock().expect("ids") = minted_ids(&wire(&client)).await;
        },
    )
    .await;
    let sink = again.clone();
    with_gemini_cassette(
        "corpus_matrix/serving_concurrent_concurrency_two",
        |client| async move {
            *sink.lock().expect("ids") = minted_ids(&wire(&client)).await;
        },
    )
    .await;
    let first = first.lock().expect("ids").clone();
    let again = again.lock().expect("ids").clone();
    assert_eq!(first.len(), 2);
    assert_eq!(first, again, "a re-run mints the same ids");
}
