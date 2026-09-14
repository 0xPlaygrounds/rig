//! Focused tool-turn checkpoint matrix on Gemini: gemini-2.5-flash-lite.
//! One producer recording is reused by every native cut with strict matching.

use super::super::support::with_gemini_cassette;
use crate::ecs_matrix::{Wire, cells, checkpoint};
use rig::completion::CompletionModel;
use rig::prelude::*;

fn wire(client: &rig::providers::gemini::Client) -> Wire<impl CompletionModel + Clone + 'static> {
    Wire {
        thinking: cells::ThinkingWire::Gemini,
        model: client.completion_model("gemini-2.5-flash-lite"),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
    }
}

#[tokio::test]
async fn multi_turn_unary() {
    with_gemini_cassette("checkpoint_matrix/multi_turn_unary", |client| async move {
        checkpoint::run_agent(&wire(&client), &checkpoint::MULTI_TURN_UNARY, |log| {
            crate::goldens::golden_effects("gemini_checkpoint_multi_turn_unary", log)
        })
        .await;
    })
    .await;
}

#[tokio::test]
async fn multi_turn_streamed() {
    with_gemini_cassette(
        "checkpoint_matrix/multi_turn_streamed",
        |client| async move {
            checkpoint::run_agent(&wire(&client), &checkpoint::MULTI_TURN_STREAMED, |log| {
                crate::goldens::golden_effects("gemini_checkpoint_multi_turn_streamed", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn parallel_batch() {
    with_gemini_cassette("checkpoint_matrix/parallel_batch", |client| async move {
        checkpoint::run_agent(&wire(&client), &checkpoint::PARALLEL_BATCH, |log| {
            crate::goldens::golden_effects("gemini_checkpoint_parallel_batch", log)
        })
        .await;
    })
    .await;
}

#[tokio::test]
async fn large_result() {
    with_gemini_cassette("checkpoint_matrix/large_result", |client| async move {
        checkpoint::run_agent(&wire(&client), &checkpoint::LARGE_RESULT, |log| {
            crate::goldens::golden_effects("gemini_checkpoint_large_result", log)
        })
        .await;
    })
    .await;
}
