//! Focused tool-turn checkpoint matrix on DeepSeek: deepseek-flash.
//! One producer recording is reused by every native cut with strict matching.

use crate::deepseek::support::with_deepseek_cassette;
use crate::ecs_matrix::{Wire, cells, checkpoint};
use rig::completion::CompletionModel;
use rig::prelude::*;

fn wire(client: &rig::providers::deepseek::Client) -> Wire<impl CompletionModel + Clone + 'static> {
    Wire {
        thinking: cells::ThinkingWire::DeepSeek,
        model: client.completion_model("deepseek-flash"),
        route: None,
        temperature: Some(0.0),
        additional_params: Some(|| serde_json::json!({"thinking":{"type":"disabled"}})),
    }
}

#[tokio::test]
async fn multi_turn_unary() {
    with_deepseek_cassette("checkpoint_matrix/multi_turn_unary", |client| async move {
        checkpoint::run_agent(&wire(&client), &checkpoint::MULTI_TURN_UNARY, |log| {
            crate::goldens::golden_effects("deepseek_checkpoint_multi_turn_unary", log)
        })
        .await;
    })
    .await;
}

#[tokio::test]
async fn multi_turn_streamed() {
    with_deepseek_cassette(
        "checkpoint_matrix/multi_turn_streamed",
        |client| async move {
            checkpoint::run_agent(&wire(&client), &checkpoint::MULTI_TURN_STREAMED, |log| {
                crate::goldens::golden_effects("deepseek_checkpoint_multi_turn_streamed", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn parallel_batch() {
    with_deepseek_cassette("checkpoint_matrix/parallel_batch", |client| async move {
        checkpoint::run_agent(&wire(&client), &checkpoint::PARALLEL_BATCH, |log| {
            crate::goldens::golden_effects("deepseek_checkpoint_parallel_batch", log)
        })
        .await;
    })
    .await;
}

#[tokio::test]
async fn large_result() {
    with_deepseek_cassette("checkpoint_matrix/large_result", |client| async move {
        checkpoint::run_agent(&wire(&client), &checkpoint::LARGE_RESULT, |log| {
            crate::goldens::golden_effects("deepseek_checkpoint_large_result", log)
        })
        .await;
    })
    .await;
}
