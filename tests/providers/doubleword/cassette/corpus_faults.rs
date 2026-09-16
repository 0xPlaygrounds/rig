//! The failure rows' producers on the Doubleword wire (`Qwen/Qwen3.5-397B-A17B-FP8`, `reasoning_effort: none`): rig-agent over the
//! wire's own recording, writing the golden the world cell
//! (`ecs_faults.rs`) is compared to. Only the recorded rows
//! have a producer here; a scripted row's oracle is the runner in the world
//! cell's own test.

use rig::completion::CompletionModel;
use rig::providers::doubleword::QWEN3_5_397B_A17B;

use super::super::support::{BoundDoubleword, with_doubleword_cassette};
use crate::ecs_matrix::{Wire, agent::run_agent, cells, faults};

fn wire(client: &BoundDoubleword) -> Wire<impl CompletionModel + Clone + 'static> {
    Wire {
        thinking: crate::ecs_matrix::cells::ThinkingWire::Doubleword,
        model: client.completion(QWEN3_5_397B_A17B),
        route: None,
        temperature: Some(0.0),
        additional_params: Some(cells::reasoning_off),
    }
}

/// The wire over the model it refuses: the setup cells' request.
fn missing(client: &BoundDoubleword) -> Wire<impl CompletionModel + Clone + 'static> {
    Wire {
        thinking: crate::ecs_matrix::cells::ThinkingWire::Doubleword,
        model: client.completion("rig/definitely-not-a-doubleword-model"),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
    }
}

#[tokio::test]
async fn setup_unary() {
    with_doubleword_cassette("corpus_faults/setup_unary", |client| async move {
        run_agent(&missing(&client), &super::ecs_faults::SETUP_UNARY, |log| {
            crate::goldens::golden_effects("doubleword_fault_setup_unary", log)
        })
        .await;
    })
    .await;
}

#[tokio::test]
async fn setup_streamed() {
    with_doubleword_cassette(
        "error_matrix/unknown_model_streaming",
        |client| async move {
            run_agent(
                &missing(&client),
                &super::ecs_faults::SETUP_STREAMED,
                |log| crate::goldens::golden_effects("doubleword_fault_setup_streamed", log),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn tool_error() {
    with_doubleword_cassette("corpus_faults/tool_error", |client| async move {
        run_agent(&wire(&client), &faults::TOOL_ERROR, |log| {
            crate::goldens::golden_effects("doubleword_fault_tool_error", log)
        })
        .await;
    })
    .await;
}

#[tokio::test]
async fn tool_error_streamed() {
    with_doubleword_cassette("corpus_faults/tool_error_streamed", |client| async move {
        run_agent(&wire(&client), &faults::TOOL_ERROR_STREAMED, |log| {
            crate::goldens::golden_effects("doubleword_fault_tool_error_streamed", log)
        })
        .await;
    })
    .await;
}

#[tokio::test]
async fn batch_second_fails() {
    with_doubleword_cassette("corpus_faults/batch_second_fails", |client| async move {
        run_agent(&wire(&client), &faults::BATCH_SECOND_FAILS, |log| {
            crate::goldens::golden_effects("doubleword_fault_batch_second_fails", log)
        })
        .await;
    })
    .await;
}

#[tokio::test]
async fn batch_second_fails_concurrent() {
    with_doubleword_cassette("corpus_faults/batch_second_fails", |client| async move {
        run_agent(
            &wire(&client),
            &faults::BATCH_SECOND_FAILS_CONCURRENT,
            |log| {
                crate::goldens::golden_effects(
                    "doubleword_fault_batch_second_fails_concurrent",
                    log,
                )
            },
        )
        .await;
    })
    .await;
}
