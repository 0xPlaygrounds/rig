//! The failure rows' producers on the Gemini REST wire (`gemini-3-flash-preview`): rig-agent over the
//! wire's own recording, writing the golden the world cell
//! (`ecs_faults.rs`) is compared to. Only the recorded rows
//! have a producer here; a scripted row's oracle is the runner in the world
//! cell's own test.

use rig::completion::CompletionModel;
use rig::driver::{Bound, Socket};
use rig::providers::gemini::Gemini;
use rig::providers::gemini::completion::GEMINI_3_FLASH_PREVIEW;

use super::super::support::with_gemini_cassette;
use crate::ecs_matrix::{Wire, agent::run_agent, faults};

fn wire<H: Socket>(client: &Bound<Gemini, H>) -> Wire<impl CompletionModel + Clone + 'static> {
    Wire {
        thinking: crate::ecs_matrix::cells::ThinkingWire::Gemini,
        model: client.completion(GEMINI_3_FLASH_PREVIEW),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
    }
}

/// The wire over the model it refuses: the setup cells' request.
fn missing<H: Socket>(client: &Bound<Gemini, H>) -> Wire<impl CompletionModel + Clone + 'static> {
    Wire {
        thinking: crate::ecs_matrix::cells::ThinkingWire::Gemini,
        model: client.completion("gemini-nonexistent-rig-test"),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
    }
}

#[rig_test_support::cassette(rig_test_support::recording::Scenario::live(
    "gemini/corpus_faults/setup_unary"
))]
#[tokio::test]
async fn setup_unary() {
    with_gemini_cassette("corpus_faults/setup_unary", |client| async move {
        run_agent(&missing(&client), &super::ecs_faults::SETUP_UNARY, |log| {
            crate::goldens::golden_effects("gemini_fault_setup_unary", log)
        })
        .await;
    })
    .await;
}

#[rig_test_support::cassette(rig_test_support::recording::Scenario::live(
    "gemini/error_envelope/nonexistent_model_streaming_error_preserves_status_and_body"
))]
#[tokio::test]
async fn setup_streamed() {
    with_gemini_cassette(
        "error_envelope/nonexistent_model_streaming_error_preserves_status_and_body",
        |client| async move {
            run_agent(
                &missing(&client),
                &super::ecs_faults::SETUP_STREAMED,
                |log| crate::goldens::golden_effects("gemini_fault_setup_streamed", log),
            )
            .await;
        },
    )
    .await;
}

#[rig_test_support::cassette(rig_test_support::recording::Scenario::live(
    "gemini/corpus_faults/tool_error"
))]
#[tokio::test]
async fn tool_error() {
    with_gemini_cassette("corpus_faults/tool_error", |client| async move {
        run_agent(&wire(&client), &faults::TOOL_ERROR, |log| {
            crate::goldens::golden_effects("gemini_fault_tool_error", log)
        })
        .await;
    })
    .await;
}

#[rig_test_support::cassette(rig_test_support::recording::Scenario::live(
    "gemini/corpus_faults/tool_error_streamed"
))]
#[tokio::test]
async fn tool_error_streamed() {
    with_gemini_cassette("corpus_faults/tool_error_streamed", |client| async move {
        run_agent(&wire(&client), &faults::TOOL_ERROR_STREAMED, |log| {
            crate::goldens::golden_effects("gemini_fault_tool_error_streamed", log)
        })
        .await;
    })
    .await;
}

#[rig_test_support::cassette(rig_test_support::recording::Scenario::live(
    "gemini/corpus_faults/batch_second_fails"
))]
#[tokio::test]
async fn batch_second_fails() {
    with_gemini_cassette("corpus_faults/batch_second_fails", |client| async move {
        run_agent(&wire(&client), &faults::BATCH_SECOND_FAILS, |log| {
            crate::goldens::golden_effects("gemini_fault_batch_second_fails", log)
        })
        .await;
    })
    .await;
}

#[rig_test_support::cassette(rig_test_support::recording::Scenario::live(
    "gemini/corpus_faults/batch_second_fails"
))]
#[tokio::test]
async fn batch_second_fails_concurrent() {
    with_gemini_cassette("corpus_faults/batch_second_fails", |client| async move {
        run_agent(
            &wire(&client),
            &faults::BATCH_SECOND_FAILS_CONCURRENT,
            |log| crate::goldens::golden_effects("gemini_fault_batch_second_fails_concurrent", log),
        )
        .await;
    })
    .await;
}
