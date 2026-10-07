//! The failure rows' producers on the Gemini REST wire (`gemini-3-flash-preview`): rig-agent over the
//! wire's own recording, writing the golden the world cell
//! (`ecs_faults.rs`) is compared to. Only the recorded rows
//! have a producer here; a scripted row's oracle is the runner in the world
//! cell's own test.

use rig::providers::gemini::completion::GEMINI_3_FLASH_PREVIEW;
use rig_test_support::cassette_models::GeminiModels;

use super::super::support::with_gemini_cassette;
use crate::ecs_matrix::{Wire, agent::run_agent, faults};

fn wire(
    client: &GeminiModels,
) -> Wire<
    rig::Model<
        rig::providers::gemini::completion::GenerateContent,
        rig::http_client::DynHttpClient,
    >,
> {
    Wire {
        thinking: crate::ecs_matrix::cells::ThinkingWire::Gemini,
        model: client.completion(GEMINI_3_FLASH_PREVIEW),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
        options: None,
    }
}

/// The wire over the model it refuses: the setup cells' request.
fn missing(
    client: &GeminiModels,
) -> Wire<
    rig::Model<
        rig::providers::gemini::completion::GenerateContent,
        rig::http_client::DynHttpClient,
    >,
> {
    Wire {
        thinking: crate::ecs_matrix::cells::ThinkingWire::Gemini,
        model: client.completion("gemini-nonexistent-rig-test"),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
        options: None,
    }
}

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
