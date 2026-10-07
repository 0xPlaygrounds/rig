//! The failure rows' producers on the Doubleword wire (`Qwen/Qwen3.5-397B-A17B-FP8`, `reasoning_effort: none`): rig-agent over the
//! wire's own recording, writing the golden the world cell
//! (`ecs_faults.rs`) is compared to. Only the recorded rows
//! have a producer here; a scripted row's oracle is the runner in the world
//! cell's own test.

use rig_test_support::cassette_models::OpenAiModels;

use super::super::support::with_doubleword_cassette;
use crate::ecs_matrix::{Wire, agent::run_agent};

/// The wire over the model it refuses: the setup cells' request.
fn missing(client: &OpenAiModels) -> Wire<rig::Model<rig::providers::openai::wire::OpenAiWire>> {
    Wire {
        thinking: crate::ecs_matrix::cells::ThinkingWire::Doubleword,
        model: client.completion("rig/definitely-not-a-doubleword-model"),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
        options: None,
    }
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
