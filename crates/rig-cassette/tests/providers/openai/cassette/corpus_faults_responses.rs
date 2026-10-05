//! The failure rows' producers on the OpenAI Responses wire (`gpt-5-mini`): rig-agent over the
//! wire's own recording, writing the golden the world cell
//! (`ecs_faults_responses.rs`) is compared to. Only the recorded rows
//! have a producer here; a scripted row's oracle is the runner in the world
//! cell's own test.

use super::super::support::{OpenAiCassette, with_openai_cassette};
use crate::ecs_matrix::{Wire, agent::run_agent};

/// The wire over the model it refuses: the setup cells' request.
fn missing(client: &OpenAiCassette) -> Wire<rig::Model<rig::providers::openai::wire::OpenAiWire>> {
    Wire {
        thinking: crate::ecs_matrix::cells::ThinkingWire::OpenAiResponses,
        model: client.openai.completion("gpt-4o-mini-nonexistent-rig-test"),
        route: None,
        temperature: None,
        additional_params: None,
    }
}

#[tokio::test]
async fn setup_unary() {
    with_openai_cassette("corpus_faults_responses/setup_unary", |client| async move {
        run_agent(
            &missing(&client),
            &super::ecs_faults_responses::SETUP_UNARY,
            |log| crate::goldens::golden_effects("openai_responses_fault_setup_unary", log),
        )
        .await;
    })
    .await;
}

#[tokio::test]
async fn setup_streamed() {
    with_openai_cassette(
        "error_envelope/nonexistent_model_streaming_error_preserves_status_and_body",
        |client| async move {
            run_agent(
                &missing(&client),
                &super::ecs_faults_responses::SETUP_STREAMED,
                |log| crate::goldens::golden_effects("openai_responses_fault_setup_streamed", log),
            )
            .await;
        },
    )
    .await;
}
