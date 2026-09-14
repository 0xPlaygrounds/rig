//! The failure rows' producers on the OpenAI Responses wire (`gpt-5-mini`): rig-agent over the
//! wire's own recording, writing the golden the world cell
//! (`ecs_faults_responses.rs`) is compared to. Only the recorded rows
//! have a producer here; a scripted row's oracle is the runner in the world
//! cell's own test.

use rig::completion::CompletionModel;
use rig::prelude::*;
use rig::providers::openai::GPT_5_MINI;

use super::super::support::with_openai_cassette;
use crate::ecs_matrix::{Wire, agent::run_agent, faults};

fn wire(client: &rig::providers::openai::Client) -> Wire<impl CompletionModel + Clone + 'static> {
    Wire {
        thinking: crate::ecs_matrix::cells::ThinkingWire::OpenAiResponses,
        model: client.completion_model(GPT_5_MINI),
        route: None,
        temperature: None,
        additional_params: None,
    }
}

/// The wire over the model it refuses: the setup cells' request.
fn missing(
    client: &rig::providers::openai::Client,
) -> Wire<impl CompletionModel + Clone + 'static> {
    Wire {
        thinking: crate::ecs_matrix::cells::ThinkingWire::OpenAiResponses,
        model: client.completion_model("gpt-4o-mini-nonexistent-rig-test"),
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

#[tokio::test]
async fn tool_error() {
    with_openai_cassette("corpus_faults_responses/tool_error", |client| async move {
        run_agent(&wire(&client), &faults::TOOL_ERROR, |log| {
            crate::goldens::golden_effects("openai_responses_fault_tool_error", log)
        })
        .await;
    })
    .await;
}

#[tokio::test]
async fn tool_error_streamed() {
    with_openai_cassette(
        "corpus_faults_responses/tool_error_streamed",
        |client| async move {
            run_agent(&wire(&client), &faults::TOOL_ERROR_STREAMED, |log| {
                crate::goldens::golden_effects("openai_responses_fault_tool_error_streamed", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn batch_second_fails() {
    with_openai_cassette(
        "corpus_faults_responses/batch_second_fails",
        |client| async move {
            run_agent(&wire(&client), &faults::BATCH_SECOND_FAILS, |log| {
                crate::goldens::golden_effects("openai_responses_fault_batch_second_fails", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn batch_second_fails_concurrent() {
    with_openai_cassette(
        "corpus_faults_responses/batch_second_fails",
        |client| async move {
            run_agent(
                &wire(&client),
                &faults::BATCH_SECOND_FAILS_CONCURRENT,
                |log| {
                    crate::goldens::golden_effects(
                        "openai_responses_fault_batch_second_fails_concurrent",
                        log,
                    )
                },
            )
            .await;
        },
    )
    .await;
}
