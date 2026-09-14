//! The failure rows' producers on the OpenAI Chat Completions wire (`gpt-5-mini`): rig-agent over the
//! wire's own recording, writing the golden the world cell
//! (`ecs_faults_chat.rs`) is compared to. Only the recorded rows
//! have a producer here; a scripted row's oracle is the runner in the world
//! cell's own test.

use rig::completion::CompletionModel;
use rig::prelude::*;
use rig::providers::openai::GPT_5_MINI;

use super::super::support::with_openai_cassette;
use crate::ecs_matrix::{Wire, agent::run_agent, faults};

fn wire(client: &rig::providers::openai::Client) -> Wire<impl CompletionModel + Clone + 'static> {
    Wire {
        thinking: crate::ecs_matrix::cells::ThinkingWire::OpenAiChat,
        model: client
            .clone()
            .completions_api()
            .completion_model(GPT_5_MINI),
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
        thinking: crate::ecs_matrix::cells::ThinkingWire::OpenAiChat,
        model: client
            .clone()
            .completions_api()
            .completion_model("gpt-5-mini-nonexistent-rig-test"),
        route: None,
        temperature: None,
        additional_params: None,
    }
}

#[tokio::test]
async fn setup_unary() {
    with_openai_cassette("corpus_faults_chat/setup_unary", |client| async move {
        run_agent(
            &missing(&client),
            &super::ecs_faults_chat::SETUP_UNARY,
            |log| crate::goldens::golden_effects("openai_chat_fault_setup_unary", log),
        )
        .await;
    })
    .await;
}

#[tokio::test]
async fn setup_streamed() {
    with_openai_cassette(
        "corpus_matrix_chat/error_facts_streamed",
        |client| async move {
            run_agent(
                &missing(&client),
                &super::ecs_faults_chat::SETUP_STREAMED,
                |log| crate::goldens::golden_effects("openai_chat_fault_setup_streamed", log),
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn tool_error() {
    with_openai_cassette("corpus_faults_chat/tool_error", |client| async move {
        run_agent(&wire(&client), &faults::TOOL_ERROR, |log| {
            crate::goldens::golden_effects("openai_chat_fault_tool_error", log)
        })
        .await;
    })
    .await;
}

#[tokio::test]
async fn tool_error_streamed() {
    with_openai_cassette(
        "corpus_faults_chat/tool_error_streamed",
        |client| async move {
            run_agent(&wire(&client), &faults::TOOL_ERROR_STREAMED, |log| {
                crate::goldens::golden_effects("openai_chat_fault_tool_error_streamed", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn batch_second_fails() {
    with_openai_cassette(
        "corpus_faults_chat/batch_second_fails",
        |client| async move {
            run_agent(&wire(&client), &faults::BATCH_SECOND_FAILS, |log| {
                crate::goldens::golden_effects("openai_chat_fault_batch_second_fails", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn batch_second_fails_concurrent() {
    with_openai_cassette(
        "corpus_faults_chat/batch_second_fails",
        |client| async move {
            run_agent(
                &wire(&client),
                &faults::BATCH_SECOND_FAILS_CONCURRENT,
                |log| {
                    crate::goldens::golden_effects(
                        "openai_chat_fault_batch_second_fails_concurrent",
                        log,
                    )
                },
            )
            .await;
        },
    )
    .await;
}
