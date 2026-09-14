//! The long tool loop's producer column on OpenAiResponses: gpt-4.1-mini (Responses).
//! One recording per live cell; the native twin (`ecs_matrix_long_loop_responses.rs`)
//! reuses each with strict matching. Programs, toolset and assertions are
//! `tests/common/ecs_matrix/long_loop.rs`'s; this file holds the scenario
//! literals and the wire's model.

use super::super::support::with_openai_cassette;
use crate::ecs_matrix::{Wire, cells, long_loop};
use rig::completion::CompletionModel;
use rig::prelude::*;

fn wire(client: &rig::providers::openai::Client) -> Wire<impl CompletionModel + Clone + 'static> {
    Wire {
        thinking: cells::ThinkingWire::OpenAiResponses,
        model: client.completion_model("gpt-4.1-mini"),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
    }
}

#[tokio::test]
async fn long_unary() {
    with_openai_cassette(
        "long_loop_matrix_responses/long_unary",
        |client| async move {
            long_loop::run_agent(&wire(&client), &long_loop::LONG_UNARY, |log| {
                crate::goldens::golden_effects("openai_responses_long_loop_long_unary", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn long_streamed() {
    with_openai_cassette(
        "long_loop_matrix_responses/long_streamed",
        |client| async move {
            long_loop::run_agent(&wire(&client), &long_loop::LONG_STREAMED, |log| {
                crate::goldens::golden_effects("openai_responses_long_loop_long_streamed", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn parallel_calls() {
    with_openai_cassette(
        "long_loop_matrix_responses/parallel_calls",
        |client| async move {
            long_loop::run_agent(&wire(&client), &long_loop::PARALLEL_CALLS, |log| {
                crate::goldens::golden_effects("openai_responses_long_loop_parallel_calls", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn big_result() {
    with_openai_cassette(
        "long_loop_matrix_responses/big_result",
        |client| async move {
            long_loop::run_agent(&wire(&client), &long_loop::BIG_RESULT, |log| {
                crate::goldens::golden_effects("openai_responses_long_loop_big_result", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn tool_error_midway() {
    with_openai_cassette(
        "long_loop_matrix_responses/tool_error_midway",
        |client| async move {
            long_loop::run_agent(&wire(&client), &long_loop::TOOL_ERROR_MIDWAY, |log| {
                crate::goldens::golden_effects("openai_responses_long_loop_tool_error_midway", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn max_turns_midway() {
    with_openai_cassette(
        "long_loop_matrix_responses/max_turns_midway",
        |client| async move {
            long_loop::run_agent(&wire(&client), &long_loop::MAX_TURNS_MIDWAY, |log| {
                crate::goldens::golden_effects("openai_responses_long_loop_max_turns_midway", log)
            })
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn output_cap_midway() {
    with_openai_cassette(
        "long_loop_matrix_responses/output_cap_midway",
        |client| async move {
            // Failed(Response): the cut write_file call is a response error
            // on this wire (recording confirms).
            long_loop::run_agent(&wire(&client), &long_loop::OUTPUT_CAP_MIDWAY, |log| {
                crate::goldens::golden_effects("openai_responses_long_loop_output_cap_midway", log)
            })
            .await;
        },
    )
    .await;
}
