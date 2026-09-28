//! The setup failure rows on the Gemini REST wire (`gemini-3-flash-preview`):
//! rig-agent over the wire's own recording of a model it refuses.

use rig::providers::gemini::completion::GEMINI_3_FLASH_PREVIEW;
use rig_test_support::cassette_models::GeminiModels;

use super::super::support::with_gemini_cassette;
use crate::ecs_matrix::{
    Wire,
    agent::run_agent,
    cells::Cell,
    corpus::Program,
    faults::{self, Fault},
};

/// The setup cells with this wire's recorded facts: the model the wire
/// refuses, the recorded status, the body's own code.
const SETUP_UNARY: Cell = Cell {
    fault: Some(Fault::Setup {
        status: 404,
        code: Some("NOT_FOUND"),
    }),
    ..faults::SETUP_UNARY
};
const SETUP_STREAMED: Cell = Cell {
    program: Program {
        streamed: true,
        ..SETUP_UNARY.program
    },
    name: faults::SETUP_STREAMED.name,
    fault: Some(Fault::Setup {
        status: 404,
        code: Some("NOT_FOUND"),
    }),
    ..SETUP_UNARY
};

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
    }
}

#[tokio::test]
async fn setup_unary() {
    with_gemini_cassette("corpus_faults/setup_unary", |client| async move {
        run_agent(&missing(&client), &SETUP_UNARY, |log| {
            crate::goldens::golden_effects("gemini_fault_setup_unary", log)
        })
        .await;
    })
    .await;
}

#[tokio::test]
async fn setup_streamed() {
    with_gemini_cassette(
        "error_envelope/nonexistent_model_streaming_error_preserves_status_and_body",
        |client| async move {
            run_agent(&missing(&client), &SETUP_STREAMED, |log| {
                crate::goldens::golden_effects("gemini_fault_setup_streamed", log)
            })
            .await;
        },
    )
    .await;
}

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
