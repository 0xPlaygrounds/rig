//! The failure rows' producers on the Doubleword wire (`Qwen/Qwen3.5-397B-A17B-FP8`, `reasoning_effort: none`): rig-agent over the
//! wire's own recording, writing the golden. Only the recorded rows have a
//! producer here; the scripted rows run in the `runtime` target.

use rig_test_support::cassette_models::OpenAiModels;

use super::super::support::with_doubleword_cassette;
use crate::corpus_matrix::{Wire, agent::run_agent, cells::Cell, corpus::Program, faults};

/// The setup cell with this wire's own request: the model it refuses asked
/// for a short probe.
const SETUP_STREAMED: Cell = Cell {
    program: Program {
        prompt: "Reply with error-probe.",
        max_tokens: Some(8),
        streamed: true,
        ..faults::SETUP_UNARY.program
    },
    ..faults::SETUP_STREAMED
};

/// The wire over the model it refuses: the setup cells' request.
fn missing(client: &OpenAiModels) -> Wire<rig::Model<rig::providers::openai::wire::OpenAiWire>> {
    Wire {
        thinking: crate::corpus_matrix::cells::ThinkingWire::Doubleword,
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
            run_agent(&missing(&client), &SETUP_STREAMED, |log| {
                crate::goldens::golden_effects("doubleword_fault_setup_streamed", log)
            })
            .await;
        },
    )
    .await;
}
