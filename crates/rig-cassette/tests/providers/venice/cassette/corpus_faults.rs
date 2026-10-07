//! The failure rows' producers on the Venice wire (`mistral-small-3-2-24b-instruct`): rig-agent over the
//! wire's own recording, writing the golden. Only the recorded rows have a
//! producer here; the scripted rows run in the `runtime` target.

use rig_test_support::cassette_models::OpenAiModels;

use super::super::support::with_venice_cassette;
use crate::corpus_matrix::{Wire, agent::run_agent, faults};

/// The wire over the model it refuses: the setup cells' request.
fn missing(client: &OpenAiModels) -> Wire<rig::Model<rig::providers::openai::wire::OpenAiWire>> {
    Wire {
        thinking: crate::corpus_matrix::cells::ThinkingWire::Venice,
        model: client.completion("venice-nonexistent-rig-test"),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
        options: None,
    }
}

#[tokio::test]
async fn setup_streamed() {
    with_venice_cassette(
        "error_envelope/nonexistent_model_streaming_error_preserves_status_and_body",
        |client| async move {
            run_agent(&missing(&client), &faults::SETUP_STREAMED, |log| {
                crate::goldens::golden_effects("venice_fault_setup_streamed", log)
            })
            .await;
        },
    )
    .await;
}
