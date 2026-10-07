//! The failure rows' producers on the DeepSeek wire (`deepseek-chat`): rig-agent over the
//! wire's own recording, writing the golden. Only the recorded rows have a
//! producer here; the scripted rows run in the `runtime` target.

use crate::corpus_matrix::{Wire, agent::run_agent, faults};
use crate::deepseek::support::with_deepseek_cassette;
use rig_test_support::cassette_models::OpenAiModels;

fn wire(client: &OpenAiModels) -> Wire<rig::Model<rig::providers::openai::wire::OpenAiWire>> {
    Wire {
        thinking: crate::corpus_matrix::cells::ThinkingWire::DeepSeek,
        model: client.completion("deepseek-chat"),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
        options: None,
    }
}

#[tokio::test]
async fn tool_error_streamed() {
    with_deepseek_cassette("corpus_faults/tool_error_streamed", |client| async move {
        run_agent(&wire(&client), &faults::TOOL_ERROR_STREAMED, |log| {
            crate::goldens::golden_effects("deepseek_fault_tool_error_streamed", log)
        })
        .await;
    })
    .await;
}
