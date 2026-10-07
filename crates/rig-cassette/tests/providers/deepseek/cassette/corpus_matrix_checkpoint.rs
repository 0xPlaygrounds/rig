//! Focused tool-turn checkpoint matrix on DeepSeek: deepseek-flash.

use crate::corpus_matrix::{Wire, cells, checkpoint};
use crate::deepseek::support::with_deepseek_cassette;
use rig_test_support::cassette_models::OpenAiModels;

fn wire(client: &OpenAiModels) -> Wire<rig::Model<rig::providers::openai::wire::OpenAiWire>> {
    Wire {
        thinking: cells::ThinkingWire::DeepSeek,
        model: client.completion("deepseek-flash"),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
        options: Some(|| {
            crate::corpus_matrix::corpus::TypedOptions::generation(
                rig::completion::GenerationOptions::default()
                    .reasoning(rig::completion::Reasoning::Off),
            )
        }),
    }
}

crate::matrix::golden_matrix! {
    wrapper: with_deepseek_cassette, wire: wire, run: checkpoint::run_agent, oracle: crate::goldens::golden_effects;
    #[tokio::test]
    parallel_batch: ("checkpoint_matrix/parallel_batch", checkpoint::PARALLEL_BATCH, "deepseek_checkpoint_parallel_batch");
}
