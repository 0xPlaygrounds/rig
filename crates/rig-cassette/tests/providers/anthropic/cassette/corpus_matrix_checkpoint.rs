//! Focused tool-turn checkpoint matrix on Anthropic: claude-haiku-4-5-20251001.

use super::super::support::with_anthropic_cassette;
use crate::corpus_matrix::{Wire, cells, checkpoint};
use rig_test_support::cassette_models::AnthropicModels;

fn wire(client: &AnthropicModels) -> Wire<rig::Model<rig::providers::anthropic::wire::Messages>> {
    Wire {
        thinking: cells::ThinkingWire::Anthropic,
        model: client.completion("claude-haiku-4-5-20251001"),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
        options: None,
    }
}

crate::matrix::golden_matrix! {
    wrapper: with_anthropic_cassette, wire: wire, run: checkpoint::run_agent, oracle: crate::goldens::golden_effects;
    #[tokio::test]
    parallel_batch: ("checkpoint_matrix/parallel_batch", checkpoint::PARALLEL_BATCH, "anthropic_checkpoint_parallel_batch");
}
