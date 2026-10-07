//! The long tool loop's producer column on Anthropic: claude-haiku-4-5-20251001.
//! One recording per live cell. Programs, toolset and assertions are
//! `tests/common/corpus_matrix/long_loop.rs`'s; this file holds the scenario
//! literals and the wire's model.

use super::super::support::with_anthropic_cassette;
use crate::corpus_matrix::{Wire, cells, long_loop};
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
    wrapper: with_anthropic_cassette, wire: wire, run: long_loop::run_agent, oracle: crate::goldens::golden_effects;
    #[tokio::test]
    parallel_calls: ("long_loop_matrix/parallel_calls", long_loop::PARALLEL_CALLS, "anthropic_long_loop_parallel_calls");
}
