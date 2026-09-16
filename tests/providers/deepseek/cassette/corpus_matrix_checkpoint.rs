//! Focused tool-turn checkpoint matrix on DeepSeek: deepseek-flash.
//! One producer recording is reused by every native cut with strict matching.

use crate::deepseek::support::{BoundDeepSeek, with_deepseek_cassette};
use crate::ecs_matrix::{Wire, cells, checkpoint};
use rig::completion::CompletionModel;

fn wire(client: &BoundDeepSeek) -> Wire<impl CompletionModel + Clone + 'static> {
    Wire {
        thinking: cells::ThinkingWire::DeepSeek,
        model: client.completion("deepseek-flash"),
        route: None,
        temperature: Some(0.0),
        additional_params: Some(|| serde_json::json!({"thinking":{"type":"disabled"}})),
    }
}

crate::matrix::golden_matrix! {
    wrapper: with_deepseek_cassette, wire: wire, run: checkpoint::run_agent, oracle: crate::goldens::golden_effects;
    #[tokio::test]
    multi_turn_unary: ("checkpoint_matrix/multi_turn_unary", checkpoint::MULTI_TURN_UNARY, "deepseek_checkpoint_multi_turn_unary");
    #[tokio::test]
    multi_turn_streamed: ("checkpoint_matrix/multi_turn_streamed", checkpoint::MULTI_TURN_STREAMED, "deepseek_checkpoint_multi_turn_streamed");
    #[tokio::test]
    parallel_batch: ("checkpoint_matrix/parallel_batch", checkpoint::PARALLEL_BATCH, "deepseek_checkpoint_parallel_batch");
    #[tokio::test]
    large_result: ("checkpoint_matrix/large_result", checkpoint::LARGE_RESULT, "deepseek_checkpoint_large_result");
}
