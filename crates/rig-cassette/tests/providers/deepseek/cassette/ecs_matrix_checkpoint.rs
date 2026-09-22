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

crate::matrix::resume_matrix! {
    wrapper: with_deepseek_cassette, wire: wire, run: crate::ecs_matrix::checkpoint_world::run_world;
    #[tokio::test]
    multi_turn_unary: ("checkpoint_matrix/multi_turn_unary", checkpoint::MULTI_TURN_UNARY, None);
    #[tokio::test]
    multi_turn_unary_cut_1: ("checkpoint_matrix/multi_turn_unary", checkpoint::MULTI_TURN_UNARY, Some(1));
    #[tokio::test]
    multi_turn_unary_cut_2: ("checkpoint_matrix/multi_turn_unary", checkpoint::MULTI_TURN_UNARY, Some(2));
    #[tokio::test]
    multi_turn_unary_cut_3: ("checkpoint_matrix/multi_turn_unary", checkpoint::MULTI_TURN_UNARY, Some(3));
    #[tokio::test]
    multi_turn_unary_cut_final: ("checkpoint_matrix/multi_turn_unary", checkpoint::MULTI_TURN_UNARY, Some(usize::MAX));
    #[tokio::test]
    multi_turn_streamed: ("checkpoint_matrix/multi_turn_streamed", checkpoint::MULTI_TURN_STREAMED, None);
    #[tokio::test]
    multi_turn_streamed_cut_1: ("checkpoint_matrix/multi_turn_streamed", checkpoint::MULTI_TURN_STREAMED, Some(1));
    #[tokio::test]
    multi_turn_streamed_cut_2: ("checkpoint_matrix/multi_turn_streamed", checkpoint::MULTI_TURN_STREAMED, Some(2));
    #[tokio::test]
    multi_turn_streamed_cut_3: ("checkpoint_matrix/multi_turn_streamed", checkpoint::MULTI_TURN_STREAMED, Some(3));
    #[tokio::test]
    multi_turn_streamed_cut_final: ("checkpoint_matrix/multi_turn_streamed", checkpoint::MULTI_TURN_STREAMED, Some(usize::MAX));
    #[tokio::test]
    parallel_batch: ("checkpoint_matrix/parallel_batch", checkpoint::PARALLEL_BATCH, None);
    #[tokio::test]
    parallel_batch_cut_final: ("checkpoint_matrix/parallel_batch", checkpoint::PARALLEL_BATCH, Some(usize::MAX));
    #[tokio::test]
    large_result: ("checkpoint_matrix/large_result", checkpoint::LARGE_RESULT, None);
    #[tokio::test]
    large_result_cut_final: ("checkpoint_matrix/large_result", checkpoint::LARGE_RESULT, Some(usize::MAX));
}

/// Negative matcher evidence only; fresh-world continuation uses the native tests above.
#[tokio::test]
async fn large_result_last_byte_mismatch() {
    checkpoint::assert_large_request_rejected("deepseek", "checkpoint_matrix/large_result").await;
}

/// Negative matcher probe against the same streamed loop used by native consumers.
#[tokio::test]
async fn streamed_tool_result_mismatch() {
    checkpoint::assert_stream_request_rejected("deepseek", "checkpoint_matrix/multi_turn_streamed")
        .await;
}
