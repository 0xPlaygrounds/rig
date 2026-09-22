//! Focused tool-turn checkpoint matrix on Anthropic: claude-haiku-4-5-20251001.
//! One producer recording is reused by every native cut with strict matching.

use super::super::support::with_anthropic_cassette;
use crate::ecs_matrix::{Wire, cells, checkpoint};
use rig::completion::CompletionModel;
use rig::driver::Bound;
use rig::providers::anthropic::wire::Anthropic;

fn wire(client: &Bound<Anthropic>) -> Wire<impl CompletionModel + Clone + 'static> {
    Wire {
        thinking: cells::ThinkingWire::Anthropic,
        model: client.completion("claude-haiku-4-5-20251001"),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
    }
}

crate::matrix::resume_matrix! {
    wrapper: with_anthropic_cassette, wire: wire, run: crate::ecs_matrix::checkpoint_world::run_world;
    #[tokio::test]
    multi_turn_unary: ("checkpoint_matrix/multi_turn_unary", checkpoint::MULTI_TURN_UNARY, None, "anthropic_multi_turn_unary");
    #[tokio::test]
    multi_turn_unary_cut_1: ("checkpoint_matrix/multi_turn_unary", checkpoint::MULTI_TURN_UNARY, Some(1), "anthropic_multi_turn_unary_cut_1");
    #[tokio::test]
    multi_turn_unary_cut_2: ("checkpoint_matrix/multi_turn_unary", checkpoint::MULTI_TURN_UNARY, Some(2), "anthropic_multi_turn_unary_cut_2");
    #[tokio::test]
    multi_turn_unary_cut_3: ("checkpoint_matrix/multi_turn_unary", checkpoint::MULTI_TURN_UNARY, Some(3), "anthropic_multi_turn_unary_cut_3");
    #[tokio::test]
    multi_turn_unary_cut_final: ("checkpoint_matrix/multi_turn_unary", checkpoint::MULTI_TURN_UNARY, Some(usize::MAX), "anthropic_multi_turn_unary_cut_final");
    #[tokio::test]
    multi_turn_streamed: ("checkpoint_matrix/multi_turn_streamed", checkpoint::MULTI_TURN_STREAMED, None, "anthropic_multi_turn_streamed");
    #[tokio::test]
    multi_turn_streamed_cut_1: ("checkpoint_matrix/multi_turn_streamed", checkpoint::MULTI_TURN_STREAMED, Some(1), "anthropic_multi_turn_streamed_cut_1");
    #[tokio::test]
    multi_turn_streamed_cut_2: ("checkpoint_matrix/multi_turn_streamed", checkpoint::MULTI_TURN_STREAMED, Some(2), "anthropic_multi_turn_streamed_cut_2");
    #[tokio::test]
    multi_turn_streamed_cut_3: ("checkpoint_matrix/multi_turn_streamed", checkpoint::MULTI_TURN_STREAMED, Some(3), "anthropic_multi_turn_streamed_cut_3");
    #[tokio::test]
    multi_turn_streamed_cut_final: ("checkpoint_matrix/multi_turn_streamed", checkpoint::MULTI_TURN_STREAMED, Some(usize::MAX), "anthropic_multi_turn_streamed_cut_final");
    #[tokio::test]
    parallel_batch: ("checkpoint_matrix/parallel_batch", checkpoint::PARALLEL_BATCH, None, "anthropic_parallel_batch");
    #[tokio::test]
    parallel_batch_cut_final: ("checkpoint_matrix/parallel_batch", checkpoint::PARALLEL_BATCH, Some(usize::MAX), "anthropic_parallel_batch_cut_final");
    #[tokio::test]
    large_result: ("checkpoint_matrix/large_result", checkpoint::LARGE_RESULT, None, "anthropic_large_result");
    #[tokio::test]
    large_result_cut_final: ("checkpoint_matrix/large_result", checkpoint::LARGE_RESULT, Some(usize::MAX), "anthropic_large_result_cut_final");
}

/// Negative matcher evidence only; fresh-world continuation uses the native tests above.
#[tokio::test]
async fn large_result_last_byte_mismatch() {
    checkpoint::assert_large_request_rejected("anthropic", "checkpoint_matrix/large_result").await;
}

/// Negative matcher probe against the same streamed loop used by native consumers.
#[tokio::test]
async fn streamed_tool_result_mismatch() {
    checkpoint::assert_stream_request_rejected(
        "anthropic",
        "checkpoint_matrix/multi_turn_streamed",
    )
    .await;
}
