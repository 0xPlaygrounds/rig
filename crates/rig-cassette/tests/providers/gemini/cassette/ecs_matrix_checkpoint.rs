//! Focused tool-turn checkpoint matrix on Gemini: gemini-2.5-flash-lite.
//! One producer recording is reused by every native cut with strict matching.

use super::super::support::with_gemini_cassette;
use crate::ecs_matrix::{Wire, cells, checkpoint};
use rig::completion::CompletionModel;
use rig::driver::{Bound, Socket};
use rig::providers::gemini::Gemini;

fn wire<H: Socket>(client: &Bound<Gemini, H>) -> Wire<impl CompletionModel + Clone + 'static> {
    Wire {
        thinking: cells::ThinkingWire::Gemini,
        model: client.completion("gemini-2.5-flash-lite"),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
    }
}

crate::matrix::resume_matrix! {
    wrapper: with_gemini_cassette, wire: wire, run: crate::ecs_matrix::checkpoint_world::run_world;
    #[tokio::test]
    multi_turn_unary: ("checkpoint_matrix/multi_turn_unary", checkpoint::MULTI_TURN_UNARY, None, "gemini_multi_turn_unary");
    #[tokio::test]
    multi_turn_unary_cut_1: ("checkpoint_matrix/multi_turn_unary", checkpoint::MULTI_TURN_UNARY, Some(1), "gemini_multi_turn_unary_cut_1");
    #[tokio::test]
    multi_turn_unary_cut_2: ("checkpoint_matrix/multi_turn_unary", checkpoint::MULTI_TURN_UNARY, Some(2), "gemini_multi_turn_unary_cut_2");
    #[tokio::test]
    multi_turn_unary_cut_3: ("checkpoint_matrix/multi_turn_unary", checkpoint::MULTI_TURN_UNARY, Some(3), "gemini_multi_turn_unary_cut_3");
    #[tokio::test]
    multi_turn_unary_cut_final: ("checkpoint_matrix/multi_turn_unary", checkpoint::MULTI_TURN_UNARY, Some(usize::MAX), "gemini_multi_turn_unary_cut_final");
    #[tokio::test]
    multi_turn_streamed: ("checkpoint_matrix/multi_turn_streamed", checkpoint::MULTI_TURN_STREAMED, None, "gemini_multi_turn_streamed");
    #[tokio::test]
    multi_turn_streamed_cut_1: ("checkpoint_matrix/multi_turn_streamed", checkpoint::MULTI_TURN_STREAMED, Some(1), "gemini_multi_turn_streamed_cut_1");
    #[tokio::test]
    multi_turn_streamed_cut_2: ("checkpoint_matrix/multi_turn_streamed", checkpoint::MULTI_TURN_STREAMED, Some(2), "gemini_multi_turn_streamed_cut_2");
    #[tokio::test]
    multi_turn_streamed_cut_3: ("checkpoint_matrix/multi_turn_streamed", checkpoint::MULTI_TURN_STREAMED, Some(3), "gemini_multi_turn_streamed_cut_3");
    #[tokio::test]
    multi_turn_streamed_cut_final: ("checkpoint_matrix/multi_turn_streamed", checkpoint::MULTI_TURN_STREAMED, Some(usize::MAX), "gemini_multi_turn_streamed_cut_final");
    #[tokio::test]
    parallel_batch: ("checkpoint_matrix/parallel_batch", checkpoint::PARALLEL_BATCH, None, "gemini_parallel_batch");
    #[tokio::test]
    parallel_batch_cut_final: ("checkpoint_matrix/parallel_batch", checkpoint::PARALLEL_BATCH, Some(usize::MAX), "gemini_parallel_batch_cut_final");
    #[tokio::test]
    large_result: ("checkpoint_matrix/large_result", checkpoint::LARGE_RESULT, None, "gemini_large_result");
    #[tokio::test]
    large_result_cut_final: ("checkpoint_matrix/large_result", checkpoint::LARGE_RESULT, Some(usize::MAX), "gemini_large_result_cut_final");
}

/// Negative matcher evidence only; fresh-world continuation uses the native tests above.
#[tokio::test]
async fn large_result_last_byte_mismatch() {
    checkpoint::assert_large_request_rejected("gemini", "checkpoint_matrix/large_result").await;
}

/// Negative matcher probe against the same streamed loop used by native consumers.
#[tokio::test]
async fn streamed_tool_result_mismatch() {
    checkpoint::assert_stream_request_rejected("gemini", "checkpoint_matrix/multi_turn_streamed")
        .await;
}
