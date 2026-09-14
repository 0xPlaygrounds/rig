//! Focused tool-turn checkpoint matrix on Gemini: gemini-2.5-flash-lite.
//! One producer recording is reused by every native cut with strict matching.

use super::super::support::with_gemini_cassette;
use crate::ecs_matrix::{Wire, cells, checkpoint};
use rig::completion::CompletionModel;
use rig::prelude::*;

fn wire(client: &rig::providers::gemini::Client) -> Wire<impl CompletionModel + Clone + 'static> {
    Wire {
        thinking: cells::ThinkingWire::Gemini,
        model: client.completion_model("gemini-2.5-flash-lite"),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
    }
}

crate::matrix::resume_matrix! {
    wrapper: with_gemini_cassette, wire: wire, run: crate::ecs_matrix::checkpoint_world::run_world;
    #[tokio::test]
    multi_turn_unary: ("checkpoint_matrix/multi_turn_unary", checkpoint::MULTI_TURN_UNARY, None, golden_gemini_checkpoint_multi_turn_unary);
    #[tokio::test]
    multi_turn_unary_cut_1: ("checkpoint_matrix/multi_turn_unary", checkpoint::MULTI_TURN_UNARY, Some(1), golden_gemini_checkpoint_multi_turn_unary);
    #[tokio::test]
    multi_turn_unary_cut_2: ("checkpoint_matrix/multi_turn_unary", checkpoint::MULTI_TURN_UNARY, Some(2), golden_gemini_checkpoint_multi_turn_unary);
    #[tokio::test]
    multi_turn_unary_cut_3: ("checkpoint_matrix/multi_turn_unary", checkpoint::MULTI_TURN_UNARY, Some(3), golden_gemini_checkpoint_multi_turn_unary);
    #[tokio::test]
    multi_turn_unary_cut_final: ("checkpoint_matrix/multi_turn_unary", checkpoint::MULTI_TURN_UNARY, Some(usize::MAX), golden_gemini_checkpoint_multi_turn_unary);
    #[tokio::test]
    multi_turn_streamed: ("checkpoint_matrix/multi_turn_streamed", checkpoint::MULTI_TURN_STREAMED, None, golden_gemini_checkpoint_multi_turn_streamed);
    #[tokio::test]
    multi_turn_streamed_cut_1: ("checkpoint_matrix/multi_turn_streamed", checkpoint::MULTI_TURN_STREAMED, Some(1), golden_gemini_checkpoint_multi_turn_streamed);
    #[tokio::test]
    multi_turn_streamed_cut_2: ("checkpoint_matrix/multi_turn_streamed", checkpoint::MULTI_TURN_STREAMED, Some(2), golden_gemini_checkpoint_multi_turn_streamed);
    #[tokio::test]
    multi_turn_streamed_cut_3: ("checkpoint_matrix/multi_turn_streamed", checkpoint::MULTI_TURN_STREAMED, Some(3), golden_gemini_checkpoint_multi_turn_streamed);
    #[tokio::test]
    multi_turn_streamed_cut_final: ("checkpoint_matrix/multi_turn_streamed", checkpoint::MULTI_TURN_STREAMED, Some(usize::MAX), golden_gemini_checkpoint_multi_turn_streamed);
    #[tokio::test]
    parallel_batch: ("checkpoint_matrix/parallel_batch", checkpoint::PARALLEL_BATCH, None, golden_gemini_checkpoint_parallel_batch);
    #[tokio::test]
    parallel_batch_cut_final: ("checkpoint_matrix/parallel_batch", checkpoint::PARALLEL_BATCH, Some(usize::MAX), golden_gemini_checkpoint_parallel_batch);
    #[tokio::test]
    large_result: ("checkpoint_matrix/large_result", checkpoint::LARGE_RESULT, None, golden_gemini_checkpoint_large_result);
    #[tokio::test]
    large_result_cut_final: ("checkpoint_matrix/large_result", checkpoint::LARGE_RESULT, Some(usize::MAX), golden_gemini_checkpoint_large_result);
}

fn golden_gemini_checkpoint_multi_turn_unary(log: &rig::effect_log::EffectLog) {
    crate::ecs_goldens::golden_effects("gemini_checkpoint_multi_turn_unary", log);
}

fn golden_gemini_checkpoint_multi_turn_streamed(log: &rig::effect_log::EffectLog) {
    crate::ecs_goldens::golden_effects("gemini_checkpoint_multi_turn_streamed", log);
}

fn golden_gemini_checkpoint_parallel_batch(log: &rig::effect_log::EffectLog) {
    crate::ecs_goldens::golden_effects("gemini_checkpoint_parallel_batch", log);
}

fn golden_gemini_checkpoint_large_result(log: &rig::effect_log::EffectLog) {
    crate::ecs_goldens::golden_effects("gemini_checkpoint_large_result", log);
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
