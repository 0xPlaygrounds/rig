//! Focused tool-turn checkpoint matrix on OpenAiChat: gpt-4.1-mini.
//! One producer recording is reused by every native cut with strict matching.

use super::super::support::{OpenAiCassette, with_openai_cassette};
use crate::ecs_matrix::{Wire, cells, checkpoint};
use rig::completion::CompletionModel;

fn wire(client: &OpenAiCassette) -> Wire<impl CompletionModel + Clone + 'static> {
    Wire {
        thinking: cells::ThinkingWire::OpenAiChat,
        model: client.openai.chat("gpt-4.1-mini"),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
    }
}

crate::matrix::resume_matrix! {
    wrapper: with_openai_cassette, wire: wire, run: crate::ecs_matrix::checkpoint_world::run_world;
    #[tokio::test]
    multi_turn_unary: ("checkpoint_matrix_chat/multi_turn_unary", checkpoint::MULTI_TURN_UNARY, None, "openai_chat_multi_turn_unary");
    #[tokio::test]
    multi_turn_unary_cut_1: ("checkpoint_matrix_chat/multi_turn_unary", checkpoint::MULTI_TURN_UNARY, Some(1), "openai_chat_multi_turn_unary_cut_1");
    #[tokio::test]
    multi_turn_unary_cut_2: ("checkpoint_matrix_chat/multi_turn_unary", checkpoint::MULTI_TURN_UNARY, Some(2), "openai_chat_multi_turn_unary_cut_2");
    #[tokio::test]
    multi_turn_unary_cut_3: ("checkpoint_matrix_chat/multi_turn_unary", checkpoint::MULTI_TURN_UNARY, Some(3), "openai_chat_multi_turn_unary_cut_3");
    #[tokio::test]
    multi_turn_unary_cut_final: ("checkpoint_matrix_chat/multi_turn_unary", checkpoint::MULTI_TURN_UNARY, Some(usize::MAX), "openai_chat_multi_turn_unary_cut_final");
    #[tokio::test]
    multi_turn_streamed: ("checkpoint_matrix_chat/multi_turn_streamed", checkpoint::MULTI_TURN_STREAMED, None, "openai_chat_multi_turn_streamed");
    #[tokio::test]
    multi_turn_streamed_cut_1: ("checkpoint_matrix_chat/multi_turn_streamed", checkpoint::MULTI_TURN_STREAMED, Some(1), "openai_chat_multi_turn_streamed_cut_1");
    #[tokio::test]
    multi_turn_streamed_cut_2: ("checkpoint_matrix_chat/multi_turn_streamed", checkpoint::MULTI_TURN_STREAMED, Some(2), "openai_chat_multi_turn_streamed_cut_2");
    #[tokio::test]
    multi_turn_streamed_cut_3: ("checkpoint_matrix_chat/multi_turn_streamed", checkpoint::MULTI_TURN_STREAMED, Some(3), "openai_chat_multi_turn_streamed_cut_3");
    #[tokio::test]
    multi_turn_streamed_cut_final: ("checkpoint_matrix_chat/multi_turn_streamed", checkpoint::MULTI_TURN_STREAMED, Some(usize::MAX), "openai_chat_multi_turn_streamed_cut_final");
    #[tokio::test]
    parallel_batch: ("checkpoint_matrix_chat/parallel_batch", checkpoint::PARALLEL_BATCH, None, "openai_chat_parallel_batch");
    #[tokio::test]
    parallel_batch_cut_final: ("checkpoint_matrix_chat/parallel_batch", checkpoint::PARALLEL_BATCH, Some(usize::MAX), "openai_chat_parallel_batch_cut_final");
    #[tokio::test]
    large_result: ("checkpoint_matrix_chat/large_result", checkpoint::LARGE_RESULT, None, "openai_chat_large_result");
    #[tokio::test]
    large_result_cut_final: ("checkpoint_matrix_chat/large_result", checkpoint::LARGE_RESULT, Some(usize::MAX), "openai_chat_large_result_cut_final");
}

/// Negative matcher evidence only; fresh-world continuation uses the native tests above.
#[tokio::test]
async fn large_result_last_byte_mismatch() {
    checkpoint::assert_large_request_rejected("openai", "checkpoint_matrix_chat/large_result")
        .await;
}

/// Negative matcher probe against the same streamed loop used by native consumers.
#[tokio::test]
async fn streamed_tool_result_mismatch() {
    checkpoint::assert_stream_request_rejected(
        "openai",
        "checkpoint_matrix_chat/multi_turn_streamed",
    )
    .await;
}
