//! Focused tool-turn checkpoint matrix on OpenAiChat: gpt-4.1-mini.
//! One producer recording is reused by every native cut with strict matching.

use super::super::support::with_openai_cassette;
use crate::ecs_matrix::{Wire, cells, checkpoint};
use rig::completion::CompletionModel;
use rig::prelude::*;

fn wire(client: &rig::providers::openai::Client) -> Wire<impl CompletionModel + Clone + 'static> {
    Wire {
        thinking: cells::ThinkingWire::OpenAiChat,
        model: client
            .clone()
            .completions_api()
            .completion_model("gpt-4.1-mini"),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
    }
}

#[tokio::test]
async fn multi_turn_unary() {
    with_openai_cassette(
        "checkpoint_matrix_chat/multi_turn_unary",
        |client| async move {
            let cell = cells::Cell {
                resume_after: None,
                ..checkpoint::MULTI_TURN_UNARY
            };
            crate::ecs_matrix::checkpoint_world::run_world(
                &wire(&client),
                &cell,
                golden_openai_chat_checkpoint_multi_turn_unary,
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn multi_turn_unary_cut_1() {
    with_openai_cassette(
        "checkpoint_matrix_chat/multi_turn_unary",
        |client| async move {
            let cell = cells::Cell {
                resume_after: Some(1),
                ..checkpoint::MULTI_TURN_UNARY
            };
            crate::ecs_matrix::checkpoint_world::run_world(
                &wire(&client),
                &cell,
                golden_openai_chat_checkpoint_multi_turn_unary,
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn multi_turn_unary_cut_2() {
    with_openai_cassette(
        "checkpoint_matrix_chat/multi_turn_unary",
        |client| async move {
            let cell = cells::Cell {
                resume_after: Some(2),
                ..checkpoint::MULTI_TURN_UNARY
            };
            crate::ecs_matrix::checkpoint_world::run_world(
                &wire(&client),
                &cell,
                golden_openai_chat_checkpoint_multi_turn_unary,
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn multi_turn_unary_cut_3() {
    with_openai_cassette(
        "checkpoint_matrix_chat/multi_turn_unary",
        |client| async move {
            let cell = cells::Cell {
                resume_after: Some(3),
                ..checkpoint::MULTI_TURN_UNARY
            };
            crate::ecs_matrix::checkpoint_world::run_world(
                &wire(&client),
                &cell,
                golden_openai_chat_checkpoint_multi_turn_unary,
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn multi_turn_unary_cut_final() {
    with_openai_cassette(
        "checkpoint_matrix_chat/multi_turn_unary",
        |client| async move {
            let cell = cells::Cell {
                resume_after: Some(usize::MAX),
                ..checkpoint::MULTI_TURN_UNARY
            };
            crate::ecs_matrix::checkpoint_world::run_world(
                &wire(&client),
                &cell,
                golden_openai_chat_checkpoint_multi_turn_unary,
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn multi_turn_streamed() {
    with_openai_cassette(
        "checkpoint_matrix_chat/multi_turn_streamed",
        |client| async move {
            let cell = cells::Cell {
                resume_after: None,
                ..checkpoint::MULTI_TURN_STREAMED
            };
            crate::ecs_matrix::checkpoint_world::run_world(
                &wire(&client),
                &cell,
                golden_openai_chat_checkpoint_multi_turn_streamed,
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn multi_turn_streamed_cut_1() {
    with_openai_cassette(
        "checkpoint_matrix_chat/multi_turn_streamed",
        |client| async move {
            let cell = cells::Cell {
                resume_after: Some(1),
                ..checkpoint::MULTI_TURN_STREAMED
            };
            crate::ecs_matrix::checkpoint_world::run_world(
                &wire(&client),
                &cell,
                golden_openai_chat_checkpoint_multi_turn_streamed,
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn multi_turn_streamed_cut_2() {
    with_openai_cassette(
        "checkpoint_matrix_chat/multi_turn_streamed",
        |client| async move {
            let cell = cells::Cell {
                resume_after: Some(2),
                ..checkpoint::MULTI_TURN_STREAMED
            };
            crate::ecs_matrix::checkpoint_world::run_world(
                &wire(&client),
                &cell,
                golden_openai_chat_checkpoint_multi_turn_streamed,
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn multi_turn_streamed_cut_3() {
    with_openai_cassette(
        "checkpoint_matrix_chat/multi_turn_streamed",
        |client| async move {
            let cell = cells::Cell {
                resume_after: Some(3),
                ..checkpoint::MULTI_TURN_STREAMED
            };
            crate::ecs_matrix::checkpoint_world::run_world(
                &wire(&client),
                &cell,
                golden_openai_chat_checkpoint_multi_turn_streamed,
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn multi_turn_streamed_cut_final() {
    with_openai_cassette(
        "checkpoint_matrix_chat/multi_turn_streamed",
        |client| async move {
            let cell = cells::Cell {
                resume_after: Some(usize::MAX),
                ..checkpoint::MULTI_TURN_STREAMED
            };
            crate::ecs_matrix::checkpoint_world::run_world(
                &wire(&client),
                &cell,
                golden_openai_chat_checkpoint_multi_turn_streamed,
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn parallel_batch() {
    with_openai_cassette(
        "checkpoint_matrix_chat/parallel_batch",
        |client| async move {
            let cell = cells::Cell {
                resume_after: None,
                ..checkpoint::PARALLEL_BATCH
            };
            crate::ecs_matrix::checkpoint_world::run_world(
                &wire(&client),
                &cell,
                golden_openai_chat_checkpoint_parallel_batch,
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn parallel_batch_cut_final() {
    with_openai_cassette(
        "checkpoint_matrix_chat/parallel_batch",
        |client| async move {
            let cell = cells::Cell {
                resume_after: Some(usize::MAX),
                ..checkpoint::PARALLEL_BATCH
            };
            crate::ecs_matrix::checkpoint_world::run_world(
                &wire(&client),
                &cell,
                golden_openai_chat_checkpoint_parallel_batch,
            )
            .await;
        },
    )
    .await;
}

#[tokio::test]
async fn large_result() {
    with_openai_cassette("checkpoint_matrix_chat/large_result", |client| async move {
        let cell = cells::Cell {
            resume_after: None,
            ..checkpoint::LARGE_RESULT
        };
        crate::ecs_matrix::checkpoint_world::run_world(
            &wire(&client),
            &cell,
            golden_openai_chat_checkpoint_large_result,
        )
        .await;
    })
    .await;
}

#[tokio::test]
async fn large_result_cut_final() {
    with_openai_cassette("checkpoint_matrix_chat/large_result", |client| async move {
        let cell = cells::Cell {
            resume_after: Some(usize::MAX),
            ..checkpoint::LARGE_RESULT
        };
        crate::ecs_matrix::checkpoint_world::run_world(
            &wire(&client),
            &cell,
            golden_openai_chat_checkpoint_large_result,
        )
        .await;
    })
    .await;
}

fn golden_openai_chat_checkpoint_multi_turn_unary(log: &rig::effect_log::EffectLog) {
    crate::ecs_goldens::golden_effects("openai_chat_checkpoint_multi_turn_unary", log);
}

fn golden_openai_chat_checkpoint_multi_turn_streamed(log: &rig::effect_log::EffectLog) {
    crate::ecs_goldens::golden_effects("openai_chat_checkpoint_multi_turn_streamed", log);
}

fn golden_openai_chat_checkpoint_parallel_batch(log: &rig::effect_log::EffectLog) {
    crate::ecs_goldens::golden_effects("openai_chat_checkpoint_parallel_batch", log);
}

fn golden_openai_chat_checkpoint_large_result(log: &rig::effect_log::EffectLog) {
    crate::ecs_goldens::golden_effects("openai_chat_checkpoint_large_result", log);
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
