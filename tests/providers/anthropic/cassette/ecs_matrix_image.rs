//! The image matrix on the Anthropic Messages wire (`claude-sonnet-4-6`, the existing image test's model): the image cells of
//! `tests/common/ecs_matrix/cells.rs` as agent graphs in a Bevy `World`,
//! served by the real adapter over the same recording as their producers in
//! `corpus_matrix_image*.rs` and asserted against those producers' goldens,
//! then by their history, their live-resumed cut and their despawn (the
//! driver is `tests/common/ecs_matrix/world.rs`). This file holds the
//! scenario literals, the wire's model and the wire's `#[ignore]` reasons.

use rig::completion::CompletionModel;
use rig::prelude::*;
use rig::providers::anthropic::completion::CLAUDE_SONNET_4_6;

use super::super::support::with_anthropic_cassette;
use crate::ecs_matrix::{Wire, cells, world::run_world};

fn wire(
    client: &rig::providers::anthropic::Client,
) -> Wire<impl CompletionModel + Clone + 'static> {
    Wire {
        thinking: cells::ThinkingWire::Anthropic,
        model: client.completion_model(CLAUDE_SONNET_4_6),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
    }
}

#[tokio::test]
async fn inline_text_unary() {
    with_anthropic_cassette("image_matrix/inline_text_unary", |client| async move {
        run_world(&wire(&client), &cells::IMAGE_INLINE_TEXT_UNARY, |log| {
            crate::ecs_goldens::golden_effects("anthropic_image_inline_text_unary", log)
        })
        .await;
    })
    .await;
}

#[tokio::test]
async fn inline_text_streamed() {
    with_anthropic_cassette("image_matrix/inline_text_streamed", |client| async move {
        run_world(&wire(&client), &cells::IMAGE_INLINE_TEXT_STREAMED, |log| {
            crate::ecs_goldens::golden_effects("anthropic_image_inline_text_streamed", log)
        })
        .await;
    })
    .await;
}

#[tokio::test]
async fn inline_mixed_order() {
    with_anthropic_cassette("image_matrix/inline_mixed_order", |client| async move {
        run_world(&wire(&client), &cells::IMAGE_INLINE_MIXED_ORDER, |log| {
            crate::ecs_goldens::golden_effects("anthropic_image_inline_mixed_order", log)
        })
        .await;
    })
    .await;
}

#[tokio::test]
async fn inline_tool_unary() {
    with_anthropic_cassette("image_matrix/inline_tool_unary", |client| async move {
        run_world(&wire(&client), &cells::IMAGE_INLINE_TOOL_UNARY, |log| {
            crate::ecs_goldens::golden_effects("anthropic_image_inline_tool_unary", log)
        })
        .await;
    })
    .await;
}

#[tokio::test]
async fn inline_tool_streamed() {
    with_anthropic_cassette("image_matrix/inline_tool_streamed", |client| async move {
        run_world(&wire(&client), &cells::IMAGE_INLINE_TOOL_STREAMED, |log| {
            crate::ecs_goldens::golden_effects("anthropic_image_inline_tool_streamed", log)
        })
        .await;
    })
    .await;
}

#[tokio::test]
async fn inline_followup() {
    with_anthropic_cassette("image_matrix/inline_followup", |client| async move {
        run_world(&wire(&client), &cells::IMAGE_INLINE_FOLLOWUP, |log| {
            crate::ecs_goldens::golden_effects("anthropic_image_inline_followup", log)
        })
        .await;
    })
    .await;
}

#[tokio::test]
async fn url_text_unary() {
    with_anthropic_cassette("image_matrix/url_text_unary", |client| async move {
        run_world(&wire(&client), &cells::IMAGE_URL_TEXT_UNARY, |log| {
            crate::ecs_goldens::golden_effects("anthropic_image_url_text_unary", log)
        })
        .await;
    })
    .await;
}

#[tokio::test]
async fn url_tool_unary() {
    with_anthropic_cassette("image_matrix/url_tool_unary", |client| async move {
        run_world(&wire(&client), &cells::IMAGE_URL_TOOL_UNARY, |log| {
            crate::ecs_goldens::golden_effects("anthropic_image_url_tool_unary", log)
        })
        .await;
    })
    .await;
}
