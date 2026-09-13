//! The image matrix on the Anthropic Messages wire (`claude-sonnet-4-6`, the existing image test's model): the rig-agent producers of the image cells of
//! `tests/common/ecs_matrix/cells.rs` over the shared driver
//! (`tests/common/ecs_matrix/agent.rs`), their logs written as the goldens
//! the world cells (`ecs_matrix_image*.rs`) are compared to. This file holds
//! the scenario literals, the wire's model and the wire's `#[ignore]` reasons.

use rig::completion::CompletionModel;
use rig::prelude::*;
use rig::providers::anthropic::completion::CLAUDE_SONNET_4_6;

use super::super::support::with_anthropic_cassette;
use crate::ecs_matrix::{Wire, agent::run_agent, cells};

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
        run_agent(&wire(&client), &cells::IMAGE_INLINE_TEXT_UNARY, |log| {
            crate::goldens::golden_effects("anthropic_image_inline_text_unary", log)
        })
        .await;
    })
    .await;
}

#[tokio::test]
async fn inline_text_streamed() {
    with_anthropic_cassette("image_matrix/inline_text_streamed", |client| async move {
        run_agent(&wire(&client), &cells::IMAGE_INLINE_TEXT_STREAMED, |log| {
            crate::goldens::golden_effects("anthropic_image_inline_text_streamed", log)
        })
        .await;
    })
    .await;
}

#[tokio::test]
async fn inline_mixed_order() {
    with_anthropic_cassette("image_matrix/inline_mixed_order", |client| async move {
        run_agent(&wire(&client), &cells::IMAGE_INLINE_MIXED_ORDER, |log| {
            crate::goldens::golden_effects("anthropic_image_inline_mixed_order", log)
        })
        .await;
    })
    .await;
}

#[tokio::test]
async fn inline_tool_unary() {
    with_anthropic_cassette("image_matrix/inline_tool_unary", |client| async move {
        run_agent(&wire(&client), &cells::IMAGE_INLINE_TOOL_UNARY, |log| {
            crate::goldens::golden_effects("anthropic_image_inline_tool_unary", log)
        })
        .await;
    })
    .await;
}

#[tokio::test]
async fn inline_tool_streamed() {
    with_anthropic_cassette("image_matrix/inline_tool_streamed", |client| async move {
        run_agent(&wire(&client), &cells::IMAGE_INLINE_TOOL_STREAMED, |log| {
            crate::goldens::golden_effects("anthropic_image_inline_tool_streamed", log)
        })
        .await;
    })
    .await;
}

#[tokio::test]
async fn inline_followup() {
    with_anthropic_cassette("image_matrix/inline_followup", |client| async move {
        run_agent(&wire(&client), &cells::IMAGE_INLINE_FOLLOWUP, |log| {
            crate::goldens::golden_effects("anthropic_image_inline_followup", log)
        })
        .await;
    })
    .await;
}

#[tokio::test]
async fn url_text_unary() {
    with_anthropic_cassette("image_matrix/url_text_unary", |client| async move {
        run_agent(&wire(&client), &cells::IMAGE_URL_TEXT_UNARY, |log| {
            crate::goldens::golden_effects("anthropic_image_url_text_unary", log)
        })
        .await;
    })
    .await;
}

#[tokio::test]
async fn url_tool_unary() {
    with_anthropic_cassette("image_matrix/url_tool_unary", |client| async move {
        run_agent(&wire(&client), &cells::IMAGE_URL_TOOL_UNARY, |log| {
            crate::goldens::golden_effects("anthropic_image_url_tool_unary", log)
        })
        .await;
    })
    .await;
}
