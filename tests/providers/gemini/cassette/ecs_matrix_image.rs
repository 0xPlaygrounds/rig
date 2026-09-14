//! The image matrix on the Gemini REST wire (`gemini-3-flash-preview`): the image cells of
//! `tests/common/ecs_matrix/cells.rs` as agent graphs in a Bevy `World`,
//! served by the real adapter over the same recording as their producers in
//! `corpus_matrix_image*.rs` and asserted against those producers' goldens,
//! then by their history, their live-resumed cut and their despawn (the
//! driver is `tests/common/ecs_matrix/world.rs`). This file holds the
//! scenario literals, the wire's model and the wire's `#[ignore]` reasons.

use rig::completion::CompletionModel;
use rig::prelude::*;
use rig::providers::gemini::completion::GEMINI_3_FLASH_PREVIEW;

use super::super::support::with_gemini_cassette;
use crate::ecs_matrix::{Wire, cells, world::run_world};

fn wire(client: &rig::providers::gemini::Client) -> Wire<impl CompletionModel + Clone + 'static> {
    Wire {
        thinking: cells::ThinkingWire::Gemini,
        model: client.completion_model(GEMINI_3_FLASH_PREVIEW),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
    }
}

#[tokio::test]
async fn inline_text_unary() {
    with_gemini_cassette("image_matrix/inline_text_unary", |client| async move {
        run_world(&wire(&client), &cells::IMAGE_INLINE_TEXT_UNARY, |log| {
            crate::ecs_goldens::golden_effects("gemini_image_inline_text_unary", log)
        })
        .await;
    })
    .await;
}

#[tokio::test]
async fn inline_text_streamed() {
    with_gemini_cassette("image_matrix/inline_text_streamed", |client| async move {
        run_world(&wire(&client), &cells::IMAGE_INLINE_TEXT_STREAMED, |log| {
            crate::ecs_goldens::golden_effects("gemini_image_inline_text_streamed", log)
        })
        .await;
    })
    .await;
}

#[tokio::test]
async fn inline_mixed_order() {
    with_gemini_cassette("image_matrix/inline_mixed_order", |client| async move {
        run_world(&wire(&client), &cells::IMAGE_INLINE_MIXED_ORDER, |log| {
            crate::ecs_goldens::golden_effects("gemini_image_inline_mixed_order", log)
        })
        .await;
    })
    .await;
}

#[tokio::test]
async fn inline_tool_unary() {
    with_gemini_cassette("image_matrix/inline_tool_unary", |client| async move {
        run_world(&wire(&client), &cells::IMAGE_INLINE_TOOL_UNARY, |log| {
            crate::ecs_goldens::golden_effects("gemini_image_inline_tool_unary", log)
        })
        .await;
    })
    .await;
}

#[tokio::test]
async fn inline_tool_streamed() {
    with_gemini_cassette("image_matrix/inline_tool_streamed", |client| async move {
        run_world(&wire(&client), &cells::IMAGE_INLINE_TOOL_STREAMED, |log| {
            crate::ecs_goldens::golden_effects("gemini_image_inline_tool_streamed", log)
        })
        .await;
    })
    .await;
}

#[tokio::test]
async fn inline_followup() {
    with_gemini_cassette("image_matrix/inline_followup", |client| async move {
        run_world(&wire(&client), &cells::IMAGE_INLINE_FOLLOWUP, |log| {
            crate::ecs_goldens::golden_effects("gemini_image_inline_followup", log)
        })
        .await;
    })
    .await;
}

#[tokio::test]
#[ignore = "Gemini's `fileData.fileUri` takes Files API and Cloud Storage URIs, not an arbitrary HTTPS image (image-understanding docs, retrieved 2026-09-13); rig renders `DocumentSourceKind::Url` as `fileData`, so the HTTPS row is documented-unsupported on this wire and unrecorded"]
async fn url_text_unary() {
    with_gemini_cassette("image_matrix/url_text_unary", |client| async move {
        run_world(&wire(&client), &cells::IMAGE_URL_TEXT_UNARY, |_| {
            panic!("unrecorded image scenario: see this test's ignore disposition")
        })
        .await;
    })
    .await;
}

#[tokio::test]
#[ignore = "Gemini's `fileData.fileUri` takes Files API and Cloud Storage URIs, not an arbitrary HTTPS image (image-understanding docs, retrieved 2026-09-13); rig renders `DocumentSourceKind::Url` as `fileData`, so the HTTPS row is documented-unsupported on this wire and unrecorded"]
async fn url_tool_unary() {
    with_gemini_cassette("image_matrix/url_tool_unary", |client| async move {
        run_world(&wire(&client), &cells::IMAGE_URL_TOOL_UNARY, |_| {
            panic!("unrecorded image scenario: see this test's ignore disposition")
        })
        .await;
    })
    .await;
}
