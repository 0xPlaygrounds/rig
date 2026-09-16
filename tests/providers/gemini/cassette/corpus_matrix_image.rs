//! The image matrix on the Gemini REST wire (`gemini-3-flash-preview`): the rig-agent producers of the image cells of
//! `tests/common/ecs_matrix/cells.rs` over the shared driver
//! (`tests/common/ecs_matrix/agent.rs`), their logs written as the goldens
//! the world cells (`ecs_matrix_image*.rs`) are compared to. This file holds
//! the scenario literals, the wire's model and the wire's `#[ignore]` reasons.

use rig::completion::CompletionModel;
use rig::driver::{Bound, Socket};
use rig::providers::gemini::Gemini;
use rig::providers::gemini::completion::GEMINI_3_FLASH_PREVIEW;

use super::super::support::with_gemini_cassette;
use crate::ecs_matrix::{Wire, agent::run_agent, cells};

fn wire<H: Socket>(client: &Bound<Gemini, H>) -> Wire<impl CompletionModel + Clone + 'static> {
    Wire {
        thinking: cells::ThinkingWire::Gemini,
        model: client.completion(GEMINI_3_FLASH_PREVIEW),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
    }
}

crate::matrix::golden_matrix! {
    wrapper: with_gemini_cassette, wire: wire, run: run_agent, oracle: crate::goldens::golden_effects;
    #[tokio::test]
    inline_text_unary: ("image_matrix/inline_text_unary", cells::IMAGE_INLINE_TEXT_UNARY, "gemini_image_inline_text_unary");
    #[tokio::test]
    inline_text_streamed: ("image_matrix/inline_text_streamed", cells::IMAGE_INLINE_TEXT_STREAMED, "gemini_image_inline_text_streamed");
    #[tokio::test]
    inline_mixed_order: ("image_matrix/inline_mixed_order", cells::IMAGE_INLINE_MIXED_ORDER, "gemini_image_inline_mixed_order");
    #[tokio::test]
    inline_tool_unary: ("image_matrix/inline_tool_unary", cells::IMAGE_INLINE_TOOL_UNARY, "gemini_image_inline_tool_unary");
    #[tokio::test]
    inline_tool_streamed: ("image_matrix/inline_tool_streamed", cells::IMAGE_INLINE_TOOL_STREAMED, "gemini_image_inline_tool_streamed");
    #[tokio::test]
    inline_followup: ("image_matrix/inline_followup", cells::IMAGE_INLINE_FOLLOWUP, "gemini_image_inline_followup");
}

#[tokio::test]
#[ignore = "Gemini's `fileData.fileUri` takes Files API and Cloud Storage URIs, not an arbitrary HTTPS image (image-understanding docs, retrieved 2026-09-13); rig renders `DocumentSourceKind::Url` as `fileData`, so the HTTPS row is documented-unsupported on this wire and unrecorded"]
async fn url_text_unary() {
    with_gemini_cassette("image_matrix/url_text_unary", |client| async move {
        run_agent(&wire(&client), &cells::IMAGE_URL_TEXT_UNARY, |_| {
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
        run_agent(&wire(&client), &cells::IMAGE_URL_TOOL_UNARY, |_| {
            panic!("unrecorded image scenario: see this test's ignore disposition")
        })
        .await;
    })
    .await;
}
