//! The image matrix on the Gemini REST wire (`gemini-3-flash-preview`): the rig-agent producers of the image cells of
//! `tests/common/ecs_matrix/cells.rs` over the shared driver
//! (`tests/common/ecs_matrix/agent.rs`), their logs written as the goldens
//! the world cells (`ecs_matrix_image*.rs`) are compared to. This file holds
//! the scenario literals, the wire's model and the wire's `#[ignore]` reasons.

use rig::providers::gemini::completion::GEMINI_3_FLASH_PREVIEW;
use rig_test_support::cassette_models::GeminiModels;

use super::super::support::with_gemini_cassette;
use crate::ecs_matrix::{Wire, agent::run_agent, cells};

fn wire(
    client: &GeminiModels,
) -> Wire<
    rig::Model<
        rig::providers::gemini::completion::GenerateContent,
        rig::http_client::DynHttpClient,
    >,
> {
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
    inline_text_streamed: ("image_matrix/inline_text_streamed", cells::IMAGE_INLINE_TEXT_STREAMED, "gemini_image_inline_text_streamed");
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
