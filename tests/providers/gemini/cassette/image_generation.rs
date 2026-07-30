//! Gemini image generation cassette tests.

use rig::image_generation::ImageGenerationRequest;
use rig::providers::gemini;

#[tokio::test]
async fn nano_banana_image_generation_smoke() {
    super::super::support::with_gemini_cassette(
        "image_generation/nano_banana_image_generation_smoke",
        |client| async move {
            let response = gemini::functions::generate_image(
                &client.config(gemini::GEMINI_2_5_FLASH_IMAGE),
                &client.http(),
                ImageGenerationRequest::new(
                    "Generate a simple flat icon of a yellow banana on a white background.",
                )
                .with_width(256)
                .with_height(256),
            )
            .await
            .expect("Nano Banana image generation should succeed");

            assert!(
                response.image.len() > 100,
                "expected non-empty generated image bytes"
            );
            assert_eq!(
                response.response.model_version.as_deref(),
                Some(gemini::GEMINI_2_5_FLASH_IMAGE),
                "expected Gemini response to identify the image model"
            );
        },
    )
    .await;
}
